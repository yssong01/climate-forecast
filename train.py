"""
train.py — Phase 3-4: 3축 Tri-CHEF 학습 (Gram-Schmidt 직교화 + 학습 안정화)
논문 Eq.1: s = √(A² + (α·B)² + (φ·C)²)

Re 축 (A): 위성 이미지 → ResNet18 4채널
Im 축 (B): 기상 문서   → MiniLM 384 프로젝션
Z  축 (C): 수치 센서   → MLP 인코더

Phase 3-4 수정 사항
  1. Gram-Schmidt 직교화 (ORTHOGONALIZE 플래그로 ablation)
  2. dying-clamp 버그 수정 (pipeline_model: clamp → softplus)
  3. 손실 스케일 정규화 — 기온 MSE 가 강수 MSE 를 1000배 압도하던 문제
  4. 축 직교성·강수 예측 분산 진단 로그
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from weather_collector import RobustWeatherCollector, STATIONS, STATION_COORDS, KST
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from pipeline_model import TriCHEFPipeline

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM  = 64
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-3
ALPHA_INIT = 0.4    # Im 축(텍스트) 가중치 초기값 — 학습 중 자동 조정
PHI_INIT   = 0.2    # Z  축(수치)  가중치 초기값 — 학습 중 자동 조정
N_HOURS    = 720    # 관측소당 수집 기간: 30일 × 24시간 (장마철 포함 확보)
PATIENCE   = 20     # Early stopping
VAL_RATIO  = 0.2

# 데이터 확장 — 서울 단일 관측소(표본 572개)는 이번 세션 대부분의 이상 현상
# (위성 인코더 과적합, 게이트 왜곡)의 근본 원인이었다. 12개 관측소 전체로
# 확장해 표본을 약 15배(≈8,600개)로 늘린다.
STATIONS_TO_COLLECT = list(STATIONS.values())          # 12개 관측소 전체
STATION_NAMES = {v: k for k, v in STATIONS.items()}    # 코드 → 한글명

# 예보 시계 — t 시각 관측으로 t+LEAD_HOURS 를 예측한다.
# 1시간에서는 퍼시스턴스(T(t+1)≈T(t))가 사실상 무적이라 모델 기여를 측정할
# 수 없었다(실측: 기온 +0.7%). 6시간이면 일주기가 지배적이 되어 시각 특성과
# 기압 추세를 학습한 모델이 baseline 을 이길 여지가 생긴다.
LEAD_HOURS = 6

# Gram-Schmidt 직교화 — ablation.py 실측 결과 기본값 False.
#   ON  : 기온 0.8886 / 강수 0.3267
#   OFF : 기온 0.8452 / 강수 0.2889   ← 우세
# OFF 조건에서 네트워크가 학습한 Re-Im 상관도가 0.4809(ON은 0.0846)로,
# 축 간 공유 정보가 회귀 성능에 기여함을 뜻한다. 검색 태스크(논문 원본)와
# 달리 예보 태스크에서는 직교성 강제가 손해다.
ORTHOGONALIZE = False
PERSISTENCE_RESIDUAL = True   # 기온을 delta 예측으로 전환 (아래 설명 참조)

# Phase 3-5 동적 게이팅 — 입력마다 축 가중치를 다르게 산출한다.
DYNAMIC_GATE  = True
# 게이트 엔트로피 정규화 계수. 가중치가 소수 축에 집중되도록 압력을 준다.
GATE_ENTROPY_WEIGHT = 0.05
# 엔트로피 항 워밍업 에폭. 이 구간에서는 λ=0 으로 두고 이후 선형 증가시킨다.
# 에폭 1부터 집중 압력을 걸면 모델이 어느 축이 유용한지 배우기 전에 초기
# 순서가 고착된다(Phase 3-5 1차 실측: 위성 축 98% 로 붕괴). 먼저 배우게 한
# 뒤 집중시킨다.
GATE_WARMUP_EPOCHS = 50
# 위성 인코더 용량. ResNet18(1,100만)은 표본 572개를 암기해 게이트를 왜곡시킨다.
COMPACT_SATELLITE = True
PRECIP_WEIGHT = 1.0    # 강수 손실 가중치 (기온 손실은 σ² 로 정규화되어 O(1))
SEED          = 42     # 학습/검증 분할 고정 → ablation 비교 가능

CHECKPOINT  = "./checkpoints/numerical_trichef.pt"
DATA_CACHE  = "./cache/historical_data.json"
os.makedirs("./checkpoints", exist_ok=True)


# ── 1. 수치 벡터 변환 ─────────────────────────────────────────────

def record_to_vec(r: dict) -> list[float]:
    """
    관측 레코드 → 12차원 수치 벡터.

    인덱스 8·9(시각 sin/cos)는 Phase 3-4 에서 추가했다. 기온의 일주기
    (diurnal cycle)는 기온 예측의 지배적 요인인데 기존 8차원에는 시각
    정보가 전혀 없어 모델이 낮/밤을 구분할 수 없었다.
    인덱스 10·11(위도·경도)은 다중 관측소 확장(1단계)에서 추가했다.
    관측소마다 기후가 다르므로(강릉의 동해 영향, 제주의 해양성 기후 등)
    지역 좌표 없이는 하나의 모델이 12개 관측소를 구분할 수 없다.
    인덱스 0(기온)은 퍼시스턴스 잔차 계산에 쓰이므로 위치를 바꾸지 말 것.
    """
    wd_rad = np.deg2rad(r.get("wind_dir", 0.0))

    ts = str(r.get("timestamp", ""))            # YYYYMMDDHHmm
    hour = int(ts[8:10]) if len(ts) >= 10 else 0
    h_rad = 2.0 * np.pi * hour / 24.0

    lat, lon = STATION_COORDS.get(str(r.get("stn", "108")), (36.5, 127.5))

    return [
        r.get("temperature",   20.0),   # 0: 기온      ← 퍼시스턴스 기준
        r.get("precipitation",  0.0),   # 1: 강수량
        r.get("humidity",      50.0),   # 2: 습도
        r.get("wind_speed",     1.5),   # 3: 풍속
        float(np.sin(wd_rad)),          # 4: 풍향 sin
        float(np.cos(wd_rad)),          # 5: 풍향 cos
        r.get("pressure",    1013.0),   # 6: 기압
        float(r.get("precip_type", 0)), # 7: 강수형태
        float(np.sin(h_rad)),           # 8: 시각 sin  ← 일주기
        float(np.cos(h_rad)),           # 9: 시각 cos  ← 일주기
        lat,                            # 10: 위도     ← 관측소 위치
        lon,                            # 11: 경도     ← 관측소 위치
    ]


NUM_FEATURES = 12   # record_to_vec 출력 차원


# ── 2. 과거 데이터 수집 ───────────────────────────────────────────

def collect_historical(n_hours: int = N_HOURS,
                       stations: list = None) -> list[dict]:
    """
    ASOS API로 각 관측소별 최근 n_hours 시간의 관측 데이터 수집.
    관측소를 순회하며 관측소 블록 단위로 이어붙인다(블록 내부는 시간순).
    """
    stations = stations or STATIONS_TO_COLLECT
    target_total = n_hours * len(stations)

    # 캐시 재사용 (API 호출 절약) — 레코드 수뿐 아니라 관측소 구성도 검증한다.
    # 이전 세션의 단일 관측소 캐시(720개)는 여기서 자동으로 부족 판정되어
    # 재수집이 트리거된다.
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_stations = {r.get("stn") for r in cached}
        if len(cached) >= int(target_total * 0.9) and cached_stations >= set(stations):
            print(f"[캐시] {len(cached)}개 로드 완료 "
                  f"({len(cached_stations)}개 관측소, API 절약)")
            return cached
        print(f"[캐시] {len(cached)}개 ({len(cached_stations)}개 관측소) "
              f"— 목표({target_total}개, {len(stations)}개 관측소) 부족, 재수집")

    all_records = []
    # ASOS 시각은 KST 기준 — 컨테이너 기본 시간대(UTC)에 의존하지 않도록
    # 명시한다 (weather_collector.KST 참조). 여기서 datetime.now()를 그대로
    # 쓰면 Docker에서 수집 창 전체가 9시간 밀려 어긋난다.
    now = datetime.now(KST)

    print(f"ASOS 과거 {n_hours}시간 × {len(stations)}개 관측소 수집 시작 "
          f"(예상 {target_total}개, 약 {target_total * 0.3 / 60:.0f}분 소요)...")

    for si, stn in enumerate(stations, 1):
        stn_name = STATION_NAMES.get(stn, stn)
        collector = RobustWeatherCollector(stn=stn)
        records = []

        for i in range(n_hours, 0, -1):
            dt = now - timedelta(hours=i)
            tm = dt.strftime("%Y%m%d%H00")
            data = collector.fetch_at(tm)

            if data.get("status") == "SUCCESS_LIVE":
                records.append(data)

            time.sleep(0.3)  # API 과부하 방지

        print(f"  [{si:2d}/{len(stations)}] {stn_name}({stn}): "
              f"{len(records)}/{n_hours}개 성공")
        all_records.extend(records)

    print(f"\n전체 수집 완료: {len(all_records)}/{target_total}개 "
          f"({len(stations)}개 관측소)\n")

    with open(DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return all_records


# ── 3. 데이터셋 ───────────────────────────────────────────────────

def _parse_ts(ts) -> datetime:
    """YYYYMMDDHHmm 문자열 → datetime."""
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


class WeatherDataset(Dataset):
    """
    입력 X_num: 시각 t 의 12차원 수치 벡터 (표준화)
    입력 X_img: 시각 t 의 합성 위성 이미지 (4, 32, 32)
    입력 X_txt: 시각 t 의 MiniLM 텍스트 임베딩 (384)
    타깃 y    : [기온_{t+L}, 강수량_{t+L}]      L = lead_hours

    수집 실패로 시각이 건너뛴 구간이 있으면 (t, t+L) 쌍의 실제 간격이 L 이
    아닐 수 있다. 타임스탬프를 검사해 정확히 L 시간 떨어진 쌍만 사용한다.

    다중 관측소 확장(1단계) 이후 records 는 관측소 블록을 이어붙인 리스트다
    (서울 720개, 인천 720개, ...). 블록 경계에서는 시각 간격 조건만으로는
    걸러지지 않는 오염 쌍이 생길 수 있다 — 각 관측소가 "지금부터 n_hours 전"
    까지 동일한 실시간 구간을 수집하므로, 한 관측소 블록의 마지막 레코드와
    다음 관측소 블록의 초반 레코드가 우연히 L시간 간격을 만족할 수 있다.
    관측소 코드가 같은 쌍만 사용해 이를 원천 차단한다.
    """

    def __init__(self, records: list[dict],
                 sat_collector: SimulatedSatelliteCollector = None,
                 txt_collector: SimulatedTextCollector = None,
                 lead_hours: int = 1,
                 mean: np.ndarray = None, std: np.ndarray = None):
        L = lead_hours
        self.lead_hours = L

        # ── 유효한 (t, t+L) 쌍만 선별 — 동일 관측소 + 정확히 L시간 간격 ──
        times = [_parse_ts(r["timestamp"]) for r in records]
        stns  = [r.get("stn") for r in records]
        src_idx = [i for i in range(len(records) - L)
                   if stns[i] == stns[i + L]
                   and (times[i + L] - times[i]).total_seconds() == L * 3600]
        if not src_idx:
            raise ValueError(f"lead={L}시간 유효 쌍이 없습니다. 데이터 연속성 확인 필요.")
        dropped = (len(records) - L) - len(src_idx)
        if dropped > 0:
            print(f"  [경고] 관측소 경계·시각 불연속으로 {dropped}개 쌍 제외 "
                  f"(사용 {len(src_idx)}개)")

        src_records = [records[i] for i in src_idx]
        tgt_records = [records[i + L] for i in src_idx]

        # ── Z축: 수치 특성 ───────────────────────────────────────
        vecs = np.array([record_to_vec(r) for r in src_records], dtype=np.float32)
        if mean is None:
            self.mean = vecs.mean(axis=0)
            self.std  = np.where(vecs.std(axis=0) > 1e-6, vecs.std(axis=0), 1.0)
        else:
            self.mean, self.std = mean, std
        self.X_num = torch.tensor((vecs - self.mean) / self.std, dtype=torch.float32)

        # ── Re축 / Im축 ──────────────────────────────────────────
        self.X_img = (torch.tensor(sat_collector.get_batch(src_records),
                                   dtype=torch.float32)
                      if sat_collector is not None else None)
        self.X_txt = (torch.tensor(txt_collector.get_batch(src_records),
                                   dtype=torch.float32)
                      if txt_collector is not None else None)

        # ── 타깃 ─────────────────────────────────────────────────
        temp_tgt   = np.array([r["temperature"]   for r in tgt_records], dtype=np.float32)
        precip_tgt = np.array([r["precipitation"] for r in tgt_records], dtype=np.float32)
        self.y = torch.tensor(np.stack([temp_tgt, precip_tgt], axis=1),
                              dtype=torch.float32)

        # 퍼시스턴스 기준선을 샘플 단위로 보관 → 검증셋과 동일한 표본에서
        # baseline 을 계산해야 공정한 비교가 된다(전체 평균과 비교하면 안 됨).
        temp_now = np.array([r["temperature"] for r in src_records], dtype=np.float32)
        self.temp_persist_abs = torch.tensor(np.abs(temp_tgt - temp_now),
                                             dtype=torch.float32)
        self.temp_persistence_mae = float(self.temp_persist_abs.mean())

        # 타깃 통계 — 헤드 편향 초기화 및 손실 스케일 정규화에 사용
        self.temp_mean   = float(temp_tgt.mean())
        self.temp_std    = float(max(temp_tgt.std(), 1e-3))
        self.precip_mean = float(precip_tgt.mean())

        # 관측소 구성 진단 — 다중 관측소 확장 검증용
        self.n_stations = len({r.get("stn") for r in src_records})

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        return self.X_num[idx], self.X_img[idx], self.X_txt[idx], self.y[idx]


# ── 4. 모델: pipeline_model.py의 TriCHEFPipeline 사용 ─────────────
#   Phase 3-1: Z축(수치) 활성, Re/Im 축 stub
#   Phase 3-2~: 순차적으로 실제 인코더 교체


# ── 5. 학습 ───────────────────────────────────────────────────────

def train(orthogonalize: bool = ORTHOGONALIZE,
          persistence_residual: bool = PERSISTENCE_RESIDUAL,
          lead_hours: int = LEAD_HOURS,
          use_re: bool = True, use_im: bool = True,
          early_stop: bool = True,
          dynamic_gate: bool = DYNAMIC_GATE,
          gate_entropy_weight: float = GATE_ENTROPY_WEIGHT,
          checkpoint: str = CHECKPOINT,
          verbose: bool = True) -> dict:
    """
    3축 Tri-CHEF 학습. 플래그를 바꿔 호출하면 ablation 대조군이 된다.

    use_re / use_im : 해당 축 입력을 차단(영벡터)해 기여도를 측정한다.
                      Z축(수치)은 퍼시스턴스 기준선에 필요하므로 항상 활성.
    early_stop      : ablation 비교 시 False 로 두어 모든 조건을 동일 에폭수로
                      학습한다. 조기 종료는 조건마다 학습 길이를 다르게 만들어
                      '성능 차이'와 '학습량 차이'를 구분할 수 없게 한다.
    반환: 최종 성능 지표 dict (ablation.py 에서 비교용으로 사용)
    """
    axes = "Z" + ("+Re" if use_re else "") + ("+Im" if use_im else "")
    if verbose:
        print(f"{'='*70}")
        print(f" Tri-CHEF Phase 3-4 — 예보 시계 +{lead_hours}시간 | 활성 축: {axes}")
        print(f" 디바이스: {DEVICE} | embed_dim: {EMBED_DIM}")
        print(f" Gram-Schmidt 직교화: {'ON' if orthogonalize else 'OFF'}"
              f" | 기온 Δ 예측: {'ON' if persistence_residual else 'OFF'}")
        gdesc = (f"동적 게이트 (균등 초기화, λ={gate_entropy_weight}, "
                 f"워밍업 {GATE_WARMUP_EPOCHS}에폭)" if dynamic_gate else "정적 α/φ")
        print(f" 축 가중치: {gdesc}")
        print(f" 위성 인코더: {'소형 CNN' if COMPACT_SATELLITE else 'ResNet18'}")
        print(f"{'='*70}\n")

    torch.manual_seed(SEED)
    np.random.seed(SEED)

    records = collect_historical()
    if len(records) < 20:
        print("[ERROR] 데이터 부족 (최소 20개 필요). API 연결을 확인하세요.")
        return {}

    if verbose:
        print("위성 이미지 생성 중 (합성, 4ch 32×32)...")
    sat_collector = SimulatedSatelliteCollector()
    if verbose:
        print("기상 예보문 MiniLM 임베딩 생성 중...")
    txt_collector = SimulatedTextCollector()

    full_ds = WeatherDataset(records,
                             sat_collector=sat_collector,
                             txt_collector=txt_collector,
                             lead_hours=lead_hours)
    n_val   = max(2, int(len(full_ds) * VAL_RATIO))
    n_train = len(full_ds) - n_val
    # SEED 고정 → 직교화 ON/OFF 비교 시 동일한 분할 사용
    train_ds, val_ds = random_split(
        full_ds, [n_train, n_val],
        generator=torch.Generator().manual_seed(SEED)
    )

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    # baseline 은 반드시 검증셋과 동일한 표본에서 계산한다.
    # 전체 평균과 비교하면 표본 차이만으로 '개선'처럼 보일 수 있다.
    val_idx      = val_ds.indices
    temp_naive   = full_ds.temp_persist_abs[val_idx].mean().item()
    precip_naive = full_ds.y[val_idx, 1].abs().mean().item()

    if verbose:
        print(f"\n학습: {n_train}개 | 검증: {n_val}개 | 배치 크기: {BATCH_SIZE} "
              f"| 관측소: {full_ds.n_stations}개")
        print(f"기준선 — 검증셋 {n_val}개 기준 (모델과 동일 표본)")
        print(f"  기온 퍼시스턴스 T(t+{lead_hours})≈T(t) : {temp_naive:.4f} °C   "
              f"[전체 {full_ds.temp_persistence_mae:.4f}]")
        print(f"  강수 상시 0 예측            : {precip_naive:.4f} mm   "
              f"[전체 {full_ds.precip_mean:.4f}]\n")

    # 모델 — 타깃 평균을 헤드 편향에 주입해 편향 학습 낭비 제거
    model = TriCHEFPipeline(
        embed_dim=EMBED_DIM, num_features=NUM_FEATURES,
        alpha_init=ALPHA_INIT, phi_init=PHI_INIT,
        orthogonalize=orthogonalize,
        temp_mean=full_ds.temp_mean, precip_mean=full_ds.precip_mean,
        persistence_residual=persistence_residual,
        feat_mean=full_ds.mean.tolist(), feat_std=full_ds.std.tolist(),
        dynamic_gate=dynamic_gate, compact_satellite=COMPACT_SATELLITE,
    ).to(DEVICE)

    # 차단된 축의 인코더는 forward 에서 호출되지 않아 gradient 를 받지 않는다.
    # 옵티마이저에 넘기기 전에 동결해 '실제 학습되는 파라미터'만 집계한다.
    if not use_re:
        for p in model.enc_re.parameters():
            p.requires_grad_(False)
    if not use_im:
        for p in model.enc_im.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    if verbose:
        print(f"학습 파라미터: {n_params:,}개 / 표본 {n_train}개  "
              f"(비율 {n_params / n_train:,.0f}:1)")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )
    mse = nn.MSELoss()

    # 손실 스케일 정규화: 기온 MSE 는 σ²(≈수십) 규모, 강수 MSE 는 O(1) 규모라
    # 그대로 더하면 강수 항이 전체의 0.1% 미만이 되어 학습되지 않는다.
    temp_var = full_ds.temp_std ** 2

    best_score = float("inf")
    best_stats = {}
    no_improve = 0
    # 지표별 최저치를 독립 추적 — 합산 score 기반 조기 종료가 한쪽 지표를
    # 가려버리는 문제를 피하고, ablation 조건 간 공정한 비교를 가능하게 한다.
    best_temp_only   = float("inf")
    best_precip_only = float("inf")

    if verbose:
        whead = "w Re/Im/Z" if dynamic_gate else "  α     φ"
        print(f"{'Epoch':>6} | {'Loss':>8} | {'기온MAE':>8} | {'강수MAE':>8} | "
              f"{'강수σ':>6} | {'|Δ기온|':>6} | {whead} | LR")
        print("-" * 92)

    for epoch in range(1, EPOCHS + 1):

        # 엔트로피 워밍업 — warmup 구간은 λ=0, 이후 목표값까지 선형 증가.
        if epoch <= GATE_WARMUP_EPOCHS:
            lam_now = 0.0
        else:
            ramp = min(1.0, (epoch - GATE_WARMUP_EPOCHS) / max(1, GATE_WARMUP_EPOCHS))
            lam_now = gate_entropy_weight * ramp

        model.train()
        train_loss = 0.0
        for x_num, x_img, x_txt, y_b in train_loader:
            x_num, y_b = x_num.to(DEVICE), y_b.to(DEVICE)
            x_img = x_img.to(DEVICE) if use_re else None
            x_txt = x_txt.to(DEVICE) if use_im else None

            optimizer.zero_grad()
            pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            loss = (mse(pred[:, 0:1], y_b[:, 0:1]) / temp_var
                    + PRECIP_WEIGHT * mse(pred[:, 1:2], y_b[:, 1:2]))
            # 게이트 엔트로피 정규화 — 가중치를 소수 축에 집중시키는 압력.
            # 이 항이 없으면 게이트도 정적 α/φ 와 마찬가지로 무용 축의
            # 가중치를 키우는 방향으로 학습된다(Phase 3-4 실측).
            if dynamic_gate and lam_now > 0:
                loss = loss + lam_now * model.gate_entropy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── 검증 ──────────────────────────────────────────────────
        model.eval()
        temp_abs, precip_abs, precip_preds, deltas = [], [], [], []
        with torch.no_grad():
            for i, (x_num, x_img, x_txt, y_b) in enumerate(val_loader):
                x_num, y_b = x_num.to(DEVICE), y_b.to(DEVICE)
                x_img = x_img.to(DEVICE) if use_re else None
                x_txt = x_txt.to(DEVICE) if use_im else None
                pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt,
                             collect_diagnostics=(i == 0))
                temp_abs.append((pred[:, 0] - y_b[:, 0]).abs())
                precip_abs.append((pred[:, 1] - y_b[:, 1]).abs())
                precip_preds.append(pred[:, 1])
                if persistence_residual:
                    # 모델이 퍼시스턴스에 실제로 얹은 보정량 |Δ|
                    t_now = x_num[:, 0] * model.feat_std[0] + model.feat_mean[0]
                    deltas.append((pred[:, 0] - t_now).abs())

        # 샘플 단위 평균 (배치 크기가 달라도 편향되지 않도록)
        val_temp_mae   = torch.cat(temp_abs).mean().item()
        val_precip_mae = torch.cat(precip_abs).mean().item()
        precip_std     = torch.cat(precip_preds).std().item()
        delta_mag      = torch.cat(deltas).mean().item() if deltas else 0.0

        # 조기 종료 기준: 두 baseline 대비 상대 오차의 합 (skill score)
        # 1.0 미만이면 해당 지표가 baseline 을 이긴 것.
        val_score = val_temp_mae / temp_naive + val_precip_mae / precip_naive

        best_temp_only   = min(best_temp_only,   val_temp_mae)
        best_precip_only = min(best_precip_only, val_precip_mae)

        scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        if verbose and (epoch % 10 == 0 or epoch == 1):
            w = model.axis_weights()
            mark = " ★" if val_score < best_score else ""
            if dynamic_gate:
                # 게이트가 각 축에 실제로 배분한 비중 (합 = 1)
                wcol = (f"{w.get('w_re',0):.3f}/{w.get('w_im',0):.3f}/"
                        f"{w.get('w_z',0):.3f}")
            else:
                wcol = f"{w['alpha']:5.3f} {w['phi']:5.3f}"
            print(f"{epoch:6d} | {train_loss:8.4f} | {val_temp_mae:7.4f}° | "
                  f"{val_precip_mae:7.4f}m | {precip_std:6.3f} | {delta_mag:6.3f} | "
                  f"{wcol} | {current_lr:.1e}{mark}")

        if val_score < best_score:
            best_score = val_score
            no_improve = 0
            w = model.axis_weights()
            best_stats = {
                "epoch":          epoch,
                "val_temp_mae":   val_temp_mae,
                "val_precip_mae": val_precip_mae,
                "precip_pred_std": precip_std,
                "delta_magnitude": delta_mag,
                "val_score":      val_score,
                "alpha_learned":  w["alpha"],
                "phi_learned":    w["phi"],
                "diagnostics":    dict(model.last_diagnostics),
            }
            torch.save({
                **best_stats,
                "model_state":  model.state_dict(),
                "embed_dim":    EMBED_DIM,
                "orthogonalize": orthogonalize,
                "persistence_residual": persistence_residual,
                # dynamic_gate/compact_satellite 는 모델 아키텍처 자체를 바꾼다
                # (게이트 네트워크 유무, ResNet18 vs 소형 CNN). 이 값 없이는
                # 체크포인트만으로 TriCHEFPipeline 을 재구성할 수 없어
                # load_state_dict 가 shape mismatch 로 실패한다.
                "dynamic_gate": dynamic_gate,
                "compact_satellite": COMPACT_SATELLITE,
                "lead_hours":   lead_hours,
                "num_features": NUM_FEATURES,
                "alpha_init":   ALPHA_INIT,
                "phi_init":     PHI_INIT,
                "mean":         full_ds.mean.tolist(),
                "std":          full_ds.std.tolist(),
                "temp_mean":    full_ds.temp_mean,
                "precip_mean":  full_ds.precip_mean,
            }, checkpoint)
        else:
            no_improve += 1
            if early_stop and no_improve >= PATIENCE:
                if verbose:
                    print(f"\nEarly stopping — epoch {epoch}")
                break

    # ── 결과 요약 ────────────────────────────────────────────────
    d = best_stats.get("diagnostics", {})
    result = {**best_stats,
              "temp_naive": temp_naive, "precip_naive": precip_naive,
              "orthogonalize": orthogonalize,
              "persistence_residual": persistence_residual,
              "lead_hours": lead_hours, "axes": axes,
              "epochs_run": epoch, "n_params": n_params,
              # 지표별 독립 최저치 — ablation 비교의 기준
              "best_temp_mae":   best_temp_only,
              "best_precip_mae": best_precip_only,
              "temp_gain_pct":   (1 - best_temp_only   / temp_naive)   * 100,
              "precip_gain_pct": (1 - best_precip_only / precip_naive) * 100}

    if verbose:
        print(f"\n{'='*70}")
        print(f" Phase 3-4 학습 완료  (+{lead_hours}h 예보 | 직교화 "
              f"{'ON' if orthogonalize else 'OFF'}"
              f" | 기온Δ {'ON' if persistence_residual else 'OFF'})")
        print(f"{'-'*70}")
        print(f" 기온 MAE : {best_temp_only:.4f} °C  vs 퍼시스턴스 "
              f"{temp_naive:.4f} °C  → {result['temp_gain_pct']:+.1f}%")
        print(f" 강수 MAE : {best_precip_only:.4f} mm  vs 상시0예측 "
              f"{precip_naive:.4f} mm  → {result['precip_gain_pct']:+.1f}%")
        print(f" 학습 에폭: {epoch}  (조기종료 {'ON' if early_stop else 'OFF'})")
        print(f" 강수 예측 표준편차: {best_stats['precip_pred_std']:.4f} mm  "
              f"({'정상 — 입력에 반응함' if best_stats['precip_pred_std'] > 1e-3 else '⚠ 붕괴 — 상수 출력'})")
        if persistence_residual:
            dm = best_stats['delta_magnitude']
            print(f" 기온 보정량 평균 |Δ|: {dm:.4f} °C  "
                  f"({'퍼시스턴스에 실질 보정 추가' if dm > 0.05 else '⚠ ≈0 — 사실상 퍼시스턴스 복사'})")
        print(f"{'-'*70}")
        d_final = best_stats.get("diagnostics", {})
        if dynamic_gate and "w_re" in d_final:
            print(f" 게이트 축 배분 (합=1): Re(위성) {d_final['w_re']:.4f} | "
                  f"Im(텍스트) {d_final['w_im']:.4f} | Z(수치) {d_final['w_z']:.4f}")
            print(f" 게이트 엔트로피: {d_final.get('gate_entropy', 0):.4f}  "
                  f"(최대 1.0986=균등배분, 낮을수록 한 축에 집중)")
        else:
            print(f" 학습된 축 가중치: α(Im)={best_stats['alpha_learned']:.4f} "
                  f"(초기 {ALPHA_INIT}) | φ(Z)={best_stats['phi_learned']:.4f} "
                  f"(초기 {PHI_INIT})")
        if d:
            print(f" 축 직교성 |cos| (0=직교, 무작위 기준 {1/np.sqrt(EMBED_DIM):.4f})")
            print(f"   직교화 전: Re-Im {d.get('cos_re_im_pre',0):.4f} | "
                  f"Re-Z {d.get('cos_re_z_pre',0):.4f} | Im-Z {d.get('cos_im_z_pre',0):.4f}")
            print(f"   직교화 후: Re-Im {d.get('cos_re_im_post',0):.4f} | "
                  f"Re-Z {d.get('cos_re_z_post',0):.4f} | Im-Z {d.get('cos_im_z_post',0):.4f}")
        print(f" 체크포인트: {checkpoint}")
        print(f"{'='*70}")

    return result


if __name__ == "__main__":
    train()
