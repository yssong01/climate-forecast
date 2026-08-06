"""
train.py — Phase 2: 수치 단독 기준 모델 학습
논문 Eq.1 2축 버전: s = √(A² + (0.4·B)²) → 다음 시각 [기온, 강수량] 동시 예측

Re 축 (안정 대기): 기온·습도·기압·강수형태  → 광역 패턴
Im 축 (동적 기상): 강수량·풍속·풍향sin/cos → 국지 변동

기온 헤드: 항상 변화 있음 → 모델 동작 검증 (퍼시스턴스 대비 MAE)
강수 헤드: 강수 이벤트 예측 → Phase 3 멀티모달 확장 기반
"""
import os
import json
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader, random_split
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from weather_collector import RobustWeatherCollector

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM  = 64
BATCH_SIZE = 32
EPOCHS     = 150
LR         = 1e-3
ALPHA      = 0.4    # Im 축 감쇠 계수 (논문 deployed default α=0.4)
N_HOURS    = 720    # 수집 기간: 30일 × 24시간 (장마철 포함 확보)
PATIENCE   = 20     # Early stopping
VAL_RATIO  = 0.2

CHECKPOINT  = "./checkpoints/numerical_trichef.pt"
DATA_CACHE  = "./cache/historical_data.json"
os.makedirs("./checkpoints", exist_ok=True)


# ── 1. 수치 벡터 변환 ─────────────────────────────────────────────

def record_to_vec(r: dict) -> list[float]:
    """관측 레코드 → 8차원 수치 벡터."""
    wd_rad = np.deg2rad(r.get("wind_dir", 0.0))
    return [
        r.get("temperature",   20.0),   # 0: 기온 (Re)
        r.get("precipitation",  0.0),   # 1: 강수량 (Im)
        r.get("humidity",      50.0),   # 2: 습도 (Re)
        r.get("wind_speed",     1.5),   # 3: 풍속 (Im)
        float(np.sin(wd_rad)),          # 4: 풍향 sin (Im)
        float(np.cos(wd_rad)),          # 5: 풍향 cos (Im)
        r.get("pressure",    1013.0),   # 6: 기압 (Re)
        float(r.get("precip_type", 0)), # 7: 강수형태 (Re)
    ]


# ── 2. 과거 데이터 수집 ───────────────────────────────────────────

def collect_historical(n_hours: int = N_HOURS) -> list[dict]:
    """ASOS API로 최근 n_hours 시간의 관측 데이터 수집."""

    # 캐시 재사용 (API 호출 절약)
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        if len(cached) >= int(n_hours * 0.9):
            print(f"[캐시] {len(cached)}개 로드 완료 (API 절약)")
            return cached
        print(f"[캐시] {len(cached)}개 — 부족하여 재수집")

    collector = RobustWeatherCollector()
    records   = []
    now       = datetime.now()

    print(f"ASOS 과거 {n_hours}시간 데이터 수집 시작 ({collector.stn}번 관측소)...")

    for i in range(n_hours, 0, -1):
        dt = now - timedelta(hours=i)
        tm = dt.strftime("%Y%m%d%H00")
        data = collector.fetch_at(tm)

        if data.get("status") == "SUCCESS_LIVE":
            records.append(data)

        if i % 24 == 0:
            done = n_hours - i
            print(f"  {done:3d}/{n_hours} 처리 | 성공: {len(records)}개")

        time.sleep(0.3)  # API 과부하 방지

    print(f"수집 완료: {len(records)}/{n_hours}개 성공\n")

    with open(DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False, indent=2)

    return records


# ── 3. 데이터셋 ───────────────────────────────────────────────────

class WeatherDataset(Dataset):
    """
    입력 X: 시각 t의 8차원 수치 벡터 (표준화)
    타깃 y: [기온_t+1, 강수량_t+1] — 2개 동시 예측
      - 기온: 항상 변화 → 모델 동작 검증 (퍼시스턴스 기준 MAE 제공)
      - 강수: 강수 이벤트 예측
    """

    def __init__(self, records: list[dict], mean: np.ndarray = None, std: np.ndarray = None):
        vecs = np.array([record_to_vec(r) for r in records], dtype=np.float32)

        if mean is None:
            self.mean = vecs.mean(axis=0)
            self.std  = np.where(vecs.std(axis=0) > 1e-6, vecs.std(axis=0), 1.0)
        else:
            self.mean, self.std = mean, std

        vecs_norm = (vecs - self.mean) / self.std
        self.X = torch.tensor(vecs_norm[:-1], dtype=torch.float32)

        temp_next   = np.array([r["temperature"]   for r in records[1:]], dtype=np.float32)
        precip_next = np.array([r["precipitation"] for r in records[1:]], dtype=np.float32)
        self.y = torch.tensor(
            np.stack([temp_next, precip_next], axis=1), dtype=torch.float32
        )  # shape: (N, 2)

        # 기온 퍼시스턴스 기준: T_{t+1} ≈ T_t (단순 지속 예보)
        temp_curr = np.array([r["temperature"] for r in records[:-1]], dtype=np.float32)
        self.temp_persistence_mae = float(np.mean(np.abs(temp_next - temp_curr)))

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


# ── 4. 모델 ───────────────────────────────────────────────────────

class NumericalTriCHEF(nn.Module):
    """
    Phase 2: 2축 Tri-CHEF 수치 회귀 모델 (논문 Eq.1 적용).

    score = √( v_re² + (α·v_im)² )  →  회귀 헤드

    Re 축: 기온[0]·습도[2]·기압[6]·강수형태[7] — 안정·광역 대기 상태
    Im 축: 강수량[1]·풍속[3]·풍향sin[4]·풍향cos[5] — 동적·국지 기상 이벤트
    """

    def __init__(self, embed_dim: int = EMBED_DIM, alpha: float = ALPHA):
        super().__init__()
        self.alpha = alpha

        self.enc_re = nn.Sequential(
            nn.Linear(4, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )
        self.enc_im = nn.Sequential(
            nn.Linear(4, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )
        # 기온 예측 헤드 (항상 유효 신호 → 모델 동작 검증)
        self.head_temp = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(),
            nn.Linear(32, 1),
        )
        # 강수량 예측 헤드 (희소 이벤트 → Phase 3 확장 기반)
        self.head_precip = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(),
            nn.Linear(32, 1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_re = x[:, [0, 2, 6, 7]]
        x_im = x[:, [1, 3, 4, 5]]

        v_re = F.normalize(self.enc_re(x_re), dim=-1)
        v_im = F.normalize(self.enc_im(x_im), dim=-1)

        magnitude = torch.sqrt(v_re ** 2 + (self.alpha * v_im) ** 2 + 1e-7)

        temp_pred   = self.head_temp(magnitude)
        precip_pred = torch.clamp(self.head_precip(magnitude), min=0.0, max=200.0)
        return torch.cat([temp_pred, precip_pred], dim=-1)  # (N, 2)

    def encode(self, x: torch.Tensor):
        """Phase 3 멀티모달 통합용: (v_re, v_im) 임베딩 반환."""
        x_re = x[:, [0, 2, 6, 7]]
        x_im = x[:, [1, 3, 4, 5]]
        v_re = F.normalize(self.enc_re(x_re), dim=-1)
        v_im = F.normalize(self.enc_im(x_im), dim=-1)
        return v_re, v_im


# ── 5. 학습 ───────────────────────────────────────────────────────

def train():
    print(f"{'='*50}")
    print(f" Tri-CHEF Phase 2 — 수치 기준 모델 학습")
    print(f" 디바이스: {DEVICE} | embed_dim: {EMBED_DIM} | α: {ALPHA}")
    print(f"{'='*50}\n")

    # 데이터 수집
    records = collect_historical()
    if len(records) < 20:
        print("[ERROR] 데이터 부족 (최소 20개 필요). API 연결을 확인하세요.")
        return

    # 데이터셋
    full_ds = WeatherDataset(records)
    n_val   = max(2, int(len(full_ds) * VAL_RATIO))
    n_train = len(full_ds) - n_val
    train_ds, val_ds = random_split(full_ds, [n_train, n_val])

    # 검증셋도 학습셋의 mean/std 기준으로 표준화 (data leakage 방지)
    val_ds.dataset.mean = full_ds.mean
    val_ds.dataset.std  = full_ds.std

    train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True,  drop_last=False)
    val_loader   = DataLoader(val_ds,   batch_size=BATCH_SIZE, shuffle=False)

    print(f"학습: {n_train}개 | 검증: {n_val}개 | 배치 크기: {BATCH_SIZE}")
    print(f"기온 퍼시스턴스 기준 MAE: {full_ds.temp_persistence_mae:.4f} °C  "
          f"(T_{{t+1}} ≈ T_t 예측 시)")
    print(f"평균 강수량: {full_ds.y[:, 1].mean().item():.4f} mm  "
          f"(강수량 0 예측 시 MAE)\n")

    # 모델·옵티마이저
    model     = NumericalTriCHEF(embed_dim=EMBED_DIM, alpha=ALPHA).to(DEVICE)
    optimizer = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", patience=8, factor=0.5
    )
    criterion = nn.MSELoss()

    best_val_mae = float("inf")
    no_improve   = 0

    print(f"{'Epoch':>6} | {'Train Loss':>10} | {'기온 MAE':>9} | {'강수 MAE':>9} | LR")
    print("-" * 60)

    for epoch in range(1, EPOCHS + 1):

        # Train — loss = MSE(기온) + 0.5 * MSE(강수량)
        model.train()
        train_loss = 0.0
        for x_b, y_b in train_loader:
            x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
            optimizer.zero_grad()
            pred = model(x_b)
            loss = criterion(pred[:, 0:1], y_b[:, 0:1]) + \
                   0.5 * criterion(pred[:, 1:2], y_b[:, 1:2])
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # Validation — 기온 MAE (학습 조기종료 기준)
        model.eval()
        val_temp_mae = 0.0
        val_precip_mae = 0.0
        with torch.no_grad():
            for x_b, y_b in val_loader:
                x_b, y_b = x_b.to(DEVICE), y_b.to(DEVICE)
                pred = model(x_b)
                val_temp_mae   += torch.mean(torch.abs(pred[:, 0:1] - y_b[:, 0:1])).item()
                val_precip_mae += torch.mean(torch.abs(pred[:, 1:2] - y_b[:, 1:2])).item()
        val_temp_mae   /= len(val_loader)
        val_precip_mae /= len(val_loader)
        val_mae = val_temp_mae  # Early stopping 기준: 기온 MAE

        scheduler.step(val_mae)
        current_lr = optimizer.param_groups[0]["lr"]

        if epoch % 10 == 0 or epoch == 1:
            mark = " ★" if val_mae < best_val_mae else ""
            print(f"{epoch:6d} | {train_loss:10.4f} | {val_temp_mae:8.4f}°C | "
                  f"{val_precip_mae:8.4f}mm | {current_lr:.2e}{mark}")

        if val_mae < best_val_mae:
            best_val_mae = val_mae
            no_improve   = 0
            torch.save({
                "epoch":          epoch,
                "model_state":    model.state_dict(),
                "val_temp_mae":   val_temp_mae,
                "val_precip_mae": val_precip_mae,
                "embed_dim":      EMBED_DIM,
                "alpha":          ALPHA,
                "mean":           full_ds.mean.tolist(),
                "std":            full_ds.std.tolist(),
            }, CHECKPOINT)
        else:
            no_improve += 1
            if no_improve >= PATIENCE:
                print(f"\nEarly stopping — epoch {epoch} (patience {PATIENCE} 초과)")
                break

    print(f"\n{'='*55}")
    print(f" 학습 완료")
    print(f" Best Val 기온 MAE  : {best_val_mae:.4f} °C")
    print(f" 퍼시스턴스 기준    : {full_ds.temp_persistence_mae:.4f} °C  "
          f"({'개선' if best_val_mae < full_ds.temp_persistence_mae else '미달성 — 에폭 늘리거나 데이터 확인'})")
    print(f" Best Val 강수 MAE  : (위 저장된 체크포인트 기준)")
    print(f" 체크포인트         : {CHECKPOINT}")
    print(f"{'='*55}")


if __name__ == "__main__":
    train()
