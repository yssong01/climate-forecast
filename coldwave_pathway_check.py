"""
coldwave_pathway_check.py — 한파 헤드가 어느 입력 경로에 의존하는지 정량화.

배경 — A안(연중 시각 sin/cos 추가, 12→14차원)은 seed 4회 중 2회에서 한파
헤드가 기온에 거꾸로 반응했다(README '시도했다가 원복한 시험'). 집계 지표
(F1·MAE)로는 전혀 잡히지 않았고, 오히려 깨진 쪽이 지표가 더 좋았다.

가설 — 연중 시각 특징이 '계절'이라는 강한 지름길을 제공하면서, 한파 헤드가
기온 경로 대신 계절 경로에 의존하게 된다. 그 결합이 학습 초기값에 따라
불안정하게 형성되어 어떤 실행에서는 기온 반응이 뒤집힌다.

이 스크립트는 구조를 고치기 전에 그 가설을 검증한다. 방법은 단순하다 —
다른 조건을 고정한 채 ① 기온만 흔들었을 때와 ② 계절만 흔들었을 때 한파
확률이 각각 얼마나 움직이는지 재고, 그 비율을 본다.

  기온 진폭  = 계절마다 기온을 −15~35°C로 흔들었을 때 확률 변동폭의 평균
  계절 진폭  = 기온마다 날짜를 1~12월로 흔들었을 때 확률 변동폭의 평균
  계절 의존도 = 계절 진폭 / 기온 진폭

가설이 맞다면 역전된 체크포인트에서 이 비율이 크게 나온다.
방향성도 함께 잰다 — 기온이 오를 때 확률이 내려가야(음의 상관) 정상이다.

12차원 체크포인트는 연중 시각 특징이 잘려 나가므로 계절 진폭이 0에 가깝게
나온다 — 시험이 의도대로 동작함을 보이는 대조군이다.

실행: python coldwave_pathway_check.py [체크포인트 ...] [--head=heatwave] [--live]
      기본은 학습 캐시에서 계절이 고루 섞이도록 균등 표집한 기준 레코드
      12개로 탐침하고 **최악값**으로 판정한다(재현 가능·보수적).
      --live 는 실시간 관측 1개만 쓴다(현장 진단용, 판정이 시각에 따라 달라짐).
"""
import sys

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import record_to_vec
from weather_collector import RobustWeatherCollector, STATIONS, STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

TEMPS = list(range(-15, 36, 5))                      # −15 ~ 35 °C
MONTHS = [(m, f"{m:02d}15") for m in range(1, 13)]   # 각 달 15일


def head_prob(model, ckpt, record, img, txt, head="coldwave", nwp_fixed=None):
    """`nwp_fixed` 는 기준 레코드에서 한 번 뽑은 예보값이다.

    기온·계절을 흔드는 동안 예보를 다시 조회하지 않는 이유는 두 가지다.
    ① 한 번에 하나만 바꾼다는 원칙 — 예보까지 함께 바뀌면 확률 변화가
       기온 때문인지 예보 때문인지 가릴 수 없다. ② 흔드는 날짜에는 애초에
       예보가 없다(미래 날짜를 포함한다).
    """
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    nf = ckpt.get("num_features", len(mean))
    vec = np.array(record_to_vec(record), dtype=np.float32)
    if ckpt.get("use_climatology_anomaly", False):
        # predict.py 서빙 경로와 같은 순서(record_to_vec 뒤에 이어붙임)로
        # 맞춘다 — 안 그러면 mean/std 와 차원이 안 맞아 셰이프 에러가 난다.
        from train import climatology_anomaly
        table = ckpt.get("climatology_table") or {}
        vec = np.concatenate([vec, [climatology_anomaly(record, table)]]).astype(np.float32)
    if ckpt.get("use_nwp", False):
        from nwp_collector import NWPForecastCollector
        nv = NWPForecastCollector.encode_from(
            nwp_fixed, record, ckpt.get("nwp_feature_set", "full14"))
        if nv is None:
            raise RuntimeError("고정 예보값으로 NWP 특징을 만들 수 없다 — "
                               "기준 레코드의 관측값이 결측이다.")
        vec = np.concatenate([vec, nv]).astype(np.float32)
    vec = vec[:nf]
    x = torch.tensor((vec - mean) / std, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        model(num_x=x,
              img_x=torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE),
              txt_x=torch.tensor(txt, dtype=torch.float32).unsqueeze(0).to(DEVICE))
        return torch.sigmoid(getattr(model, f"_last_{head}_logit")).item()


def build_grid(model, ckpt, base, img, txt, head="coldwave"):
    """(달, 기온) → 한파확률 격자. 시각(시)은 고정한다."""
    hh = str(base.get("timestamp", "202601011200"))[8:12] or "1200"
    # NWP 예보값은 기준 레코드에서 한 번만 뽑아 격자 전체에 고정한다
    # (head_prob docstring 참고).
    nwp_fixed = None
    if ckpt.get("use_nwp", False):
        from nwp_collector import shared as nwp_shared
        nwp_fixed = nwp_shared().raw_forecast(base, ckpt["lead_hours"])
        if nwp_fixed is None:
            raise RuntimeError(
                f"탐침 기준 레코드({base.get('stn')} {base.get('timestamp')})의 "
                f"수치예보가 아카이브에 없다 — 단조성 시험을 수행할 수 없다.")
    grid = np.zeros((len(MONTHS), len(TEMPS)))
    for i, (_, mmdd) in enumerate(MONTHS):
        for j, t in enumerate(TEMPS):
            r = dict(base)
            r["temperature"] = t
            r["timestamp"] = f"2026{mmdd}{hh}"
            grid[i, j] = head_prob(model, ckpt, r, img, txt, head, nwp_fixed)
    return grid


# 계절의존도(계절진폭 ÷ 기온진폭) 판정 기준(2026-09-07 신설).
#
# **왜 상관만으로는 부족한가.** 종전 판정은 기온 순위와 확률 순위의 상관
# 하나만 봤다. 그런데 `INIT_SEED` 만 바꿔 2구성 × 3seed 를 학습해 보니
# (한파 헤드) 판정을 가르는 것은 상관이 아니라 **계절의존도**였다:
#
#   구성/seed        판정   최악상관   계절의존   기온진폭
#   full14 / 42      PASS    −0.40      0.47      0.833
#   full14 / 43      PASS    −0.70      0.47      0.841
#   full14 / 44      WARN    +0.27      0.99      0.387
#   compact6 / 42    FAIL    +0.79      1.08      0.490
#   compact6 / 43    FAIL    +0.31      0.64      0.494
#   compact6 / 44    WARN    −0.14      0.68      0.637
#
# PASS 인 두 건은 계절의존 0.47 · 기온진폭 0.84 로 같고, 나머지 넷은
# 0.64~1.08 · 0.39~0.64 다 — **겹치는 구간이 없다.** 해석: 헤드가 기온을
# 충분히 보지 않으면(기온진폭이 작으면) 순위가 계절 성분의 잡음으로
# 결정되고, 그 결과가 우연히 음수면 통과하고 양수면 실패한다. 즉 상관은
# 결과이고 계절의존도가 원인에 가깝다. `compact6 / 44` 는 상관이 −0.14 로
# 통과 방향인데 계절의존 0.68 이라 불안정한 통과다 — 종전 게이트는 이걸
# 잡지 못했다.
#
# **임계값의 근거.** 1.0 은 물리적 의미가 있다 — 한파는 정의상 기온으로
# 규정되는 사건인데 계절에 그만큼 이상 의존한다면 그 자체로 부적합이다.
# 0.6 은 위 실측에서 PASS(0.47)와 실패(최소 0.64) 사이의 구간이며, 6개
# 표본으로 정한 값이라 **경계로만 쓰고 차단하지 않는다**(WARN). 표본이
# 늘면 재검토할 것 — 이 값을 FAIL 로 올리려면 고르는 표본과 채점하는
# 표본을 분리해 다시 정해야 한다(CLAUDE.md 2절).
SEASON_DEP_WARN = 0.60
SEASON_DEP_FAIL = 1.00

N_PROBE_BASES = 12   # 계절이 고루 섞이도록 캐시 전 구간에서 균등 표집
# 심각도 = (역전 상관) × (그 달의 확률 진폭). A안 사고(7월 한파확률 0.999,
# 진폭 1.0, 상관 +0.33 → 심각도 0.33)를 잡되, 진폭이 작아 실질 영향이
# 없는 잡음성 역전은 통과시키는 수준으로 잡았다.
# (판정에는 아직 쓰지 않는다 — 아래 main() 주석 참고)
SEVERITY_REF = 0.30


def pick_base_records(n: int = N_PROBE_BASES, ckpt: dict = None):
    """단조성 탐침에 쓸 기준 레코드 n 개를 **결정적으로** 고른다.

    **왜 여러 개인가(2026-08-31 추가).** 이 격자는 14개 입력 중 기온과
    연중시각 둘만 흔들고 나머지 12개(습도·기압·풍속·풍향·위경도 등)는
    기준 레코드 값으로 고정한다. 즉 한 번의 탐침은 입력공간의 **한 점**만
    본다. 실측으로 같은 체크포인트가 기준 레코드에 따라 정상 10회·혼재
    2회로 갈렸고, 특정 기준(2024-11-25)에서는 역전 월이 5개나 드러났다 —
    한 점만 찌르면 그 결함을 통째로 놓친다.

    실시간 관측을 쓰지 않는 이유도 같다. 종전 구현은 매 실행마다 현재
    관측을 기준으로 삼아, 같은 체크포인트의 판정이 실행 시각에 따라
    갈렸다(실측: 폭염 0.2167 WARN vs 0.5140 정상).
    """
    from train import collect_historical

    recs = collect_historical()
    by_ts: dict = {}
    for r in recs:
        by_ts.setdefault(str(r["timestamp"])[:12], []).append(r)
    full = sorted(ts for ts, rs in by_ts.items() if len(rs) >= len(STATIONS))
    if ckpt is not None and (ckpt.get("use_nwp") or ckpt.get("use_nwp_subset")):
        # 수치예보를 쓰는 체크포인트는 **그 모델이 실제로 처리할 수 있는
        # 시각**에서만 탐침해야 한다(2026-09-06). 캐시 전 구간에서 균등
        # 표집하면 아카이브 소급 한계(2016-01-01) 이전 시각이 뽑혀 게이트가
        # 실행조차 되지 않는다 — 그러면 단조성이 "통과"가 아니라 "미측정"인
        # 채로 승격 절차가 진행될 위험이 있다.
        from nwp_collector import shared as _nwp_shared
        _tbl = _nwp_shared().table
        _lead = ckpt.get("lead_hours", 6)
        def _covered(ts):
            b = next((r for r in by_ts[ts] if str(r.get("stn")) == "108"), None)
            return b is not None and _nwp_shared().raw_forecast(b, _lead) is not None
        full = [ts for ts in full if _covered(ts)]
    if not full:
        raise RuntimeError("12관측소가 모두 있는 시각이 캐시에 없다 — --live 로 실행할 것")
    idx = np.linspace(0, len(full) - 1, min(n, len(full))).astype(int)
    out = []
    for i in sorted(set(idx.tolist())):
        ts = full[i]
        allr = by_ts[ts]
        b = next((r for r in allr if str(r.get("stn")) == "108"), None)
        if b is not None:
            out.append((ts, b, allr))
    return out


def analyse(grid):
    """
    기온 진폭 = 각 달에서 기온을 훑었을 때 변동폭 → 달 평균
    계절 진폭 = 각 기온에서 달을 훑었을 때 변동폭 → 기온 평균
    기온 상관 = 각 달에서 (기온, 확률)의 순위상관 → **진폭 가중** 평균

    **왜 진폭으로 가중하는가(2026-08-31 수정).** 종전에는 진폭 1e-6 이상인
    달을 전부 같은 무게로 평균했다. 그런데 확률이 0 에 눌린 달(진폭 0.01
    등)의 순위상관은 수치 잡음이지 물리적 방향성이 아니다 — 그런 달이
    실제로 동작하는 달(진폭 1.0)과 같은 무게를 가지면 판정이 희석된다.
    CLAUDE.md 1절 8항이 "단조성 시험은 그 헤드가 실제로 동작하는 구간에서
    해야 한다"고 요구하는데, 정작 이 게이트가 그 요구를 지키지 않고 있었다.

    실측(2026-08-31): `numerical_trichef_ctrl_retry_20260831.pt` 의 폭염
    헤드는 종전 방식으로 0.51(혼재 경계)이었으나, 진폭 0.20 이상인 달만
    보면 0.755(뚜렷한 정상)다. 잠잠한 달 4개가 판정을 흐리고 있었다.
    반대로 배포본 한파는 −0.620 → −0.936 으로 정상 판정이 더 뚜렷해진다.

    가중 평균을 쓰는 이유(하드 컷오프 대신) — 임계값을 하나 더 고르면 그
    값 자체가 임의 선택이 된다. 진폭으로 가중하면 잠잠한 달이 자동으로
    0 에 가까운 무게를 갖고, 경계에서 판정이 튀지 않는다.
    """
    amp_temp = float((grid.max(axis=1) - grid.min(axis=1)).mean())
    amp_season = float((grid.max(axis=0) - grid.min(axis=0)).mean())

    ranks_t = np.argsort(np.argsort(np.array(TEMPS, dtype=float)))
    cors, weights = [], []
    for i in range(grid.shape[0]):
        row = grid[i]
        amp = float(row.max() - row.min())
        if amp < 1e-6:
            continue                      # 무반응 구간은 상관을 정의하지 않는다
        rr = np.argsort(np.argsort(row))
        if rr.std() < 1e-9:
            continue
        cors.append(float(np.corrcoef(ranks_t, rr)[0, 1]))
        weights.append(amp)
    if not cors:
        return amp_temp, amp_season, float("nan"), 0.0
    w = np.array(weights)
    c = np.array(cors)
    corr = float(np.sum(c * w) / w.sum())
    # 심각도 = max over 월 of (역전 정도 × 그 달의 진폭).
    # 평균 상관은 "한 달에서 확신에 찬 역전"과 "여러 달이 어정쩡함"을 구분하지
    # 못한다 — A안 사고(7월 한파확률 0.999, 진폭 1.0, 상관 +0.33)가 정확히
    # 전자인데 평균으로는 희석돼 잡히지 않는다. 진폭을 곱해 "그 구간에서
    # 실제로 얼마나 확신에 차서 틀렸는가"를 잰다. 부호는 호출부가 준다.
    sev = float(np.max(np.clip(c, 0, None) * w))          # 한파 기준(양수가 역전)
    sev_up = float(np.max(np.clip(-c, 0, None) * w))      # 폭염 기준(음수가 역전)
    return amp_temp, amp_season, corr, (sev, sev_up)


def main():
    args = [a for a in sys.argv[1:]
            if not a.startswith("--head=") and a != "--live"]
    head = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--head=")),
                "coldwave")
    # 폭염은 기온이 오를수록 확률이 올라가야 정상이라 판정 부호가 반대다.
    expect_up = (head == "heatwave")
    paths = args or [CHECKPOINT]

    # Re/Im축 입력은 한 번만 만들어 모든 체크포인트에 동일하게 쓴다
    # (API 호출 절약 + 체크포인트 간 비교 조건 동일화).
    # 기준 레코드는 **재현 가능**해야 한다(2026-08-31 수정). 종전에는 실시간
    # 관측을 그대로 썼는데, 그러면 같은 체크포인트도 실행 시각에 따라 판정이
    # 갈린다 — 실측으로 `ctrl_retry` 폭염이 0.2167(WARN)과 0.5140(정상)으로
    # 갈렸다. 승격을 막거나 허용하는 게이트가 실행 시각에 흔들리면 안 된다.
    # 기본값은 학습 캐시의 고정 시각 레코드이고, --live 를 주면 종전처럼
    # 실시간 관측을 쓴다(현장 진단용).
    use_live = "--live" in sys.argv
    if use_live:
        base = RobustWeatherCollector(stn="108").fetch()
        allr = [RobustWeatherCollector(stn=s).fetch() for s in STATIONS.values()]
        probes = [(str(base.get("timestamp"))[:12], base, allr)]
    else:
        # 비교 대상 중 하나라도 수치예보를 쓰면 **전부** 그 아카이브가 덮는
        # 구간에서만 탐침한다 — 체크포인트마다 다른 탐침을 쓰면 비교가
        # 성립하지 않고, 안 걸러내면 게이트가 실행조차 못 한다.
        _meta = None
        for _p in paths:
            try:
                _c = torch.load(_p, map_location="cpu", weights_only=True)
            except Exception:
                continue
            if _c.get("use_nwp") or _c.get("use_nwp_subset"):
                _meta = _c
                break
        probes = pick_base_records(ckpt=_meta)
    print(f"기준 레코드 {len(probes)}개로 탐침"
          + (" (--live: 실시간 관측 1개)" if use_live else " (학습 캐시 균등 표집, 재현 가능)"))
    print(f"  {probes[0][0]} ~ {probes[-1][0]}\n")

    # 각 기준 레코드의 Re/Im 입력을 미리 만들어 체크포인트 간 조건을 동일화한다.
    prepared = []
    for ts, b, allr in probes:
        sat = InterpolatedFieldCollector(allr, STATION_COORDS)
        tnd = TendencyCollector(allr)
        prepared.append((ts, b, sat.get_image(b), tnd.encode_single(b)))

    print(f"{'체크포인트':<44}{'기온진폭':>10}{'계절진폭':>10}{'계절의존':>10}"
          f"{'최악상관':>10}{'심각도':>10}{'정상/탐침':>10}  판정")
    print("-" * 120)
    for p in paths:
        try:
            model, ckpt = load_model(p, DEVICE)
        except Exception as e:
            print(f"{p:<44}  로드 실패: {str(e)[:40]}")
            continue
        model.eval()

        # 탐침마다 판정하고 **최악값**으로 게이트를 정한다 — 승격 게이트는
        # 보수적이어야 하고, 한 기준에서만 드러나는 역전을 놓치면 안 된다.
        ats, ass, corrs, sevs, worst_ts = [], [], [], [], None
        for ts, b, img, txt in prepared:
            g = build_grid(model, ckpt, b, img, txt, head)
            at_i, as_i, c_i, sev_i = analyse(g)
            ats.append(at_i); ass.append(as_i); corrs.append(c_i)
            sevs.append(sev_i[1] if expect_up else sev_i[0])
        arr = np.array(corrs, dtype=float)
        valid = ~np.isnan(arr)
        if not valid.any():
            corr = float("nan")
        else:
            # expect_up 이면 가장 작은 상관이, 아니면 가장 큰 상관이 최악이다.
            k = int(np.nanargmin(arr)) if expect_up else int(np.nanargmax(arr))
            corr = float(arr[k]); worst_ts = prepared[k][0]
        at, as_ = float(np.mean(ats)), float(np.mean(ass))
        ratio = as_ / at if at > 1e-6 else float("inf")

        good_v = (arr > 0.3) if expect_up else (arr < -0.3)
        n_ok = int(np.sum(good_v & valid))

        # 기온 상관이 양수면 "기온이 오를수록 한파확률 상승" — 물리적으로 역전.
        # 심각도는 **정보 제공용**이다(2026-08-31). FAIL 기준으로 쓰려면
        # 임계값을 정해야 하는데, 체크포인트 3개로 정하면 원하는 답이 나오게
        # 값을 맞추는 순환 논리가 된다(CLAUDE.md 2절: 고르는 표본과 채점하는
        # 표본을 분리한다). 실측 참고값 — 배포본 한파 0.288·폭염 0.078,
        # ctrl_retry 한파 0.391, ctrl_island 한파 0.555·폭염 0.559.
        # 판정 자체는 종전과 같은 상관 기준을 쓰되, 단일 시각이 아니라
        # 12개 기준 레코드의 **최악값**에 적용해 종전보다 엄격해졌다.
        severity = float(np.max(sevs)) if sevs else 0.0
        good = (corr > 0.3) if expect_up else (corr < -0.3)
        bad = (corr < -0.3) if expect_up else (corr > 0.3)
        verdict = "무반응" if np.isnan(corr) else ("정상" if good else ("★역전★" if bad else "혼재"))
        name = p.split("/")[-1]
        # 계절의존도 판정(2026-09-07 추가) — 아래 SEASON_DEP_* 주석 참고.
        dep_code = ("FAIL" if ratio > SEASON_DEP_FAIL
                    else ("WARN" if ratio > SEASON_DEP_WARN else "PASS"))
        mark = "" if dep_code == "PASS" else f"  [계절의존 {dep_code}]"
        print(f"{name:<44}{at:>10.4f}{as_:>10.4f}{ratio:>10.2f}{corr:>10.2f}"
              f"{severity:>10.3f}{n_ok:>6}/{int(valid.sum()):<4}  {verdict}{mark}"
              + (f"  (최악 기준 {worst_ts})" if worst_ts and not good else ""))
        # promote_checkpoint.py 가 이 줄을 파싱한다 — 형식을 바꾸지 말 것.
        # 역전이면 FAIL, 방향이 뚜렷하지 않으면(혼재·무반응) WARN.
        corr_code = "FAIL" if verdict.startswith("★") else ("PASS" if verdict == "정상" else "WARN")
        # 두 판정 중 나쁜 쪽을 채택한다 — 게이트는 보수적이어야 한다.
        rank = {"PASS": 0, "WARN": 1, "FAIL": 2}
        code = max(corr_code, dep_code, key=lambda c: rank[c])
        print(f"VERDICT monotonicity_{head} {code} corr={corr:.4f} season_dep={ratio:.2f}")

    print(f"\n대상 헤드: {head}")
    print(f"계절의존 = 계절 진폭 ÷ 기온 진폭. 클수록 기온보다 계절에 의존한다 "
          f"(경계 {SEASON_DEP_WARN} · 차단 {SEASON_DEP_FAIL} — 상수 주석에 근거 실측 있음).")
    print("기온상관 = 기온 순위와 확률 순위의 상관. "
          + ("양수가 정상(기온↑ → 폭염확률↑)." if expect_up
             else "음수가 정상(기온↑ → 한파확률↓)."))


if __name__ == "__main__":
    main()
