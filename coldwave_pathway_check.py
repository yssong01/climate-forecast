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

실행: python coldwave_pathway_check.py [체크포인트 ...]
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


def head_prob(model, ckpt, record, img, txt, head="coldwave"):
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    nf = ckpt.get("num_features", len(mean))
    vec = np.array(record_to_vec(record), dtype=np.float32)[:nf]
    x = torch.tensor((vec - mean) / std, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    with torch.no_grad():
        model(num_x=x,
              img_x=torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE),
              txt_x=torch.tensor(txt, dtype=torch.float32).unsqueeze(0).to(DEVICE))
        return torch.sigmoid(getattr(model, f"_last_{head}_logit")).item()


def build_grid(model, ckpt, base, img, txt, head="coldwave"):
    """(달, 기온) → 한파확률 격자. 시각(시)은 고정한다."""
    hh = str(base.get("timestamp", "202601011200"))[8:12] or "1200"
    grid = np.zeros((len(MONTHS), len(TEMPS)))
    for i, (_, mmdd) in enumerate(MONTHS):
        for j, t in enumerate(TEMPS):
            r = dict(base)
            r["temperature"] = t
            r["timestamp"] = f"2026{mmdd}{hh}"
            grid[i, j] = head_prob(model, ckpt, r, img, txt, head)
    return grid


def analyse(grid):
    """
    기온 진폭 = 각 달에서 기온을 훑었을 때 변동폭 → 달 평균
    계절 진폭 = 각 기온에서 달을 훑었을 때 변동폭 → 기온 평균
    기온 상관 = 각 달에서 (기온, 확률)의 순위상관 → 달 평균(음수가 정상)
    """
    amp_temp = float((grid.max(axis=1) - grid.min(axis=1)).mean())
    amp_season = float((grid.max(axis=0) - grid.min(axis=0)).mean())

    ranks_t = np.argsort(np.argsort(np.array(TEMPS, dtype=float)))
    cors = []
    for i in range(grid.shape[0]):
        row = grid[i]
        if row.max() - row.min() < 1e-6:
            continue                      # 무반응 구간은 상관을 정의하지 않는다
        rr = np.argsort(np.argsort(row))
        if rr.std() < 1e-9:
            continue
        cors.append(float(np.corrcoef(ranks_t, rr)[0, 1]))
    corr = float(np.mean(cors)) if cors else float("nan")
    return amp_temp, amp_season, corr


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--head=")]
    head = next((a.split("=", 1)[1] for a in sys.argv[1:] if a.startswith("--head=")),
                "coldwave")
    # 폭염은 기온이 오를수록 확률이 올라가야 정상이라 판정 부호가 반대다.
    expect_up = (head == "heatwave")
    paths = args or [CHECKPOINT]

    # Re/Im축 입력은 한 번만 만들어 모든 체크포인트에 동일하게 쓴다
    # (API 호출 절약 + 체크포인트 간 비교 조건 동일화).
    base = RobustWeatherCollector(stn="108").fetch()
    all_records = [RobustWeatherCollector(stn=s).fetch() for s in STATIONS.values()]
    sat = InterpolatedFieldCollector(all_records, STATION_COORDS)
    tnd = TendencyCollector(all_records)
    img, txt = sat.get_image(base), tnd.encode_single(base)
    print(f"기준 레코드: 서울 {base.get('timestamp')} · 기온 {base.get('temperature')}°C\n")

    print(f"{'체크포인트':<44}{'기온진폭':>10}{'계절진폭':>10}{'계절의존':>10}{'기온상관':>10}  판정")
    print("-" * 100)
    for p in paths:
        try:
            model, ckpt = load_model(p, DEVICE)
        except Exception as e:
            print(f"{p:<44}  로드 실패: {str(e)[:40]}")
            continue
        model.eval()
        grid = build_grid(model, ckpt, base, img, txt, head)
        at, as_, corr = analyse(grid)
        ratio = as_ / at if at > 1e-6 else float("inf")

        # 기온 상관이 양수면 "기온이 오를수록 한파확률 상승" — 물리적으로 역전.
        good = (corr > 0.3) if expect_up else (corr < -0.3)
        bad = (corr < -0.3) if expect_up else (corr > 0.3)
        verdict = "무반응" if np.isnan(corr) else ("정상" if good else ("★역전★" if bad else "혼재"))
        name = p.split("/")[-1]
        print(f"{name:<44}{at:>10.4f}{as_:>10.4f}{ratio:>10.2f}{corr:>10.2f}  {verdict}")
        # promote_checkpoint.py 가 이 줄을 파싱한다 — 형식을 바꾸지 말 것.
        # 역전이면 FAIL, 방향이 뚜렷하지 않으면(혼재·무반응) WARN.
        code = "FAIL" if verdict.startswith("★") else ("PASS" if verdict == "정상" else "WARN")
        print(f"VERDICT monotonicity_{head} {code} corr={corr:.4f}")

    print(f"\n대상 헤드: {head}")
    print("계절의존 = 계절 진폭 ÷ 기온 진폭. 클수록 기온보다 계절에 의존한다.")
    print("기온상관 = 기온 순위와 확률 순위의 상관. "
          + ("양수가 정상(기온↑ → 폭염확률↑)." if expect_up
             else "음수가 정상(기온↑ → 한파확률↓)."))


if __name__ == "__main__":
    main()
