"""
null_calibration_check.py — Risk_Prediction의 널 캘리브레이션(non-event 구간
기준 μ+z·σ)을 관측소별 임계값 후보로 시험하고, 기존 F1 그리드서치
(threshold_validation.py의 best_thresh())와 같은 보정용/평가용 분리
절차로 head-to-head 비교한다.

**왜 필요한가.** 현재 관측소별 임계값(`predict.STATION_EVENT_THRESH_OVERRIDES`)
은 calibration_plot_diagnose.py로 이상치를 찾고 station_threshold_check.py의
F1 그리드서치로 검증한다. 그리드서치는 표본이 적은 관측소(사건 단위로 세면
더 적다, CLAUDE.md 2절)에서 특정 표본 배치에 맞춰진 값을 고를 위험이 있다.
널 캘리브레이션은 "무사건 구간에서 이 헤드가 정상적으로 얼마나 흔들리는가"
만으로 임계값을 역산한다 — F1을 직접 최적화하지 않으므로 표본 노이즈에
덜 취약할 수 있다(cross-project-technique-survey 메모 2단계 항목 5,
Risk_Prediction calibration.py fit_null() 이식).

**함정 — 반드시 점검한다.** Risk_Prediction 자신도 "무사건 표본이 정확히
0으로 퇴화"하는 문제를 겪었다. climate-forecast 극한기상 헤드도 헤드가
비활성 구간에서 로짓이 0 근처로 눌리는 특성이 있다(CLAUDE.md 1절 8항) —
무사건 구간에서 σ_null이 0에 가까우면 τ=μ+z·σ가 사실상 μ 하나로 붕괴해
아주 작은 확률 변동에도 판정이 뒤집힌다. 이 스크립트는 널 표본 수가
MIN_NULL 미만이거나 σ_null이 MIN_NULL_STD 미만인 관측소를 자동으로
건너뛰고 그 사실을 출력한다 — 조용히 무효한 임계값을 내지 않는다.

**보정용/평가용 분리·z 선택 절차.** threshold_validation.py와 같은
CALIB_SEED(1234)로 헤드별 표본을 절반씩 재분할한다. z(널 통계에 곱하는
표준편차 배수)도 F1을 직접 보고 고르면 안 되므로, 후보 그리드(Z_GRID)
중 보정용 절반에서 F1이 가장 높은 z만 고르고, 평가용 절반에서만 채점한다
— μ_null·σ_null 도 보정용 절반의 무사건 표본에서만 계산한다.

**이 스크립트는 진단만 한다** — predict.py의 STATION_EVENT_THRESH_OVERRIDES
를 건드리지 않는다. 순이득(널 F1 − t=0.5 F1) 0.01 미만이면 채택 후보가
아니다(CLAUDE.md 규약과 동일 기준).

실행 (GPU 컨테이너, eval_cache 재사용 — CLAUDE.md 3절):
  python null_calibration_check.py [--event heatwave|coldwave|dust]
"""
import argparse

import numpy as np

import eval_cache
from predict import CHECKPOINT
from threshold_validation import prf, best_thresh, CALIB_SEED
from train import STATION_NAMES

EVENTS = {
    "heatwave": ("heat_prob", "y_heatwave", "heat_mask"),
    "coldwave": ("cold_prob", "y_coldwave", "cold_mask"),
    "dust":     ("dust_prob", "y_dust", "dust_mask"),
}

MIN_NULL = 20          # 무사건 표본이 이보다 적으면 그 관측소는 건너뛴다
MIN_NULL_STD = 1e-4    # σ_null 이 이보다 작으면 무사건 구간이 완전히 눌려있다는 뜻
Z_GRID = np.arange(0.5, 3.01, 0.25)
NET_GAIN_GATE = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="heatwave", choices=list(EVENTS))
    args = ap.parse_args()

    prob_key, y_key, mask_key = EVENTS[args.event]
    d = eval_cache.load(CHECKPOINT)
    mask = d[mask_key].astype(bool)
    probs = d[prob_key][mask]
    labels = d[y_key][mask].astype(int)
    stns = d["stn"][mask]

    rng = np.random.RandomState(CALIB_SEED)
    split = rng.rand(len(probs)) < 0.5   # True=보정용

    print(f"[{args.event}] 전체 {len(probs):,}개(판정가능), "
          f"보정용 {split.sum():,} / 평가용 {(~split).sum():,}\n")
    print(f"{'관측소':<8}{'널N':>6}{'z*':>6}{'μ_null':>9}{'σ_null':>9}{'τ_null':>9}"
          f"{'F1(널)':>9}{'F1(그리드)':>11}{'F1(t=0.5)':>10}{'순이득(널−0.5)':>15}{'채택':>6}")

    uniq = sorted(set(stns.tolist()))
    for stn in uniq:
        m = stns == stn
        pc, lc = probs[m & split], labels[m & split]
        pt, lt = probs[m & ~split], labels[m & ~split]
        name = STATION_NAMES.get(stn, stn)
        if lc.sum() == 0 or lt.sum() == 0:
            print(f"{name:<8} 양성 표본이 보정/평가 한쪽에 없음 — 건너뜀")
            continue

        null_c = pc[lc == 0]
        if len(null_c) < MIN_NULL:
            print(f"{name:<8} 널표본 {len(null_c)}개 — 최소 {MIN_NULL} 미달, 건너뜀")
            continue
        mu, sigma = float(null_c.mean()), float(null_c.std())
        if sigma < MIN_NULL_STD:
            print(f"{name:<8} σ_null={sigma:.2e} — 무사건 구간이 완전히 눌려있음, 건너뜀")
            continue

        # z 는 보정용 절반의 F1로만 고른다(평가용을 훔쳐보지 않는다).
        z_scored = [(z, prf(pc, lc, float(np.clip(mu + z * sigma, 0.001, 0.999)))[2])
                    for z in Z_GRID]
        z_opt, _ = max(z_scored, key=lambda x: x[1])
        tau = float(np.clip(mu + z_opt * sigma, 0.001, 0.999))

        _, _, f1_null = prf(pt, lt, tau)
        t_opt, _ = best_thresh(pc, lc)
        _, _, f1_grid = prf(pt, lt, t_opt)
        _, _, f1_05 = prf(pt, lt, 0.5)

        gain = f1_null - f1_05
        adopt = "예" if gain >= NET_GAIN_GATE else ""
        print(f"{name:<8}{len(null_c):>6}{z_opt:>6.2f}{mu:>9.4f}{sigma:>9.4f}{tau:>9.4f}"
              f"{f1_null:>9.4f}{f1_grid:>11.4f}{f1_05:>10.4f}{gain:>+15.4f}{adopt:>6}")

    print(f"\n채택 기준: 순이득(F1(널)−F1(t=0.5)) ≥ {NET_GAIN_GATE}. "
          f"F1(그리드)는 기존 station_threshold_check.py 식 방법과의 비교 참고용 — "
          f"널 방식이 그리드서치보다 나은지는 별도로 판단할 것.")


if __name__ == "__main__":
    main()
