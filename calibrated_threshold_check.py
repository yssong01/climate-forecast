"""
calibrated_threshold_check.py — 판정 임계값을 '보정 곡선으로 옮기는' 방식이
판정을 보존하는지 확인하고, 보존하지 못하면 얼마나 회복 가능한지 잰다.

배경(2026-08-17). `predict.event_threshold()`는 원본 임계값을 확률 보정
곡선에 통과시켜 판정선으로 쓴다. 근거는 "보정은 단조 증가 함수이므로 순위가
보존되고, 임계값도 같은 곡선으로 옮기면 판정이 그대로다"였다. 이 논리는
곡선이 **순증가**일 때만 성립한다. 등장성 회귀(isotonic regression)의 결과는
계단 모양이라 평탄 구간(plateau)을 갖는다 — 원본 확률이 서로 다른 표본들이
같은 보정값으로 접힌다. 임계값이 그 평탄 구간 안에 떨어지면 구간 전체가
한쪽으로 몰려 판정되므로, 실효 임계값이 의도한 위치에서 벗어난다.

한파 헤드 디커플링(2026-08-17) 직후 실제로 발생했다 — 임계값 0.500을 0.196
으로 옮기자 F1 이 0.4566에서 0.4484로 떨어졌다. 디커플링으로 얻은 이득
(+0.009)과 같은 크기의 손실이다.

이 스크립트는 헤드별로 세 가지를 같은 평가용 표본에서 비교한다.
  ① 원본 공간 판정(이상적) — 보정 전 확률에 원본 임계값
  ② 현행 배포 방식        — 보정 후 확률에 '곡선으로 옮긴' 임계값
  ③ 보정 공간 직접 선정   — 보정용 절반에서 F1 최대 임계값을 고르고
                            평가용 절반에서 채점(과적합 없이 회복 가능한 상한)

③이 ①에 근접하면, 임계값은 판정이 실제로 이뤄지는 공간(보정 후)에서 골라야
한다는 결론이 된다.

실행: python calibrated_threshold_check.py [체크포인트]
"""
import sys

import numpy as np

from probability_calibration_fit import (
    CALIB_SEED, apply_calibration, fit_isotonic, load_probs, prf,
)
from predict import CHECKPOINT, EXTREME_EVENT_THRESH, event_threshold

KO = {"rain": "강수", "heatwave": "폭염", "coldwave": "한파", "dust": "황사"}

# 서빙에서 보정을 적용하는 헤드만 본다 — 강수는 확률이 아니라 mm 로 나가고
# 양(量) 헤드가 보정 전 확률과 곱해지도록 학습돼 서빙 경로에서 보정하지 않는다.
SERVED_HEADS = ("heatwave", "coldwave", "dust")


def best_threshold(probs, labels, n_grid=400):
    """보정 공간에서 F1 을 최대화하는 임계값을 고른다(격자 탐색)."""
    lo, hi = float(probs.min()), float(probs.max())
    grid = np.linspace(lo, hi, n_grid)
    best_t, best_f1 = lo, -1.0
    for t in grid:
        f1 = prf(probs, labels, float(t))[2]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t, best_f1


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    heads, ckpt = load_probs(ckpt_path)

    # probability_calibration_fit.main() 과 동일한 난수열·순회 순서를 써야
    # 같은 보정용/평가용 분할이 재현된다.
    rng = np.random.RandomState(CALIB_SEED)

    print(f"체크포인트: {ckpt_path}")
    print(f"\n{'헤드':<6}{'①원본공간':>12}{'②현행배포':>12}{'③보정공간선정':>14}"
          f"{'②손실':>10}{'③회복':>10}")
    print("-" * 66)

    rows = []
    for key, (probs, labels) in heads.items():
        m = rng.rand(len(probs)) < 0.5     # 보정용 절반 — 순회 순서에 의존한다
        if key not in SERVED_HEADS:
            continue
        pc, lc = probs[m], labels[m]
        pe, le = probs[~m], labels[~m]

        xs, ys = fit_isotonic(pc, lc)
        pc_cal = apply_calibration(pc, xs, ys)
        pe_cal = apply_calibration(pe, xs, ys)

        t_raw = EXTREME_EVENT_THRESH.get(key, 0.5)
        t_mapped = float(apply_calibration(np.array([t_raw]), xs, ys)[0])

        f_ideal = prf(pe, le, t_raw)[2]            # ① 원본 공간
        f_current = prf(pe_cal, le, t_mapped)[2]   # ② 현행 배포
        t_pick, _ = best_threshold(pc_cal, lc)     # ③ 보정용에서 선정
        f_picked = prf(pe_cal, le, t_pick)[2]      # 평가용에서 채점

        rows.append((key, t_raw, t_mapped, t_pick, f_ideal, f_current, f_picked,
                     pe_cal, le))
        print(f"{KO[key]:<6}{f_ideal:>12.4f}{f_current:>12.4f}{f_picked:>14.4f}"
              f"{f_current - f_ideal:>+10.4f}{f_picked - f_current:>+10.4f}")

    print(f"\n{'헤드':<6}{'원본 t':>10}{'옮긴 t':>10}{'선정 t':>10}")
    print("-" * 36)
    for key, t_raw, t_mapped, t_pick, *_ in rows:
        print(f"{KO[key]:<6}{t_raw:>10.3f}{t_mapped:>10.3f}{t_pick:>10.3f}")

    print("\n판정 기준 — ②손실이 0 이면 현행 방식이 판정을 보존한다. 손실이 있고"
          "\n③이 그 손실을 되찾으면, 임계값은 보정 공간에서 직접 골라야 한다.")

    # ── 판정선 민감도 ────────────────────────────────────────
    # 원본 공간에서는 threshold_validation.py 가 ±0.02 흔들기를 재고 있으나,
    # 판정이 실제로 이뤄지는 보정 공간에서는 재본 적이 없다. 등온 회귀는
    # 평탄 구간을 가지므로 임계값을 조금만 옮겨도 구간 하나가 통째로 넘어가
    # 판정이 계단식으로 바뀔 수 있다 — 그 취약성을 직접 확인한다.
    print(f"\n{'=' * 74}\n 보정 공간 판정선 민감도 — 배포 판정선을 ±0.02 흔들면"
          f"\n{'=' * 74}")
    print(f"  {'헤드':<6}{'배포 t':>9}{'F1':>9}{'−0.02':>9}{'+0.02':>9}"
          f"{'F1 범위':>10}{'동점 표본':>11}{'판정':>10}")
    worst = 0.0
    for key, t_raw, t_mapped, t_pick, f_ideal, f_current, f_picked, pe_cal, le in rows:
        t_dep = event_threshold(key, "108", ckpt)
        f_dep = prf(pe_cal, le, t_dep)[2]
        f_lo = prf(pe_cal, le, max(0.0, t_dep - 0.02))[2]
        f_hi = prf(pe_cal, le, min(1.0, t_dep + 0.02))[2]
        rng_f1 = max(f_dep, f_lo, f_hi) - min(f_dep, f_lo, f_hi)
        worst = max(worst, rng_f1)
        # 배포 판정선과 정확히 같은 보정값을 갖는 표본 수 — 평탄 구간의 크기다.
        ties = int(np.isclose(pe_cal, t_dep, atol=1e-9).sum())
        mark = "취약" if rng_f1 > 0.02 else "안정"
        print(f"  {KO[key]:<6}{t_dep:>9.3f}{f_dep:>9.4f}{f_lo:>9.4f}{f_hi:>9.4f}"
              f"{rng_f1:>10.4f}{ties:>11,}{mark:>10}")
    print(f"\n  최악 F1 변동폭 {worst:.4f} — 0.02 이하면 배포 동작점이 안정적이다.")
    print(f"VERDICT calibrated_threshold_sensitivity "
          f"{'PASS' if worst <= 0.02 else 'WARN'} worst_f1_range={worst:.4f}")


if __name__ == "__main__":
    main()
