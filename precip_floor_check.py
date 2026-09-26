"""
precip_floor_check.py — 강수 전체 MAE 의 남은 격차를 **주 지표를 팔지 않고**
줄일 수 있는지 잰다(2026-09-27).

## 왜 이 질문이 새로운가

이 저장소는 "전체 MAE 를 낮추려면 게이팅 임계값(τ)을 올리면 되지만 그러면
발생 F1 과 강수 구간 MAE 가 함께 나빠지므로 하지 않는다"고 결론지어 두었다.
그건 **τ 를 움직였을 때**의 이야기다. 여기서 보는 것은 다른 손잡이다.

화면·산출물의 발생 판정 F1 은 **서빙 강수량**을 `HIT_PRECIP_THRESH`(0.1mm)
로 잘라 계산한다(`metrics_report.accuracy_block`). 즉 **서빙값이 0.1mm 미만인
행은 F1 채점에서 이미 '무강수 예측'으로 세어진다.** 그런데 그 작은 값이
전체 MAE 에는 그대로 더해진다. 그러면 그 구간을 0 으로 만드는 것은
**F1 에 정의상 무료**다 — τ 를 건드리지 않으므로 확률 판정도 그대로다.

바뀔 수 있는 것은 강수 구간 조건부 MAE 뿐이다(실제로 비가 왔는데 모델이
0.1mm 미만을 낸 행에서 오차가 조금 커진다). 그 대가가 얼마인지가 이
측정의 전부다. **공짜로 보이는 것이 정말 공짜인지 확인하지 않고 채택하지
않는다.**

## 재는 방법

배포 모델(`precip_gbm.npz`)의 출력을 그대로 받아, 서빙 후처리에 바닥값
`a_min` 을 하나 더 얹고 `a_min` 을 훑는다. 채점은 배포와 같은 표본
(검증셋 전체)과 같은 정의(`metrics_report` 와 동일한 0.1mm 기준)다.

실행: python precip_floor_check.py       (LEAD_HOURS 환경변수로 6/12)
"""
import os

import numpy as np

import eval_cache

LEAD_HOURS = int(os.getenv("LEAD_HOURS", "6"))
CKPT = ("./checkpoints/numerical_trichef.pt" if LEAD_HOURS == 6
        else "./checkpoints/numerical_trichef_12h.pt")
GBM = ("./checkpoints/precip_gbm.npz" if LEAD_HOURS == 6
       else "./checkpoints/precip_gbm_12h.npz")
HIT_PRECIP_THRESH = 0.1     # metrics_report 와 같은 값이어야 한다


def score(served, truth, wet_thresh):
    """`metrics_report.accuracy_block` 과 **같은 정의**로 채점한다."""
    wet_pred = served >= HIT_PRECIP_THRESH
    wet_true = truth >= HIT_PRECIP_THRESH
    tp = int((wet_pred & wet_true).sum())
    fp = int((wet_pred & ~wet_true).sum())
    fn = int((~wet_pred & wet_true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    wt = truth >= wet_thresh
    return {
        "f1": f1, "precision": p, "recall": r,
        "mae": float(np.abs(served - truth).mean()),
        "mae_wet": float(np.abs(served[wt] - truth[wt]).mean()),
        "mae_dry": float(np.abs(served[~wt] - truth[~wt]).mean()),
        "n_zeroed": 0,
    }


def main():
    import torch
    from precip_gbm import NumpyGBM

    print(f"\n{'=' * 82}\n 강수 바닥값 — 주 지표를 팔지 않고 전체 MAE 를 줄일 수 있는가"
          f" (+{LEAD_HOURS}h)\n{'=' * 82}")

    ckpt = torch.load(CKPT, map_location="cpu", weights_only=True)
    src = ckpt.get("precip_source")
    if not src:
        raise SystemExit("이 체크포인트는 강수를 전용 GBM 으로 내지 않는다 — "
                         "이 측정은 배포 구성에서만 뜻이 있다.")

    f = eval_cache.load_features(CKPT)
    d = np.load(GBM, allow_pickle=True)
    x_va = f["x_val"]
    truth = f["precip_true_val"].astype(np.float64)
    wet_thresh = float(d["meta_wet_thresh"])
    tau = float(d["meta_gate_tau"])

    amt = np.clip(NumpyGBM(d, "amt").predict(x_va), 0, None).astype(np.float64)
    occ = NumpyGBM(d, "occ").predict_proba1(x_va).astype(np.float64)
    base = np.where(occ >= tau, amt, 0.0)          # 현행 서빙값

    naive = float(np.abs(truth).mean())
    print(f"  표본 {len(truth):,} · 판정선 τ={tau:.3f} · 상시 무강수 기준선 "
          f"{naive:.4f}mm")
    print(f"\n  {'바닥값':>8}{'0으로 바뀐 행':>14}{'발생F1':>9}{'구간MAE':>10}"
          f"{'전체MAE':>10}{'기준선대비':>11}")

    rows = []
    for a_min in (0.0, 0.02, 0.05, 0.10, 0.15, 0.20, 0.30, 0.50):
        served = np.where(base < a_min, 0.0, base)
        n_zero = int(((base > 0) & (base < a_min)).sum())
        s = score(served, truth, wet_thresh)
        gap = (s["mae"] - naive) / naive * 100
        mark = ""
        if a_min == 0.0:
            ref = s
        else:
            # 주 지표 둘이 **나빠지지 않아야** 채택 후보다.
            if (s["f1"] >= ref["f1"] - 1e-12
                    and s["mae_wet"] <= ref["mae_wet"] + 1e-12
                    and s["mae"] < ref["mae"]):
                mark = "  ← 주 지표 손해 없이 개선"
        rows.append((a_min, s, gap))
        print(f"  {a_min:8.2f}{n_zero:14,}{s['f1']:9.4f}{s['mae_wet']:10.4f}"
              f"{s['mae']:10.4f}{gap:+10.1f}%{mark}")

    ref_a, ref_s, ref_gap = rows[0]
    # F1 이 정확히 같은 구간(정의상 무료인 구간)만 추려 가장 좋은 것을 고른다.
    free = [(a, s, g) for a, s, g in rows[1:]
            if abs(s["f1"] - ref_s["f1"]) < 1e-12]
    print(f"\n  현행(바닥값 없음) — F1 {ref_s['f1']:.4f} · 구간MAE "
          f"{ref_s['mae_wet']:.4f} · 전체MAE {ref_s['mae']:.4f}({ref_gap:+.1f}%)")
    if not free:
        print("  F1 이 정확히 보존되는 바닥값이 없다 — 이 손잡이도 거래다.")
        return
    a, s, g = min(free, key=lambda t: t[1]["mae"])
    print(f"  F1 이 **정확히 보존되는** 최선 바닥값 {a:.2f}mm — "
          f"전체MAE {s['mae']:.4f}({g:+.1f}%)")
    print(f"    전체MAE 개선 {ref_s['mae'] - s['mae']:+.4f}mm "
          f"({ref_gap - g:.1f}%p) · 구간MAE 대가 "
          f"{s['mae_wet'] - ref_s['mae_wet']:+.4f}mm")
    if s["mae_wet"] > ref_s["mae_wet"] + 1e-12:
        print("    → **공짜가 아니다.** 구간 조건부 MAE(주 지표)가 나빠진다.")
    else:
        print("    → 주 지표 둘 다 손해가 없다. 채택 후보.")


if __name__ == "__main__":
    main()
