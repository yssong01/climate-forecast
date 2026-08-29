"""
precip_prob_gate_sweep.py — 강수 예측을 rain_prob(분류 확률) 공간에서 게이팅했을 때
발생 판정 F1을 유지하면서 무강수 잔여오차(MAE)가 줄어드는지 검증한다.

**배경.** metrics_report.py 의 강수 발생 판정 F1 은 amount(= rain_prob × softplus
출력)를 PRECIP_CLIP_THRESH(predict.py, 0.2mm)로 자른 뒤 HIT_PRECIP_THRESH(0.1mm)로
이진화한 값이다 — 즉 F1과 서빙 MAE가 매그니튜드(mm) 임계값 하나를 공유한다.
error_breakdown.py 가 실측한 무강수 구간 잔여오차(평균 0.03mm/표본, 전체 격차의
201%)를 줄이려고 이 매그니튜드 임계값을 올리면(t=2.95) MAE는 기준선을 이기지만
F1이 0.430→0.103으로 무너진다(README '후처리 임계값을 올리면 이기지만, 채택하지
않는다' 참고) — 이미 기각된 함정이다.

이 스크립트는 매그니튜드 대신 **rain_prob 자체**에 별도 임계값(tau_p)을 걸어
분류 결정과 회귀 출력의 영/비영 여부를 하나로 묶는다 — 분류기가 "무강수"라고 더
확신할수록(매그니튜드와 무관하게) amount를 강제로 0으로 보낸다. 매그니튜드 임계값
스윕과는 다른 축이므로 아직 검증되지 않았다. 재학습 없이 cache/eval_cache.npz
(이미 배포 체크포인트 기준으로 최신)만 재사용하므로 체크포인트 위험이 없다.

**절차**(12절 규약과 동일한 보정용/평가용 분리). CALIB_SEED로 검증셋을 반씩 나누고,
보정용 절반에서 F1이 현재 서빙값 이상인 tau 중 무강수 MAE가 가장 낮은 값을 고른
뒤, 그 tau를 평가용 절반에서만 다시 잰다 — 채점에 쓰지 않은 표본으로 검증해야
순환 논리를 피한다.

실행: python precip_prob_gate_sweep.py [체크포인트]
"""
import sys

import numpy as np

import eval_cache
from predict import CHECKPOINT, PRECIP_CLIP_THRESH

CALIB_SEED = 1234         # probability_calibration_fit.py / threshold_validation.py 와 동일
HIT_PRECIP_THRESH = 0.1   # metrics_report.py 와 동일 기준


def prf(pred_bool, true_bool):
    tp = int((pred_bool & true_bool).sum())
    fp = int((pred_bool & ~true_bool).sum())
    fn = int((~pred_bool & true_bool).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f1


def score(precip_gated, precip_true, wet_true):
    mae = float(np.abs(precip_gated - precip_true).mean())
    mae_dry = float(np.abs(precip_gated[~wet_true] - precip_true[~wet_true]).mean())
    mae_wet = (float(np.abs(precip_gated[wet_true] - precip_true[wet_true]).mean())
               if wet_true.any() else float("nan"))
    wet_pred = precip_gated >= HIT_PRECIP_THRESH
    p, r, f1 = prf(wet_pred, wet_true)
    return dict(mae=mae, mae_dry=mae_dry, mae_wet=mae_wet, precision=p, recall=r, f1=f1)


def main(ckpt_path):
    d = eval_cache.load(ckpt_path)
    pp = np.asarray(d["precip_pred"])       # rain_prob * amount, 후처리(clip) 전
    pt = np.asarray(d["precip_true"])
    rp = np.asarray(d["rain_prob"])
    wet_thresh = float(d["wet_thresh"]) if np.ndim(d["wet_thresh"]) == 0 else float(d["wet_thresh"].item())
    wet_true_all = pt >= wet_thresh

    n = len(pt)
    rng = np.random.RandomState(CALIB_SEED)
    is_calib = rng.rand(n) < 0.5
    print(f"검증 표본 {n:,}개 — 보정용 {is_calib.sum():,} / 평가용 {(~is_calib).sum():,}")

    # 현재 서빙 기준선(매그니튜드 clip) — 두 반쪽 모두에서 참고용으로 낸다.
    served = np.where(pp < PRECIP_CLIP_THRESH, 0.0, pp)
    base_calib = score(served[is_calib], pt[is_calib], wet_true_all[is_calib])
    base_eval = score(served[~is_calib], pt[~is_calib], wet_true_all[~is_calib])
    print(f"\n[기준선 · 매그니튜드 clip {PRECIP_CLIP_THRESH}mm]")
    print(f"  보정용 F1={base_calib['f1']:.4f} MAE={base_calib['mae']:.4f} "
          f"MAE_dry={base_calib['mae_dry']:.4f}")
    print(f"  평가용 F1={base_eval['f1']:.4f} MAE={base_eval['mae']:.4f} "
          f"MAE_dry={base_eval['mae_dry']:.4f}")

    # 후보: rain_prob >= tau_p 이면 원본(비클립) 곱을 그대로 쓰고, 아니면 0.
    taus = np.linspace(0.02, 0.95, 48)
    rows = []
    for tau in taus:
        gated = np.where(rp[is_calib] >= tau, pp[is_calib], 0.0)
        rows.append((tau, score(gated, pt[is_calib], wet_true_all[is_calib])))

    print("\n[전체 sweep · 보정용, 참고용]")
    print(f"{'tau_p':>7}{'F1':>8}{'prec':>8}{'recall':>8}{'MAE':>9}{'MAE_dry':>9}{'MAE_wet':>9}")
    for tau, s in rows:
        print(f"{tau:7.3f}{s['f1']:8.4f}{s['precision']:8.4f}{s['recall']:8.4f}"
              f"{s['mae']:9.4f}{s['mae_dry']:9.4f}{s['mae_wet']:9.4f}")

    # 채택 기준: 보정용 F1이 기준선 이상이면서 MAE_dry가 최소인 tau.
    candidates = [(tau, s) for tau, s in rows if s["f1"] >= base_calib["f1"]]
    if not candidates:
        print("\n결론: F1을 기준선 이상으로 유지하는 tau_p가 없음 — "
              "확률 게이팅 단독으로는 개선 여지 없음(가설 기각).")
        return

    best_tau, best_s = min(candidates, key=lambda t: t[1]["mae_dry"])
    print(f"\n[후보 선정 · 보정용] tau_p={best_tau:.3f} "
          f"F1={best_s['f1']:.4f}(기준선 {base_calib['f1']:.4f}) "
          f"MAE_dry={best_s['mae_dry']:.4f}(기준선 {base_calib['mae_dry']:.4f}) "
          f"MAE={best_s['mae']:.4f}(기준선 {base_calib['mae']:.4f})")

    gated_eval = np.where(rp[~is_calib] >= best_tau, pp[~is_calib], 0.0)
    eval_s = score(gated_eval, pt[~is_calib], wet_true_all[~is_calib])
    print(f"\n[평가용 재확인 — 채점에 쓰지 않은 절반] tau_p={best_tau:.3f}")
    print(f"  F1={eval_s['f1']:.4f}(기준선 {base_eval['f1']:.4f}) "
          f"MAE_dry={eval_s['mae_dry']:.4f}(기준선 {base_eval['mae_dry']:.4f}) "
          f"MAE={eval_s['mae']:.4f}(기준선 {base_eval['mae']:.4f})")
    if eval_s["f1"] < base_eval["f1"]:
        print("  주의: 평가용에서는 F1이 기준선보다 낮음 — 보정용 선정이 과적합했을 가능성.")


if __name__ == "__main__":
    ckpt = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    main(ckpt)
