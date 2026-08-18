"""
conformal_interval_fit.py — 기온·강수 점추정에 분포무관 예측구간을 부여한다
(split conformal prediction).

**왜 필요한가.** `head_dropout` 도입 검토 당시 이미 지적된 한계 — 대시보드의
"±"는 검증셋 평균오차일 뿐 이 예측 한 건의 신뢰구간이 아니다. MC Dropout으로
예측별 불확실성을 내려던 시도는 계절 오탐 재발로 기각됐다(README '기각한
시험 — 강수 헤드 부호 표현·헤드 드롭아웃' 참고). 이 스크립트는 **가중치를
전혀 바꾸지 않는 사후 보정**이라 그 실패와 무관하다 — 이미 계산된 점추정에
비순응도 점수(nonconformity score)로 구간 폭만 씌운다.

**방법(split conformal, 부호 있는 잔차).** 검증셋을 보정용/평가용 절반씩
무작위 분할한다(probability_calibration_fit.py와 같은 CALIB_SEED — 같은
재분할을 재사용). 보정용에서 부호 있는 잔차 e = 실측 − 예측 의 (α/2, 1−α/2)
분위수(q_lo, q_hi)를 구해 구간 [예측+q_lo, 예측+q_hi]를 만든다. 대칭 잔차
가정(|e|의 단일 분위수)이 아니라 부호별로 따로 구하는 이유 — 강수는 무강수
쪽 잔차가 항상 작은 음수 근처로 쏠려 있고 호우 쪽만 크게 벌어지는
비대칭이라, 대칭 구간을 쓰면 무강수 쪽 폭이 불필요하게 넓어진다.

**층화(Mondrian conformal) — 축이 헤드마다 다르다.**
기온은 관측소마다 물리적 기준이 다를 수 있다는 원칙(README 검증 규약)을
그대로 적용해 관측소별로 분위수를 구한다.
강수는 전역(관측소만) 층화로 먼저 시도했으나 실측(2026-08-18)에서 실패했다
— 검증셋의 93.9%가 무강수라 전역 분위수가 그 다수에 지배되고, 실제
강수(습윤) 표본의 조건부 커버리지가 7%까지 떨어졌다(목표 90%). 원인은
head_rain(hurdle)이 이미 겪은 것과 같다 — "오는가"와 "얼마나"를 뭉뚱그리면
안 된다. 그래서 강수는 **예측된 강수확률(rain_prob) 구간**으로 층화한다.
관측소·시각별로 손쉽게 다른 물리적 조건을 반영하면서도, 추론 시점에 실제로
쓸 수 있는 값(참값 습윤 여부가 아니라 모델이 낸 확률)만 쓴다는 게 핵심이다.
버킷팅 후에도 각 구간은 목표 커버리지에 근접했지만, **습윤 조건부
커버리지는 여전히 90%에 못 미친다**(버킷팅 후 61%) — 이는 정합적 예측의
결함이 아니라 "참 라벨로 조건화된 커버리지는 모델-무관 방법으로 보장할 수
없다"는 알려진 한계(distribution-free conditional coverage 불가능성)가
드러난 것이며, 예측확률이 낮았는데 실제로 비가 온 사건(강수 헤드가 놓친
사건)은 애초에 좁은 구간을 받으므로 못 잡는다. 이 한계를 README에 명시한다.

**채택 기준.** 정합적 예측의 이론적 보장은 "한 번도 보지 않은 표본에서
목표 커버리지 이상"이다. 평가용(적합에 전혀 쓰이지 않은 절반)에서 실측
커버리지가 목표(1−α)에 못 미치면(허용오차 0.02) 채택하지 않는다 — 큰
표본에서는 정상적으로는 거의 항상 충족되므로, 미달은 계산 버그나
분포 이동을 의심할 신호다. 이 기준은 버킷/관측소 전체 평균 커버리지에
적용하며, "습윤 조건부" 같은 참값 기반 하위집단 커버리지는 위 한계 때문에
채택 기준에 넣지 않는다 — 대신 실측값을 보고해 투명하게 남긴다.

실행: python conformal_interval_fit.py [체크포인트] [--apply]
      --apply 없이 실행하면 측정만 하고 체크포인트를 건드리지 않는다.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from predict import CHECKPOINT
import eval_cache

CALIB_SEED = 1234       # probability_calibration_fit.py 와 같은 값 — 같은 재분할
ALPHA = float(os.getenv("CONFORMAL_ALPHA", "0.1"))   # 목표 커버리지 1-ALPHA = 90%
MIN_GROUP_N = 200       # 층화 그룹별 분위수를 따로 낼 최소 보정 표본 수
COVERAGE_TOL = 0.02     # 평가용 실측 커버리지가 목표보다 이만큼 이상 못 미치면 기각
# 강수 층화 — 예측된 강수확률(rain_prob) 구간. 참값 습윤 여부가 아니라 추론
# 시점에 실제로 쓸 수 있는 값만 층화 축으로 쓴다(위 모듈 docstring 참고).
RAIN_PROB_EDGES = [0.0, 0.1, 0.3, 0.6, 1.01]


def _signed_quantiles(e, alpha):
    """부호 있는 잔차 e 에서 (α/2, 1−α/2) 분위수 — 유한표본 보정 포함."""
    n = len(e)
    e_sorted = np.sort(e)
    lo_rank = int(np.ceil((n + 1) * (alpha / 2))) - 1
    hi_rank = int(np.ceil((n + 1) * (1 - alpha / 2))) - 1
    lo_rank = min(max(lo_rank, 0), n - 1)
    hi_rank = min(max(hi_rank, 0), n - 1)
    return float(e_sorted[lo_rank]), float(e_sorted[hi_rank])


def _fit_one(name, y_true, y_pred, group, group_label, calib_mask, alpha,
             clip_min=None, wet_mask=None):
    """
    group      — 층화 축 값 배열(관측소 코드 또는 rain_prob 버킷 인덱스).
    group_label— 로그에 찍을 그룹 이름 사전(선택).
    wet_mask   — 있으면 "습윤 조건부 커버리지"를 별도로 보고한다(참값 기반이라
                 채택 기준에는 넣지 않음, 투명성 목적으로만 출력).
    """
    ec, ee = calib_mask, ~calib_mask
    e_all = y_true - y_pred

    q_lo, q_hi = _signed_quantiles(e_all[ec], alpha)
    lo_global = y_pred + q_lo
    hi_global = y_pred + q_hi
    if clip_min is not None:
        lo_global = np.clip(lo_global, clip_min, None)

    per_group = {}
    group_eval_cov = {}
    lo_strat = np.empty(len(y_pred)); hi_strat = np.empty(len(y_pred))
    for g in sorted(set(group.tolist())):
        g_calib = ec & (group == g)
        g_eval = ee & (group == g)
        g_all = (group == g)
        gname = group_label.get(g, str(g)) if group_label else str(g)
        if g_calib.sum() < MIN_GROUP_N:
            ql, qh, fallback = q_lo, q_hi, True
        else:
            ql, qh = _signed_quantiles(e_all[g_calib], alpha)
            fallback = False
        per_group[gname] = {"q_lo": ql, "q_hi": qh, "n_calib": int(g_calib.sum()),
                             "fallback_global": fallback}
        lo_strat[g_all] = y_pred[g_all] + ql
        hi_strat[g_all] = y_pred[g_all] + qh
        lo_g = np.clip(lo_strat[g_all], clip_min, None) if clip_min is not None else lo_strat[g_all]
        g_eval_local = ee[g_all]
        group_eval_cov[gname] = (
            float(((y_true[g_all][g_eval_local] >= lo_g[g_eval_local])
                   & (y_true[g_all][g_eval_local] <= hi_strat[g_all][g_eval_local])).mean())
            if g_eval_local.sum() else float("nan"))
    if clip_min is not None:
        lo_strat = np.clip(lo_strat, clip_min, None)

    cov_strat = float(((y_true[ee] >= lo_strat[ee]) & (y_true[ee] <= hi_strat[ee])).mean())
    width_strat = float((hi_strat[ee] - lo_strat[ee]).mean())

    print(f"\n{'='*74}\n[{name}]  보정용 {int(ec.sum()):,}개 / 평가용 {int(ee.sum()):,}개")
    print(f"  전역 구간폭 [{q_lo:+.4f}, {q_hi:+.4f}]  |  층화 평가 커버리지 {cov_strat:.4f}"
          f"  (목표 {1-alpha:.2f})  |  평균 구간폭 {width_strat:.4f}")
    for gname, v in sorted(per_group.items()):
        print(f"    {gname:>12}: n_calib={v['n_calib']:>7,}  "
              f"커버리지={group_eval_cov[gname]:.3f}"
              f"{'  (전역 대체)' if v['fallback_global'] else ''}")

    cov_wet = None
    if wet_mask is not None:
        we = ee & wet_mask
        cov_wet = float(((y_true[we] >= lo_strat[we]) & (y_true[we] <= hi_strat[we])).mean())
        print(f"  [참고, 채택 기준 아님] 습윤 조건부 커버리지 {cov_wet:.4f}  "
              f"(n={int(we.sum()):,}) — 참값 기반이라 분포무관 방법으로 보장 불가")

    adopt = cov_strat >= (1 - alpha) - COVERAGE_TOL
    print(f"  판정: {'채택' if adopt else f'기각(커버리지 {cov_strat:.4f} < 목표 허용치)'}")

    return {
        "alpha": alpha,
        "q_lo_global": q_lo, "q_hi_global": q_hi,
        "coverage_eval_stratified": cov_strat,
        "width_eval_stratified": width_strat,
        "coverage_eval_wet_conditional": cov_wet,
        "per_group": per_group,
        "clip_min": clip_min,
    }, adopt


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_apply = "--apply" in sys.argv
    ckpt_path = args[0] if args else CHECKPOINT

    data = eval_cache.load(ckpt_path)
    stn = data["stn"]
    n = len(stn)
    rng = np.random.RandomState(CALIB_SEED)
    calib_mask = rng.rand(n) < 0.5

    temp_result, temp_adopt = _fit_one(
        "기온", data["temp_true"], data["temp_pred"], stn, None, calib_mask, ALPHA)

    rain_prob = data["rain_prob"]
    wet_thresh = float(data["wet_thresh"])
    bucket = np.digitize(rain_prob, RAIN_PROB_EDGES) - 1
    bucket_label = {b: f"[{RAIN_PROB_EDGES[b]:.1f},{RAIN_PROB_EDGES[b+1]:.1f})"
                     for b in range(len(RAIN_PROB_EDGES) - 1)}
    wet_mask = data["precip_true"] >= wet_thresh
    precip_result, precip_adopt = _fit_one(
        "강수", data["precip_true"], data["precip_pred"], bucket, bucket_label,
        calib_mask, ALPHA, clip_min=0.0, wet_mask=wet_mask)

    print(f"\n{'='*74}\n요약")
    print(f"  기온 — 목표 커버리지 {1-ALPHA:.2f}, 실측 {temp_result['coverage_eval_stratified']:.4f}, "
          f"{'채택' if temp_adopt else '기각'}")
    print(f"  강수 — 목표 커버리지 {1-ALPHA:.2f}, 실측 {precip_result['coverage_eval_stratified']:.4f}, "
          f"{'채택' if precip_adopt else '기각'} "
          f"(습윤 조건부 참고값 {precip_result['coverage_eval_wet_conditional']:.4f})")

    if not do_apply:
        print("\n(--apply 를 주지 않아 체크포인트는 그대로 둔다)")
        return

    if not (temp_adopt or precip_adopt):
        print("\n채택된 항목이 없어 체크포인트를 바꾸지 않는다.")
        return

    backup = ckpt_path.replace(".pt", "_before_conformal.pt")
    if not os.path.exists(backup):
        shutil.copy2(ckpt_path, backup)
        print(f"\n백업: {backup}")

    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    ckpt["conformal_interval"] = {
        "method": "split conformal, signed residual quantiles (Mondrian-stratified)",
        "calib_seed": CALIB_SEED,
        "alpha": ALPHA,
        "temp": {"strata": "station", **temp_result} if temp_adopt else None,
        "precip": ({"strata": "rain_prob_bucket", "rain_prob_edges": RAIN_PROB_EDGES,
                     **precip_result} if precip_adopt else None),
    }
    # 공유 파일 저장은 고유 tmp + os.replace 로 원자적으로(CLAUDE.md 6절).
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(ckpt_path) or ".", suffix=".pt")
    os.close(fd)
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)
    print(f"체크포인트에 반영: {ckpt_path}")


if __name__ == "__main__":
    main()
