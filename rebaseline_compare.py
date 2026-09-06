"""
rebaseline_compare.py — 검증셋이 서로 다른 두 체크포인트를 공정하게 비교한다.

**왜 필요한가.** `promote_checkpoint.py` 게이트 3 은 후보와 배포본의
기준선(naive MAE)이 다르면 즉시 FAIL 시킨다. 옳은 검사다 — 기준선이 다르면
검증셋이 다르다는 뜻이고, 다른 표본에서 잰 수치를 나란히 놓는 것은 비교가
아니기 때문이다(CLAUDE.md 2절).

그런데 이 검사 때문에 **표본 구성이 바뀐 재학습은 원리적으로 승격할 수
없다.** 실제로 세 번 연속 막혔다 — 분할 알고리즘 교체(2026-08-31), 그래디언트
누적(2026-09-01), 수치예보(2026-09-06). 그때마다 "다음에 일괄 처리"로
미뤄졌고, 그 사이 개선이 하나도 배포되지 못했다.

**해결 — 두 모델 모두에게 학습 밖인 표본에서만 비교한다.**

  후보의 검증셋 ∩ 배포본의 검증셋

이 교집합은 정의상 어느 쪽도 학습에 쓰지 않은 표본이다. 따라서 누수가
없고, 한쪽에 유리한 편향도 없다. 분할 알고리즘이 달라도(구 randperm vs
날짜 해시), 표본 구성이 달라도(2013~ vs 2016~) 성립한다 — 교집합만 쓰면
되기 때문이다.

대가는 표본 수다. 20% 검증셋 둘의 교집합이므로 대략 전체의 4% 로 줄어든다.
그래서 **날짜 블록 부트스트랩으로 신뢰구간을 함께 낸다** — 표본이 줄면
차이가 유의하지 않게 나올 수 있고, 그 사실 자체가 판단에 필요한 정보다.

**극한기상 헤드는 각자의 판정선에서 채점한다.** t=0.5 고정 비교는
성립하지 않는다 — 손실 구성(pos_weight)이 바뀌면 확률 눈금 자체가 이동해
같은 임계값이 다른 동작점을 뜻하기 때문이다. 판정선은 보정용 절반에서
고르고 평가용 절반에서만 채점한다(CLAUDE.md 규칙 9 와 같은 절차).
아울러 **공식 라벨 표본으로만 채점한다** — `EXTREME_OFFSEASON_NEGATIVE`
를 켠 체크포인트는 채점 표본이 늘어나므로, 안 맞추면 비교가 성립하지 않는다.

실행 (GPU 컨테이너, 순차):
  python rebaseline_compare.py <배포본> <후보>
"""
import argparse

import numpy as np

import eval_cache
from predict import PRECIP_PROB_GATE, PRECIP_PROB_GATE_BY_LEAD

WET_THRESH = 0.1
N_BOOT = 2000
BOOT_SEED = 20260907
CALIB_SEED = 1234       # threshold_validation.py 와 같은 분할 시드


def _key(d):
    return np.char.add(np.char.add(d["stn"].astype(str), "@"), d["tgt_ts"].astype(str))


def _served_precip(d):
    gate = PRECIP_PROB_GATE_BY_LEAD.get(int(d["lead_hours"]), PRECIP_PROB_GATE)
    return np.where(d["rain_prob"] < gate, 0.0, d["precip_pred"])


def _pick_threshold(prob, y, calib):
    """보정용 절반에서 F1 최대 임계값을 고른다(평가용은 보지 않는다)."""
    if calib.sum() == 0 or y[calib].sum() == 0:
        return 0.5
    grid = np.unique(np.round(np.quantile(prob[calib], np.linspace(0.5, 0.9999, 200)), 4))
    best, best_f1 = 0.5, -1.0
    for t in grid:
        pred = prob[calib] >= t
        tp = int((pred & (y[calib] == 1)).sum())
        fp = int((pred & (y[calib] == 0)).sum())
        fn = int((~pred & (y[calib] == 1)).sum())
        f1 = 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0
        if f1 > best_f1:
            best, best_f1 = float(t), f1
    return best


def _f1(prob, y, t):
    pred = prob >= t
    tp = int((pred & (y == 1)).sum())
    fp = int((pred & (y == 0)).sum())
    fn = int((~pred & (y == 1)).sum())
    return 2 * tp / (2 * tp + fp + fn) if (2 * tp + fp + fn) else 0.0


def metrics(d, idx, thr, eval_mask):
    """교집합 표본(idx)에서의 지표. thr 은 헤드별 판정선."""
    out = {}
    tt, tp_ = d["temp_true"][idx], d["temp_pred"][idx]
    out["temp_mae"] = float(np.abs(tp_ - tt).mean())
    pt = d["precip_true"][idx]
    pp = _served_precip(d)[idx]
    out["precip_mae_served"] = float(np.abs(pp - pt).mean())
    wet_t, wet_p = pt >= WET_THRESH, pp >= WET_THRESH
    tp_w = int((wet_p & wet_t).sum()); fp_w = int((wet_p & ~wet_t).sum())
    fn_w = int((~wet_p & wet_t).sum())
    out["precip_f1"] = (2 * tp_w / (2 * tp_w + fp_w + fn_w)
                        if (2 * tp_w + fp_w + fn_w) else 0.0)
    out["precip_mae_wet"] = (float(np.abs(pp[wet_t] - pt[wet_t]).mean())
                             if wet_t.any() else float("nan"))
    for head, short in (("heatwave", "heat"), ("coldwave", "cold"), ("dust", "dust")):
        m = eval_mask[short]
        if m.sum() == 0:
            continue
        out[f"{short}_f1"] = _f1(d[f"{short}_prob"][idx][m], d[f"y_{head}"][idx][m].astype(int),
                                 thr[short])
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", help="배포본(현행)")
    ap.add_argument("ckpt_b", help="후보")
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    print(f"현행 추론: {args.ckpt_a}")
    A = eval_cache.build(args.ckpt_a, args.batch)
    print(f"후보 추론: {args.ckpt_b}")
    B = eval_cache.build(args.ckpt_b, args.batch)

    ka, kb = _key(A), _key(B)
    common = np.intersect1d(ka, kb)
    if len(common) == 0:
        raise SystemExit("두 검증셋의 교집합이 없다 — 비교할 수 없다.")
    ia = np.searchsorted(ka, common, sorter=np.argsort(ka))
    ia = np.argsort(ka)[ia]
    ib = np.argsort(kb)[np.searchsorted(kb, common, sorter=np.argsort(kb))]
    assert (ka[ia] == kb[ib]).all(), "표본 정렬이 어긋났다"

    dates = np.array([s[:8] for s in A["tgt_ts"][ia].astype(str)])
    print(f"\n교집합 표본 {len(common):,}개 · 고유 날짜 {len(np.unique(dates)):,}개")
    print(f"  (현행 검증 {len(ka):,} · 후보 검증 {len(kb):,} — 양쪽 모두 학습 밖)")

    # 공식 라벨 마스크 — 없으면(구버전 캐시) 일반 마스크로 폴백한다.
    def _mask(d, idx, short):
        k = f"{short}_mask_official"
        m = d[k] if k in d else d[f"{short}_mask"]
        return m[idx].astype(bool)

    rng = np.random.RandomState(CALIB_SEED)
    uniq = np.unique(dates)
    calib_dates = set(rng.choice(uniq, size=len(uniq) // 2, replace=False).tolist())
    is_calib = np.array([x in calib_dates for x in dates])

    thr, masks = {}, {}
    for label, d, idx in (("A", A, ia), ("B", B, ib)):
        t, mk = {}, {}
        for head, short in (("heatwave", "heat"), ("coldwave", "cold"), ("dust", "dust")):
            # 두 모델의 채점 표본을 **교집합으로 다시 맞춘다** — 한쪽만
            # offseason 을 채웠으면 마스크가 달라 같은 질문이 되지 않는다.
            mk[short] = _mask(A, ia, short) & _mask(B, ib, short)
            y = d[f"y_{head}"][idx].astype(int)
            p = d[f"{short}_prob"][idx]
            sel = mk[short] & is_calib
            t[short] = _pick_threshold(p, y, sel)
        thr[label], masks[label] = t, mk
    print("  판정선(보정용 절반에서 선정) — "
          + " · ".join(f"{s}: 현행 {thr['A'][s]:.3f} / 후보 {thr['B'][s]:.3f}"
                       for s in ("heat", "cold", "dust")))

    # 채점은 보정에 쓰지 않은 절반에서만 한다.
    ev = ~is_calib
    eva = {s: masks["A"][s] & ev for s in masks["A"]}
    ia_e, ib_e = ia[ev], ib[ev]
    eva_e = {s: masks["A"][s][ev] for s in masks["A"]}
    ma = metrics(A, ia_e, thr["A"], eva_e)
    mb = metrics(B, ib_e, thr["B"], eva_e)

    dates_e = dates[ev]
    uniq_e = np.unique(dates_e)
    by_date = {u: np.flatnonzero(dates_e == u) for u in uniq_e}
    boot = {k: [] for k in ma}
    rs = np.random.RandomState(BOOT_SEED)
    for b in range(N_BOOT):
        pick = rs.choice(uniq_e, size=len(uniq_e), replace=True)
        sel = np.concatenate([by_date[u] for u in pick])
        sa = {s: eva_e[s][sel] for s in eva_e}
        x = metrics(A, ia_e[sel], thr["A"], sa)
        y = metrics(B, ib_e[sel], thr["B"], sa)
        for k in boot:
            if k in x and k in y:
                boot[k].append(y[k] - x[k])
        if (b + 1) % 500 == 0:
            print(f"  {b + 1}/{N_BOOT}")

    print(f"\n{'지표':<20}{'현행':>10}{'후보':>10}{'Δ':>10}{'95% CI':>22}  유의")
    lower_better = {"temp_mae", "precip_mae_served", "precip_mae_wet"}
    for k in ma:
        if k not in mb or not boot[k]:
            continue
        arr = np.array(boot[k])
        lo, hi = np.percentile(arr, [2.5, 97.5])
        sig = "*" if (lo > 0 or hi < 0) else " "
        print(f"{k:<20}{ma[k]:10.4f}{mb[k]:10.4f}{mb[k] - ma[k]:+10.4f}"
              f"{f'[{lo:+.4f}, {hi:+.4f}]':>22}  {sig}")
    print("\n* = 95% CI 가 0 을 포함하지 않음. "
          + f"{'·'.join(sorted(lower_better))} 는 낮을수록, 나머지 F1 은 높을수록 후보가 우세.")
    print("표본은 두 모델 모두의 검증셋 교집합이라 어느 쪽에도 학습 누수가 없다.")


if __name__ == "__main__":
    main()
