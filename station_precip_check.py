"""
station_precip_check.py — 관측소별 강수 발생 F1 을 검증셋과 온라인 기록에서
**같은 정의로** 재어 대조한다(2026-10-21).

## 왜 필요한가

2026-10-21 온라인 대조에서 강수 발생 F1 이 전체 0.617 로 검증값(0.628)과
일치했는데, **제주(184)만 0.357** 이고 나머지 11곳은 0.615~0.727 이었다.
관측소별 **검증** F1 을 잰 적이 없어 두 해석을 가를 수 없었다.

- **본래 수준** — 검증셋에서도 그 관측소가 낮다. 서빙 문제가 아니라 입력의
  한계다(기록만 한다).
- **서빙 이상** — 검증셋에서는 다른 관측소 수준인데 온라인만 낮다. 그
  관측소의 서빙 입력(실시간 관측·수치예보 창)을 추적해야 한다.

## 재는 방법

검증 쪽은 `eval_cache.load_features()` 를 재사용하고(데이터셋을 다시 만들지
않는다), 서빙값은 `metrics_report.py --precip-gbm` 과 같은 정의다 —
`occ ≥ τ` 이면 양(amount), 아니면 0 이고, 서빙값·실측 모두 0.1mm 이상을
강수로 본다. **전체 F1 이 배포 기록값과 같은지 먼저 대조하고** 어긋나면
멈춘다(정의가 다르면 관측소별 표도 의미가 없다).

온라인 쪽은 적중률 로그 스냅숏(`--log`)에서 현재 배포 신원(`model_id`)의
대조 완료 기록만 쓴다. 판정은 화면과 같은 `accuracy._is_wet` 이다.

온라인 표본은 특정 계절에 몰려 있으므로 검증 쪽에 **같은 목표월로 거른
부분집합**도 함께 낸다. 신뢰구간은 날짜 블록 부트스트랩이다(연속 시간 표본은
독립이 아니다 — 2절 규약).

실행: python station_precip_check.py --log cache/accuracy_log_snapshot.json
      (온라인 로그는 +6h 만 기록하므로 +6h 만 대조한다)
"""
import argparse
import json
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

import eval_cache

CKPT = "./checkpoints/numerical_trichef.pt"
GBM = "./checkpoints/precip_gbm.npz"
HIT_PRECIP_THRESH = 0.1     # metrics_report 와 같은 값이어야 한다
EVENT_GAP_HOURS = 3        # 이 간격 이내로 이어진 강수 시간은 한 사건
B, SEED = 2000, 0


def counts(wet_pred, wet_true):
    tp = int((wet_pred & wet_true).sum())
    fp = int((wet_pred & ~wet_true).sum())
    fn = int((~wet_pred & wet_true).sum())
    return tp, fp, fn


def f1_of(tp, fp, fn):
    return 2 * tp / (2 * tp + fp + fn) if tp + fp + fn else float("nan")


def n_events(stn, ts, wet_true):
    """관측소별로 강수 시간을 `EVENT_GAP_HOURS` 이내 연속 구간으로 묶어 센다."""
    n = 0
    by = defaultdict(list)
    for s, t in zip(stn[wet_true], ts[wet_true]):
        by[s].append(datetime.strptime(str(t)[:12], "%Y%m%d%H%M"))
    for lst in by.values():
        lst.sort()
        n += 1 + sum(1 for a, b in zip(lst, lst[1:])
                     if b - a > timedelta(hours=EVENT_GAP_HOURS))
    return n


def block_ci(day, wet_pred, wet_true):
    """날짜 블록 부트스트랩 95% CI — 날짜 단위로 (tp,fp,fn) 을 모아 재표본한다."""
    days, inv = np.unique(day, return_inverse=True)
    agg = np.zeros((len(days), 3), dtype=np.int64)
    np.add.at(agg[:, 0], inv, wet_pred & wet_true)
    np.add.at(agg[:, 1], inv, wet_pred & ~wet_true)
    np.add.at(agg[:, 2], inv, ~wet_pred & wet_true)
    rng = np.random.default_rng(SEED)
    s = agg[rng.integers(0, len(days), size=(B, len(days)))].sum(axis=1)
    den = 2 * s[:, 0] + s[:, 1] + s[:, 2]
    f = np.where(den > 0, 2 * s[:, 0] / np.maximum(den, 1), np.nan)
    lo, hi = np.nanpercentile(f, [2.5, 97.5])
    return float(lo), float(hi), len(days)


def table(title, stn, ts, wet_pred, wet_true, focus):
    """관측소별 표. 반환: {stn: (f1, ci, n_event)}"""
    print(f"\n── {title} — 표본 {len(stn):,}")
    tp, fp, fn = counts(wet_pred, wet_true)
    print(f"  전체 F1 {f1_of(tp, fp, fn):.4f} (TP {tp:,} · FP {fp:,} · FN {fn:,}"
          f" · 사건 {n_events(stn, ts, wet_true):,})")
    print(f"  {'관측소':>6}{'F1':>8}{'정밀도':>9}{'재현율':>9}{'기저율':>9}"
          f"{'강수시간':>9}{'사건':>6}   95% CI(날짜 블록)")
    day = np.array([str(t)[:8] for t in ts])
    out = {}
    for s in sorted(set(stn.tolist())):
        m = stn == s
        tp, fp, fn = counts(wet_pred[m], wet_true[m])
        f1 = f1_of(tp, fp, fn)
        p = tp / (tp + fp) if tp + fp else float("nan")
        r = tp / (tp + fn) if tp + fn else float("nan")
        ev = n_events(stn[m], ts[m], wet_true[m])
        ci = block_ci(day[m], wet_pred[m], wet_true[m]) if s == focus else None
        ci_txt = f"   [{ci[0]:.3f}, {ci[1]:.3f}] ({ci[2]}일)" if ci else ""
        mark = " ◀" if s == focus else ""
        print(f"  {s:>6}{f1:8.3f}{p:9.3f}{r:9.3f}{(tp + fn) / m.sum():9.3f}"
              f"{tp + fn:9,}{ev:6,}{ci_txt}{mark}")
        out[s] = (f1, ci, ev)
    return out


def rank_note(res, focus):
    others = sorted(v[0] for k, v in res.items() if k != focus)
    f = res[focus][0]
    below = sum(1 for o in others if o < f)
    return f"{focus} 은 {len(others) + 1}곳 중 F1 {f:.3f} 로 하위에서 {below + 1}번째"


def station_tau(occ_p, amt_p, truth, stn, ts, tau, min_gain=0.01):
    """관측소별 판정선을 따로 고르면 나아지는가(2026-10-22, NEXT.md P3-6).

    2절 규약대로 **보정용(목표일 짝수)에서 고르고 평가용(홀수)에서만 채점**한다.
    순이득이 `min_gain` 미만이면 채택하지 않는다. 한 관측소만 보지 않고 전부
    재는 이유 — 기준을 넘는 관측소가 몇 곳 나오는지와 손해 본 곳의 수를 함께
    보아야 선택 잡음과 진짜 이득을 가를 수 있다.
    """
    cand = np.arange(0.05, 0.96, 0.005)
    calib = (ts // 10 ** 4) % 2 == 0
    true = truth >= HIT_PRECIP_THRESH

    def f1(t, m):
        return f1_of(*counts(np.where(occ_p[m] >= t, amt_p[m], 0.0) >= HIT_PRECIP_THRESH,
                             true[m]))
    deltas = []
    print(f"\n관측소별 판정선(보정용에서 선택 → 평가용 채점, 전역 τ={tau:.3f})")
    for s_ in sorted(set(stn)):
        m = stn == s_
        best = cand[int(np.argmax([f1(t, m & calib) for t in cand]))]
        g0, g1 = f1(tau, m & ~calib), f1(best, m & ~calib)
        deltas.append(g1 - g0)
        print(f"  {s_}: τ {best:.3f} · 평가용 F1 {g0:.4f} → {g1:.4f} (Δ {g1 - g0:+.4f})"
              + ("  ← 기준 충족" if g1 - g0 >= min_gain else ""))
    d = np.array(deltas)
    print(f"  기준 충족 {int((d >= min_gain).sum())}곳 · 손해 ≥{min_gain} {int((d <= -min_gain).sum())}곳 · "
          f"평균 Δ {d.mean():+.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station-tau", action="store_true",
                    help="관측소별 판정선 선택의 순이득을 보정용/평가용 분리로 잰다")
    ap.add_argument("--log", default=None, help="적중률 로그 스냅숏(JSON)")
    ap.add_argument("--station", default="184")
    ap.add_argument("--model-id", default=None,
                    help="온라인 기록을 거를 신원(생략하면 배포 체크포인트에서 계산)")
    args = ap.parse_args()
    focus = args.station

    import accuracy
    from precip_gbm import load

    print(f"\n{'=' * 82}\n 관측소별 강수 발생 F1 — 검증셋 vs 온라인 (+6h, 초점 {focus})"
          f"\n{'=' * 82}")

    # ── 검증셋 ────────────────────────────────────────────────
    f = eval_cache.load_features(CKPT)
    amt, occ, meta = load(GBM)
    tau = float(meta["meta_gate_tau"])
    x = f["x_val"]
    served = np.where(occ.predict_proba1(x) >= tau,
                      np.clip(amt.predict(x), 0, None), 0.0)
    truth = f["precip_true_val"].astype(np.float64)
    v_pred = served >= HIT_PRECIP_THRESH
    v_true = truth >= HIT_PRECIP_THRESH
    v_stn = f["stn_val"].astype(int).astype(str)
    v_ts = f["tgt_ts_val"].astype(np.int64)

    # 기대값은 상수가 아니라 체크포인트의 기록값이다 — `metrics_report
    # --patch-checkpoint` 가 같은 정의로 적어 둔 값이므로 모델이 바뀌어도 따라온다.
    import torch
    expected = float(torch.load(CKPT, map_location="cpu",
                                weights_only=True)["val_precip_wet_f1"])
    f1_all = f1_of(*counts(v_pred, v_true))
    print(f"  판정선 τ={tau:.3f} · 검증셋 전체 F1 {f1_all:.4f} (기록값 {expected:.4f})")
    if abs(f1_all - expected) > 5e-4:
        raise SystemExit("전체 F1 이 배포 기록값과 다르다 — 채점 정의가 어긋났으므로 "
                         "관측소별 표를 해석하지 않는다.")

    if args.station_tau:
        station_tau(occ.predict_proba1(x), np.clip(amt.predict(x), 0, None),
                    truth, v_stn, v_ts, tau)
        return

    res_val = table("검증셋 전체", v_stn, v_ts, v_pred, v_true, focus)

    # ── 온라인 ────────────────────────────────────────────────
    if not args.log:
        print(f"\n  {rank_note(res_val, focus)} (검증셋 전체)")
        return
    mid = args.model_id or accuracy.model_fingerprint(CKPT)
    with open(args.log, encoding="utf-8") as fp:
        recs = [r for r in json.load(fp)
                if r.get("model_id") == mid and r.get("actual_precip") is not None]
    if not recs:
        raise SystemExit(f"로그에 {mid} 의 대조 완료 기록이 없다.")
    o_stn = np.array([str(r["station"]) for r in recs])
    o_ts = np.array([int(str(r["target_time"])[:12]) for r in recs], dtype=np.int64)
    o_pred = np.array([accuracy._is_wet(r["pred_precip"]) for r in recs])
    o_true = np.array([accuracy._is_wet(r["actual_precip"]) for r in recs])

    # 같은 계절 — 온라인 표본의 목표월로 검증셋을 거른다.
    months = sorted({int(str(t)[4:6]) for t in o_ts})
    m = np.isin((v_ts // 10 ** 6) % 100, months)
    res_season = table(f"검증셋 같은 목표월({','.join(map(str, months))}월)",
                       v_stn[m], v_ts[m], v_pred[m], v_true[m], focus)

    span = f"{str(o_ts.min())[:8]}~{str(o_ts.max())[:8]}"
    res_on = table(f"온라인 {mid[:17]} ({span})",
                   o_stn, o_ts, o_pred, o_true, focus)

    # ── 판정 ──────────────────────────────────────────────────
    print(f"\n── 판정({focus})")
    if focus not in res_on or res_on[focus][1] is None:
        print(f"  온라인 기록에 {focus} 가 없어 판정하지 않는다.")
        return
    for name, r in (("검증셋 전체", res_val), ("검증셋 같은 목표월", res_season),
                    ("온라인", res_on)):
        print(f"  {name:<14} {rank_note(r, focus)} · 사건 {r[focus][2]}")
    lo_on, hi_on = res_on[focus][1][:2]
    lo_s, hi_s = res_season[focus][1][:2]
    overlap = lo_on <= hi_s and lo_s <= hi_on
    print(f"  온라인 CI [{lo_on:.3f}, {hi_on:.3f}] vs 검증 같은 목표월 CI "
          f"[{lo_s:.3f}, {hi_s:.3f}] → {'겹친다' if overlap else '겹치지 않는다'}")
    if overlap:
        # 신뢰구간 겹침은 약한 검정이다 — "같다"가 아니라 "구별되지 않는다"만 말한다.
        print("  → 온라인 값은 검증셋의 같은 계절 분포와 구별되지 않는다. "
              "서빙 이상의 근거는 없다(약한 검정 — 동일성을 보이지는 않는다).")
    else:
        print("  → 온라인 값이 검증셋의 같은 계절 분포를 벗어난다. "
              "이 관측소의 서빙 입력을 추적할 근거가 있다.")


if __name__ == "__main__":
    main()
