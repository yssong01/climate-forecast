"""
stack_compare.py — 두 **서빙 구성**(주 체크포인트 + 강수 GBM + 극한기상 GBM)을
같은 검증셋에서 서빙과 같은 정의로 채점하고, 차이를 날짜 블록 부트스트랩으로
검정한다(2026-10-22).

## 왜 필요한가

`bootstrap_ci_compare.py` 는 **신경망 체크포인트 둘**의 출력을 비교한다. 그런데
지금 화면의 강수·극한기상은 GBM 이 내고, GBM 은 주 체크포인트의 정규화 통계로
표준화한 입력을 받는다. 재학습으로 그 통계가 바뀌면 GBM 도 함께 바뀌므로, 비교
단위는 체크포인트 하나가 아니라 **(주 체크포인트, 강수 GBM, 극한 GBM) 묶음**이다.

## 채점 정의 — 서빙·`metrics_report.py` 와 같다

- 강수: `occ ≥ τ`(모델 파일의 `meta_gate_tau`)이면 양(amount), 아니면 0. 서빙값·실측
  모두 0.1mm 이상을 강수로 본다. 발생 F1 과 강수 구간 조건부 MAE.
- 극한기상: 모델 파일의 보정 곡선을 적용한 확률이 그 파일의 판정선 이상이면 발생.
  채점 표본은 `metrics_report.precision_block` 과 같이 공식 라벨이 있는 표본이다.
- 두 묶음의 검증 표본은 **행 단위로 같아야** 한다(관측소·목표 시각 대조) — 다르면
  짝지은 비교가 성립하지 않으므로 멈춘다.

평가용 절반(목표일 홀수 — 판정선·보정·하이퍼파라미터 선택에 쓰지 않은 날)의 값도
함께 낸다. 신뢰구간은 날짜 단위 블록 부트스트랩(B=2,000, 짝지음)이다.

실행:
    python stack_compare.py <주A> <강수GBM_A> <극한GBM_A> <주B> <강수GBM_B> <극한GBM_B>
"""
import argparse
import sys

import numpy as np

import eval_cache
import extreme_gbm as eg
import precip_gbm as pg

HIT = 0.1
N_BOOT = 2000


def _f1(pred, true):
    tp = np.sum(pred & true)
    fp = np.sum(pred & ~true)
    fn = np.sum(~pred & true)
    return 2 * tp / max(2 * tp + fp + fn, 1)


def stack_outputs(main, pgbm, egbm):
    """한 서빙 구성의 검증셋 출력(서빙 정의)."""
    f = eval_cache.load_features(main)
    amt, occ, pmeta = pg.load(pgbm)
    pg.check_input_contract(pmeta, f, pgbm, "강수 GBM")
    models, emeta = eg.load(egbm)
    pg.check_input_contract(emeta, f, egbm, "극한기상 GBM")
    import gbm_extra                          # GBM 전용 추가 특징 — 묶음마다 사양이 다를 수 있다
    xp = gbm_extra.augment(pmeta, f["x_val"], f["stn_val"], f["src_ts_val"], int(f["lead_hours"]))
    xe = gbm_extra.augment(emeta, f["x_val"], f["stn_val"], f["src_ts_val"], int(f["lead_hours"]))
    served = np.where(occ.predict_proba1(xp) >= float(pmeta["meta_gate_tau"]),
                      np.clip(amt.predict(xp), 0, None), 0.0)
    out = {"stn": f["stn_val"], "ts": f["tgt_ts_val"].astype(np.int64),
           "precip_served": served, "precip_true": f["precip_true_val"]}
    for head, short in (("heatwave", "heat"), ("coldwave", "cold"), ("dust", "dust")):
        mask = f[f"{short}_mask_val"].astype(bool)
        off = f.get(f"{short}_mask_official_val")
        if off is not None and int(off.sum()) and int(off.sum()) != int(mask.sum()):
            mask = mask & off.astype(bool)
        p = eg.calibrated_batch(models, emeta, head, xe)
        out[f"{head}_pred"] = p >= eg.threshold(emeta, head)
        out[f"{head}_true"] = f[f"y_{head}_val"] > 0
        out[f"{head}_mask"] = mask
    return out


def metrics(o, idx):
    pt = o["precip_true"][idx] >= HIT
    ps = o["precip_served"][idx] >= HIT
    wet = o["precip_true"][idx] >= HIT
    m = {"precip_f1": _f1(ps, pt),
         "precip_mae_wet": float(np.mean(np.abs(o["precip_served"][idx][wet]
                                                 - o["precip_true"][idx][wet]))) if wet.any() else np.nan}
    for head in ("heatwave", "coldwave", "dust"):
        mk = o[f"{head}_mask"][idx]
        m[f"{head}_f1"] = _f1(o[f"{head}_pred"][idx][mk], o[f"{head}_true"][idx][mk])
    return m


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("stacks", nargs=6)
    ap.add_argument("--n-boot", type=int, default=N_BOOT)
    args = ap.parse_args()
    a = stack_outputs(*args.stacks[:3])
    b = stack_outputs(*args.stacks[3:])
    if not (np.array_equal(a["stn"], b["stn"]) and np.array_equal(a["ts"], b["ts"])):
        raise SystemExit("두 구성의 검증 표본이 다르다 — 짝지은 비교가 성립하지 않는다.")
    n = len(a["ts"])
    day = a["ts"] // 10 ** 4
    days = np.unique(day)
    rows_of = {d_: np.flatnonzero(day == d_) for d_ in days}
    half = (day % 2 == 1)
    full = np.arange(n)
    ma, mb = metrics(a, full), metrics(b, full)
    ha, hb = metrics(a, np.flatnonzero(half)), metrics(b, np.flatnonzero(half))
    rng = np.random.default_rng(0)
    boots = {k: [] for k in ma}
    for _ in range(args.n_boot):
        idx = np.concatenate([rows_of[d_] for d_ in rng.choice(days, len(days), replace=True)])
        xa, xb = metrics(a, idx), metrics(b, idx)
        for k in ma:
            boots[k].append(xb[k] - xa[k])
    print(f"\n검증 표본 {n:,}행 · 날짜 {len(days):,}개 · 부트스트랩 B={args.n_boot}")
    print(f"  A = {' · '.join(args.stacks[:3])}\n  B = {' · '.join(args.stacks[3:])}")
    print(f"\n{'지표':<16}{'A':>9}{'B':>9}{'Δ(B−A)':>10}   95% CI(날짜 블록)      "
          f"{'평가용 A':>9}{'평가용 B':>9}")
    for k in ma:
        lo, hi = np.nanpercentile(boots[k], [2.5, 97.5])
        sig = "유의" if (lo > 0 or hi < 0) else "—"
        print(f"{k:<16}{ma[k]:>9.4f}{mb[k]:>9.4f}{mb[k] - ma[k]:>+10.4f}   "
              f"[{lo:+.4f}, {hi:+.4f}] {sig:<4}{ha[k]:>9.4f}{hb[k]:>9.4f}")
    print("\n  precip_mae_wet 는 낮을수록, 나머지는 높을수록 B 가 낫다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
