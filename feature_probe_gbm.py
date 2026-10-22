"""
feature_probe_gbm.py — 저장소가 이미 가진 정보 중 현행 GBM 이 쓰지 않는 것이 강수·
극한기상을 개선하는지 재는 탐침(2026-10-22).

## 무엇을 더해 보는가 — 전부 서빙에서 추가 API 없이 만들 수 있는 것

- **O(자기 경향)**: 그 관측소의 1·3·6시간 전 대비 기압·기온·습도 변화, 최근 3시간 강수 합.
  서빙은 저장소 관측 창(`recent_window.json`)에서 만든다.
- **N(이웃 현재값)**: 같은 시각 다른 11개 관측소의 강수 최대·강수 관측소 수, 서해안
  상류(인천·전주·광주) 강수, 이웃 평균 3시간 기압 변화. 역시 저장소 창.
- **W(수치예보 시간창)**: `jma_gsm` `_previous_day1` 의 목표 시각 ±2시간 강수 합·최대,
  예측 시점 이후 24시간 강수 합·최대, 같은 구간 최저·최고 기온, 24시간 뒤 기온 − 현재
  관측 기온. **유효시각이 예측 시점 +24시간 이내인 previous_day1 값은 그 시각보다 24시간
  앞서 발표돼 예측 시점에 이미 존재한다** — 학습·서빙 리드타임이 같다(4절 규약).

## 재는 방법

현행 28특징 GBM(배포와 같은 하이퍼파라미터)과 특징만 더한 GBM 을 같은 학습 표본으로
학습하고, 판정선은 보정용 절반(목표일 짝수)에서 고른 뒤 평가용 절반(홀수)에서만 채점한다.
차이는 평가용 절반의 날짜 블록 짝지은 부트스트랩(B=1,000)으로 검정한다. 결측(창에 값이
없는 시각)은 채우지 않고 NaN 으로 두어 GBM 이 결측 가지로 보낸다 — 탐침이므로 서빙
결측 처리는 채택 시 따로 정한다.

실행: python feature_probe_gbm.py <주 체크포인트> [--groups O,N,W]
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np

import eval_cache

SEED = 42
UPSTREAM = ("112", "146", "156")          # 인천·전주·광주 — 편서풍 상류(서해안)


def _ts(i):
    return datetime.strptime(str(int(i))[:12], "%Y%m%d%H%M")


def _key(dt):
    return dt.strftime("%Y%m%d%H00")


def kma_apparent_temp(ta, rh):
    """기상청 여름철 체감온도(2022-06-02~): Stull 습구온도 기반."""
    tw = (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(ta + rh)
          - np.arctan(rh - 1.67633) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0


# 깨끗한 창의 끝 — previous_day1 의 미래 유효시각 값은 실시간으로 확정값이 아니다.
# data 브랜치 스냅숏 대조(2026-10-22, 조사)에서 유효시각까지 3~9시간이면 85% 가 학습
# 아카이브와 같고 11~13시간이면 34%, 17시간 이상이면 0% 였다. 그래서 V 그룹은 예측
# 시점 +9시간까지만 쓴다.
CLEAN_H = 9
DAY2 = {}
_DAY2_PATH = os.environ.get("DAY2_ARCHIVE", "")
if _DAY2_PATH and os.path.exists(_DAY2_PATH):
    with open(_DAY2_PATH, encoding="utf-8") as _fp:
        DAY2 = json.load(_fp)
# day2 창에 쓸 유효시각 상한(예측 시점 기준 시간). 비우면 상한 없음 = 아카이브 확정값을 그대로
# 쓰는 **상한 측정**이다(서빙 재현성은 따로 재야 한다 — nwp_drift_check.py).
DAY2_CLEAN_H = int(os.environ["DAY2_CLEAN_H"]) if os.environ.get("DAY2_CLEAN_H") else None
# jma_gsm previous_day1 강수의 관측소 주변 ±0.5° 8개 격자점(2016~) — G 그룹.
GSM_NB = {}
_NB_PATH = os.environ.get("GSM_NB_ARCHIVE", "")
if _NB_PATH and os.path.exists(_NB_PATH):
    with open(_NB_PATH, encoding="utf-8") as _fp:
        GSM_NB = json.load(_fp)
LABELS = {}
if os.path.exists("./cache/weather_issue_labels.json"):
    with open("./cache/weather_issue_labels.json", encoding="utf-8") as _fp:
        LABELS = json.load(_fp)
MSM = {}
_MSM_PATH = os.environ.get("MSM_ARCHIVE", "")
if _MSM_PATH and os.path.exists(_MSM_PATH):
    with open(_MSM_PATH, encoding="utf-8") as _fp:
        MSM = json.load(_fp)


def build_extra(stn, src, lead, obs, nwp, groups):
    n = len(stn)
    cols = {}
    if "O" in groups:
        for name in ("dp1", "dp3", "dp6", "dt3", "drh3", "p3sum"):
            cols["O_" + name] = np.full(n, np.nan)
    if "N" in groups:
        for name in ("pmax", "nwet", "pup", "dp3mean"):
            cols["N_" + name] = np.full(n, np.nan)
    if "W" in groups:
        for name in ("pwin_sum", "pwin_max", "p24_sum", "p24_max", "t24_min", "t24_max", "dT24"):
            cols["W_" + name] = np.full(n, np.nan)
    if "V" in groups:
        for name in ("p9_sum", "p9_max", "t9_min", "t9_max", "at9_max", "dT9min"):
            cols["V_" + name] = np.full(n, np.nan)
    if "G" in groups:          # GSM 공간 이웃(3×3)·시간 창(±3h) 강수
        for name in ("mean_V", "max_V", "mean_w3", "ctr_w3", "west_east_w3"):
            cols["G_" + name] = np.full(n, np.nan)
    if "D" in groups:          # jma_gsm previous_day2 — 특보 기준에 맞춘 '목표일·다음날' 창
        for name in ("c_min_d0", "c_min_d1", "c_2day", "c_drop", "h_at_d0", "h_at_d1", "h_2day"):
            cols["D_" + name] = np.full(n, np.nan)
    if "M" in groups:          # jma_msm previous_day1(2022-07~) — 깨끗한 창(+9h) 안만
        for name in ("p_V", "pwin_sum", "pwin_max", "t_V", "rh_V", "cc_V"):
            cols["M_" + name] = np.full(n, np.nan)
    stations = sorted({s for s, _ in obs})
    for i in range(n):
        s, T = str(int(stn[i])), _ts(src[i])
        cur = obs.get((s, _key(T)))
        if "O" in groups and cur is not None:
            for h, nm in ((1, "dp1"), (3, "dp3"), (6, "dp6")):
                past = obs.get((s, _key(T - timedelta(hours=h))))
                if past is not None:
                    cols["O_" + nm][i] = cur[2] - past[2]
                    if h == 3:
                        cols["O_dt3"][i] = cur[0] - past[0]
                        cols["O_drh3"][i] = cur[1] - past[1]
            ps = [obs.get((s, _key(T - timedelta(hours=h)))) for h in (0, 1, 2)]
            if all(p is not None for p in ps):
                cols["O_p3sum"][i] = sum(p[3] for p in ps)
        if "N" in groups:
            nb = [obs.get((o, _key(T))) for o in stations if o != s]
            nb = [r for r in nb if r is not None]
            if nb:
                pr = [r[3] for r in nb]
                cols["N_pmax"][i] = max(pr)
                cols["N_nwet"][i] = sum(p >= 0.1 for p in pr)
                up = [obs.get((o, _key(T))) for o in UPSTREAM if o != s]
                up = [r[3] for r in up if r is not None]
                if up:
                    cols["N_pup"][i] = max(up)
                dps = []
                for o in stations:
                    if o == s:
                        continue
                    a, b = obs.get((o, _key(T))), obs.get((o, _key(T - timedelta(hours=3))))
                    if a is not None and b is not None:
                        dps.append(a[2] - b[2])
                if dps:
                    cols["N_dp3mean"][i] = float(np.mean(dps))
        if "W" in groups:
            row = nwp.get(s, {})
            V = T + timedelta(hours=lead)
            win = [row.get(_key(V + timedelta(hours=h))) for h in range(-2, 3)]
            win = [w for w in win if w is not None]
            if len(win) == 5:
                cols["W_pwin_sum"][i] = sum(w[0] for w in win)
                cols["W_pwin_max"][i] = max(w[0] for w in win)
            day = [row.get(_key(T + timedelta(hours=h))) for h in range(1, 25)]
            day = [w for w in day if w is not None]
            if len(day) == 24:
                cols["W_p24_sum"][i] = sum(w[0] for w in day)
                cols["W_p24_max"][i] = max(w[0] for w in day)
                cols["W_t24_min"][i] = min(w[1] for w in day)
                cols["W_t24_max"][i] = max(w[1] for w in day)
                if cur is not None:
                    cols["W_dT24"][i] = day[-1][1] - cur[0]
        if "G" in groups and GSM_NB:
            nb = GSM_NB.get(s, {})
            ring, nrows, crow = nb.get("ring", []), nb.get("rows", {}), nwp.get(s, {})
            V = T + timedelta(hours=lead)

            def _pts(t):            # 중심 + 8 이웃 강수(9개) — 하나라도 없으면 None
                c, r = crow.get(_key(t)), nrows.get(_key(t))
                return None if (c is None or r is None) else [c[0]] + list(r)
            pv = _pts(V)
            if pv is not None:
                cols["G_mean_V"][i] = float(np.mean(pv))
                cols["G_max_V"][i] = max(pv)
            win = [_pts(V + timedelta(hours=h)) for h in range(-3, 4)]
            if all(w is not None for w in win):
                sums = np.sum(np.array(win), axis=0)          # 지점별 ±3h 합(9개)
                cols["G_mean_w3"][i] = float(sums.mean())
                cols["G_ctr_w3"][i] = float(sums[0])
                west = [sums[1 + j] for j, (dy, dx) in enumerate(ring) if dx < 0]
                east = [sums[1 + j] for j, (dy, dx) in enumerate(ring) if dx > 0]
                cols["G_west_east_w3"][i] = float(np.mean(west) - np.mean(east))
        if "D" in groups and DAY2:
            row = DAY2.get(s, {})
            V0 = T + timedelta(hours=lead)
            d0 = V0.replace(hour=0, minute=0)

            def _window(day, h0, h1, idx):
                vals = []
                for h in range(h0, h1 + 1):
                    V = day + timedelta(hours=h)
                    if DAY2_CLEAN_H is not None and (V - T).total_seconds() / 3600 > DAY2_CLEAN_H:
                        continue
                    w = row.get(_key(V))
                    if w is not None:
                        vals.append(w if idx is None else w[idx])
                return vals
            m0 = _window(d0, 0, 9, 0)
            m1 = _window(d0 + timedelta(days=1), 0, 9, 0)
            if m0:
                cols["D_c_min_d0"][i] = min(m0)
            if m1:
                cols["D_c_min_d1"][i] = min(m1)
            if m0 and m1:
                cols["D_c_2day"][i] = max(min(m0), min(m1))
            past = [obs.get((s, _key(T - timedelta(hours=h)))) for h in range(0, 24)]
            past = [p_[0] for p_ in past if p_ is not None]
            fut = [v for v in (m0 + m1)]
            if past and fut:
                cols["D_c_drop"][i] = min(fut) - min(past)
            a0 = [float(kma_apparent_temp(w[0], w[1])) for w in _window(d0, 12, 17, None)]
            a1 = [float(kma_apparent_temp(w[0], w[1])) for w in _window(d0 + timedelta(days=1), 12, 17, None)]
            if a0:
                cols["D_h_at_d0"][i] = max(a0)
            if a1:
                cols["D_h_at_d1"][i] = max(a1)
            if a0 and a1:
                cols["D_h_2day"][i] = min(max(a0), max(a1))
        if "M" in groups and MSM:
            row = MSM.get(s, {})
            V = T + timedelta(hours=lead)
            if lead + 2 <= CLEAN_H:
                win = [row.get(_key(V + timedelta(hours=h))) for h in range(-2, 3)]
                if all(w is not None for w in win):
                    cols["M_pwin_sum"][i] = sum(w[0] for w in win)
                    cols["M_pwin_max"][i] = max(w[0] for w in win)
            v = row.get(_key(V))
            if v is not None:
                cols["M_p_V"][i], cols["M_t_V"][i], cols["M_rh_V"][i], cols["M_cc_V"][i] = v
        if "V" in groups:
            row = nwp.get(s, {})
            win = [row.get(_key(T + timedelta(hours=h))) for h in range(1, CLEAN_H + 1)]
            win = [w for w in win if w is not None]
            if len(win) == CLEAN_H:
                cols["V_p9_sum"][i] = sum(w[0] for w in win)
                cols["V_p9_max"][i] = max(w[0] for w in win)
                cols["V_t9_min"][i] = min(w[1] for w in win)
                cols["V_t9_max"][i] = max(w[1] for w in win)
                cols["V_at9_max"][i] = max(float(kma_apparent_temp(w[1], w[2])) for w in win)
                if cur is not None:
                    cols["V_dT9min"][i] = cols["V_t9_min"][i] - cur[0]
    names = sorted(cols)
    return np.column_stack([cols[k] for k in names]) if names else np.zeros((n, 0)), names


def f1(pred, true):
    tp = np.sum(pred & true)
    return 2 * tp / max(2 * tp + np.sum(pred & ~true) + np.sum(~pred & true), 1)


def pick_tau(p, y, calib):
    cand = np.arange(0.02, 0.98, 0.005)
    return float(cand[int(np.argmax([f1(p[calib] >= t, y[calib]) for t in cand]))])


def boot_diff(score_fn, days, rows_of, n_boot=1000, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for _ in range(n_boot):
        idx = np.concatenate([rows_of[d] for d in rng.choice(days, len(days), replace=True)])
        out.append(score_fn(idx))
    return np.percentile(out, [2.5, 97.5])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--groups", default="O,N,W")
    args = ap.parse_args()
    groups = set(args.groups.split(","))
    from sklearn.ensemble import HistGradientBoostingClassifier as HC

    f = eval_cache.load_features(args.ckpt)
    lead = int(f["lead_hours"])
    with open("./cache/historical_data_1y.json", encoding="utf-8") as fp:
        recs = json.load(fp)
    obs = {(str(r["stn"]), str(r["timestamp"])[:12]): (float(r["temperature"]), float(r["humidity"]),
                                                       float(r["pressure"]), float(r["precipitation"]))
           for r in recs}
    del recs
    with open("./cache/nwp_archive.json", encoding="utf-8") as fp:
        nwp = json.load(fp)
    print(f"리드 +{lead}h · 관측 {len(obs):,} · 그룹 {sorted(groups)}", flush=True)

    sets = {}
    extra_names = []
    for part in ("train", "val"):
        ex, names = build_extra(f[f"stn_{part}"], f[f"src_ts_{part}"], lead, obs, nwp, groups)
        sets[part] = ex
        extra_names = names
        print(f"  {part}: 추가 특징 {len(names)}개 결측률 "
              + ", ".join(f"{n} {np.isnan(ex[:, j]).mean():.1%}" for j, n in enumerate(names)), flush=True)
    del obs, nwp

    xtr0, xva0 = f["x_train"], f["x_val"]
    day = f["tgt_ts_val"].astype(np.int64) // 10 ** 4
    calib = day % 2 == 0
    # CROSSFIT=1 — 각 절반의 판정선을 **다른 절반**에서 골라 양쪽을 모두 평가에 쓴다. 모델은
    # 학습 분할로만 적합하므로 표본 내 이득이 없고, 평가 표본이 두 배가 된다(한파 착수처럼
    # 사건이 수십 개뿐인 지표의 신뢰구간을 좁히려는 것 — 조사 권고의 날짜 그룹 k-fold).
    CROSSFIT = os.environ.get("CROSSFIT") == "1"
    ev = np.ones(len(day), bool) if CROSSFIT else ~calib
    ev_days = np.unique(day[ev])
    rows_of = {d: np.flatnonzero(ev & (day == d)) for d in ev_days}

    def fit_eval(xtr, xva, ytr, yva, mtr, mva, cst=None):
        m = HC(random_state=SEED, max_iter=400, learning_rate=0.05, monotonic_cst=cst)
        m.fit(xtr[mtr], ytr[mtr])
        p = np.full(len(xva), np.nan)
        p[mva] = m.predict_proba(xva[mva])[:, 1]
        tau = pick_tau(p[mva], yva[mva], calib[mva])
        if CROSSFIT:
            tau_o = pick_tau(p[mva], yva[mva], ~calib[mva])
            return np.where(calib, p >= tau_o, p >= tau) & mva, tau
        return (p >= tau) & mva, tau

    results = []
    PREV = {}
    # ── 강수 발생 ──
    wet = 0.1
    ytr, yva = f["precip_true_train"] >= wet, f["precip_true_val"] >= wet
    allm_tr, allm_va = np.ones(len(ytr), bool), np.ones(len(yva), bool)
    pa, _ = fit_eval(xtr0, xva0, ytr, yva, allm_tr, allm_va)
    pb, _ = fit_eval(np.hstack([xtr0, sets["train"]]), np.hstack([xva0, sets["val"]]),
                     ytr, yva, allm_tr, allm_va)
    results.append(("강수 발생", pa, pb, yva, allm_va))
    # ── 폭염·한파 — 배포와 같은 마스크·단조 제약. 제약 열은 extreme_gbm 과 **같은
    # 함수**로 구한다(열 번호를 여기서 세면 특징 집합이 바뀔 때 조용히 어긋난다).
    import importlib
    import extreme_gbm as _eg
    import train as _t
    os.environ["USE_NWP"] = "1" if bool(f["use_nwp"]) else "0"
    importlib.reload(_t)
    _t.NWP_FEATURE_SET = str(f["nwp_feature_set"])
    temp_cols = _t.extreme_temp_neutral_index(xtr0.shape[1])
    for head, short, sign in (("heatwave", "heat", +1), ("coldwave", "cold", -1)):
        mtr = f[f"{short}_mask_train"].astype(bool)
        off = f.get(f"{short}_mask_official_val")
        mva = (off if off is not None else f[f"{short}_mask_val"]).astype(bool)
        ytr_e, yva_e = f[f"y_{head}_train"] > 0, f[f"y_{head}_val"] > 0
        nf = xtr0.shape[1]
        _c = _eg._mono_cst(head, nf, temp_cols)
        cst0 = [int(v) for v in _c] if _c is not None else [0] * nf
        pa, _ = fit_eval(xtr0, xva0, ytr_e, yva_e, mtr, mva, cst0)
        # 열 이름은 특징을 만든 함수가 돌려준 순서 그대로 쓴다 — 여기서 목록을 따로
        # 적으면 그룹을 늘릴 때 제약 벡터 길이가 어긋난다(M 그룹 추가 때 실제로 그랬다).
        names = extra_names
        # 새 기온 열도 같은 방향으로 단조 제약한다(폭염 +, 한파 −). day2 열은 자기 헤드에만 건다 —
        # 한파 열(D_c_*)은 한파 헤드에서 −(더 춥고 더 크게 떨어질수록 ↑), 폭염 열(D_h_*)은 폭염 헤드에서 +.
        def _c(n):
            if n in ("W_t24_min", "W_t24_max", "V_t9_min", "V_t9_max", "V_at9_max"):
                return sign
            if n.startswith("D_c_"):
                return -1 if head == "coldwave" else 0
            if n.startswith("D_h_"):
                return +1 if head == "heatwave" else 0
            return 0
        cst1 = cst0 + [_c(n) for n in names]
        pb, _ = fit_eval(np.hstack([xtr0, sets["train"]]), np.hstack([xva0, sets["val"]]),
                         ytr_e, yva_e, mtr, mva, cst1)
        results.append((head, pa, pb, yva_e, mva))
        # 목표일 '전날' 라벨 — 착수(전날 0)·지속(전날 1) 분해용. 없으면 NaN(분해에서 빠진다).
        from datetime import date as _date
        prev = np.full(len(yva_e), np.nan)
        lab = f"{head}_advisory"
        for k_ in np.flatnonzero(mva):
            dd = str(int(f["tgt_ts_val"][k_]))[:8]
            d_prev = (_date(int(dd[:4]), int(dd[4:6]), int(dd[6:8])) - timedelta(days=1)).isoformat()
            rec = LABELS.get(str(int(f["stn_val"][k_])), {}).get(d_prev)
            if rec is not None and lab in rec:
                prev[k_] = rec[lab]
        PREV[head] = prev

    era = f["tgt_ts_val"].astype(np.int64) >= 202207100000
    if "M" in groups:
        print(f"\n[MSM 시대(2022-07-10~) 평가용 행만] — 같은 두 모델, 판정선도 같다")
        for name, pa, pb, y, m in results:
            e = np.flatnonzero(ev & m & era)
            ed = np.unique(day[e]); ro = {d: e[day[e] == d] for d in ed}
            def sc2(idx, pa=pa, pb=pb, y=y):
                return f1(pb[idx], y[idx]) - f1(pa[idx], y[idx])
            lo, hi = boot_diff(sc2, ed, ro)
            print(f"  {name:<10}{f1(pa[e], y[e]):>10.4f}{f1(pb[e], y[e]):>12.4f}"
                  f"{f1(pb[e], y[e]) - f1(pa[e], y[e]):>+9.4f}   [{lo:+.4f}, {hi:+.4f}] (날짜 {len(ed)})")
    if PREV:
        print(f"\n[착수·지속 분해 — {'검증셋 전체(교차 적합)' if CROSSFIT else '평가용 절반'}] "
              "착수 = 목표일 전날 특보 0, 지속 = 전날 1")
        for name, pa, pb, y, m in results:
            if name not in PREV:
                continue
            for lab_, cond in (("착수", PREV[name] == 0), ("지속", PREV[name] == 1)):
                e = np.flatnonzero(ev & m & cond)
                ed = np.unique(day[e]); ro = {d: e[day[e] == d] for d in ed}
                def sc3(idx, pa=pa, pb=pb, y=y):
                    return f1(pb[idx], y[idx]) - f1(pa[idx], y[idx])
                lo, hi = boot_diff(sc3, ed, ro)
                print(f"  {name:<10}{lab_}  양성 {int(y[e].sum()):>5} · F1 {f1(pa[e], y[e]):.4f} → "
                      f"{f1(pb[e], y[e]):.4f} ({f1(pb[e], y[e]) - f1(pa[e], y[e]):+.4f}) [{lo:+.4f}, {hi:+.4f}]")
    print(f"\n{'교차 적합 — 검증셋 전체(각 절반의 판정선은 다른 절반에서)' if CROSSFIT else '평가용 절반(목표일 홀수) — 판정선은 보정용 절반에서 고름'}"
          f" · 부트스트랩 날짜 {len(ev_days)}개")
    print(f"{'출력':<10}{'현행 28':>10}{'+'+args.groups:>12}{'Δ':>9}   95% CI")
    for name, pa, pb, y, m in results:
        def sc(idx, pa=pa, pb=pb, y=y, m=m):
            idx = idx[m[idx]]
            return f1(pb[idx], y[idx]) - f1(pa[idx], y[idx])
        e = np.flatnonzero(ev & m)
        a, b = f1(pa[e], y[e]), f1(pb[e], y[e])
        lo, hi = boot_diff(sc, ev_days, rows_of)
        print(f"{name:<10}{a:>10.4f}{b:>12.4f}{b - a:>+9.4f}   [{lo:+.4f}, {hi:+.4f}]"
              + ("  유의" if lo > 0 or hi < 0 else ""))
    return 0


if __name__ == "__main__":
    sys.exit(main())
