"""
nwp_feature_probe.py — data_expansion_probe.py ③(특징 절제)의 확장.

data_expansion_probe.py 는 기존 12개 관측소의 지상 실측만으로 만들 수 있는
특징군을 단계적으로 넣어 AUC 증분을 쟀고, 마지막 단계(이웃 경향)의 증분이
+0.0010 으로 사실상 0 임을 보였다. 결론은 "기존 관측망의 정보는 소진됐고,
필요한 정보는 물리적으로 관측망 밖(서해상 약 404km 상류)에 있다"였다.

이 스크립트는 그 결론이 지목한 대상을 **실제로 붙여서** 잰다. 관측망 밖의
상류 상태를 대리하는 가장 값싼 자료는 수치예보모델(NWP)의 예보장이다 —
NWP 는 전 지구 자료동화로 서해·중국 대륙의 상태를 이미 반영하고 있다.

자료원: Open-Meteo Previous Runs API(무료·인증키 불필요·CC BY 4.0).
  · `<변수>_previous_day1` = **유효시각 24시간 전에 발표된** 예보값.
  · 따라서 T 시점에서 T+6h·T+12h 를 예측할 때 이 값은 이미 발표돼 있다
    (발표 시각이 T 보다 18h·12h 앞선다) — 시간 누수가 없다. 오히려
    실서빙에서 쓸 수 있는 리드타임(+6h·+12h)보다 **불리한** 예보이므로,
    여기서 나오는 증분은 실사용 시 기대치의 하한이다.
  · 수집 스크립트는 이 파일 하단 `fetch_archive()` 참고.

실행: python nwp_feature_probe.py           (LEAD_HOURS 환경변수로 6/12 선택)
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from weather_collector import STATION_COORDS

DATA_PATH = "./cache/historical_data_1y.json"
NWP_PATH = "./cache/nwp_prevrun_openmeteo.json"
WET_THRESH = 0.1
LEAD_HOURS = int(os.getenv("LEAD_HOURS", "6"))
LAGS = [1, 3, 6]
SEED = 7
NWP_VARS = ["precipitation", "temperature_2m", "relative_humidity_2m", "cloud_cover",
            "wind_speed_10m", "wind_direction_10m", "surface_pressure"]


def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def load_matrices():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    stns = sorted(STATION_COORDS.keys())
    sidx = {s: i for i, s in enumerate(stns)}
    by_ts = defaultdict(dict)
    for r in records:
        s = str(r.get("stn"))
        if s in sidx:
            by_ts[str(r["timestamp"])][s] = r
    timestamps = sorted(by_ts.keys())
    T, S = len(timestamps), len(stns)
    fields = {k: np.full((T, S), np.nan, dtype=np.float32)
              for k in ("precip", "temp", "humid", "wspd", "wdir", "press")}
    keymap = dict(precip="precipitation", temp="temperature", humid="humidity",
                  wspd="wind_speed", wdir="wind_dir", press="pressure")
    for t, ts in enumerate(timestamps):
        for s, r in by_ts[ts].items():
            i = sidx[s]
            for k, src in keymap.items():
                v = r.get(src)
                if v is not None:
                    fields[k][t, i] = v
    import datetime as dt
    base = dt.datetime.strptime(timestamps[0][:12], "%Y%m%d%H%M")
    hour_idx = np.array([
        int((dt.datetime.strptime(ts[:12], "%Y%m%d%H%M") - base).total_seconds() // 3600)
        for ts in timestamps])
    dates = np.array([ts[:8] for ts in timestamps])
    print(f"실측 정렬 — 시각 {T:,}개 × 관측소 {S}개  ({timestamps[0]}~{timestamps[-1]})")
    return timestamps, dates, hour_idx, stns, fields


def load_nwp(timestamps, stns):
    """NWP 예보장을 실측과 같은 (시각 × 관측소 × 변수) 격자에 정렬한다.

    Open-Meteo 의 시각 문자열은 `YYYY-MM-DDTHH:MM`(Asia/Seoul)이고 실측
    타임스탬프는 `YYYYMMDDHHMM` 이다. 벽시계 표기가 둘 다 KST 라 문자열만
    맞추면 되지만, 그 사실을 코드가 확인하도록 정렬 성공률을 출력한다.
    """
    with open(NWP_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    T, S, V = len(timestamps), len(stns), len(NWP_VARS)
    grid = np.full((T, S, V), np.nan, dtype=np.float32)
    tpos = {ts[:12]: t for t, ts in enumerate(timestamps)}
    matched = 0
    for si, s in enumerate(stns):
        h = raw.get(s)
        if h is None:
            print(f"  [WARN] NWP 결측 관측소 {s}")
            continue
        keys = [t.replace("-", "").replace("T", "").replace(":", "")[:12] for t in h["time"]]
        idx = np.array([tpos.get(k, -1) for k in keys])
        ok = idx >= 0
        matched += int(ok.sum())
        for vi, v in enumerate(NWP_VARS):
            arr = np.array([np.nan if x is None else x
                            for x in h[f"{v}_previous_day1"]], dtype=np.float32)
            grid[idx[ok], si, vi] = arr[ok]
    print(f"NWP 정렬 — 시각축 매칭 {matched:,}건 · "
          f"강수 유효값 {int(np.isfinite(grid[:, :, 0]).sum()):,}건")
    return grid


def build_features(stns, fields, nwp, hour_idx, dates, k_neighbor=5):
    """(국지현재, 국지경향, 이웃현재, 이웃경향, NWP) 특징군.

    앞 4개는 data_expansion_probe.build_features 와 동일하다 — 증분을 그
    표와 직접 비교할 수 있어야 하므로 정의를 바꾸지 않는다.
    """
    S = len(stns)
    pos = {h: t for t, h in enumerate(hour_idx)}
    precip, temp = fields["precip"], fields["temp"]
    humid, wspd, wdir, press = fields["humid"], fields["wspd"], fields["wdir"], fields["press"]
    u = -np.sin(np.radians(wdir)) * wspd
    v = -np.cos(np.radians(wdir)) * wspd

    n_pr, n_tp, n_rh, n_cc, n_ws, n_wd, n_sp = range(7)

    groups = {"local_now": [], "local_tend": [], "nb_now": [], "nb_tend": [], "nwp": []}
    ys, amts, ds = [], [], []
    for tgt in range(S):
        order = sorted([j for j in range(S) if j != tgt],
                       key=lambda j: haversine(STATION_COORDS[stns[tgt]],
                                               STATION_COORDS[stns[j]]))[:k_neighbor]
        for t, h in enumerate(hour_idx):
            t_f = pos.get(h + LEAD_HOURS)
            if t_f is None or np.isnan(precip[t_f, tgt]):
                continue
            lag_ts = [pos.get(h - L) for L in LAGS]
            if any(x is None for x in lag_ts):
                continue
            base = [temp[t, tgt], precip[t, tgt], humid[t, tgt],
                    u[t, tgt], v[t, tgt], press[t, tgt]]
            if any(np.isnan(base)):
                continue
            tend = []
            for lt in lag_ts:
                tend += [temp[t, tgt] - temp[lt, tgt], press[t, tgt] - press[lt, tgt],
                         humid[t, tgt] - humid[lt, tgt], precip[t, tgt] - precip[lt, tgt]]
            nb_now, nb_tend = [], []
            for j in order:
                nb_now += [precip[t, j], humid[t, j], u[t, j], v[t, j]]
                nb_tend += [precip[t, j] - precip[lag_ts[1], j],
                            press[t, j] - press[lag_ts[1], j],
                            humid[t, j] - humid[lag_ts[1], j]]

            # NWP 특징 — 모두 T 시점에 이미 발표돼 있는 값이다(24h 전 발표).
            f_now = nwp[t, tgt]                       # 유효시각 T 의 예보
            f_tgt = nwp[t_f, tgt]                     # 유효시각 T+L 의 예보 ← 핵심
            t_m1, t_p1 = pos.get(h + LEAD_HOURS - 1), pos.get(h + LEAD_HOURS + 1)
            if t_m1 is None or t_p1 is None:
                continue
            win = [nwp[t_m1, tgt, n_pr], nwp[t_f, tgt, n_pr], nwp[t_p1, tgt, n_pr]]
            wu = -np.sin(np.radians(f_tgt[n_wd])) * f_tgt[n_ws]
            wv = -np.cos(np.radians(f_tgt[n_wd])) * f_tgt[n_ws]
            nwp_feat = [
                f_tgt[n_pr], f_tgt[n_tp], f_tgt[n_rh], f_tgt[n_cc], wu, wv, f_tgt[n_sp],
                float(np.nansum(win)), float(np.nanmax(win)),      # 타이밍 오차 완충
                f_tgt[n_pr] - f_now[n_pr],                          # 예보된 변화
                f_tgt[n_sp] - f_now[n_sp],
                # 편향 보정 신호 — 같은 시각의 예보와 실측 차이
                f_now[n_tp] - temp[t, tgt], f_now[n_pr] - precip[t, tgt],
                f_now[n_rh] - humid[t, tgt],
            ]
            row = dict(local_now=base, local_tend=tend, nb_now=nb_now,
                       nb_tend=nb_tend, nwp=nwp_feat)
            if any(np.isnan(np.array(vv, dtype=np.float64)).any() for vv in row.values()):
                continue
            for kk in groups:
                groups[kk].append(row[kk])
            ys.append(1 if precip[t_f, tgt] >= WET_THRESH else 0)
            amts.append(float(precip[t_f, tgt]))
            ds.append(dates[t])
    out = {k: np.asarray(vv, dtype=np.float32) for k, vv in groups.items()}
    return out, np.asarray(ys), np.asarray(amts, dtype=np.float32), np.asarray(ds)


def part_intensity(groups, amt, is_val):
    """강도(amount) 축 — 습윤 표본 조건부 MAE.

    발생 판정(위 절제)과 별개로, 배포 모델의 진짜 남은 과제는 amount 헤드가
    강도를 표현하지 못하는 것이다(README '강수 축 상한 분석' ②: 10mm 이상
    구간에서 실측 17.2mm 대비 모델 1.54mm, 습윤구간 상수보다 나쁨). NWP 가
    그 축에도 정보를 주는지 같은 표본에서 잰다.

    평가는 MAE 이므로 손실도 절대오차(=조건부 중앙값)로 맞춘다 — MSE 로
    학습하고 MAE 로 채점하던 기존 불일치를 이 프로브에서 재현하지 않는다.
    """
    from sklearn.ensemble import HistGradientBoostingRegressor
    wet = amt >= WET_THRESH
    tr, va = (~is_val) & wet, is_val & wet
    print(f"\n  [강도 축] 습윤 표본 학습 {int(tr.sum()):,} / 검증 {int(va.sum()):,}")
    med = float(np.median(amt[tr]))
    print(f"  {'설정':<24}{'습윤 MAE':>11}{'상수 대비':>11}")
    print(f"  {'상수(학습 중앙값 %.2fmm)' % med:<24}"
          f"{np.abs(amt[va] - med).mean():11.4f}{0.0:+11.2%}")
    base_mae = float(np.abs(amt[va] - med).mean())
    OBS = ["local_now", "local_tend", "nb_now", "nb_tend"]
    ALL = OBS + ["nwp"]
    # 손실을 바꿔가며 같은 특징으로 학습한다 — "amount 헤드가 강도를 표현하지
    # 못한다"가 헤드의 결함인지, MAE(=조건부 중앙값)라는 목표 자체의 성질인지
    # 가르기 위해서다. 중앙값 회귀는 두꺼운 오른쪽 꼬리를 구조적으로 축소한다.
    for tag, keys, loss, q in (
            ("관측만 · 중앙값", OBS, "absolute_error", None),
            ("+NWP · 중앙값", ALL, "absolute_error", None),
            ("+NWP · 평균(MSE)", ALL, "squared_error", None),
            ("+NWP · 90분위", ALL, "quantile", 0.9)):
        X = np.concatenate([groups[k] for k in keys], axis=1)
        kw = dict(quantile=q) if q is not None else {}
        reg = HistGradientBoostingRegressor(loss=loss, max_iter=200,
                                            learning_rate=0.1, max_depth=6,
                                            random_state=SEED, **kw)
        reg.fit(X[tr], amt[tr])
        pred = np.maximum(reg.predict(X[va]), 0.0)
        mae = float(np.abs(amt[va] - pred).mean())
        print(f"  {tag:<24}{mae:11.4f}{(base_mae - mae) / base_mae:+11.2%}")
        if q is not None:
            cov = float((amt[va] <= pred).mean())
            print(f"      (90분위 예측의 실측 포함률 {cov:.3f} — 목표 0.90)")
        # 강도 표현력 — 강한 강수 구간에서 예측 평균이 실측을 얼마나 따라가는가
        for lo, hi in ((1, 5), (5, 10), (10, 1e9)):
            m = (amt[va] >= lo) & (amt[va] < hi)
            if m.sum() > 30:
                print(f"      {lo}~{hi if hi < 1e9 else '∞'}mm  n={int(m.sum()):5d}  "
                      f"실측평균 {amt[va][m].mean():6.2f}  예측평균 {pred[m].mean():6.2f}")


def main():
    timestamps, dates, hour_idx, stns, fields = load_matrices()
    nwp = load_nwp(timestamps, stns)
    print(f"\n{'=' * 88}\n NWP 예보 특징 절제 — 리드타임 +{LEAD_HOURS}h\n{'=' * 88}")
    groups, y, amt, ds = build_features(stns, fields, nwp, hour_idx, dates)
    print(f"  표본 {len(y):,}개 · 양성률 {y.mean():.1%} "
          f"(NWP 아카이브가 있는 구간으로 한정됨)")

    uniq = np.unique(ds)
    rng = np.random.RandomState(SEED)
    val_dates = set(rng.choice(uniq, size=int(len(uniq) * 0.2), replace=False).tolist())
    is_val = np.array([d in val_dates for d in ds])
    print(f"  날짜 그룹 분할 — 학습 {int((~is_val).sum()):,} / 검증 {int(is_val.sum()):,}")

    stages = [
        ("국지 현재", ["local_now"]),
        ("+ 국지 경향", ["local_now", "local_tend"]),
        ("+ 이웃 현재", ["local_now", "local_tend", "nb_now"]),
        ("+ 이웃 경향", ["local_now", "local_tend", "nb_now", "nb_tend"]),
        ("+ NWP 예보 (신규)", ["local_now", "local_tend", "nb_now", "nb_tend", "nwp"]),
        ("[참고] NWP 단독", ["nwp"]),
    ]
    print(f"\n  {'특징군':<24}{'AUC':>9}{'ΔAUC':>9}{'AP':>9}{'최대F1':>9}{'차원':>6}")
    prev_auc, results = None, []
    for tag, keys in stages:
        X = np.concatenate([groups[k] for k in keys], axis=1)
        clf = HistGradientBoostingClassifier(
            max_iter=200, learning_rate=0.1, max_depth=6, random_state=SEED)
        clf.fit(X[~is_val], y[~is_val])
        p = clf.predict_proba(X[is_val])[:, 1]
        auc = roc_auc_score(y[is_val], p)
        ap = average_precision_score(y[is_val], p)
        prec, rec, _ = precision_recall_curve(y[is_val], p)
        f1s = 2 * prec * rec / np.maximum(prec + rec, 1e-12)
        d = (auc - prev_auc) if (prev_auc is not None and not tag.startswith("[참고]")) else 0.0
        print(f"  {tag:<24}{auc:9.4f}{d:+9.4f}{ap:9.4f}{np.nanmax(f1s):9.4f}{X.shape[1]:6d}")
        results.append((tag, auc, ap, float(np.nanmax(f1s))))
        if not tag.startswith("[참고]"):
            prev_auc = auc

    print(f"\n  핵심 — NWP 증분 ΔAUC = {results[4][1] - results[3][1]:+.4f}, "
          f"최대F1 {results[3][3]:.4f} → {results[4][3]:.4f}")
    print(f"  대조 — 이웃 경향 증분(기존 측정) = {results[3][1] - results[2][1]:+.4f}")
    print(f"\n  주의: 여기 쓴 NWP 는 유효시각 24h 전 발표분이다. 실서빙에서는")
    print(f"  +{LEAD_HOURS}h 리드의 최신 예보를 쓸 수 있으므로 위 증분은 하한이다.")

    part_intensity(groups, amt, is_val)


if __name__ == "__main__":
    main()
