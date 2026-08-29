"""
data_expansion_probe.py — data_expansion_analysis.py 의 후속. 결정적 실험 세 가지.

1차 분석(data_expansion_analysis.py)에서 드러난 것:
  · 이웃 11곳까지 써도 조건부 상호정보량이 k=7~8에서 포화(11번째 증분 0.5%)
  · 이웃의 현재 강수만으로 만든 oracle F1 0.337 < 국지 지속성 0.444 < 모델 0.463
  · 모델이 놓친 강수의 rain_prob 중앙값이 0.674 — 신호를 못 본 게 아니라 잘린 것

이 결과는 "공간 정보가 이미 포화"를 시사하지만, 두 가지를 아직 못 재고 있다.

  ① 너깃(nugget) — 지수 감쇠를 원점을 지나도록 적합했으나, 실측은 30km 에서
     이미 r=0.705 다. 즉 최소 관측 간격(30km) 아래에 미해상 분산이 남아 있고,
     그 크기가 곧 "AWS 같은 조밀 관측이 새로 볼 수 있는 몫"의 상한이다.
     r(d) = c0·exp(−d/L) 로 절편을 함께 적합해 c0 를 구한다. 1−c0 가 너깃이다.
  ② 이류 지연(advection lag) — 관측소 쌍의 교차상관을 시차 0~12h 로 훑어
     최대가 되는 시차를 찾으면 강수계 이동속도가 나온다. 그 속도로 6시간이면
     지금 어디에 있는지가 정해지고, 그 지점이 관측망 **밖**(서해상)이면
     지상 관측을 아무리 조밀하게 해도 볼 수 없다는 결론이 나온다.

세 번째는 "새 데이터가 필요한가, 기존 데이터를 덜 쓰고 있는가"를 가르는 실험이다.
  ③ 특징 절제(feature ablation) — 같은 분류기에 특징군을 단계적으로 넣어 AUC 증분을
     잰다. 국지현재 → +국지경향(Im축이 가진 것) → +이웃현재(Re축이 가진 것) →
     +이웃경향(현재 어느 축도 갖지 않은 시공간 정보). 마지막 단계가 유의하게
     올라가면 **새 데이터 없이** 개선 여지가 남아 있다는 뜻이다.

분할은 날짜 단위 그룹 분할을 쓴다(CLAUDE.md 검증 규약 — 시각 단위 무작위 분할은
같은 날이 학습·검증에 쪼개져 누수가 생긴다).

실행: python data_expansion_probe.py
"""
import json
import math
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from weather_collector import STATION_COORDS

DATA_PATH = "./cache/historical_data_1y.json"
WET_THRESH = 0.1
LEAD_HOURS = 6
LAGS = [1, 3, 6]      # tendency_collector.LAGS_HOURS 와 동일
SEED = 7

STATION_NAMES = {
    "101": "춘천", "105": "강릉", "108": "서울", "112": "인천", "119": "수원",
    "131": "청주", "133": "대전", "143": "대구", "146": "전주", "156": "광주",
    "159": "부산", "184": "제주",
}


def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def bearing(a, b):
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


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
    print(f"정렬 완료 — 시각 {T:,}개 × 관측소 {S}개")
    return timestamps, dates, hour_idx, stns, fields


def corr_pair(x, y, min_n=100):
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < min_n:
        return np.nan
    xv, yv = x[m], y[m]
    if xv.std() < 1e-9 or yv.std() < 1e-9:
        return np.nan
    return float(np.corrcoef(xv, yv)[0, 1])


# ── ① 너깃 보정 변동도 ───────────────────────────────────────
def part1_nugget(stns, precip, temp):
    print(f"\n{'=' * 88}\n ① 너깃(nugget) 보정 — 조밀 관측이 새로 볼 수 있는 분산은 얼마인가\n{'=' * 88}")
    S = len(stns)
    d_list, occ, amt, tmp = [], [], [], []
    for i in range(S):
        for j in range(i + 1, S):
            d = haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            wi = np.where(np.isnan(precip[:, i]), np.nan, (precip[:, i] >= WET_THRESH))
            wj = np.where(np.isnan(precip[:, j]), np.nan, (precip[:, j] >= WET_THRESH))
            d_list.append(d)
            occ.append(corr_pair(wi, wj))
            amt.append(corr_pair(precip[:, i], precip[:, j]))
            tmp.append(corr_pair(temp[:, i], temp[:, j]))
    d_arr = np.array(d_list)
    res = {}
    print(f"  모형 r(d) = c₀·exp(−d/L) — c₀ 는 d→0 극한의 상관(공간 일관 성분),")
    print(f"  1−c₀ 가 너깃(최소 관측 간격 아래에 숨은 미해상 분산 비율)이다.\n")
    print(f"  {'변수':<12}{'c₀':>8}{'너깃 1−c₀':>12}{'L(km)':>9}{'설명분산 R²':>13}")
    for tag, arr in (("강수 발생", np.array(occ)), ("강수량", np.array(amt)),
                     ("기온(대조)", np.array(tmp))):
        m = (arr > 0.01) & ~np.isnan(arr)
        X = np.column_stack([np.ones(m.sum()), -d_arr[m]])
        y = np.log(arr[m])
        beta, *_ = np.linalg.lstsq(X, y, rcond=None)
        c0, L = float(np.exp(beta[0])), float(1.0 / beta[1]) if beta[1] > 0 else np.inf
        pred = X @ beta
        r2 = 1 - ((y - pred) ** 2).sum() / ((y - y.mean()) ** 2).sum()
        res[tag] = (c0, L)
        print(f"  {tag:<12}{c0:8.3f}{1 - c0:12.3f}{L:9.0f}{r2:13.3f}")

    nn = [min(haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
              for j in range(S) if j != i) for i in range(S)]
    print(f"\n  현재 관측망 최근접 이웃 거리: 평균 {np.mean(nn):.0f} km · 최소 {np.min(nn):.0f} km")
    c0_occ = res["강수 발생"][0]
    print(f"  → 강수 발생의 너깃 {1 - c0_occ:.1%}: 최소 간격 {np.min(nn):.0f} km 아래 규모에 "
          f"남아 있는 분산 비율.")
    print(f"    이것이 조밀 관측(AWS)이 **새로 볼 수 있는 몫의 상한**이다. 단, 그 분산이")
    print(f"    예측 가능한 신호인지 순수 잡음인지는 이 분석만으로 구분되지 않는다.")
    return res


# ── ② 이류 지연 ─────────────────────────────────────────────
def part2_advection(stns, precip, hour_idx):
    print(f"\n{'=' * 88}\n ② 이류 지연 — 6시간 뒤 도달할 강수계는 지금 어디에 있는가\n{'=' * 88}")
    S = len(stns)
    pos = {h: t for t, h in enumerate(hour_idx)}
    wet = np.where(np.isnan(precip), np.nan, (precip >= WET_THRESH).astype(np.float32))

    lag_grid = list(range(0, 13))
    best = []
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            d = haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            b = bearing(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            curve = []
            for lag in lag_grid:
                src, tgt = [], []
                for t, h in enumerate(hour_idx):
                    t2 = pos.get(h + lag)
                    if t2 is not None:
                        src.append(t)
                        tgt.append(t2)
                r = corr_pair(wet[np.array(src), i], wet[np.array(tgt), j])
                curve.append(r)
            curve = np.array(curve)
            k = int(np.nanargmax(curve))
            best.append(dict(i=i, j=j, d=d, b=b, lag=lag_grid[k], r=float(curve[k]),
                             r0=float(curve[0])))

    # 시차>0 에서 최대가 되는 쌍만 — 이류가 실제로 잡힌 경우
    moving = [x for x in best if x["lag"] > 0]
    print(f"  관측소 순서쌍 {len(best)}개 중 최대 상관 시차가 0보다 큰 쌍: {len(moving)}개")
    if moving:
        speeds = [x["d"] / x["lag"] for x in moving]
        print(f"  그 쌍들의 함축 이동속도: 중앙값 {np.median(speeds):.0f} km/h "
              f"(사분위 {np.percentile(speeds, 25):.0f}~{np.percentile(speeds, 75):.0f})")
        v = float(np.median(speeds))
        print(f"\n  → 이동속도 {v:.0f} km/h 이면 +{LEAD_HOURS}h 뒤 도달할 계는 "
              f"지금 약 {v * LEAD_HOURS:.0f} km 상류에 있다.")
        print(f"     한반도 동서 폭은 약 300 km, 남북 약 500 km 다.")
        if v * LEAD_HOURS > 300:
            print(f"     ★ {v * LEAD_HOURS:.0f} km > 300 km — 서풍 계열에서는 그 지점이 "
                  f"**서해 위**다. 육상 관측소를 아무리 조밀하게 깔아도 볼 수 없다.")
        else:
            print(f"     {v * LEAD_HOURS:.0f} km ≤ 300 km — 국내 관측망 범위 안에 있을 수 있다.")

    top = sorted(moving, key=lambda x: -x["r"])[:8]
    print(f"\n  시차 상관이 가장 큰 쌍")
    print(f"  {'상류→하류':<16}{'거리km':>8}{'방위':>7}{'최적시차h':>10}{'r(시차)':>9}{'r(동시)':>9}")
    for x in top:
        print(f"  {STATION_NAMES[stns[x['i']]] + '→' + STATION_NAMES[stns[x['j']]]:<16}"
              f"{x['d']:8.0f}{x['b']:7.0f}{x['lag']:10d}{x['r']:9.3f}{x['r0']:9.3f}")

    # 방향 비대칭 vs 동향 성분 — 표본이 66쌍뿐이라 min_n 을 낮춰 계산한다.
    lookup = {(x["i"], x["j"]): x for x in best}
    a_vals, e_vals = [], []
    for i in range(S):
        for j in range(i + 1, S):
            f, r = lookup[(i, j)], lookup[(j, i)]
            src, tgt = [], []
            for t, h in enumerate(hour_idx):
                t2 = pos.get(h + LEAD_HOURS)
                if t2 is not None:
                    src.append(t)
                    tgt.append(t2)
            src, tgt = np.array(src), np.array(tgt)
            fwd = corr_pair(wet[src, i], wet[tgt, j])
            bwd = corr_pair(wet[src, j], wet[tgt, i])
            if not (np.isnan(fwd) or np.isnan(bwd)):
                a_vals.append(fwd - bwd)
                e_vals.append(math.sin(math.radians(f["b"])))
    a_vals, e_vals = np.array(a_vals), np.array(e_vals)
    r_dir = float(np.corrcoef(a_vals, e_vals)[0, 1])
    print(f"\n  +{LEAD_HOURS}h 방향 비대칭 vs 동향 성분 sin(방위): r = {r_dir:+.3f} "
          f"(쌍 {len(a_vals)}개)")
    print(f"    양수 = 서쪽 관측소가 동쪽의 미래를 안다(편서풍 이류). "
          f"{'신호 있음' if r_dir > 0.2 else '약함'}")
    return dict(r_dir=r_dir, n_moving=len(moving))


# ── ③ 특징 절제 ─────────────────────────────────────────────
def build_features(stns, fields, hour_idx, dates, k_neighbor=5):
    """대상 관측소별로 (국지현재, 국지경향, 이웃현재, 이웃경향) 특징군을 만든다."""
    S = len(stns)
    pos = {h: t for t, h in enumerate(hour_idx)}
    precip, temp = fields["precip"], fields["temp"]
    humid, wspd, wdir, press = fields["humid"], fields["wspd"], fields["wdir"], fields["press"]
    u = -np.sin(np.radians(wdir)) * wspd     # 동향 성분
    v = -np.cos(np.radians(wdir)) * wspd     # 북향 성분

    groups = {"local_now": [], "local_tend": [], "nb_now": [], "nb_tend": []}
    ys, ds = [], []
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
                # 이웃 경향 — 시공간 정보(현재 어느 축도 갖지 않음)
                nb_tend += [precip[t, j] - precip[lag_ts[1], j],
                            press[t, j] - press[lag_ts[1], j],
                            humid[t, j] - humid[lag_ts[1], j]]
            row = dict(local_now=base, local_tend=tend, nb_now=nb_now, nb_tend=nb_tend)
            if any(np.isnan(np.array(vv, dtype=np.float64)).any() for vv in row.values()):
                continue
            for kk in groups:
                groups[kk].append(row[kk])
            ys.append(1 if precip[t_f, tgt] >= WET_THRESH else 0)
            ds.append(dates[t])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in groups.items()}
    return out, np.asarray(ys), np.asarray(ds)


def part3_ablation(stns, fields, hour_idx, dates):
    print(f"\n{'=' * 88}\n ③ 특징 절제 — 새 데이터가 필요한가, 기존 데이터를 덜 쓰고 있는가\n{'=' * 88}")
    groups, y, ds = build_features(stns, fields, hour_idx, dates)
    print(f"  표본 {len(y):,}개 · 양성률 {y.mean():.1%}")

    # 날짜 단위 그룹 분할(CLAUDE.md 검증 규약)
    uniq = np.unique(ds)
    rng = np.random.RandomState(SEED)
    val_dates = set(rng.choice(uniq, size=int(len(uniq) * 0.2), replace=False).tolist())
    is_val = np.array([d in val_dates for d in ds])
    print(f"  날짜 그룹 분할 — 학습 {int((~is_val).sum()):,} / 검증 {int(is_val.sum()):,} "
          f"(공유 날짜 0개)")

    stages = [
        ("국지 현재", ["local_now"]),
        ("+ 국지 경향 (Im축이 가진 것)", ["local_now", "local_tend"]),
        ("+ 이웃 현재 (Re축이 가진 것)", ["local_now", "local_tend", "nb_now"]),
        ("+ 이웃 경향 (어느 축도 없음)", ["local_now", "local_tend", "nb_now", "nb_tend"]),
    ]
    print(f"\n  {'특징군':<32}{'AUC':>8}{'ΔAUC':>8}{'AP':>8}{'최대F1':>8}")
    prev_auc = None
    results = []
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
        d_auc = auc - prev_auc if prev_auc is not None else 0.0
        print(f"  {tag:<32}{auc:8.4f}{d_auc:+8.4f}{ap:8.4f}{np.nanmax(f1s):8.4f}")
        results.append((tag, auc, ap, float(np.nanmax(f1s)), X.shape[1]))
        prev_auc = auc

    print(f"\n  해석")
    d_nb_now = results[2][1] - results[1][1]
    d_nb_tend = results[3][1] - results[2][1]
    print(f"    이웃 현재값이 주는 AUC 증분   : {d_nb_now:+.4f}  "
          f"(= Re축이 이미 담당하는 몫)")
    print(f"    이웃 경향이 추가로 주는 증분  : {d_nb_tend:+.4f}  "
          f"(= 현재 어느 축도 안 쓰는 시공간 정보)")
    if d_nb_tend > 0.005:
        print(f"    → 새 데이터 없이도 남은 여지가 있다. 이웃의 시간 변화(이류 방향)를")
        print(f"      입력에 넣는 것이 AWS 신규 수집보다 먼저다.")
    else:
        print(f"    → 이웃 경향의 추가 이득이 미미하다. 기존 12개 관측소에서 짜낼 수 있는")
        print(f"      공간·시공간 정보는 사실상 소진됐다 — 개선하려면 다른 종류의 관측이 필요하다.")
    print(f"\n  참고 — 현재 배포 모델의 강수 발생 F1 0.463 과 위 '최대F1' 은 표본·분할이")
    print(f"  달라 직접 비교하지 않는다. 여기서 유효한 것은 단계 간 **증분**이다.")
    return results


def main():
    timestamps, dates, hour_idx, stns, fields = load_matrices()
    part1_nugget(stns, fields["precip"], fields["temp"])
    part2_advection(stns, fields["precip"], hour_idx)
    part3_ablation(stns, fields, hour_idx, dates)


if __name__ == "__main__":
    main()
