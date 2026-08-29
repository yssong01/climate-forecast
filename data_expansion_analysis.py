"""
data_expansion_analysis.py — 추가 관측 데이터의 기대 이득을 기존 데이터에서 정량화한다.

**왜 필요한가.** "관측망을 조밀하게 하면 강수 예측이 좋아질 것"은 지금까지 추측이었다.
근거로 든 것은 Re축을 합성 위성에서 실측 IDW로 교체했을 때 황사 F1이 0.257→0.661로
오른 사례뿐이며, 그것이 강수에도 이전된다는 보장은 없다. 실제로 AWS 자료를 받아
재학습해야만 알 수 있는 것처럼 보이지만, **일부는 지금 있는 12개 관측소 자료만으로
측정할 수 있다.**

핵심 착상 — 관측소를 k개 쓸 때의 정보 이득을 k=1..11로 재면 그 증가 곡선의 모양이
"관측망을 더 늘리면 이득이 있는가"에 답한다. 곡선이 k=11에서도 계속 오르고 있으면
포화 전이므로 조밀화의 기대 이득이 크고, 이미 평평하면 12개로 이미 포화된 것이라
AWS를 붙여도 강수 예측은 개선되지 않는다. 이는 외삽이지만 **데이터에 근거한** 외삽이다.

다섯 부분으로 구성한다.
  A. 공간 상관 감쇠 길이(decorrelation length) — 강수장이 12개 관측소 간격으로
     표본화 가능한 규모인지. 기온과 대조해 강수의 공간 규모가 얼마나 작은지 본다.
  B. 이류(advection) 신호 — 시차 교차상관이 방위(bearing)에 따라 비대칭인지.
     비대칭하면 "상류 관측소가 하류의 미래를 안다"는 뜻이고, 현재 Re축(등방
     IDW 보간)은 그 방향성을 표현하지 못하므로 미사용 신호가 남아 있다는 증거다.
  C. 정보 이득 스케일링 — 조건부 상호정보량 I(Y ; 이웃 k개 | 국지관측)를 k에 대해
     측정. 위 착상의 핵심.
  D. 예측 상한(oracle) — +6시간 후 강수를 "지금 어딘가에서 이미 관측되고 있는가"로
     맞힐 때의 F1. 현재 모델(0.463)과 비교해 남은 여지를 잰다.
  E. 실패 사건 진단 — 모델이 놓친 강수(false negative)에서 이웃·상류에 신호가
     있었는지. 있었다면 입력 부족이 아니라 표현/학습 문제다.

실행: python data_expansion_analysis.py
"""
import json
import math
from collections import defaultdict

import numpy as np

from weather_collector import STATION_COORDS

DATA_PATH = "./cache/historical_data_1y.json"
WET_THRESH = 0.1     # mm — train.WET_THRESH 와 동일 기준
LEAD_HOURS = 6       # 배포 +6h 경로와 동일

STATION_NAMES = {
    "101": "춘천", "105": "강릉", "108": "서울", "112": "인천", "119": "수원",
    "131": "청주", "133": "대전", "143": "대구", "146": "전주", "156": "광주",
    "159": "부산", "184": "제주",
}


# ── 기하 ─────────────────────────────────────────────────────
def haversine(a, b):
    """두 (위도, 경도) 사이 거리(km)."""
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def bearing(a, b):
    """a에서 b를 향하는 방위각(도, 북=0, 동=90)."""
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlon = lon2 - lon1
    y = math.sin(dlon) * math.cos(lat2)
    x = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360) % 360


# ── 데이터 적재 ───────────────────────────────────────────────
def load_matrices():
    """(시각 × 관측소) 행렬로 정렬. 결측은 NaN 으로 둔다(채우지 않는다)."""
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    print(f"레코드 {len(records):,}개 적재")

    stns = sorted(STATION_COORDS.keys())
    sidx = {s: i for i, s in enumerate(stns)}

    by_ts = defaultdict(dict)
    for r in records:
        s = str(r.get("stn"))
        if s not in sidx:
            continue
        by_ts[str(r["timestamp"])][s] = r

    timestamps = sorted(by_ts.keys())
    T, S = len(timestamps), len(stns)
    precip = np.full((T, S), np.nan, dtype=np.float32)
    temp = np.full((T, S), np.nan, dtype=np.float32)
    wdir = np.full((T, S), np.nan, dtype=np.float32)
    wspd = np.full((T, S), np.nan, dtype=np.float32)

    for t, ts in enumerate(timestamps):
        for s, r in by_ts[ts].items():
            i = sidx[s]
            precip[t, i] = r.get("precipitation", np.nan)
            temp[t, i] = r.get("temperature", np.nan)
            wdir[t, i] = r.get("wind_dir", np.nan)
            wspd[t, i] = r.get("wind_speed", np.nan)

    print(f"정렬 완료 — 시각 {T:,}개 × 관측소 {S}개")
    print(f"강수 결측률 {np.isnan(precip).mean():.1%} · 기온 결측률 {np.isnan(temp).mean():.1%}")
    return timestamps, stns, precip, temp, wdir, wspd


def contiguous_index(timestamps):
    """타임스탬프를 시간 단위 정수 인덱스로. +6h 시차를 정확히 잡기 위함."""
    import datetime as dt
    base = dt.datetime.strptime(timestamps[0][:12], "%Y%m%d%H%M")
    return np.array([
        int((dt.datetime.strptime(ts[:12], "%Y%m%d%H%M") - base).total_seconds() // 3600)
        for ts in timestamps
    ])


# ── 통계 도구 ─────────────────────────────────────────────────
def nan_corr(x, y):
    """두 벡터의 피어슨 상관 — 양쪽 모두 유효한 표본만."""
    m = ~np.isnan(x) & ~np.isnan(y)
    if m.sum() < 100:
        return np.nan, int(m.sum())
    xv, yv = x[m], y[m]
    if xv.std() < 1e-9 or yv.std() < 1e-9:
        return np.nan, int(m.sum())
    return float(np.corrcoef(xv, yv)[0, 1]), int(m.sum())


def discrete_mi(x, y, nx, ny):
    """이산 변수 두 개의 상호정보량(bit). x∈[0,nx), y∈[0,ny)."""
    joint = np.zeros((nx, ny), dtype=np.float64)
    np.add.at(joint, (x, y), 1.0)
    n = joint.sum()
    if n < 1:
        return 0.0
    p = joint / n
    px = p.sum(axis=1, keepdims=True)
    py = p.sum(axis=0, keepdims=True)
    nz = p > 0
    return float((p[nz] * np.log2(p[nz] / (px @ py)[nz])).sum())


def conditional_mi(x, y, z, nx, ny, nz_):
    """조건부 상호정보량 I(X;Y|Z) = Σ_z P(z) I(X;Y | Z=z), bit 단위."""
    total = 0.0
    n = len(z)
    for zv in range(nz_):
        m = z == zv
        k = int(m.sum())
        if k < 200:
            continue
        total += (k / n) * discrete_mi(x[m], y[m], nx, ny)
    return total


def f1_binary(pred, true):
    tp = int((pred & true).sum())
    fp = int((pred & ~true).sum())
    fn = int((~pred & true).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    return (2 * p * r / (p + r) if p + r else 0.0), p, r


# ── A. 공간 상관 감쇠 ─────────────────────────────────────────
def part_a(stns, precip, temp):
    print(f"\n{'=' * 88}\n A. 공간 상관 감쇠 길이 — 강수장은 12개 관측소로 표본화되는가\n{'=' * 88}")
    S = len(stns)
    rows = []
    for i in range(S):
        for j in range(i + 1, S):
            d = haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            # 강수는 발생 여부(이진)와 양(연속) 둘 다 본다 — 성격이 다르다.
            wet_i = (precip[:, i] >= WET_THRESH).astype(np.float32)
            wet_j = (precip[:, j] >= WET_THRESH).astype(np.float32)
            wet_i[np.isnan(precip[:, i])] = np.nan
            wet_j[np.isnan(precip[:, j])] = np.nan
            r_occ, n_occ = nan_corr(wet_i, wet_j)
            r_amt, _ = nan_corr(precip[:, i], precip[:, j])
            r_tmp, _ = nan_corr(temp[:, i], temp[:, j])
            rows.append((d, r_occ, r_amt, r_tmp, n_occ, stns[i], stns[j]))

    rows.sort()
    print(f"  {'거리(km)':>9}{'강수발생 r':>12}{'강수량 r':>11}{'기온 r':>10}   관측소쌍")
    for d, r_occ, r_amt, r_tmp, _, si, sj in rows[:6]:
        print(f"  {d:9.0f}{r_occ:12.3f}{r_amt:11.3f}{r_tmp:10.3f}   "
              f"{STATION_NAMES[si]}–{STATION_NAMES[sj]}")
    print(f"  {'...':>9}")
    for d, r_occ, r_amt, r_tmp, _, si, sj in rows[-3:]:
        print(f"  {d:9.0f}{r_occ:12.3f}{r_amt:11.3f}{r_tmp:10.3f}   "
              f"{STATION_NAMES[si]}–{STATION_NAMES[sj]}")

    d_arr = np.array([r[0] for r in rows])
    occ = np.array([r[1] for r in rows])
    amt = np.array([r[2] for r in rows])
    tmp = np.array([r[3] for r in rows])

    # exp(-d/L) 감쇠 적합 — log r = -d/L (r>0 인 쌍만).
    def fit_length(r):
        m = (r > 0.01) & ~np.isnan(r)
        if m.sum() < 5:
            return np.nan
        # 절편 없는 최소제곱: log r = -d/L
        return float(-(d_arr[m] @ d_arr[m]) / (d_arr[m] @ np.log(r[m])))

    L_occ, L_amt, L_tmp = fit_length(occ), fit_length(amt), fit_length(tmp)
    nn = []
    for i in range(S):
        ds = [haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
              for j in range(S) if j != i]
        nn.append(min(ds))
    mean_nn = float(np.mean(nn))

    print(f"\n  exp(−d/L) 적합 감쇠 길이 L")
    print(f"    강수 발생 : {L_occ:7.0f} km")
    print(f"    강수량     : {L_amt:7.0f} km")
    print(f"    기온       : {L_tmp:7.0f} km   (대조군 — 공간 규모가 훨씬 큼)")
    print(f"  현재 관측망 평균 최근접 이웃 거리: {mean_nn:.0f} km")
    print(f"  나이퀴스트(Nyquist) 관점 — 규모 L 의 구조를 분해하려면 관측 간격이 L/2 이하여야 한다.")
    print(f"    강수 발생 기준 요구 간격 {L_occ/2:.0f} km  vs  현재 {mean_nn:.0f} km  →  "
          f"{'표본화 부족(undersampled)' if mean_nn > L_occ / 2 else '표본화 충분'}")
    return dict(L_occ=L_occ, L_amt=L_amt, L_tmp=L_tmp, mean_nn=mean_nn)


# ── B. 이류 신호 ──────────────────────────────────────────────
def part_b(stns, precip, hour_idx, wdir, wspd):
    print(f"\n{'=' * 88}\n B. 이류(advection) 신호 — 상류 관측소가 하류의 미래를 아는가\n{'=' * 88}")
    S = len(stns)
    T = len(hour_idx)
    # +LEAD 시차 정렬 — 시각 인덱스로 정확히 맞춘다(결측 시간대 존재).
    pos = {h: t for t, h in enumerate(hour_idx)}
    src_t, tgt_t = [], []
    for t, h in enumerate(hour_idx):
        t2 = pos.get(h + LEAD_HOURS)
        if t2 is not None:
            src_t.append(t)
            tgt_t.append(t2)
    src_t = np.array(src_t)
    tgt_t = np.array(tgt_t)
    print(f"  +{LEAD_HOURS}h 시차 정렬 표본 {len(src_t):,}개")

    wet = (precip >= WET_THRESH).astype(np.float32)
    wet[np.isnan(precip)] = np.nan

    # 각 순서쌍 (i→j): corr(wet_i(t), wet_j(t+6))
    recs = []
    for i in range(S):
        for j in range(S):
            if i == j:
                continue
            r, n = nan_corr(wet[src_t, i], wet[tgt_t, j])
            d = haversine(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            b = bearing(STATION_COORDS[stns[i]], STATION_COORDS[stns[j]])
            recs.append(dict(i=i, j=j, r=r, d=d, b=b, n=n))

    # 방향 비대칭 — (i→j) 와 (j→i) 의 차이. 양수면 i가 상류.
    asym = []
    lookup = {(r["i"], r["j"]): r for r in recs}
    for i in range(S):
        for j in range(i + 1, S):
            a, bb = lookup[(i, j)], lookup[(j, i)]
            if np.isnan(a["r"]) or np.isnan(bb["r"]):
                continue
            asym.append((a["r"] - bb["r"], a["b"], a["d"], i, j))

    asym.sort(reverse=True)
    print(f"\n  방향 비대칭 상위 — corr(i(t), j(t+6)) − corr(j(t), i(t+6))")
    print(f"  {'비대칭':>8}{'방위(i→j)':>11}{'거리km':>9}   상류 → 하류")
    for a, b, d, i, j in asym[:8]:
        print(f"  {a:+8.3f}{b:11.0f}{d:9.0f}   {STATION_NAMES[stns[i]]} → {STATION_NAMES[stns[j]]}")

    # 비대칭이 방위와 체계적으로 연관되는가 — 서→동이면 편서풍 이류의 증거.
    a_vals = np.array([x[0] for x in asym])
    b_vals = np.array([x[1] for x in asym])
    # 방위 90도(정동) 성분과의 상관: cos(b − 90) = sin(b)
    east_comp = np.sin(np.radians(b_vals))
    r_dir, _ = nan_corr(a_vals, east_comp)
    print(f"\n  비대칭 vs 동향 성분 sin(방위) 상관: r = {r_dir:+.3f}")
    print(f"    (양수면 '서쪽 관측소가 동쪽의 미래를 안다' = 편서풍 이류 신호)")

    # 실제 바람도 확인 — 강수 시 평균 풍향
    wet_any = np.nanmax(wet, axis=1) > 0.5
    wd = wdir[wet_any]
    ws = wspd[wet_any]
    m = ~np.isnan(wd) & ~np.isnan(ws)
    u = -np.sin(np.radians(wd[m])) * ws[m]      # 동향 성분(바람이 불어가는 쪽)
    print(f"  강수 시각의 평균 동향 바람 성분: {np.mean(u):+.2f} m/s "
          f"({'서→동 우세' if np.mean(u) > 0 else '동→서 우세'})")
    typical_speed_kmh = float(np.nanmean(ws[m]) * 3.6)
    print(f"  강수 시각 평균 풍속 {typical_speed_kmh:.1f} km/h → "
          f"{LEAD_HOURS}시간 이동거리 약 {typical_speed_kmh * LEAD_HOURS:.0f} km")
    print(f"    (지상풍은 상층 이동속도의 하한이다 — 실제 강수계 이동은 보통 이보다 빠르다)")
    return dict(r_dir=r_dir, asym_max=float(a_vals.max()),
                advect_km=typical_speed_kmh * LEAD_HOURS, src_t=src_t, tgt_t=tgt_t)


# ── C. 정보 이득 스케일링 ─────────────────────────────────────
def part_c(stns, precip, src_t, tgt_t):
    print(f"\n{'=' * 88}\n C. 정보 이득 스케일링 — 관측소를 k개 쓸 때 정보가 얼마나 늘어나는가\n{'=' * 88}")
    print("  조건부 상호정보량 I(Y ; 이웃k | 국지관측), Y = 대상 관측소의 +6h 강수 발생.")
    print("  k를 늘려도 계속 오르면 12개로 아직 포화되지 않은 것 — 조밀화 이득이 남아 있다.")
    S = len(stns)
    wet = (precip >= WET_THRESH)
    valid = ~np.isnan(precip)

    curves = {}
    for tgt in range(S):
        # 대상 관측소 기준 가까운 순 이웃 정렬
        order = sorted([j for j in range(S) if j != tgt],
                       key=lambda j: haversine(STATION_COORDS[stns[tgt]],
                                               STATION_COORDS[stns[j]]))
        y = wet[tgt_t, tgt].astype(np.int64)
        local = wet[src_t, tgt].astype(np.int64)          # 국지 현재 강수(조건)
        ok = valid[tgt_t, tgt] & valid[src_t, tgt]
        vals = []
        for k in range(0, len(order) + 1):
            if k == 0:
                vals.append(0.0)
                continue
            sel = order[:k]
            nb = wet[src_t][:, sel]
            nbv = valid[src_t][:, sel]
            # 이웃 요약통계 — 비오는 이웃 비율을 4구간으로 이산화(차원의 저주 회피)
            cnt = np.where(nbv, nb, 0).sum(axis=1)
            den = np.maximum(nbv.sum(axis=1), 1)
            frac = cnt / den
            nb_state = np.digitize(frac, [1e-9, 0.34, 0.67]).astype(np.int64)  # 0..3
            m = ok & (nbv.sum(axis=1) > 0)
            cmi = conditional_mi(y[m], nb_state[m], local[m], 2, 4, 2)
            vals.append(cmi)
        curves[tgt] = vals

    ks = list(range(0, S))
    mean_curve = np.array([np.mean([curves[t][k] for t in range(S)]) for k in ks])
    print(f"\n  {'k(이웃 수)':>10}{'평균 I(bit)':>13}{'직전 대비 증분':>15}{'포화율':>9}")
    for k in ks:
        inc = mean_curve[k] - mean_curve[k - 1] if k > 0 else mean_curve[k]
        sat = mean_curve[k] / mean_curve[-1] if mean_curve[-1] > 0 else 0.0
        print(f"  {k:10d}{mean_curve[k]:13.5f}{inc:15.5f}{sat:9.1%}")

    # 마지막 증분이 전체의 몇 %인가 — 포화 판정
    last_inc = mean_curve[-1] - mean_curve[-2]
    rel_last = last_inc / mean_curve[-1] if mean_curve[-1] > 0 else 0.0
    print(f"\n  11번째 이웃의 한계 증분: {last_inc:.5f} bit "
          f"(전체의 {rel_last:.1%})")
    if rel_last > 0.02:
        print("  판정: 아직 포화 전 — 관측소를 더 늘리면 정보가 계속 늘어난다(조밀화 이득 있음).")
    else:
        print("  판정: 사실상 포화 — 12개로 이미 공간 정보를 다 뽑았다(조밀화 이득 낮음).")
    return dict(mean_curve=mean_curve.tolist(), rel_last=float(rel_last))


# ── D. 예측 상한(oracle) ─────────────────────────────────────
def part_d(stns, precip, src_t, tgt_t):
    print(f"\n{'=' * 88}\n D. 예측 상한 — '지금 어딘가에서 이미 비가 온다'만으로 +6h를 맞히면\n{'=' * 88}")
    S = len(stns)
    wet = (precip >= WET_THRESH)
    valid = ~np.isnan(precip)

    res = {}
    for tag, radius in [("최근접 1곳", 1), ("가까운 3곳", 3), ("전체 11곳", 11)]:
        preds, trues = [], []
        for tgt in range(S):
            order = sorted([j for j in range(S) if j != tgt],
                           key=lambda j: haversine(STATION_COORDS[stns[tgt]],
                                                   STATION_COORDS[stns[j]]))[:radius]
            nb = wet[src_t][:, order]
            nbv = valid[src_t][:, order]
            m = valid[tgt_t, tgt] & (nbv.sum(axis=1) > 0)
            preds.append(np.where(nbv[m], nb[m], False).any(axis=1))
            trues.append(wet[tgt_t, tgt][m])
        pred = np.concatenate(preds)
        true = np.concatenate(trues)
        f1, p, r = f1_binary(pred, true)
        res[tag] = (f1, p, r)
        print(f"  {tag:<12} F1={f1:.4f}  정밀도={p:.3f}  재현율={r:.3f}")

    # 국지 지속성(persistence) 기준선 — 지금 비 오면 6시간 뒤에도 온다
    preds, trues = [], []
    for tgt in range(S):
        m = valid[tgt_t, tgt] & valid[src_t, tgt]
        preds.append(wet[src_t, tgt][m])
        trues.append(wet[tgt_t, tgt][m])
    f1, p, r = f1_binary(np.concatenate(preds), np.concatenate(trues))
    print(f"  {'국지 지속성':<12} F1={f1:.4f}  정밀도={p:.3f}  재현율={r:.3f}   ← 가장 단순한 기준선")
    print(f"\n  참고 — 현재 배포 모델의 강수 발생 판정 F1 = 0.463 (확률 게이팅 적용 후)")
    res["persistence"] = (f1, p, r)
    return res


# ── E. 실패 사건 진단 ────────────────────────────────────────
def part_e():
    print(f"\n{'=' * 88}\n E. 모델이 놓친 강수 — 그때 이웃에는 신호가 있었는가\n{'=' * 88}")
    try:
        import eval_cache
        from predict import CHECKPOINT, PRECIP_PROB_GATE_BY_LEAD, PRECIP_PROB_GATE
        import torch
        ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
        gate = PRECIP_PROB_GATE_BY_LEAD.get(ckpt.get("lead_hours"), PRECIP_PROB_GATE)
        d = eval_cache.load(CHECKPOINT)
    except Exception as e:
        print(f"  건너뜀 — eval_cache 사용 불가: {type(e).__name__}")
        return None

    pp = np.asarray(d["precip_pred"])
    pt = np.asarray(d["precip_true"])
    rp = np.asarray(d["rain_prob"])
    served = np.where(rp < gate, 0.0, pp)
    wet_true = pt >= WET_THRESH
    wet_pred = served >= WET_THRESH

    fn = wet_true & ~wet_pred
    tp = wet_true & wet_pred
    print(f"  검증 표본 {len(pt):,}개 · 실제 강수 {int(wet_true.sum()):,}개")
    print(f"  놓친 강수(FN) {int(fn.sum()):,}개 · 맞힌 강수(TP) {int(tp.sum()):,}개")
    print(f"\n  놓친 사건의 강수량 분포(실측):")
    for lo, hi, tag in [(0.1, 0.5, "0.1~0.5mm"), (0.5, 1, "0.5~1mm"),
                        (1, 5, "1~5mm"), (5, 20, "5~20mm"), (20, 1e9, "20mm 이상")]:
        band = fn & (pt >= lo) & (pt < hi)
        band_all = wet_true & (pt >= lo) & (pt < hi)
        share = band.sum() / max(band_all.sum(), 1)
        print(f"    {tag:<12} 놓침 {int(band.sum()):>6,} / 전체 {int(band_all.sum()):>6,}  "
              f"({share:.1%} 놓침)")
    print(f"\n  모델이 낸 rain_prob 의 분포 — 놓친 사건에서")
    print(f"    중앙값 {np.median(rp[fn]):.3f} · 90백분위 {np.percentile(rp[fn], 90):.3f}")
    print(f"    맞힌 사건 중앙값 {np.median(rp[tp]):.3f}")
    print(f"    (놓친 사건의 확률이 0 근처면 '신호 자체를 못 봤다', 게이트 근처면 '봤는데 잘렸다')")
    near_gate = ((rp[fn] > gate - 0.15) & (rp[fn] < gate)).mean()
    print(f"    놓친 사건 중 게이트 바로 아래(τ−0.15~τ) 비율: {near_gate:.1%}")
    return dict(n_fn=int(fn.sum()), near_gate=float(near_gate))


def main():
    timestamps, stns, precip, temp, wdir, wspd = load_matrices()
    hour_idx = contiguous_index(timestamps)

    a = part_a(stns, precip, temp)
    b = part_b(stns, precip, hour_idx, wdir, wspd)
    c = part_c(stns, precip, b["src_t"], b["tgt_t"])
    dd = part_d(stns, precip, b["src_t"], b["tgt_t"])
    e = part_e()

    print(f"\n{'=' * 88}\n 종합\n{'=' * 88}")
    print(f"  강수 발생 공간 상관 감쇠 길이 L = {a['L_occ']:.0f} km, "
          f"현재 관측 간격 {a['mean_nn']:.0f} km")
    print(f"  이류 비대칭 방향성 r = {b['r_dir']:+.3f}, "
          f"{LEAD_HOURS}h 지상풍 이동거리 ≈ {b['advect_km']:.0f} km")
    print(f"  11번째 이웃의 한계 정보 증분 비율 = {c['rel_last']:.1%}")
    print(f"  oracle(전체 11곳 현재 강수) F1 = {dd['전체 11곳'][0]:.4f} "
          f"vs 현재 모델 0.463")


if __name__ == "__main__":
    main()
