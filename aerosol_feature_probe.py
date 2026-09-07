"""
aerosol_feature_probe.py — 황사 헤드에 외부 에어로졸 자료를 붙였을 때
변별력이 얼마나 오르는지 잰다. `nwp_feature_probe.py` 와 같은 절제 방식.

**왜 황사인가.** `improvement-levers-measured` 측정에서 네 확률 헤드 중
황사만 변별력이 병목이었다(AUC 0.824 / 폭염 0.962 · 강수 0.914 · 한파 0.878).
이 저장소는 원인을 이미 진단했다 — PM10 은 국외 장거리 수송이 지배하는데
입력은 국내 12개 지점 지상 관측뿐이다. 결론은 "외부 에어로졸이 사실상
유일한 레버"였고, 자료원이 없어 검증하지 못한 채 남아 있었다.

**설계의 핵심은 상류다.** 관측소 자신의 에어로졸 값만 넣으면 이미 입력에
있는 국내 관측과 크게 다르지 않다. 황사는 고비·내몽골에서 발원해 편서풍으로
실려 오므로, 예측 시점에 필요한 정보는 **지금 상류에 얼마나 떠 있는가**다
(`collect_aerosol_archive.UPSTREAM` 다섯 지점). 강수에서 "+6h 뒤 도달할
강수계는 지금 404km 상류에 있다"고 밝힌 것과 같은 구조다.

**이 측정은 상한이다 — 수치예보 프로브가 하한이었던 것과 정반대다.**
대기질 API 에는 리드타임을 보존하는 아카이브가 없다(`dust_previous_day1`
전부 null 로 실측 확인). 그래서
  ① 목표 시각 T+L 의 에어로졸 값은 **쓰지 않는다** — 그 시점 관측을 반영한
     분석장이라 명백한 누수다. 예측 시점 T 이하의 값만 쓴다.
  ② 그럼에도 아카이브의 T 시점 값은 분석장이고 실서빙의 T 시점 값은 그보다
     앞서 발표된 예보라, 학습이 서빙보다 유리하다.
따라서 **상한이 작으면 그대로 기각**할 수 있고, 크면 리드타임을 보존하는
자료원을 찾아 다시 재야 한다 — 이 값을 그대로 기대치로 삼으면 안 된다.

실행: python aerosol_feature_probe.py       (LEAD_HOURS 환경변수로 6/12)
"""
import json
import math
import os
from collections import defaultdict

import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from weather_collector import STATION_COORDS
from collect_aerosol_archive import VARS as AERO_VARS, UPSTREAM

# 아래 인덱스 상수는 AERO_VARS 순서에 묶여 있다 — 수집기에서 변수를
# 추가·재배열하면 여기도 함께 고쳐야 하므로, 임포트한 목록으로 실제
# 순서를 검증한다(조용히 어긋나면 특징이 뒤섞인 채 학습된다).

DATA_PATH = "./cache/historical_data_1y.json"
LABEL_PATH = "./cache/weather_issue_labels.json"
AERO_PATH = "./cache/aerosol_archive.json"
LEAD_HOURS = int(os.getenv("LEAD_HOURS", "6"))
LAGS = [3, 6, 12]        # 상류 경향은 강수보다 긴 시간 규모라 더 길게 본다
SEED = 7

I_DUST, I_PM10, I_PM25, I_AOD, I_CO = range(5)
assert AERO_VARS == ["dust", "pm10", "pm2_5", "aerosol_optical_depth",
                     "carbon_monoxide"], \
    f"collect_aerosol_archive.VARS 순서가 바뀌었다 — 인덱스 상수를 맞출 것: {AERO_VARS}" 


def haversine(a, b):
    R = 6371.0
    lat1, lon1 = math.radians(a[0]), math.radians(a[1])
    lat2, lon2 = math.radians(b[0]), math.radians(b[1])
    dlat, dlon = lat2 - lat1, lon2 - lon1
    h = math.sin(dlat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon / 2) ** 2
    return 2 * R * math.asin(math.sqrt(h))


def _ts_add(ts, hours):
    from datetime import datetime, timedelta
    return (datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")
            + timedelta(hours=hours)).strftime("%Y%m%d%H%M")


def load_all():
    with open(DATA_PATH, "r", encoding="utf-8") as f:
        records = json.load(f)
    with open(LABEL_PATH, "r", encoding="utf-8") as f:
        labels = json.load(f)
    with open(AERO_PATH, "r", encoding="utf-8") as f:
        aero_raw = json.load(f)
    aero = {k: {ts: np.asarray(v, dtype=np.float32) for ts, v in rows.items()}
            for k, rows in aero_raw.items()}
    stns = sorted(STATION_COORDS.keys())
    by_ts = defaultdict(dict)
    for r in records:
        s = str(r.get("stn"))
        if s in stns:
            by_ts[str(r["timestamp"])[:12]][s] = r
    print(f"실측 {len(records):,}건 · 에어로졸 지점 {len(aero)}곳 "
          f"(관측소 {len(STATION_COORDS)} + 상류 {len(UPSTREAM)})")
    return by_ts, labels, aero, stns


def build(by_ts, labels, aero, stns, k_neighbor=5):
    """지상 관측 특징군과 에어로졸 특징군을 같은 표본에서 만든다."""
    groups = {"local": [], "neighbor": [], "aero_local": [], "aero_up": []}
    ys, ds = [], []
    times = sorted(by_ts.keys())
    tset = set(times)
    up_names = list(UPSTREAM)

    for ts in times:
        tgt_ts = _ts_add(ts, LEAD_HOURS)
        if tgt_ts not in tset:
            continue
        lag_ts = [_ts_add(ts, -L) for L in LAGS]
        if any(x not in tset for x in lag_ts):
            continue
        # 상류 지점은 관측소와 무관하게 시각당 한 번만 만든다.
        up = []
        ok = True
        for n in up_names:
            rows = aero.get(n, {})
            cur = rows.get(ts)
            if cur is None:
                ok = False
                break
            up += [cur[I_DUST], cur[I_PM10], cur[I_AOD], cur[I_CO]]
            for lt in lag_ts:
                prev = rows.get(lt)
                if prev is None:
                    ok = False
                    break
                up.append(cur[I_DUST] - prev[I_DUST])
            if not ok:
                break
        if not ok:
            continue

        for stn, rec in by_ts[ts].items():
            day = labels.get(stn, {}).get(f"{tgt_ts[0:4]}-{tgt_ts[4:6]}-{tgt_ts[6:8]}", {})
            if "dust_observed" not in day:
                continue              # 공식 라벨이 없는 관측소·날짜는 채점 불가
            base = [rec.get("temperature"), rec.get("humidity"), rec.get("wind_speed"),
                    rec.get("pressure"), rec.get("precipitation")]
            wd = rec.get("wind_dir")
            if any(x is None for x in base) or wd is None:
                continue
            u = -math.sin(math.radians(wd)) * base[2]
            v = -math.cos(math.radians(wd)) * base[2]
            mm = int(ts[4:6])
            local = base + [u, v, math.sin(2 * math.pi * mm / 12),
                            math.cos(2 * math.pi * mm / 12)]

            order = sorted([s for s in stns if s != stn],
                           key=lambda j: haversine(STATION_COORDS[stn],
                                                   STATION_COORDS[j]))[:k_neighbor]
            nb = []
            bad = False
            for j in order:
                r2 = by_ts[ts].get(j)
                if r2 is None or r2.get("humidity") is None or r2.get("wind_speed") is None:
                    bad = True
                    break
                nb += [r2["humidity"], r2["wind_speed"], r2.get("pressure") or 1013.0]
            if bad:
                continue

            rows = aero.get(stn, {})
            cur = rows.get(ts)
            if cur is None:
                continue
            al = list(cur)
            for lt in lag_ts:
                prev = rows.get(lt)
                if prev is None:
                    al = None
                    break
                al += [cur[I_DUST] - prev[I_DUST], cur[I_PM10] - prev[I_PM10]]
            if al is None:
                continue

            groups["local"].append(local)
            groups["neighbor"].append(nb)
            groups["aero_local"].append(al)
            groups["aero_up"].append(up)
            ys.append(int(day["dust_observed"]))
            ds.append(ts[:8])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in groups.items()}
    return out, np.asarray(ys), np.asarray(ds)


def main():
    by_ts, labels, aero, stns = load_all()
    print(f"\n{'=' * 88}\n 황사 헤드 — 에어로졸 특징 절제 (리드타임 +{LEAD_HOURS}h)\n{'=' * 88}")
    g, y, ds = build(by_ts, labels, aero, stns)
    if len(y) == 0:
        raise SystemExit("표본이 0개 — 에어로졸 아카이브와 라벨의 겹치는 구간이 없다.")
    print(f"  표본 {len(y):,}개 · 양성률 {y.mean():.2%} "
          f"(에어로졸 아카이브가 있는 구간·공식 라벨 보유 관측소로 한정)")

    uniq = np.unique(ds)
    rng = np.random.RandomState(SEED)
    val_dates = set(rng.choice(uniq, size=int(len(uniq) * 0.2), replace=False).tolist())
    is_val = np.array([d in val_dates for d in ds])
    print(f"  날짜 그룹 분할 — 학습 {int((~is_val).sum()):,} / 검증 {int(is_val.sum()):,}")

    stages = [
        ("국지 지상관측", ["local"]),
        ("+ 이웃 지상관측", ["local", "neighbor"]),
        ("+ 관측소 에어로졸", ["local", "neighbor", "aero_local"]),
        ("+ 상류 에어로졸 (신규)", ["local", "neighbor", "aero_local", "aero_up"]),
        ("[참고] 상류 단독", ["aero_up"]),
    ]
    print(f"\n  {'특징군':<26}{'AUC':>9}{'ΔAUC':>9}{'AP':>9}{'최대F1':>9}{'차원':>6}")
    prev, res = None, []
    for tag, keys in stages:
        X = np.concatenate([g[k] for k in keys], axis=1)
        clf = HistGradientBoostingClassifier(max_iter=200, learning_rate=0.1,
                                             max_depth=6, random_state=SEED)
        clf.fit(X[~is_val], y[~is_val])
        p = clf.predict_proba(X[is_val])[:, 1]
        auc = roc_auc_score(y[is_val], p)
        ap = average_precision_score(y[is_val], p)
        pr, rc, _ = precision_recall_curve(y[is_val], p)
        f1 = np.nanmax(2 * pr * rc / np.maximum(pr + rc, 1e-12))
        d = (auc - prev) if (prev is not None and not tag.startswith("[참고]")) else 0.0
        print(f"  {tag:<26}{auc:9.4f}{d:+9.4f}{ap:9.4f}{f1:9.4f}{X.shape[1]:6d}")
        res.append((tag, auc, ap, float(f1)))
        if not tag.startswith("[참고]"):
            prev = auc

    print(f"\n  관측소 에어로졸 증분 : {res[2][1] - res[1][1]:+.4f}")
    print(f"  상류 에어로졸 증분   : {res[3][1] - res[2][1]:+.4f}  ← 이 실험의 핵심")
    print(f"\n  주의: 이 값은 **상한**이다. 대기질 API 에는 리드타임 보존")
    print(f"  아카이브가 없어, 학습이 쓰는 T 시점 값은 분석장이고 실서빙은")
    print(f"  그보다 앞서 발표된 예보다 — 학습이 서빙보다 유리하다.")


if __name__ == "__main__":
    main()
