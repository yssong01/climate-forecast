"""
data_expansion_probe_island.py — data_expansion_probe.py 의 후속. 서해 도서 AWS
14개 지점을 실제 특징으로 넣었을 때 강수 발생 판정 AUC가 오르는지 측정한다.

**배경.** README '다음 단계 검토' 절은 apihub 추가 관측 우선순위를 AWS 5순위·
해양부이 1순위로 정했다. 그 판정 근거(data_expansion_probe.py part3_ablation
의 이웃 경향 증분 +0.001, 포화)는 **기존 12개 관측소 내부**의 공간정보 포화를
보인 것이지, 관측망 **바깥**의 신규 지점을 넣은 실험이 아니었다. 그런데 AWS
745지점 중 인천(126.62°E)보다 서쪽 도서 관측소가 14개이고, 그 거리
(112~424km)가 이류 지연으로 역산된 상류 거리(약 404km, part2_advection 실측)
와 겹친다. 이 스크립트는 그 겹침이 실제 예측 정보로 이어지는지를 잰다.

**데이터 출처.** `C:\\yssong\\Risk_Prediction`이 같은 KMA_API_KEY로 이미 백필한
전국 AWS 정시자료(2022-01-01~2026-03-15, `data/raw/kma/awsh/*.parquet`)에서
도서 14개 지점만 추출해 `cache/island_aws_raw.parquet`으로 옮겨 왔다(신규 API
호출 없음). climate-forecast의 12개 타깃 관측소 기록(`historical_data_1y.json`)
과 시각 교집합만 쓴다.

**특징 설계 — precip(rn_hr1)만 쓴다.** 도서 지점은 humid/wind 조인 시 시각별
14개 전부 유효 커버리지가 낮아지므로(실측: rn_hr1 66.7%), 강수 이류라는 핵심
가설과 가장 직결되는 단일 필드로 좁혀 완전성을 확보한다. 이 축소가 도서 특징의
전체 정보량을 과소평가할 수 있다는 한계를 명시한다.

**사전 기각 조건(착수 전 명시, 원 스크립트의 이웃 경향 기준 0.005와 동일선상).**
ΔAUC ≤ 0.005 면 도서 AWS 편입을 보류하고 README의 AWS 5순위 판정을 유지한다.
초과하면 재학습 검토(P1)로 진행한다.

실행 (GPU 컨테이너, pandas/pyarrow 필요):
  sg docker -c "docker run --rm -v $(pwd):/app tri-chef-train-gpu:latest data_expansion_probe_island.py"
"""
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, precision_recall_curve

from data_expansion_probe import (
    load_matrices, haversine, WET_THRESH, LEAD_HOURS, LAGS, SEED, STATION_NAMES,
)
from weather_collector import STATION_COORDS

ISLAND_PARQUET = "./cache/island_aws_raw.parquet"
DELTA_AUC_GATE = 0.005  # data_expansion_probe.py part3_ablation 과 동일 기준

ISLAND_COORDS = {
    "102": (37.97396, 124.71237),  # 백령도
    "169": (34.68719, 125.45105),  # 흑산도
    "229": (36.62544, 125.55954),  # 북격렬비도
    "303": (34.07275, 125.09741),  # 가거도
    "426": (37.96611, 124.63048),  # 백령(레)
    "501": (37.65891, 125.69819),  # 대연평
    "655": (37.76024, 124.72912),  # 소청도
    "663": (36.92863, 125.78691),  # 목덕도
    "697": (34.25058, 125.91829),  # 서거차도
    "700": (36.12508, 125.96820),  # 어청도
    "743": (34.77258, 125.94688),  # 비금
    "797": (34.39498, 125.29953),  # 하태도
    "798": (34.68551, 125.19236),  # 홍도
    # "956"(가대암)은 rn_hr1 이 전 기간 -99.0 고정값(결측 sentinel)이라 실측으로
    # 제외했다 — 강수계가 없는 지점으로 추정된다(2026-08-30 실측 확인).
}
ISLAND_NAMES = {
    "102": "백령도", "169": "흑산도", "229": "북격렬비도", "303": "가거도",
    "426": "백령(레)", "501": "대연평", "655": "소청도", "663": "목덕도",
    "697": "서거차도", "700": "어청도", "743": "비금", "797": "하태도",
    "798": "홍도",
}


def load_island_matrix(timestamps):
    """도서 AWS 강수(rn_hr1)만 (T, 14) 행렬로 정렬한다. 결측(플래그 또는 미보고)은 NaN."""
    df = pd.read_parquet(ISLAND_PARQUET, columns=["tm", "stn", "rn_hr1", "rn_hr1_missing"])
    islands = sorted(ISLAND_COORDS.keys())
    df = df[df["stn"].isin(islands)].copy()
    df.loc[df["rn_hr1_missing"].astype(bool), "rn_hr1"] = np.nan
    # dropna=False: 전 기간 결측인 지점(예 956 가대암, 강수계 없음)이 있어도
    # pivot_table 이 그 컬럼을 조용히 지우지 않게 한다 — 지우면 reindex 후
    # 그 컬럼이 통째로 NaN이 되어 "14개 전부 유효" 조건이 항상 거짓이 된다
    # (2026-08-30 실측으로 드러난 함정, ISLAND_COORDS 에서 956 제외로 해결했지만
    # 재발 방지로 남긴다).
    piv = df.pivot_table(index="tm", columns="stn", values="rn_hr1", aggfunc="first", dropna=False)
    missing_stns = sorted(set(islands) - set(piv.columns))
    if missing_stns:
        raise RuntimeError(f"도서 지점 {missing_stns} 이 원본에 전혀 없다 — ISLAND_COORDS 점검 필요")
    piv = piv.reindex(index=timestamps, columns=islands)
    coverage = piv.notna().all(axis=1).mean()
    print(f"  도서 14개 지점 강수 동시 커버리지: {coverage:.1%} "
          f"(climate-forecast 12관측소 시각 기준 재정렬 후)")
    return islands, piv.to_numpy(dtype=np.float32)


def build_features_island(stns, fields, islands, island_precip, hour_idx, dates, k_neighbor=5):
    """data_expansion_probe.build_features 와 동일 로직 + island_now 그룹만 추가."""
    S = len(stns)
    pos = {h: t for t, h in enumerate(hour_idx)}
    precip, temp = fields["precip"], fields["temp"]
    humid, wspd, wdir, press = fields["humid"], fields["wspd"], fields["wdir"], fields["press"]
    u = -np.sin(np.radians(wdir)) * wspd
    v = -np.cos(np.radians(wdir)) * wspd

    groups = {"local_now": [], "local_tend": [], "nb_now": [], "nb_tend": [], "island_now": []}
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
                nb_tend += [precip[t, j] - precip[lag_ts[1], j], press[t, j] - press[lag_ts[1], j],
                            humid[t, j] - humid[lag_ts[1], j]]
            island_now = island_precip[t, :].tolist()
            row = dict(local_now=base, local_tend=tend, nb_now=nb_now, nb_tend=nb_tend,
                       island_now=island_now)
            if any(np.isnan(np.array(vv, dtype=np.float64)).any() for vv in row.values()):
                continue
            for kk in groups:
                groups[kk].append(row[kk])
            ys.append(1 if precip[t_f, tgt] >= WET_THRESH else 0)
            ds.append(dates[t])
    out = {k: np.asarray(v, dtype=np.float32) for k, v in groups.items()}
    return out, np.asarray(ys), np.asarray(ds)


def part_ablation_with_island(stns, fields, islands, island_precip, hour_idx, dates):
    print(f"\n{'=' * 88}\n ④ 도서 AWS 강수 편입 — 서해 상류 실측이 추가 정보를 주는가\n{'=' * 88}")
    groups, y, ds = build_features_island(stns, fields, islands, island_precip, hour_idx, dates)
    print(f"  표본 {len(y):,}개 · 양성률 {y.mean():.1%}")

    uniq = np.unique(ds)
    rng = np.random.RandomState(SEED)
    val_dates = set(rng.choice(uniq, size=int(len(uniq) * 0.2), replace=False).tolist())
    is_val = np.array([d in val_dates for d in ds])
    print(f"  날짜 그룹 분할 — 학습 {int((~is_val).sum()):,} / 검증 {int(is_val.sum()):,}")

    stages = [
        ("국지 현재", ["local_now"]),
        ("+ 국지 경향", ["local_now", "local_tend"]),
        ("+ 이웃 현재 (Re축)", ["local_now", "local_tend", "nb_now"]),
        ("+ 이웃 경향", ["local_now", "local_tend", "nb_now", "nb_tend"]),
        ("+ 도서 강수 (서해 상류)", ["local_now", "local_tend", "nb_now", "nb_tend", "island_now"]),
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

    d_island = results[-1][1] - results[-2][1]
    print(f"\n  도서 강수가 주는 AUC 증분: {d_island:+.4f}  (게이트 기준 {DELTA_AUC_GATE:+.4f})")
    if d_island > DELTA_AUC_GATE:
        print(f"  → 게이트 통과. 도서 AWS 편입을 P1(재학습 검토)로 진행할 근거가 있다.")
    else:
        print(f"  → 게이트 미달. 도서 AWS 단독 편입은 보류 — README AWS 5순위 판정 유지.")

    # 개별 도서 지점의 중요도(순열 없이 계수 대체 지표) — 어느 지점이 기여하는지 참고용
    X_full = np.concatenate([groups[k] for k in
                              ["local_now", "local_tend", "nb_now", "nb_tend", "island_now"]], axis=1)
    clf_full = HistGradientBoostingClassifier(
        max_iter=200, learning_rate=0.1, max_depth=6, random_state=SEED)
    clf_full.fit(X_full[~is_val], y[~is_val])
    base_p = clf_full.predict_proba(X_full[is_val])[:, 1]
    base_auc = roc_auc_score(y[is_val], base_p)
    n_base = X_full.shape[1] - len(islands)
    print(f"\n  도서 지점별 순열 중요도(AUC 하락폭, 검증셋)")
    rng2 = np.random.RandomState(SEED)
    drops = []
    for k, isl in enumerate(islands):
        Xp = X_full[is_val].copy()
        rng2.shuffle(Xp[:, n_base + k])
        p2 = clf_full.predict_proba(Xp)[:, 1]
        drops.append((ISLAND_NAMES[isl], base_auc - roc_auc_score(y[is_val], p2)))
    for name, d in sorted(drops, key=lambda x: -x[1]):
        print(f"    {name:<10} {d:+.5f}")

    return dict(delta_auc_island=d_island, gate=DELTA_AUC_GATE, passed=d_island > DELTA_AUC_GATE,
                results=results)


def main():
    timestamps, dates, hour_idx, stns, fields = load_matrices()
    islands, island_precip = load_island_matrix(timestamps)
    part_ablation_with_island(stns, fields, islands, island_precip, hour_idx, dates)


if __name__ == "__main__":
    main()
