"""
aerosol_gbm_check.py — 외부 에어로졸 특징이 **배포 중인 황사 GBM** 을
실제로 개선하는지 잰다(2026-09-27).

## 왜 다시 재는가

`aerosol_feature_probe.py` 가 2026-09-07 에 상한을 쟀고 결과는 "이득은
있으나(+6h 관측소값 ΔAUC +0.0125) 표본이 1/3 로 줄어 채택 보류"였다.
그 판단을 지금 다시 물을 이유가 둘 있다.

① **황사를 내는 모델이 바뀌었다.** 그때는 다중과제 신경망의 헤드였고
   지금은 전용 GBM 이다(`extreme_gbm.py`). 표본 감소의 비용은 모델 계열에
   따라 다르다 — 신경망은 공유 트렁크가 다른 헤드에서 표현을 얻지만,
   전용 GBM 은 그 표본만으로 서므로 감소가 더 아플 수도, 트리라서 덜
   아플 수도 있다. **추측하지 말고 잰다.**
② **프로브의 특징이 배포의 특징이 아니다.** 프로브는 자체 지상 관측
   특징군(`local`·`neighbor`)을 썼는데 배포 모델은 Z축 28특징(수치예보
   포함)을 받는다. CLAUDE.md 가 경고한 그대로다 — 단순 프로브의 절제
   결과가 전체 파이프라인에서 재현된다는 보장이 없다.

## 설계 — 비용과 이득을 같은 표본에서 가른다

세 팔을 돌려 **두 질문을 분리**한다.

    ref  : 28특징 · 전체 표본      → 배포 모델(비교 기준)
    A    : 28특징 · 축소 표본      → 표본 감소만의 비용
    B    : 28특징+에어로졸 · 축소  → 감소를 치르고 얻는 것

`B − A` 가 특징의 순수 이득이고, `A − ref` 가 표본 감소의 비용이다.
**채택 조건은 `B > ref`** — 이득이 비용을 넘어야 한다. 둘을 한 숫자로
합쳐 보면 "이득은 있는데 왜 안 좋아지는가"를 설명할 수 없다.

채점은 배포와 같은 규약이다 — 판정선을 검증셋의 **보정용 절반**에서
고르고 **평가용 절반**에서만 채점한다(`baseline_suite.honest_f1`).
`t=0.5` 고정 비교는 쓰지 않는다(CLAUDE.md 5절, 같은 함정에 두 번 빠졌다).

## 상한이라는 점은 그대로다

대기질 API 에는 리드타임을 보존하는 아카이브가 없어, 학습이 쓰는 T 시점
값은 분석장이고 실서빙의 T 시점 값은 그보다 앞서 발표된 예보다. 따라서
**여기서 나온 이득은 상한이며, 작으면 그대로 기각할 수 있다.**

실행: python aerosol_gbm_check.py        (LEAD_HOURS 환경변수로 6/12)
"""
import json
import os

import numpy as np

import eval_cache
from aerosol_feature_probe import AERO_PATH, LAGS, I_DUST, I_PM10, _ts_add
from collect_aerosol_archive import UPSTREAM

SEED = 42
LEAD_HOURS = int(os.getenv("LEAD_HOURS", "6"))
CKPT = ("./checkpoints/numerical_trichef.pt" if LEAD_HOURS == 6
        else "./checkpoints/numerical_trichef_12h.pt")


def aerosol_table():
    """`(관측소, 시각)` → 에어로졸 특징 벡터. 상류 블록은 시각당 공통이다."""
    with open(AERO_PATH, "r", encoding="utf-8") as f:
        raw = json.load(f)
    aero = {k: {ts: np.asarray(v, dtype=np.float32) for ts, v in rows.items()}
            for k, rows in raw.items()}
    up_names = list(UPSTREAM)

    def _up(ts):
        """상류 다섯 지점의 현재값 + 시차 변화량. 하나라도 없으면 None."""
        out = []
        for n in up_names:
            rows = aero.get(n, {})
            cur = rows.get(ts)
            if cur is None:
                return None
            out += list(cur)
            for L in LAGS:
                prev = rows.get(_ts_add(ts, -L))
                if prev is None:
                    return None
                out.append(float(cur[I_DUST] - prev[I_DUST]))
        return out

    def _local(stn, ts):
        rows = aero.get(str(stn), {})
        cur = rows.get(ts)
        if cur is None:
            return None
        out = list(cur)
        for L in LAGS:
            prev = rows.get(_ts_add(ts, -L))
            if prev is None:
                return None
            out += [float(cur[I_DUST] - prev[I_DUST]),
                    float(cur[I_PM10] - prev[I_PM10])]
        return out

    return aero, _up, _local


def attach(stns, ts_ints, _up, _local, up_cache):
    """(행 선택 마스크, 에어로졸 행렬). 자료가 없는 행은 제외한다."""
    keep, rows = [], []
    for s, t in zip(stns, ts_ints):
        ts = str(int(t)).zfill(12)
        if ts not in up_cache:
            up_cache[ts] = _up(ts)
        u = up_cache[ts]
        l = _local(s, ts) if u is not None else None
        if u is None or l is None:
            keep.append(False)
            continue
        keep.append(True)
        rows.append(l + u)
    m = np.asarray(keep, dtype=bool)
    return m, (np.asarray(rows, dtype=np.float32) if rows
               else np.zeros((0, 0), dtype=np.float32))


def main():
    from sklearn.ensemble import HistGradientBoostingClassifier as H
    from baseline_suite import honest_f1, max_f1
    import extreme_gbm as EG

    print(f"\n{'=' * 84}\n 황사 — 에어로졸 특징이 배포 GBM 을 개선하는가"
          f" (리드타임 +{LEAD_HOURS}h)\n{'=' * 84}")

    f = eval_cache.load_features(CKPT)
    x_tr, x_va = f["x_train"], f["x_val"]
    y_tr = f["y_dust_train"].astype(int)
    y_va = f["y_dust_val"].astype(int)
    m_tr = f["dust_mask_train"].astype(bool)
    m_va = f["dust_mask_val"].astype(bool)
    # 평가용/보정용 절반 — 배포와 같은 목표일 홀짝 분리.
    half = ((f["tgt_ts_val"].astype(np.int64) // 10 ** 4) % 2 == 0)

    _, _up, _local = aerosol_table()
    up_cache = {}
    a_tr_m, a_tr = attach(f["stn_train"], f["src_ts_train"], _up, _local, up_cache)
    a_va_m, a_va = attach(f["stn_val"], f["src_ts_val"], _up, _local, up_cache)
    print(f"  에어로졸 결합 — 학습 {a_tr_m.sum():,}/{len(a_tr_m):,}"
          f"({a_tr_m.mean():.1%}) · 검증 {a_va_m.sum():,}/{len(a_va_m):,}"
          f"({a_va_m.mean():.1%}) · 에어로졸 차원 {a_tr.shape[1]}")

    # 황사는 단조 제약이 없다(MONO_SIGN 의 부호가 0) — 배포와 같게 둔다.
    assert EG.MONO_SIGN["dust"][0] == 0, "황사에 단조 제약이 생겼다면 여기도 고칠 것"
    grid = dict(max_iter=400, learning_rate=0.05, max_depth=6,
                min_samples_leaf=40, l2_regularization=1.0)

    def run(tag, xtr, ytr, xva, yva, hv):
        m = H(random_state=SEED, **grid).fit(xtr, ytr)
        s = m.predict_proba(xva)[:, 1]
        # `honest_f1` 은 F1 만 돌려준다 — 판정선은 보정용 절반에서 따로 얻는다.
        thr = max_f1(s[hv], yva[hv])[1]
        return tag, honest_f1(s, yva, hv), thr, len(ytr), int(yva.sum())

    tr_a = m_tr & a_tr_m
    va_a = m_va & a_va_m
    idx_tr = np.cumsum(a_tr_m) - 1          # 전체 행 → 에어로졸 행렬의 행
    idx_va = np.cumsum(a_va_m) - 1
    n_local = 5 + 2 * len(LAGS)             # 관측소 에어로졸 블록의 너비

    rows = []
    # ref — 배포 구성. **평가 표본이 다른 값을 나란히 놓지 않는다**: 같은
    # 모델을 전체 검증셋과 축소 검증셋에서 각각 채점해 둘 다 보고한다.
    # 이걸 빼면 "표본 감소의 비용"에 **채점 표본이 달라진 효과**가 섞여
    # 들어가는데, 그건 모델의 성질이 아니라 문제의 난이도 차이다.
    rows.append(run("ref  28특징·전체학습/전체채점", x_tr[m_tr], y_tr[m_tr],
                    x_va[m_va], y_va[m_va], half[m_va]))
    rows.append(run("ref' 28특징·전체학습/축소채점", x_tr[m_tr], y_tr[m_tr],
                    x_va[va_a], y_va[va_a], half[va_a]))
    rows.append(run("A    28특징·축소학습/축소채점", x_tr[tr_a], y_tr[tr_a],
                    x_va[va_a], y_va[va_a], half[va_a]))

    # B 계열 — 에어로졸 블록을 관측소/상류로 갈라서도 본다. 이득이 상류에서
    # 오면 "지금 상류에 떠 있는 것"이라는 설계 가설이 지지되고, 관측소값에서
    # 오면 **진행 중인 사건의 지속성**을 다시 읽은 것에 가깝다(황사는 며칠씩
    # 이어지므로 그 값은 서빙에서 예보로 대체되는 순간 크게 약해진다).
    def _cat(sel, idx, block):
        return np.concatenate([x_va[sel] if block is a_va else x_tr[sel],
                               block[idx[sel]]], axis=1)
    for tag, sl in (("B    +에어로졸(전부)", slice(None)),
                    ("B1   +관측소 에어로졸만", slice(0, n_local)),
                    ("B2   +상류 에어로졸만", slice(n_local, None))):
        xb_tr = np.concatenate([x_tr[tr_a], a_tr[idx_tr[tr_a]][:, sl]], axis=1)
        xb_va = np.concatenate([x_va[va_a], a_va[idx_va[va_a]][:, sl]], axis=1)
        rows.append(run(tag, xb_tr, y_tr[tr_a], xb_va, y_va[va_a], half[va_a]))

    print(f"\n  {'구성':<30}{'F1':>9}{'판정선':>9}{'학습표본':>11}{'평가양성':>10}")
    for tag, f1, thr, ntr, npos in rows:
        print(f"  {tag:<30}{f1:9.4f}{thr:9.3f}{ntr:11,}{npos:10,}")

    ref, refp, a, b, b1, b2 = [r[1] for r in rows]
    print(f"\n  [채점 표본이 같은 것끼리만 비교한다 — 아래는 전부 '축소 채점']")
    print(f"  학습 표본 감소의 비용 (A − ref')  : {a - refp:+.4f}")
    print(f"  에어로졸 전체의 이득  (B − A)     : {b - a:+.4f}")
    print(f"    · 관측소값만        (B1 − A)    : {b1 - a:+.4f}")
    print(f"    · 상류값만          (B2 − A)    : {b2 - a:+.4f}  ← 설계 가설")
    print(f"  최종 (B − ref')                   : {b - refp:+.4f}  "
          + ("← 채택 후보" if b - refp > 0.01 else "← 기각 (이득 없음)"))
    print(f"\n  참고 — 배포 전체 채점 ref {ref:.4f} 는 **표본이 달라** 위 값들과")
    print(f"  직접 비교할 수 없다(축소 채점은 양성 {rows[1][4]:,}개 · 전체는 {rows[0][4]:,}개).")

    # ── 배포 가능한 형태로 다시 — 결측 허용 단일 모델 ──────────────
    #
    # 위 B 는 "에어로졸이 있는 32% 에서만" 성립한다. 배포는 나머지 68% 에도
    # 값을 내야 하므로, 그 구성을 그대로 채택할 수 없다. `HistGradientBoosting`
    # 은 결측(NaN)을 분기에서 직접 다루므로, **없는 곳은 NaN 으로 두고 한 모델**
    # 로 학습하는 것이 배포 가능한 설계다. 이것을 **전체 검증셋**에서 배포
    # 구성과 나란히 채점한다 — 여기서 이기지 못하면 채택할 수 없다.
    def _fill(x, mask, block, idx):
        out = np.full((len(x), block.shape[1]), np.nan, dtype=np.float32)
        out[mask] = block[idx[mask]]
        return np.concatenate([x, out], axis=1)

    xd_tr = _fill(x_tr, a_tr_m, a_tr, idx_tr)
    xd_va = _fill(x_va, a_va_m, a_va, idx_va)
    tag, f1_d, thr_d, ntr_d, npos_d = run(
        "D    28+에어로졸(결측 NaN)·전체", xd_tr[m_tr], y_tr[m_tr],
        xd_va[m_va], y_va[m_va], half[m_va])

    print(f"\n  [배포 가능한 형태 — 전체 표본, 결측은 NaN]")
    print(f"  {'구성':<30}{'F1':>9}{'판정선':>9}{'학습표본':>11}{'평가양성':>10}")
    print(f"  {'ref  28특징만':<30}{ref:9.4f}{rows[0][2]:9.3f}{rows[0][3]:11,}{rows[0][4]:10,}")
    print(f"  {tag:<30}{f1_d:9.4f}{thr_d:9.3f}{ntr_d:11,}{npos_d:10,}")
    print(f"  차이 (D − ref) : {f1_d - ref:+.4f}  "
          + ("← 채택 후보 — 배포 형태에서도 이긴다" if f1_d - ref > 0.01
             else "← 기각 (배포 형태에서는 이득이 남지 않는다)"))
    print("\n  주의: 이 값은 **상한**이다. 대기질 API 에 리드타임 보존")
    print("  아카이브가 없어 학습이 서빙보다 유리하다 — 작으면 그대로 기각한다.")


if __name__ == "__main__":
    main()
