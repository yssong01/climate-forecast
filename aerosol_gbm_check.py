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
from aerosol_feature_probe import (AERO_PATH, LAGS, I_DUST, I_PM10, _ts_add)
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
    from baseline_suite import honest_f1
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
        f1, thr = honest_f1(s, yva, hv)
        return tag, f1, thr, len(ytr), int(yva.sum())

    rows = []
    # ref — 배포 구성(전체 표본)
    rows.append(run("ref  28특징·전체", x_tr[m_tr], y_tr[m_tr],
                    x_va[m_va], y_va[m_va], half[m_va]))
    # A — 표본만 줄인다
    tr_a = m_tr & a_tr_m
    va_a = m_va & a_va_m
    rows.append(run("A    28특징·축소", x_tr[tr_a], y_tr[tr_a],
                    x_va[va_a], y_va[va_a], half[va_a]))
    # B — 같은 축소 표본에 에어로졸을 붙인다
    idx_tr = np.cumsum(a_tr_m) - 1          # 전체 행 → 에어로졸 행렬의 행
    idx_va = np.cumsum(a_va_m) - 1
    xb_tr = np.concatenate([x_tr[tr_a], a_tr[idx_tr[tr_a]]], axis=1)
    xb_va = np.concatenate([x_va[va_a], a_va[idx_va[va_a]]], axis=1)
    rows.append(run("B    28+에어로졸·축소", xb_tr, y_tr[tr_a],
                    xb_va, y_va[va_a], half[va_a]))

    print(f"\n  {'구성':<24}{'F1':>9}{'판정선':>9}{'학습표본':>11}{'평가양성':>10}")
    for tag, f1, thr, ntr, npos in rows:
        print(f"  {tag:<24}{f1:9.4f}{thr:9.3f}{ntr:11,}{npos:10,}")

    ref, a, b = rows[0][1], rows[1][1], rows[2][1]
    print(f"\n  표본 감소의 비용 (A − ref) : {a - ref:+.4f}")
    print(f"  에어로졸의 이득  (B − A)   : {b - a:+.4f}")
    print(f"  최종 (B − ref)             : {b - ref:+.4f}  "
          + ("← 채택 후보" if b - ref > 0.01 else "← 기각 (배포 대비 이득 없음)"))
    print("\n  주의: 이 값은 **상한**이다. 대기질 API 에 리드타임 보존")
    print("  아카이브가 없어 학습이 서빙보다 유리하다 — 작으면 그대로 기각한다.")


if __name__ == "__main__":
    main()
