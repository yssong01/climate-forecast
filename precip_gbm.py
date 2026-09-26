"""
precip_gbm.py — 강수 전용 경사부스팅 모델의 학습·내보내기·추론(2026-09-26).

## 왜 신경망이 아닌가

`baseline_suite.py` 로 기준선을 제대로 세우자, **같은 Z축 28특징만 받는 표형
GBM 이 강수 주 지표 둘 모두에서 배포 신경망을 앞섰다**(발생 F1 0.6406 vs
0.5933 · 강수 구간 MAE 2.0471 vs 2.1014). 원인을 갈라 보니 우리가 할 수 있는
구조 변경으로는 닫히지 않는다 — 과제 분리(+0.003)와 Re·Im 제거(+0.006)를 둘
다 해도 **격차의 15%만** 닫히고 나머지 85% 는 모델 계열의 차이다. 강수는
표본의 93.4% 가 0 인 영과잉 분포라 트리의 임계값 분할이 유리한 것으로
해석한다(해석이며 검증하지 않았다).

기온을 전용 모델로 분리해 −27% 를 얻은 것과 같은 패턴이며, 다른 점은 그
모델이 신경망이 아니라는 것뿐이다.

## 왜 scikit-learn 을 서빙에 넣지 않는가

두 가지 이유다.

① **pickle 은 버전이 바뀌면 조용히 깨진다.** 배포 환경의 scikit-learn 이
   갱신되면 언피클이 실패하거나 — 더 나쁘게는 — 경고만 내고 다른 값을 낼 수
   있다. 이 저장소는 "조용히 어긋나는 경로"를 반복해 겪었다.
② 서빙 의존성을 최소로 유지해 왔다(450MB 짜리 `sentence-transformers` 를
   뺀 전례). `scikit-learn`+`scipy` 는 55MB 안팎이다.

그래서 **트리 구조를 npz 로 내보내고 순수 numpy 로 추론한다.** 학습에만
scikit-learn 이 필요하고(GPU 컨테이너에 이미 있다), 서빙은 numpy 만 쓴다.
`verify()` 가 두 경로의 출력이 **정확히 같은지** 대조하며, 내보내기 직후
자동으로 돈다 — 재구현이 원본과 다르면 그 자리에서 멈춘다.

## 하이퍼파라미터 선택

검증셋을 목표일 홀짝으로 갈라 **보정용 절반에서만** 고르고 평가용 절반에서
채점한다(§2 규약). 눈으로 보고 고르면 평가셋에 대한 선택 편향이 섞인다 —
실제로 처음 다섯 설정을 훑을 때 그 함정에 들어갈 뻔했다.

실행:
    python precip_gbm.py                 # 학습·선택·내보내기·검증
    python precip_gbm.py --report        # 내보낸 모델을 검증셋에서 채점만
"""
import argparse
import os
import sys

import numpy as np

EXPORT_PATH = "./checkpoints/precip_gbm.npz"

# 후보 격자 — 보정용 절반에서 고른다. 넓게 잡지 않는 이유는 실측에서 이
# 범위 전체가 신경망(0.5933)을 크게 넘었고 서로 간 차이는 0.013 에 그쳤기
# 때문이다(발생 F1 0.628~0.641). 성능이 이 손잡이에 민감하지 않다.
GRID = [
    {"max_iter": 200, "learning_rate": 0.10},
    {"max_iter": 400, "learning_rate": 0.05},
    {"max_iter": 600, "learning_rate": 0.05},
]
SEED = 42


# ── 순수 numpy 추론기 ────────────────────────────────────────────

class NumpyGBM:
    """내보낸 트리 배열로 예측한다 — scikit-learn 없이 동작한다.

    `HistGradientBoosting*` 의 `_predictors` 는 트리마다 구조화 배열
    `nodes` 를 갖는다. 추론은 노드 0 에서 시작해 잎에 닿을 때까지
    `x[feature] <= threshold` 로 좌우를 고르는 것이 전부다. 결측은
    `missing_go_to_left` 를 따른다(우리 입력에는 결측이 없지만 규칙을
    그대로 옮긴다).
    """

    def __init__(self, d, prefix):
        self.baseline = float(d[f"{prefix}_baseline"])
        self.offset = d[f"{prefix}_offset"]          # 트리별 시작 인덱스
        self.feature = d[f"{prefix}_feature"]
        self.threshold = d[f"{prefix}_threshold"]
        self.left = d[f"{prefix}_left"]
        self.right = d[f"{prefix}_right"]
        self.is_leaf = d[f"{prefix}_is_leaf"]
        self.value = d[f"{prefix}_value"]
        self.missing_left = d[f"{prefix}_missing_left"]

    def raw(self, x: np.ndarray) -> np.ndarray:
        x = np.asarray(x, dtype=np.float64)
        if x.ndim == 1:
            x = x[None, :]
        out = np.full(len(x), self.baseline, dtype=np.float64)
        for t in range(len(self.offset) - 1):
            s, e = int(self.offset[t]), int(self.offset[t + 1])
            feat = self.feature[s:e]
            thr = self.threshold[s:e]
            left, right = self.left[s:e], self.right[s:e]
            leaf, val = self.is_leaf[s:e], self.value[s:e]
            miss_l = self.missing_left[s:e]
            node = np.zeros(len(x), dtype=np.int64)
            active = ~leaf[node]
            while active.any():
                idx = np.flatnonzero(active)
                n = node[idx]
                v = x[idx, feat[n]]
                go_left = np.where(np.isnan(v), miss_l[n].astype(bool), v <= thr[n])
                node[idx] = np.where(go_left, left[n], right[n])
                active = ~leaf[node]
            out += val[node]
        return out

    def predict(self, x):
        """회귀 — 원시 합이 곧 예측값(제곱오차 손실)."""
        return self.raw(x)

    def predict_proba1(self, x):
        """이진분류 — 원시 합의 시그모이드."""
        return 1.0 / (1.0 + np.exp(-self.raw(x)))


def load(path: str = EXPORT_PATH):
    """(양 모델, 발생확률 모델, 메타) — 파일이 없으면 `(None, None, None)`."""
    if not os.path.exists(path):
        return None, None, None
    d = np.load(path, allow_pickle=False)
    meta = {k: d[k] for k in d.files if k.startswith("meta_")}
    return NumpyGBM(d, "amt"), NumpyGBM(d, "occ"), meta


# ── 내보내기 ─────────────────────────────────────────────────────

def _flatten(model, prefix, out):
    """scikit-learn 모델의 트리들을 평평한 배열로 옮긴다."""
    preds = model._predictors
    assert all(len(p) == 1 for p in preds), "이진/단일 출력만 지원한다"
    feats, thrs, lefts, rights, leafs, vals, miss = [], [], [], [], [], [], []
    offsets = [0]
    for stage in preds:
        n = stage[0].nodes
        feats.append(n["feature_idx"].astype(np.int32))
        thrs.append(n["num_threshold"].astype(np.float64))
        lefts.append(n["left"].astype(np.int32))
        rights.append(n["right"].astype(np.int32))
        leafs.append(n["is_leaf"].astype(bool))
        vals.append(n["value"].astype(np.float64))
        miss.append(n["missing_go_to_left"].astype(np.int8))
        offsets.append(offsets[-1] + len(n))
    out[f"{prefix}_baseline"] = np.asarray(
        float(np.ravel(model._baseline_prediction)[0]))
    out[f"{prefix}_offset"] = np.asarray(offsets, dtype=np.int64)
    out[f"{prefix}_feature"] = np.concatenate(feats)
    out[f"{prefix}_threshold"] = np.concatenate(thrs)
    out[f"{prefix}_left"] = np.concatenate(lefts)
    out[f"{prefix}_right"] = np.concatenate(rights)
    out[f"{prefix}_is_leaf"] = np.concatenate(leafs)
    out[f"{prefix}_value"] = np.concatenate(vals)
    out[f"{prefix}_missing_left"] = np.concatenate(miss)


def verify(amt_sk, occ_sk, x, tol=1e-9):
    """numpy 재구현이 scikit-learn 과 같은 값을 내는지 대조한다.

    **내보내기 직후 반드시 돈다.** 재구현이 원본과 다르면 서빙이 조용히 다른
    값을 내게 되는데, 지표는 멀쩡해 보인다 — 이 저장소가 반복해 겪은 유형이다.
    """
    d = {}
    _flatten(amt_sk, "amt", d)
    _flatten(occ_sk, "occ", d)
    amt_np, occ_np = NumpyGBM(d, "amt"), NumpyGBM(d, "occ")
    da = float(np.abs(amt_np.predict(x) - amt_sk.predict(x)).max())
    do = float(np.abs(occ_np.predict_proba1(x) - occ_sk.predict_proba(x)[:, 1]).max())
    if da > tol or do > tol:
        raise RuntimeError(
            f"numpy 재구현이 scikit-learn 과 다르다 — 양 {da:.3e} · 확률 {do:.3e}")
    return d, da, do


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", action="store_true",
                    help="내보낸 모델을 검증셋에서 채점만 한다")
    ap.add_argument("--out", default=EXPORT_PATH)
    ap.add_argument("--checkpoint", default=None,
                    help="특징을 만들 기준 체크포인트(리드타임을 결정한다). "
                         "생략하면 predict.CHECKPOINT(+6h).")
    args = ap.parse_args()

    import eval_cache
    from baseline_suite import honest_f1, conditional_mae, mae, max_f1

    f = (eval_cache.load_features(args.checkpoint) if args.checkpoint
         else eval_cache.load_features())
    x_tr, x_va = f["x_train"], f["x_val"]
    p_tr, p_va = f["precip_true_train"], f["precip_true_val"]
    wet = float(f["wet_thresh"])
    y_tr, y_va = (p_tr >= wet).astype(int), (p_va >= wet).astype(int)
    # 목표일 홀짝으로 보정용/평가용을 가른다 — 표본을 무작위로 쪼개면 같은 날
    # 다른 시각이 양쪽에 들어가 선택이 채점 표본을 엿본다.
    half = ((f["tgt_ts_val"].astype(np.int64) // 10 ** 4) % 2 == 0)

    if args.report:
        amt, occ, meta = load(args.out)
        if amt is None:
            raise SystemExit(f"내보낸 모델이 없다: {args.out}")
        a, o = amt.predict(x_va), occ.predict_proba1(x_va)
        print(f"발생 F1(절반분리) {honest_f1(o, y_va, half):.4f} · "
              f"구간 MAE {conditional_mae(a, p_va, y_va):.4f} · "
              f"원본 MAE {mae(np.clip(a, 0, None), p_va):.4f}")
        return 0

    from sklearn.ensemble import (HistGradientBoostingRegressor,
                                  HistGradientBoostingClassifier)
    print(f"학습 {len(x_tr):,} · 검증 {len(x_va):,} · 특징 {x_va.shape[1]}")

    # ── 하이퍼파라미터는 보정용 절반에서만 고른다 ──────────────
    best = (-1.0, None)
    for g in GRID:
        occ = HistGradientBoostingClassifier(random_state=SEED, **g).fit(x_tr, y_tr)
        s = occ.predict_proba(x_va)[:, 1]
        f1_calib = max_f1(s[half], y_va[half])[0]
        print(f"  후보 {g} → 보정용 최대 F1 {f1_calib:.4f}")
        if f1_calib > best[0]:
            best = (f1_calib, g)
    grid = best[1]
    print(f"선택: {grid} (보정용 F1 {best[0]:.4f})")

    occ = HistGradientBoostingClassifier(random_state=SEED, **grid).fit(x_tr, y_tr)
    amt = HistGradientBoostingRegressor(random_state=SEED, **grid).fit(x_tr, p_tr)

    d, da, do = verify(amt, occ, x_va[:20000])
    print(f"numpy 재구현 대조: 양 {da:.2e} · 확률 {do:.2e} (통과)")

    a_np = np.clip(NumpyGBM(d, "amt").predict(x_va), 0, None)
    o_np = NumpyGBM(d, "occ").predict_proba1(x_va)
    f1 = honest_f1(o_np, y_va, half)
    cmae = conditional_mae(a_np, p_va, y_va)

    # 서빙 판정선 — 보정용 절반에서 F1 최대 τ 를 고르고 평가용에서 채점한다.
    from baseline_suite import f1_at
    cand = np.arange(0.05, 0.96, 0.005)
    f1_calib = [f1_at(np.where(o_np >= t, 1.0, 0.0)[half], y_va[half], 0.5)
                for t in cand]
    tau = float(cand[int(np.argmax(f1_calib))])
    served = np.where(o_np >= tau, a_np, 0.0)
    served_mae = mae(served[~half], p_va[~half])
    base_mae = mae(np.zeros_like(p_va)[~half], p_va[~half])
    print(f"\n서빙 판정선 τ={tau:.3f} (보정용 F1 {max(f1_calib):.4f})")
    print(f"평가용 절반 — 발생 F1 {f1:.4f} · 강수 구간 MAE {cmae:.4f} · "
          f"서빙 MAE {served_mae:.4f}(기준선 {base_mae:.4f}) · "
          f"원본 MAE {mae(a_np, p_va):.4f}")

    d["meta_max_iter"] = np.asarray(grid["max_iter"])
    d["meta_learning_rate"] = np.asarray(grid["learning_rate"])
    d["meta_seed"] = np.asarray(SEED)
    d["meta_num_features"] = np.asarray(x_va.shape[1])
    d["meta_wet_thresh"] = np.asarray(wet)
    d["meta_val_precip_wet_f1"] = np.asarray(f1)
    d["meta_val_precip_mae_wet"] = np.asarray(cmae)
    # 서빙 판정선을 **모델 파일에 함께** 적는다(규약 9 — 헤드를 다시
    # 조정하면 판정선도 그 시점 표본에서 다시 고른다). 상수로 박아 두면
    # 모델을 바꿀 때마다 사람이 옮겨 적어야 하고, 그 고리가 이 저장소가
    # 반복해 겪은 드리프트의 입구다.
    d["meta_gate_tau"] = np.asarray(tau)
    d["meta_val_precip_mae_served"] = np.asarray(served_mae)
    # 같은 분할에서 나온 모델인지 대조할 수 있도록 서명을 함께 적는다.
    for k in ("ckpt_path", "split_mode", "split_algo", "lead_hours",
              "data_mtime", "data_size"):
        if k in f:
            d[f"meta_{k}"] = f[k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(args.out), suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **d)
    os.replace(tmp, args.out)
    print(f"저장: {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
