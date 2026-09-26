"""
extreme_gbm.py — 극한기상 3종(폭염·한파·황사) 전용 경사부스팅(2026-09-26).

## 왜 신경망이 아닌가 — 두 가지가 동시에 해결된다

**① 성능.** `baseline_suite.py` 기준선 측정에서 같은 Z축 28특징만 받는 표형
GBM 이 세 헤드 모두에서 신경망을 앞섰다. 강수에서 확인한 것과 같은 방향이며
(표형 데이터에서 부스팅 트리의 우위), 한파에서 특히 크다.

    폭염 F1 0.7987 → 0.8342 · 한파 0.4607 → **0.6080** · 황사 0.1410 → 0.1701

한파 +0.147 은 이 저장소가 그 헤드에 쏟은 모든 시도(중첩 전이학습·헤드
디커플링·커리큘럼·GroupDRO)를 합친 것보다 크다.

**② 단조성 — 구조적 보장.** 이 저장소의 가장 오래된 미해결 문제다. "기온이
오르면 한파 확률이 내려가야 한다"는 자명한 성질이 신경망에서는 seed 에 따라
−0.70 에서 +0.73 까지 갈렸고, 9개 실행 중 2번만 통과했다. 세 번의 처방
(헤드 디커플링·계절 중립화·magnitude 기온 중립화)이 모두 빗나갔고, 남은
후보로 "단조 제약 아키텍처"를 적어 두었으나 신경망에서 만들기에는 비용이
컸다.

**`scikit-learn` 의 `monotonic_cst` 가 그것을 정의상 강제한다.** 관측기온
(0번)과 동시각 편향(`예보기온 − 실측기온`)에 부호 제약을 걸면 학습 결과·seed
와 무관하게 단조성이 성립한다. 실측: 12개 기준 × 11개 기온 격자에서
**위반 0/120**, 한파 최악 상관 −0.946.

**제약의 비용은 거의 없다** — 무제약 대비 폭염 −0.005 · 한파 −0.003 ·
황사 0.000.

## 제약을 거는 열

`train.extreme_temp_neutral_index()` 가 돌려주는 열과 같다 — **관측 기온이
실제로 들어오는 모든 열**이다. Z축 0번만 막으면 수치예보 편향 열로 새어
들어가 제약이 반쪽이 된다.

    폭염: 0번(관측기온) +1 · 25번(예보−실측) −1
    한파: 0번 −1 · 25번 +1        (25번은 값이 클수록 실측이 차갑다)
    황사: 제약 없음 — 기온과의 물리적 방향성이 자명하지 않다

실행:
    python extreme_gbm.py [--checkpoint <주 체크포인트>] [--out <npz>]
"""
import argparse
import os
import sys

import numpy as np

EXPORT_PATH = "./checkpoints/extreme_gbm.npz"
HEADS = ("heatwave", "coldwave", "dust")
SHORT = {"heatwave": "heat", "coldwave": "cold", "dust": "dust"}
# (관측기온 열의 부호, 동시각 편향 열의 부호). 0 이면 제약 없음.
MONO_SIGN = {"heatwave": (+1, -1), "coldwave": (-1, +1), "dust": (0, 0)}
GRID = [
    {"max_iter": 200, "learning_rate": 0.10},
    {"max_iter": 400, "learning_rate": 0.05},
]
SEED = 42


def load(path: str = EXPORT_PATH):
    """{헤드: NumpyGBM} 와 메타. 파일이 없으면 `(None, None)`."""
    if not os.path.exists(path):
        return None, None
    from precip_gbm import NumpyGBM
    d = np.load(path, allow_pickle=False)
    models = {h: NumpyGBM(d, h) for h in HEADS if f"{h}_baseline" in d.files}
    meta = {k: d[k] for k in d.files
            if k.startswith("meta_") or "_cal_" in k}
    return models, meta


def calibrated(models, meta, head, x):
    """보정까지 적용한 확률(스칼라). 서빙이 쓰는 경로다.

    보정 방법은 헤드마다 다르다 — 등온·베타·안 함 중 **평가용에서 더 나은
    쪽**을 실측으로 골라 모델 파일에 적어 두었다(§12 규약, MCE 타이브레이커).
    셋 다 단조 증가 변환이거나 항등이므로 단조성 보장을 깨지 않는다.
    """
    if head not in models:
        return None
    p = float(models[head].predict_proba1(x)[0])
    method = str(meta.get(f"meta_{head}_cal_method", "none"))
    import probability_calibration_fit as _pc
    if method == "beta":
        a, b, c = [float(v) for v in meta[f"{head}_cal_beta"]]
        return float(_pc.apply_beta(np.array([p]), a, b, c)[0])
    if method == "isotonic":
        return float(_pc.apply_calibration(
            np.array([p]), meta[f"{head}_cal_xs"], meta[f"{head}_cal_ys"])[0])
    return p


def calibrated_batch(models, meta, head, x):
    """`calibrated()` 의 배치판 — 진단 스크립트가 검증셋 전체에 쓴다."""
    p = models[head].predict_proba1(x)
    method = str(meta.get(f"meta_{head}_cal_method", "none"))
    import probability_calibration_fit as _pc
    if method == "beta":
        a_, b_, c_ = [float(v) for v in meta[f"{head}_cal_beta"]]
        return _pc.apply_beta(p, a_, b_, c_)
    if method == "isotonic":
        return _pc.apply_calibration(p, meta[f"{head}_cal_xs"],
                                     meta[f"{head}_cal_ys"])
    return p


def threshold(meta, head):
    """서빙 판정선 — 보정 공간에서 재선정한 값(모델 파일에 저장)."""
    return float(meta[f"meta_{head}_tau_cal"])


def _mono_cst(head, nf, temp_cols):
    """제약 벡터. `temp_cols` 는 관측 기온이 들어오는 열 목록이다."""
    s0, s_bias = MONO_SIGN[head]
    if s0 == 0:
        return None
    cst = np.zeros(nf)
    cst[temp_cols[0]] = s0                 # 관측기온 그 자체
    for c in temp_cols[1:]:                # 수치예보 동시각 편향 열
        cst[c] = s_bias
    return cst


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default=EXPORT_PATH)
    ap.add_argument("--checkpoint", default=None)
    ap.add_argument("--patch-checkpoint", action="store_true",
                    help="화면이 이미 읽는 스키마(extreme_metrics_served·"
                         "prob_calibration.heads)에 이 모델의 값을 적어 넣는다.")
    args = ap.parse_args()

    import eval_cache
    from baseline_suite import honest_f1, max_f1, f1_at
    from precip_gbm import _flatten, NumpyGBM
    from sklearn.ensemble import HistGradientBoostingClassifier as H

    f = (eval_cache.load_features(args.checkpoint) if args.checkpoint
         else eval_cache.load_features())
    x_tr, x_va = f["x_train"], f["x_val"]
    nf = x_va.shape[1]
    half = ((f["tgt_ts_val"].astype(np.int64) // 10 ** 4) % 2 == 0)

    # 제약 열은 train 이 쓰는 것과 **같은 함수**로 구한다 — 열 번호를 여기서
    # 세면 특징 집합이 바뀔 때 조용히 어긋난다.
    import train as _t
    prev = os.environ.get("USE_NWP")
    os.environ["USE_NWP"] = "1" if bool(f["use_nwp"]) else "0"
    import importlib
    importlib.reload(_t)
    _t.NWP_FEATURE_SET = str(f["nwp_feature_set"])
    temp_cols = _t.extreme_temp_neutral_index(nf)
    if prev is None:
        os.environ.pop("USE_NWP", None)
    else:
        os.environ["USE_NWP"] = prev
    print(f"학습 {len(x_tr):,} · 검증 {len(x_va):,} · 특징 {nf} · "
          f"기온 열 {temp_cols}")

    out, summary = {}, []
    for head in HEADS:
        mk = SHORT[head]
        mtr = f[f"{mk}_mask_train"].astype(bool)
        off = f.get(f"{mk}_mask_official_val")
        mva = (off if off is not None else f[f"{mk}_mask_val"]).astype(bool)
        y_tr = f[f"y_{head}_train"][mtr].astype(int)
        y_va = f[f"y_{head}_val"][mva].astype(int)
        hv = half[mva]
        cst = _mono_cst(head, nf, temp_cols)

        best = (-1.0, None)
        for g in GRID:
            m = H(random_state=SEED, monotonic_cst=cst, **g).fit(x_tr[mtr], y_tr)
            s = m.predict_proba(x_va[mva])[:, 1]
            v = max_f1(s[hv], y_va[hv])[0]
            if v > best[0]:
                best = (v, g)
        grid = best[1]
        m = H(random_state=SEED, monotonic_cst=cst, **grid).fit(x_tr[mtr], y_tr)
        _flatten(m, head, out)

        p_np = NumpyGBM(out, head).predict_proba1(x_va[mva])
        delta = float(np.abs(p_np - m.predict_proba(x_va[mva])[:, 1]).max())
        if delta > 1e-9:
            raise RuntimeError(f"{head}: numpy 재구현이 다르다 ({delta:.3e})")

        # 판정선 — 보정용 절반에서 F1 최대를 고르고 평가용에서만 채점(규약 9).
        cand = np.arange(0.02, 0.99, 0.005)
        tau = float(cand[int(np.argmax(
            [f1_at(np.where(p_np >= t, 1.0, 0.0)[hv], y_va[hv], 0.5) for t in cand]))])
        f1_eval = f1_at(np.where(p_np >= tau, 1.0, 0.0)[~hv], y_va[~hv], 0.5)
        # 신뢰도(ECE) — 확률 보정이 필요한지 판단하려면 먼저 재야 한다.
        bins = np.linspace(0, 1, 11)
        ece = 0.0
        for lo, hi in zip(bins[:-1], bins[1:]):
            sel = (p_np >= lo) & (p_np < hi if hi < 1 else p_np <= 1)
            if sel.sum():
                ece += sel.sum() / len(p_np) * abs(p_np[sel].mean() - y_va[sel].mean())

        # 확률 보정 — 화면이 "70%라 했을 때 실제로 70%였나"를 표시하므로
        # 필요하다. 저장소의 선택 규약(§12)을 그대로 쓴다: 등온·베타를 모두
        # 적합해 **평가용에서 더 나은 쪽**을 고르되, ECE 동률이면 MCE 로
        # 가른다("베타가 항상 우월"이라는 문헌 가설을 실측 없이 채택하지 않는다).
        # 보정은 **단조 증가 변환**이라 위의 단조성 보장을 깨지 않는다.
        import probability_calibration_fit as _pc
        xs, ys = _pc.fit_isotonic(p_np[hv], y_va[hv].astype(float))
        p_iso = _pc.apply_calibration(p_np[~hv], xs, ys)
        ba, bb, bc = _pc.fit_beta(p_np[hv], y_va[hv].astype(float))
        p_beta = _pc.apply_beta(p_np[~hv], ba, bb, bc)
        e_iso, e_beta = _pc.ece(p_iso, y_va[~hv]), _pc.ece(p_beta, y_va[~hv])
        m_iso, m_beta = _pc.mce(p_iso, y_va[~hv]), _pc.mce(p_beta, y_va[~hv])
        # **후보에 "보정 안 함"을 포함한다.** GBM 의 원본 확률은 이미 꽤
        # 보정돼 있어(한파 ECE 0.0057) 곡선을 씌우면 오히려 나빠질 수 있다 —
        # 실제로 첫 실행에서 한파가 0.0057→0.0083 으로 악화됐다. 이 저장소의
        # 원칙은 "이득이 있을 때만 채택"이다.
        e_raw, m_raw = _pc.ece(p_np[~hv], y_va[~hv]), _pc.mce(p_np[~hv], y_va[~hv])
        cands = [("none", e_raw, m_raw), ("isotonic", e_iso, m_iso),
                 ("beta", e_beta, m_beta)]
        method, ece_cal, mce_cal = min(cands, key=lambda c: (round(c[1], 4), c[2]))
        if method == "beta":
            out[f"{head}_cal_beta"] = np.asarray([ba, bb, bc], dtype=np.float64)
            p_cal_all = _pc.apply_beta(p_np, ba, bb, bc)
        elif method == "isotonic":
            out[f"{head}_cal_xs"] = np.asarray(xs, dtype=np.float64)
            out[f"{head}_cal_ys"] = np.asarray(ys, dtype=np.float64)
            p_cal_all = _pc.apply_calibration(p_np, xs, ys)
        else:
            p_cal_all = p_np
        # 판정선은 **판정이 실제로 이뤄지는 공간**에서 다시 고른다(§12) —
        # 원본 임계값을 보정 곡선에 통과시키는 것만으로는 판정이 보존되지 않는다.
        tau_cal = float(cand[int(np.argmax(
            [f1_at(np.where(p_cal_all >= t, 1.0, 0.0)[hv], y_va[hv], 0.5)
             for t in cand]))])
        f1_cal = f1_at(np.where(p_cal_all >= tau_cal, 1.0, 0.0)[~hv], y_va[~hv], 0.5)
        # 화면이 쓰는 지표도 **이 모델 파일에서** 나오게 한다 — 체크포인트에
        # 적어 넣으면 다른 모델의 값이 섞이고, 사람이 옮겨 적는 고리가 생긴다.
        _pred = (p_cal_all >= tau_cal)[~hv]
        _t = y_va[~hv] > 0
        _tp = float((_pred & _t).sum()); _fp = float((_pred & ~_t).sum())
        _fn = float((~_pred & _t).sum())
        out[f"meta_{head}_precision"] = np.asarray(_tp / max(_tp + _fp, 1e-9))
        out[f"meta_{head}_recall"] = np.asarray(_tp / max(_tp + _fn, 1e-9))
        out[f"meta_{head}_n_pos"] = np.asarray(int(_t.sum()))
        out[f"meta_{head}_n"] = np.asarray(int(len(_t)))
        out[f"meta_{head}_cal_method"] = np.asarray(method)
        out[f"meta_{head}_tau_cal"] = np.asarray(tau_cal)
        out[f"meta_{head}_f1_cal"] = np.asarray(f1_cal)
        out[f"meta_{head}_ece_cal"] = np.asarray(float(ece_cal))
        out[f"meta_{head}_mce_cal"] = np.asarray(float(mce_cal))
        out[f"meta_{head}_tau"] = np.asarray(tau)
        out[f"meta_{head}_f1"] = np.asarray(f1_eval)
        # **평가용 절반 기준**으로 적는다 — 보정 후 값과 같은 표본이어야
        # 한다. 전체 표본으로 적었더니 한파가 0.0057→0.0083 으로 보여
        # "보정이 나쁘게 만들었다"로 읽혔다(실제 평가용 원본은 0.0089).
        out[f"meta_{head}_ece"] = np.asarray(float(e_raw))
        out[f"meta_{head}_ece_fullset"] = np.asarray(float(ece))
        out[f"meta_{head}_monotonic"] = np.asarray(cst is not None)
        out[f"meta_{head}_grid"] = np.asarray(
            f"{grid['max_iter']}/{grid['learning_rate']}")
        summary.append(
            f"  {head:<9} 후보 ECE — 안함 {e_raw:.4f} · 등온 {e_iso:.4f} · "
            f"베타 {e_beta:.4f} → 선택 {method}\n"
            f"            τ={tau_cal:.3f} · F1 {f1_cal:.4f} · ECE {ece_cal:.4f} · "
            f"MCE {mce_cal:.4f} · 단조제약 {'O' if cst is not None else 'X'}")

    print("\n" + "\n".join(summary))

    out["meta_num_features"] = np.asarray(nf)
    out["meta_temp_cols"] = np.asarray(temp_cols)
    for k in ("lead_hours", "split_mode", "split_algo", "ckpt_path"):
        if k in f:
            out[f"meta_{k}"] = f[k]

    os.makedirs(os.path.dirname(args.out), exist_ok=True)
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(args.out), suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **out)
    os.replace(tmp, args.out)
    print(f"\n저장: {args.out} ({os.path.getsize(args.out)/1e6:.1f}MB)")

    if args.patch_checkpoint:
        # **화면 코드를 고치지 않는다.** 화면은 이미 `extreme_metrics_served`
        # 와 `prob_calibration.heads[*]` 를 읽으므로, 그 스키마에 이 모델의
        # 값을 그대로 적으면 표·캡션·판정선이 자동으로 따라온다. 출처는
        # `extreme_source` 로 남긴다(강수의 `precip_source` 와 같은 취지) —
        # 표시가 없으면 어느 모델의 수치인지 알 수 없다.
        import torch
        ck_path = args.checkpoint or "./checkpoints/numerical_trichef.pt"
        ck = torch.load(ck_path, map_location="cpu", weights_only=True)
        served, cal = {}, (ck.get("prob_calibration") or {"heads": {}})
        cal.setdefault("heads", {})
        for head in HEADS:
            m = lambda k: out[f"meta_{head}_{k}"]
            served[head] = {
                "threshold": float(m("tau_cal")), "precision": float(m("precision")),
                "recall": float(m("recall")), "f1": float(m("f1_cal")),
                "n_pos": int(m("n_pos")), "n": int(m("n")),
            }
            h = cal["heads"].setdefault(head, {})
            h["method"] = str(m("cal_method"))
            h["threshold_decision"] = float(m("tau_cal"))
            h["threshold_raw"] = float(m("tau"))
            h["threshold_repicked"] = True
            h["eval_metrics"] = {
                "n_eval": int(m("n")),
                "ece_before": float(m("ece")), "ece_after": float(m("ece_cal")),
                "mce_after": float(m("mce_cal")),
                "f1_raw": float(m("f1")), "f1_calibrated": float(m("f1")),
                "f1_decision": float(m("f1_cal")),
            }
            # 곡선 자체는 GBM 파일에 있고 서빙이 그쪽을 읽는다. 체크포인트의
            # x/y 는 신경망용이라 지운다 — 남겨 두면 어느 곡선이 쓰이는지
            # 헷갈린다.
            h.pop("x", None); h.pop("y", None)
        ck["extreme_metrics_served"] = served
        ck["extreme_metrics"] = served
        ck["prob_calibration"] = cal
        ck["extreme_source"] = args.out
        torch.save(ck, ck_path)
        print(f"체크포인트에 반영: {ck_path} (extreme_source={args.out})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
