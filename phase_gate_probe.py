"""
phase_gate_probe.py — Risk_Prediction의 위상각 게이트를 climate-forecast의
Im축(시간 경향)에 이식할 수 있는지 시험한다.

**원 수식(Risk_Prediction `src/hermitian.py phase_angle()`).** 두 축(Re·Im)
각각에 관측(q)과 기대응답(d) 한 쌍씩, 총 4개의 실수 벡터를 쓴다.
  R_qd = ⟨q_re,d_re⟩ + ⟨q_im,d_im⟩,  I_qd = ⟨q_re,d_im⟩ − ⟨q_im,d_re⟩
  θ = atan2(I_qd, R_qd)  (전부 L2 정규화 후 계산)
Risk_Prediction에서 Re축은 강우 시계열 윈도우, Im축은 하천 수위 변화율
윈도우다 — 즉 그 프로젝트의 "Re/Im"도 실은 둘 다 시간축이고, d_re=q_re
(강우는 그대로 신뢰), d_im은 **단위도(unit hydrograph)로 강우를 변환한
"물리적으로 기대되는" 수위 반응**이다. 위상각이 실제로 의미를 가지려면
d_im이 q_re의 단순 스칼라배가 아닌, 물리적으로 독립된 변환(컨볼루션)이어야
한다 — 스칼라배면 L2 정규화 후 d_im과 d_re가 같은 방향이 되어 위상각이
cos(q_im,d_im)의 단조 재매개변수화로 붕괴한다(이 스크립트 개발 중 직접
확인한 함정, 아래 설계에서 회피).

**climate-forecast 이식 — 무엇이 대응되고 무엇이 안 되는가.**
climate-forecast에는 강우→수위 같은 독립적 "물리 기대 모델"(단위도)이
없다. 이 프로브는 그 대체로 **비중첩 과거 시간창**을 쓴다 — 최근
1시간 변화(A)를 몰수 "구동축"(q_re=d_re, Risk_Prediction 결정 1과 동일),
직전 3시간 관측 변화(q_im)와 그 이전 3~6시간 전의 독립적 변화(d_im,
6시간 전→3시간 전)를 "응답축"으로 삼는다. d_im이 q_im·A 어느 쪽의
스칼라배도 아니므로 퇴화하지 않는다. **단, 이건 진짜 물리 모델이
아니라 "추세 지속성" 대체 지표다** — Risk_Prediction의 단위도만큼
물리적 근거가 강하지 않다는 한계를 명시한다.

변수는 tendency_collector.TENDENCY_VARS(기온·기압·습도·풍속, 강수 아님 —
Im축 정의와 일치시킴)를 그대로 쓰고, 같은 _SCALE로 나눠 방향 계산 시
변수 간 크기 차이가 섞이지 않게 한다.

**시험 방법.** eval_cache.npz(배포 체크포인트, 이미 최신)에서 폭염·한파·
황사 헤드의 오탐(FP: 예측 양성·공식 라벨 음성)과 정탐(TP: 예측 양성·
공식 라벨 양성) 표본을 뽑아 위상각 분포를 비교한다. 오탐이 "노이즈성"
영역(Suspect/Reject)에 더 몰리면 이 필터가 실제로 오탐 억제에 쓸모
있다는 신호다.

실행 (GPU 불필요 — numpy만 있으면 됨, torch 서빙 이미지로 충분):
  docker run --rm --entrypoint python3 -v $(pwd):/app -w /app tri-chef-app:latest \
    phase_gate_probe.py [--event heatwave|coldwave|dust] [--n 2000]
"""
import argparse
import json
from datetime import datetime, timedelta

import numpy as np

from tendency_collector import TENDENCY_VARS, _SCALE

DATA_CACHE = "./cache/historical_data_1y.json"
EVAL_CACHE = "./cache/eval_cache.npz"

EVENTS = {
    "heatwave": ("heat_prob", "y_heatwave", "heat_mask"),
    "coldwave": ("cold_prob", "y_coldwave", "cold_mask"),
    "dust":     ("dust_prob", "y_dust", "dust_mask"),
}

THETA_TRUST_DEG = 30.0
THETA_REJECT_DEG = 80.0
N_BOOT = 2000
BOOT_SEED = 20260831


def _parse_ts(ts) -> datetime:
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


def build_index(records):
    idx = {}
    for r in records:
        key = (str(r.get("stn")), str(r["timestamp"])[:12])
        idx[key] = r
    return idx


def _diff_vec(idx, stn, t_from, t_to):
    """t_from → t_to 구간의 [기온,기압,습도,풍속] 변화량(스케일 정규화). 결측이면 None."""
    r_from = idx.get((stn, t_from.strftime("%Y%m%d%H%M")))
    r_to = idx.get((stn, t_to.strftime("%Y%m%d%H%M")))
    if r_from is None or r_to is None:
        return None
    out = []
    for var in TENDENCY_VARS:
        a, b = r_from.get(var), r_to.get(var)
        if a is None or b is None:
            return None
        out.append((b - a) / _SCALE[var])
    return np.array(out, dtype=np.float64)


def l2norm(v):
    n = np.linalg.norm(v)
    return v / n if n > 1e-12 else None


def phase_angle(stn, t, idx):
    """t 시점 표본의 위상각(도)과 판정구간을 계산한다. 결측/영벡터면 None."""
    a = _diff_vec(idx, stn, t - timedelta(hours=1), t)                 # 최근 1h — 구동축
    q_im = _diff_vec(idx, stn, t - timedelta(hours=3), t)               # 최근 3h — 응답축(관측)
    d_im = _diff_vec(idx, stn, t - timedelta(hours=6), t - timedelta(hours=3))  # 이전 3h — 응답축(기대)
    if a is None or q_im is None or d_im is None:
        return None
    a_n, q_im_n, d_im_n = l2norm(a), l2norm(q_im), l2norm(d_im)
    if a_n is None or q_im_n is None or d_im_n is None:
        return None   # 변화가 전무한 영벡터 — 방향 정의 불가

    r_qd = float(np.dot(a_n, a_n) + np.dot(q_im_n, d_im_n))
    i_qd = float(np.dot(a_n, d_im_n) - np.dot(q_im_n, a_n))
    theta = float(np.degrees(np.arctan2(i_qd, r_qd)))

    ab = abs(theta)
    zone = "Trust" if ab <= THETA_TRUST_DEG else ("Suspect" if ab <= THETA_REJECT_DEG else "Reject")
    return theta, zone


def zone_dist(zones):
    n = len(zones)
    if n == 0:
        return {}
    return {z: zones.count(z) / n for z in ("Trust", "Suspect", "Reject")}


def bootstrap_mean_ci(vals, n_boot=N_BOOT, seed=BOOT_SEED):
    rng = np.random.RandomState(seed)
    vals = np.asarray(vals)
    means = [vals[rng.randint(0, len(vals), len(vals))].mean() for _ in range(n_boot)]
    lo, hi = np.percentile(means, [2.5, 97.5])
    return float(np.mean(vals)), float(lo), float(hi)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--event", default="heatwave", choices=list(EVENTS))
    ap.add_argument("--n", type=int, default=2000, help="TP/FP 표본군당 최대 개수")
    args = ap.parse_args()

    with open(DATA_CACHE, "r", encoding="utf-8") as f:
        records = json.load(f)
    idx = build_index(records)

    d = np.load(EVAL_CACHE, allow_pickle=True)
    prob_key, y_key, mask_key = EVENTS[args.event]
    mask = d[mask_key].astype(bool)
    probs, labels = d[prob_key][mask], d[y_key][mask].astype(int)
    stns, src_ts = d["stn"][mask], d["src_ts"][mask]

    pred_pos = probs >= 0.5
    tp_idx = np.where(pred_pos & (labels == 1))[0]
    fp_idx = np.where(pred_pos & (labels == 0))[0]

    rng = np.random.RandomState(BOOT_SEED)
    if len(tp_idx) > args.n:
        tp_idx = rng.choice(tp_idx, args.n, replace=False)
    if len(fp_idx) > args.n:
        fp_idx = rng.choice(fp_idx, args.n, replace=False)

    print(f"[{args.event}] 예측양성 중 TP={len(np.where(pred_pos & (labels==1))[0]):,} "
          f"(표본 {len(tp_idx):,}개 사용) / FP={len(np.where(pred_pos & (labels==0))[0]):,} "
          f"(표본 {len(fp_idx):,}개 사용)\n")

    results = {}
    for name, sel in (("TP(정탐)", tp_idx), ("FP(오탐)", fp_idx)):
        thetas, zones = [], []
        skipped = 0
        for i in sel:
            r = phase_angle(str(stns[i]), _parse_ts(src_ts[i]), idx)
            if r is None:
                skipped += 1
                continue
            thetas.append(r[0])
            zones.append(r[1])
        results[name] = (thetas, zones, skipped)

    for name, (thetas, zones, skipped) in results.items():
        if not thetas:
            print(f"{name}: 유효 표본 없음(전부 결측/영벡터, 건너뜀 {skipped}개)")
            continue
        mean, lo, hi = bootstrap_mean_ci(np.abs(thetas))
        dist = zone_dist(zones)
        print(f"{name}: n={len(thetas):,} (결측/영벡터로 건너뜀 {skipped}개)")
        print(f"  |θ| 평균={mean:.2f}°  95% CI=[{lo:.2f}, {hi:.2f}]")
        print(f"  구간분포: Trust={dist.get('Trust',0):.1%}  "
              f"Suspect={dist.get('Suspect',0):.1%}  Reject={dist.get('Reject',0):.1%}")

    tp_t, fp_t = results["TP(정탐)"][0], results["FP(오탐)"][0]
    if tp_t and fp_t:
        m_tp, lo_tp, hi_tp = bootstrap_mean_ci(np.abs(tp_t))
        m_fp, lo_fp, hi_fp = bootstrap_mean_ci(np.abs(fp_t))
        overlap = not (hi_tp < lo_fp or hi_fp < lo_tp)
        print(f"\nTP |θ| 95%CI=[{lo_tp:.2f},{hi_tp:.2f}] vs "
              f"FP |θ| 95%CI=[{lo_fp:.2f},{hi_fp:.2f}] — "
              f"{'겹침(유의한 차이 근거 없음)' if overlap else '겹치지 않음(구분 신호 있음)'}")


if __name__ == "__main__":
    main()
