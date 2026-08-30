"""
bootstrap_ci_compare.py — 두 체크포인트의 지표 차이가 표본 재추출 노이즈
범위를 벗어나는지 날짜 블록 부트스트랩으로 검정한다.

**왜 필요한가.** climate-forecast는 지금까지 단일 seed A/B 비교만으로
채택/기각을 결정해왔다(가장 최근 사례: RE_CHANNELS 4채널 vs 5채널 대조
실험, CLAUDE.md 5절·rejected-re-channel-precip 메모). 지표 하나가 반대
방향으로 움직였을 때 그게 "진짜 효과"인지 "검증셋 표본 구성의 우연"인지
구분할 도구가 없었다. weld-opt 프로젝트의 부트스트랩 신뢰구간(N=2000)
기법을 이식한다.

**왜 날짜 블록 단위인가.** CLAUDE.md 2절 "사건 단위 표본은 시간 수가
아니라 사건 수로 센다" — 한파·폭염처럼 며칠씩 이어지는 사건은 연속 시간
표본이 서로 독립이 아니다. 개별 시간 표본을 통째로 재추출(iid 부트스트랩)
하면 사건 내부 상관을 무시해 신뢰구간이 실제보다 좁게(과신) 나온다.
train.py의 make_split이 날짜 단위로 그룹 분할하는 것과 같은 이유로,
여기서도 날짜를 재추출 단위로 삼는다(block bootstrap).

**짝지은(paired) 비교인 이유.** 두 체크포인트가 동일 SEED·동일 데이터로
학습됐다면 검증셋 구성(날짜 집합·표본 순서)이 동일하다 — 대조 실험의
전제. 이 경우 같은 재추출 날짜 집합을 두 체크포인트에 동시에 적용하는
짝지은 부트스트랩이 각자 독립적으로 재추출하는 것보다 검정력이 높다
(공통 표본 변동이 상쇄된다). 시작 시 두 체크포인트의 표본 개수·타깃
시각이 완전히 같은지 확인해 이 전제를 검증하고, 다르면 즉시 멈춘다.

**추론 재사용.** eval_cache.build()를 두 체크포인트에 각각 호출한다
(CLAUDE.md 3절 — 새 진단 스크립트가 데이터셋을 직접 다시 만들지 않는다는
규약). 두 번 호출하는 동안 공유 캐시 파일(./cache/eval_cache.npz)이
마지막 호출 결과로 남지만, 이후 다른 스크립트가 배포 체크포인트로
eval_cache.load()를 부르면 서명 불일치로 자동 재계산되므로 안전하다.

실행 (GPU 컨테이너, 순차 — CLAUDE.md 3절 "동시에 돌리지 말 것"):
  python bootstrap_ci_compare.py <체크포인트A:대조> <체크포인트B:처리> [--n-boot 2000]
"""
import argparse
import sys

import numpy as np

import eval_cache

N_BOOT_DEFAULT = 2000
# 부트스트랩 재추출 전용 시드 — 학습 SEED(42, 분할 고정)와 별개 목적이라 구분한다.
BOOT_SEED = 20260831


def f1_at(prob: np.ndarray, y: np.ndarray, mask: np.ndarray, thresh: float = 0.5) -> float:
    m = mask.astype(bool)
    if not np.any(m):
        return float("nan")
    pred = prob[m] >= thresh
    y_m = y[m].astype(bool)
    tp = np.count_nonzero(pred & y_m)
    fp = np.count_nonzero(pred & ~y_m)
    fn = np.count_nonzero(~pred & y_m)
    denom = 2 * tp + fp + fn
    return 2 * tp / denom if denom > 0 else float("nan")


def metrics(d: dict, idx: np.ndarray) -> dict:
    """주어진 표본 인덱스(중복 허용 — 부트스트랩 재추출분)에서 지표를 계산한다."""
    return {
        "temp_mae":   float(np.mean(np.abs(d["temp_pred"][idx] - d["temp_true"][idx]))),
        "precip_mae": float(np.mean(np.abs(d["precip_pred"][idx] - d["precip_true"][idx]))),
        "heat_f1": f1_at(d["heat_prob"][idx], d["y_heatwave"][idx], d["heat_mask"][idx]),
        "cold_f1": f1_at(d["cold_prob"][idx], d["y_coldwave"][idx], d["cold_mask"][idx]),
        "dust_f1": f1_at(d["dust_prob"][idx], d["y_dust"][idx], d["dust_mask"][idx]),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt_a", help="대조(control) 체크포인트")
    ap.add_argument("ckpt_b", help="처리(treatment) 체크포인트")
    ap.add_argument("--n-boot", type=int, default=N_BOOT_DEFAULT)
    ap.add_argument("--batch", type=int, default=8192)
    args = ap.parse_args()

    print(f"대조 체크포인트 추론 중: {args.ckpt_a}")
    a = eval_cache.build(args.ckpt_a, args.batch)
    print(f"처리 체크포인트 추론 중: {args.ckpt_b}")
    b = eval_cache.build(args.ckpt_b, args.batch)

    if a["stn"].shape != b["stn"].shape or not np.array_equal(a["tgt_ts"], b["tgt_ts"]):
        print("중단: 두 체크포인트의 검증셋 표본 순서/구성이 다르다 — 짝지은 비교 전제가 깨졌다.")
        print(f"  A 표본수={a['stn'].shape[0]}, B 표본수={b['stn'].shape[0]}")
        sys.exit(1)

    dates = np.array([str(ts)[:8] for ts in a["tgt_ts"]])
    uniq = np.unique(dates)
    date_to_idx = {dt: np.where(dates == dt)[0] for dt in uniq}
    print(f"검증 표본 {len(dates):,}개, 고유 날짜 {len(uniq):,}개 — 날짜 블록 단위 부트스트랩 "
          f"(B={args.n_boot})")

    rng = np.random.RandomState(BOOT_SEED)
    keys = ["temp_mae", "precip_mae", "heat_f1", "cold_f1", "dust_f1"]
    deltas = {k: [] for k in keys}

    point_a = metrics(a, np.arange(len(dates)))
    point_b = metrics(b, np.arange(len(dates)))

    for i in range(args.n_boot):
        draw = rng.choice(uniq, size=len(uniq), replace=True)
        idx = np.concatenate([date_to_idx[dt] for dt in draw])
        ma = metrics(a, idx)
        mb = metrics(b, idx)
        for k in keys:
            deltas[k].append(mb[k] - ma[k])
        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n_boot}")

    print(f"\n{'지표':<12}{'A(대조)':>10}{'B(처리)':>10}{'Δ(B-A)':>10}{'95% CI':>22}{'유의':>6}")
    for k in keys:
        arr = np.array(deltas[k])
        lo, hi = np.nanpercentile(arr, [2.5, 97.5])
        sig = "*" if (lo > 0 or hi < 0) else ""
        print(f"{k:<12}{point_a[k]:>10.4f}{point_b[k]:>10.4f}{point_b[k] - point_a[k]:>10.4f}"
              f"  [{lo:+.4f}, {hi:+.4f}]{sig:>6}")
    print("\n* = 95% CI가 0을 포함하지 않음 (표본 재추출 노이즈로는 설명되지 않는 방향성).")
    print("  temp_mae/precip_mae는 낮을수록, heat/cold/dust_f1은 높을수록 B가 우세.")


if __name__ == "__main__":
    sys.exit(main())
