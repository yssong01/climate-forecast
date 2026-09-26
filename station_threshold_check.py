"""
station_threshold_check.py — 관측소별 임계값 재보정이 통계적으로 유의한지 검증.

배경(2026-08-16): calibration_plot_diagnose.py로 관측소별 재현율×정밀도를
쪼개보니 부산만 폭염 정밀도가 확 낮았다(47%, 다른 관측소는 74~85%).
station_anomaly_investigate.py로 원인도 확인했다 — 부산은 오탐일 때 기온
(28.65°C)이 부산 자신의 실제 폭염일 평균(27.38°C)보다 높다. 즉 부산의
공식 폭염 기준 자체가 다른데 모델은 전체 관측소 공통 임계값(0.5)을 쓴다.

재학습 없이 임계값만 관측소별로 따로 고르면 나아지는지, threshold_validation.py
와 같은 보정용/평가용 분리 절차로 검증한다(과적합 방지 — 채택 기준은 같은
프로젝트 규약대로 순이득 0.01).

실행: python station_threshold_check.py [--station 159] [--event heatwave]
"""
import argparse

import numpy as np

import eval_cache
from train import STATION_NAMES
from predict import CHECKPOINT
from threshold_validation import prf, best_thresh, sensitivity, CALIB_SEED


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="159", help="관측소 코드 (기본: 부산)")
    ap.add_argument("--event", default="heatwave", choices=["heatwave", "coldwave"])
    args = ap.parse_args()

    # 검증셋 추론은 `eval_cache` 가 만든 것을 재사용한다(2026-09-27 전환).
    # 종전에는 이 스크립트가 `WeatherDataset` 을 직접 구성했는데 그 한 번이
    # RAM 약 22GiB 라, 같은 계열 스크립트 두 개만 겹쳐 돌려도 OOM 이 났다.
    # 캐시는 체크포인트 지문으로 신선도를 검사하므로 낡은 값을 쓸 위험이 없다.
    d = eval_cache.load(CHECKPOINT)
    stns = np.asarray(d["stn"])
    # 마스크를 적용하기 **전에** 관측소로 먼저 거른다 — 순서를 바꾸면
    # 마스크 인덱스와 관측소 인덱스가 어긋난다.
    pk, yk, mk, _ = eval_cache.EVENT_KEYS[args.event]
    sel = stns == args.station
    probs = np.asarray(d[pk])[sel].astype(np.float64)
    labels = np.asarray(d[yk])[sel].astype(int)
    mask = np.asarray(d[mk])[sel].astype(bool)
    stn_name = STATION_NAMES.get(args.station, args.station)
    print(f"관측소: {stn_name}({args.station}) — 검증 표본 {int(sel.sum()):,}개\n")
    probs, labels = probs[mask], labels[mask]

    rng = np.random.RandomState(CALIB_SEED)
    split = rng.rand(len(probs)) < 0.5
    pc, lc = probs[split], labels[split]
    pt, lt = probs[~split], labels[~split]

    print(f"[{stn_name} {args.event}] 판정 가능 {len(probs):,}개 "
          f"(양성 {int(labels.sum()):,}개) → 보정용 양성 {int(lc.sum())}개 / "
          f"평가용 양성 {int(lt.sum())}개")
    if lc.sum() < 5 or lt.sum() < 5:
        print("표본이 너무 적어 관측소별 재보정은 신뢰할 수 없다 — 중단")
        return

    t_opt, f1_calib = best_thresh(pc, lc)
    _, _, f1_test = prf(pt, lt, t_opt)
    _, _, f1_test05 = prf(pt, lt, 0.5)
    lo, hi = sensitivity(pt, lt, t_opt)
    real_gain = f1_test - f1_test05

    print(f"  보정용에서 고른 t = {t_opt:.2f} (보정용 F1 {f1_calib:.4f})")
    print(f"  → 평가용 F1 = {f1_test:.4f} (전역 t=0.5의 평가용 F1 {f1_test05:.4f})")
    print(f"  실제 순이득 = {real_gain:+.4f} "
          f"{'✅ 진짜 개선 — 채택 가능' if real_gain > 0.01 else '⚠️ 허수 — 과적합 의심, 채택 보류'}")
    print(f"  t±0.02 흔들 때 F1 범위 = [{lo:.4f}, {hi:.4f}] (폭 {hi-lo:.4f})"
          f"{'  ⚠️ 불안정' if hi - lo > 0.05 else ''}")

    p05, r05, _ = prf(pt, lt, 0.5)
    popt, ropt, _ = prf(pt, lt, t_opt)
    print(f"\n  참고 — t=0.50: P={p05:.3f} R={r05:.3f} | "
          f"t={t_opt:.2f}: P={popt:.3f} R={ropt:.3f}")


if __name__ == "__main__":
    main()
