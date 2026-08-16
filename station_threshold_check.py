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
import torch

from train import collect_historical, WeatherDataset, make_split, STATION_NAMES
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from predict import load_model, CHECKPOINT
from threshold_validation import prf, best_thresh, sensitivity, CALIB_SEED

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024

_LOGIT_ATTR = {"heatwave": "_last_heatwave_logit", "coldwave": "_last_coldwave_logit"}
_LABEL_ATTR = {"heatwave": "y_heatwave", "coldwave": "y_coldwave"}
_MASK_ATTR = {"heatwave": "heat_mask", "coldwave": "cold_mask"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--station", default="159", help="관측소 코드 (기본: 부산)")
    ap.add_argument("--event", default="heatwave", choices=["heatwave", "coldwave"])
    args = ap.parse_args()

    model, ckpt = load_model(CHECKPOINT, DEVICE)
    model.eval()
    records = collect_historical()
    txt = TendencyCollector(records)
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False)
    val_idx = np.array(val_ds.indices)
    stns = np.array([ds.stns[i] for i in val_idx])
    sel = stns == args.station
    idx_stn = val_idx[sel]
    stn_name = STATION_NAMES.get(args.station, args.station)
    print(f"관측소: {stn_name}({args.station}) — 검증 표본 {len(idx_stn):,}개\n")

    probs = []
    with torch.no_grad():
        for b in range(0, len(idx_stn), BATCH):
            sl = idx_stn[b:b + BATCH].tolist()
            model(num_x=ds.X_num[sl].to(DEVICE), img_x=ds.X_img[sl].to(DEVICE).float(),
                  txt_x=ds.X_txt[sl].to(DEVICE))
            logit = getattr(model, _LOGIT_ATTR[args.event])
            probs.append(torch.sigmoid(logit).squeeze(-1).cpu().numpy())
    probs = np.concatenate(probs)

    mask = getattr(ds, _MASK_ATTR[args.event])[idx_stn].numpy().astype(bool)
    labels = getattr(ds, _LABEL_ATTR[args.event])[idx_stn].numpy().astype(int)
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
