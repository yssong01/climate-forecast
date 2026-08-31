"""
precip_breakdown.py — 강수 MAE가 baseline(상시 0예측)에 지는 이유를 분해한다.

가설: 강수 헤드는 softplus(x) 출력이라 수학적으로 정확히 0을 낼 수 없다
(softplus(x) > 0 for all finite x). 검증셋의 93%가 무강수(<0.1mm)인 지금
데이터 규모에서는, 그 93% 각각에 깔리는 아주 작은 비영 오차가 누적되면
드문 강수 샘플(6.7%)에서 얻는 큰 이득을 상쇄하고도 남을 수 있다 —
"상시 0예측"은 무강수 구간에서 문자 그대로 0의 오차를 내는 반면, 모델은
그 구간에서 항상 약간의 오차를 깔고 시작한다.

검증셋을 무강수/강수 두 구간으로 쪼개 모델과 baseline 을 각각 잰다.
"""
import numpy as np
import torch

from train import collect_historical, WeatherDataset, VAL_RATIO, SEED, make_split
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from text_collector import SimulatedTextCollector
from predict import load_model, CHECKPOINT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
WET_THRESH = 0.1  # mm — train.py 이벤트 정의와 동일


def main():
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    records = collect_historical()
    # 2026-08-09: 체크포인트가 Re·Im축 모두 실제 데이터에 의존하므로 학습
    # 때와 같은 컬렉터를 써야 한다. im_dim 으로 구/신버전을 자동 판별한다.
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False, ckpt=ckpt)
    val_idx = val_ds.indices

    preds, actuals = [], []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(val_idx), BATCH):
            idx = val_idx[b:b + BATCH]
            pred = model(num_x=ds.X_num[idx].to(DEVICE),
                        img_x=ds.X_img[idx].to(DEVICE).float(),
                        txt_x=ds.X_txt[idx].to(DEVICE)).cpu()
            preds.append(pred[:, 1].numpy())
            actuals.append(ds.y[idx, 1].numpy())

    pred = np.concatenate(preds)
    actual = np.concatenate(actuals)
    n = len(actual)

    dry = actual < WET_THRESH
    wet = ~dry
    print(f"검증 표본 {n}개 — 무강수 {dry.sum()}개({dry.mean()*100:.2f}%), "
          f"강수 {wet.sum()}개({wet.mean()*100:.2f}%)\n")

    def report(mask, label):
        n_sub = mask.sum()
        model_mae = np.abs(pred[mask] - actual[mask]).mean()
        base_mae = np.abs(actual[mask]).mean()          # baseline = 항상 0
        model_total = np.abs(pred[mask] - actual[mask]).sum()
        base_total = np.abs(actual[mask]).sum()
        print(f"[{label}] n={n_sub}")
        print(f"  모델 MAE  = {model_mae:.5f} mm  (구간 내 총오차 {model_total:.1f})")
        print(f"  baseline  = {base_mae:.5f} mm  (구간 내 총오차 {base_total:.1f})")
        print(f"  모델 평균 예측값(이 구간) = {pred[mask].mean():.5f} mm")
        return model_total, base_total

    dry_m, dry_b = report(dry, "무강수 구간")
    wet_m, wet_b = report(wet, "강수 구간")

    print(f"\n전체 총오차 — 모델: {dry_m+wet_m:.1f}  baseline: {dry_b+wet_b:.1f}")
    print(f"  무강수 구간이 전체 '모델-baseline 오차 격차'에 기여한 비율: "
          f"{(dry_m-dry_b)/((dry_m+wet_m)-(dry_b+wet_b))*100:.1f}%")
    print(f"  강수 구간이 기여한 비율: "
          f"{(wet_m-wet_b)/((dry_m+wet_m)-(dry_b+wet_b))*100:.1f}%")

    print(f"\n전체 모델 MAE = {np.abs(pred-actual).mean():.5f}mm, "
          f"baseline MAE = {np.abs(actual).mean():.5f}mm")

    # ── 재학습 없이: 후처리 임계값 클리핑만 적용하면 얼마나 나아지나 ──
    # softplus 는 정확히 0을 못 내므로, 이 프로젝트 자체가 쓰는 "무강수"
    # 기준(0.1mm)보다 낮은 예측은 실질적으로 "비가 안 온다"는 뜻과 같다.
    # 재학습 없이 기존 체크포인트로 바로 검증 가능 — Hurdle 모델(더 큰
    # 구조 변경)에 투자하기 전에, 이 값싼 개입만으로 격차가 얼마나
    # 닫히는지 먼저 재는 게 우선이다.
    print(f"\n{'='*60}\n후처리 임계값 클리핑 실험 (재학습 없음)\n{'='*60}")
    for clip_thresh in (0.05, 0.1, 0.15, 0.2):
        clipped = np.where(pred < clip_thresh, 0.0, pred)
        mae = np.abs(clipped - actual).mean()
        base_mae = np.abs(actual).mean()
        gain = (1 - mae / base_mae) * 100
        print(f"  clip<{clip_thresh:.2f}mm → MAE={mae:.5f}mm "
              f"vs baseline {base_mae:.5f}mm  ({gain:+.1f}%)")


if __name__ == "__main__":
    main()
