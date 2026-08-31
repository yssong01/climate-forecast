"""
patch_extreme_metrics.py — head_decouple_finetune.py 로 만든 체크포인트의
extreme_metrics 를 실제 미세조정 결과로 재계산해 덮어쓴다.

head_decouple_finetune.py 는 val_temp_mae/val_precip_mae 는 갱신하지만
extreme_metrics 는 원본 체크포인트 값을 그대로 들고 온다(dict(ckpt) 복사).
대시보드 '성능 검증' 탭이 이 필드를 그대로 읽어 보여주므로, 패치하지 않으면
실제로 배포되는 모델의 성능이 아니라 디커플링 전 값을 화면에 내보이게 된다
(2026-08-16 첫 디커플링 때 같은 문제를 발견해 수동 패치했던 것과 동일한 함정
— 이번엔 스크립트로 고정한다).

실행: python patch_extreme_metrics.py <체크포인트 경로>
"""
import sys

import numpy as np
import torch

from train import collect_historical, WeatherDataset, make_split
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from predict import load_model
from head_decouple_finetune import evaluate, BATCH

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def main():
    ckpt_path = sys.argv[1]
    model, ckpt = load_model(ckpt_path, DEVICE)
    print(f"체크포인트: {ckpt_path}")
    print(f"  기존 저장값 — 기온MAE {ckpt['val_temp_mae']:.4f} "
          f"강수MAE {ckpt['val_precip_mae']:.4f}")
    old_em = ckpt.get("extreme_metrics") or {}
    for k, m in old_em.items():
        print(f"  기존 extreme_metrics[{k}] — "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f}")

    records = collect_historical()
    txt_collector = TendencyCollector(records)
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=True, ckpt=ckpt)
    val_idx = list(val_ds.indices)

    temp_mae, precip_mae, metrics = evaluate(model, ds, val_idx, DEVICE)

    print(f"\n  재계산값 — 기온MAE {temp_mae:.4f} 강수MAE {precip_mae:.4f}")
    for k, m in metrics.items():
        print(f"  재계산 extreme_metrics[{k}] — "
              f"P={m['precision']:.3f} R={m['recall']:.3f} F1={m['f1']:.3f} "
              f"(양성 {m['n_pos']:,}건)")

    assert abs(temp_mae - ckpt["val_temp_mae"]) < 1e-3, \
        "기온 MAE 가 저장값과 달라 재계산이 잘못됐을 수 있음 — 확인 필요"
    assert abs(precip_mae - ckpt["val_precip_mae"]) < 1e-3, \
        "강수 MAE 가 저장값과 달라 재계산이 잘못됐을 수 있음 — 확인 필요"

    ckpt["extreme_metrics"] = metrics
    torch.save(ckpt, ckpt_path)
    print(f"\n저장 완료(패치): {ckpt_path}")


if __name__ == "__main__":
    main()
