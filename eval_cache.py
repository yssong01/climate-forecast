"""
eval_cache.py — 검증셋 추론 결과를 한 번만 계산해 캐시로 남긴다.

**왜 필요한가.** 지표 스크립트들(`metrics_report.py`, `error_breakdown.py`,
`probability_calibration_*.py`)이 각자 데이터셋을 만들고 각자 forward 를 돌린다.
데이터셋 1회 구성이 RAM 약 22GiB 이므로 두 개를 동시에 띄우면 이 장비(31.2GiB)
에서 OOM 이 난다 — 학습을 순차로만 돌리는 것과 같은 이유다. 그래서 무거운
부분(데이터셋 구성 + GPU 추론)을 여기서 한 번만 수행하고, 결과를 npz 로
저장한다. 이후 리포트들은 이 캐시만 읽으므로 서로 동시에 돌려도 안전하다.

**캐시가 담는 것.** 회귀 예측·정답·퍼시스턴스 기준선, 네 헤드의 보정 전 확률,
공식 라벨과 마스크, 그리고 구간 분해에 필요한 메타(관측소·시각·현재 기온).
보정은 담지 않는다 — 곡선은 체크포인트에 있고 리포트 단계에서 적용하는 편이
보정 재적합 후 캐시를 다시 만들지 않아도 되어 낫다.

**무효화.** 체크포인트 파일의 mtime·크기와 분할 방식을 캐시에 함께 적어두고,
하나라도 다르면 다시 계산한다. `--force` 로 강제할 수 있다.

실행:
    python eval_cache.py                 # 없으면 만들고, 있으면 그대로 둔다
    python eval_cache.py --force         # 무조건 다시 계산
    python eval_cache.py --batch 16384   # VRAM 여유가 있으면 배치를 키운다
"""
import argparse
import os
import sys
import time

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import (WeatherDataset, collect_historical, make_split, WET_THRESH)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from text_collector import SimulatedTextCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_PATH = "./cache/eval_cache.npz"

# 배치 기본값 — 추론만 하므로 학습보다 크게 잡을 수 있다. 실측(2026-08-16)
# 에서 학습 중 VRAM 사용률이 12% 에 그쳤으므로 8192 로도 여유가 있다.
DEFAULT_BATCH = 8192


def cache_signature(ckpt_path: str, ckpt: dict) -> dict:
    st = os.stat(ckpt_path)
    return {
        "ckpt_path": ckpt_path,
        "ckpt_mtime": float(st.st_mtime),
        "ckpt_size": int(st.st_size),
        "split_mode": str(ckpt.get("split_mode", "random")),
        "num_features": int(ckpt["num_features"]),
        "lead_hours": int(ckpt["lead_hours"]),
    }


def is_fresh(path: str, sig: dict) -> bool:
    if not os.path.exists(path):
        return False
    try:
        z = np.load(path, allow_pickle=True)
    except Exception:
        return False
    for k, v in sig.items():
        if k not in z.files:
            return False
        got = z[k].item() if z[k].shape == () else z[k]
        if isinstance(v, float):
            if abs(float(got) - v) > 1e-6:
                return False
        elif str(got) != str(v):
            return False
    return True


def build(ckpt_path: str, batch: int):
    t0 = time.time()
    model, ckpt = load_model(ckpt_path, DEVICE)
    model.eval()
    print(f"체크포인트: {ckpt_path} (num_features={ckpt['num_features']}, "
          f"분할={ckpt.get('split_mode', 'random')})")

    records = collect_historical()
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=True)
    val_idx = np.array(val_ds.indices)
    print(f"검증 표본 {len(val_idx):,}개 · 배치 {batch:,} · 장치 {DEVICE}")

    temp_pred, precip_pred = [], []
    rain_p, heat_p, cold_p, dust_p = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(val_idx), batch):
            idx = val_idx[b:b + batch].tolist()
            out = model(num_x=ds.X_num[idx].to(DEVICE),
                        img_x=ds.X_img[idx].to(DEVICE).float(),
                        txt_x=ds.X_txt[idx].to(DEVICE))
            temp_pred.append(out[:, 0].cpu().numpy())
            precip_pred.append(out[:, 1].cpu().numpy())
            rain_p.append(torch.sigmoid(model._last_rain_logit).squeeze(-1).cpu().numpy())
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            dust_p.append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu().numpy())

    # 현재 기온(퍼시스턴스 기준선의 값이자 구간 분해의 기준축) — X_num 은
    # 표준화돼 있으므로 되돌린다. 0번 특성이 기온이다.
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    temp_now = ds.X_num[val_idx.tolist(), 0].numpy() * std[0] + mean[0]

    data = dict(
        temp_pred=np.concatenate(temp_pred),
        precip_pred=np.concatenate(precip_pred),
        temp_true=ds.y[val_idx, 0].numpy(),
        precip_true=ds.y[val_idx, 1].numpy(),
        temp_now=temp_now,
        temp_persist_abs=ds.temp_persist_abs[val_idx].numpy(),
        rain_prob=np.concatenate(rain_p),
        heat_prob=np.concatenate(heat_p),
        cold_prob=np.concatenate(cold_p),
        dust_prob=np.concatenate(dust_p),
        y_heatwave=ds.y_heatwave[val_idx].numpy(),
        y_coldwave=ds.y_coldwave[val_idx].numpy(),
        y_dust=ds.y_dust[val_idx].numpy(),
        heat_mask=ds.heat_mask[val_idx].numpy(),
        cold_mask=ds.cold_mask[val_idx].numpy(),
        dust_mask=ds.dust_mask[val_idx].numpy(),
        stn=np.array([str(ds.stns[i]) for i in val_idx]),
        src_ts=np.array([str(ds.src_timestamps[i]) for i in val_idx]),
        tgt_ts=np.array([str(ds.tgt_timestamps[i]) for i in val_idx]),
        wet_thresh=np.array(WET_THRESH),
    )
    data.update({k: np.array(v) for k, v in cache_signature(ckpt_path, ckpt).items()})

    os.makedirs(os.path.dirname(CACHE_PATH), exist_ok=True)
    # 공유 파일 저장은 고유 tmp + os.replace 로 원자적으로(CLAUDE.md 6절).
    import tempfile
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(CACHE_PATH), suffix=".npz")
    os.close(fd)
    np.savez_compressed(tmp, **data)
    os.replace(tmp, CACHE_PATH)
    size_mb = os.path.getsize(CACHE_PATH) / 1e6
    print(f"\n저장: {CACHE_PATH} ({size_mb:.1f}MB, {time.time() - t0:.0f}초)")
    return data


def load(ckpt_path: str = CHECKPOINT, batch: int = DEFAULT_BATCH, force: bool = False):
    """캐시를 읽어 dict 로 돌려준다. 없거나 낡았으면 만든다."""
    ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=True)
    sig = cache_signature(ckpt_path, ckpt)
    if not force and is_fresh(CACHE_PATH, sig):
        z = np.load(CACHE_PATH, allow_pickle=True)
        return {k: z[k] for k in z.files}
    return build(ckpt_path, batch)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--batch", type=int, default=DEFAULT_BATCH)
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    sig = cache_signature(args.ckpt, ckpt)
    if not args.force and is_fresh(CACHE_PATH, sig):
        print(f"캐시가 최신이다 — 다시 계산하지 않는다: {CACHE_PATH}")
        print("(강제로 다시 만들려면 --force)")
        return
    build(args.ckpt, args.batch)


if __name__ == "__main__":
    sys.exit(main())
