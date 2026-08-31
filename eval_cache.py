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
from train import (WeatherDataset, collect_historical, make_split, WET_THRESH,
                   DATA_CACHE)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from island_collector import IslandPrecipCollector
from tendency_collector import TendencyCollector
from text_collector import SimulatedTextCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
CACHE_PATH = "./cache/eval_cache.npz"

# 배치 기본값 — 추론만 하므로 학습보다 크게 잡을 수 있다. 실측(2026-08-16)
# 에서 학습 중 VRAM 사용률이 12% 에 그쳤으므로 8192 로도 여유가 있다.
DEFAULT_BATCH = 8192


def cache_signature(ckpt_path: str, ckpt: dict) -> dict:
    st = os.stat(ckpt_path)
    sig = {
        "ckpt_path": ckpt_path,
        "ckpt_mtime": float(st.st_mtime),
        "ckpt_size": int(st.st_size),
        "split_mode": str(ckpt.get("split_mode", "random")),
        "num_features": int(ckpt["num_features"]),
        "lead_hours": int(ckpt["lead_hours"]),
    }
    # 학습 데이터 파일도 서명에 넣는다(2026-08-29 추가). 종전에는 체크포인트
    # 속성만 봤는데, build() 는 매번 collect_historical() 로 이 파일을 다시
    # 읽으므로 파일이 바뀌면 캐시 내용이 실제와 어긋난다 — 특히 make_split
    # 의 그룹 분할이 **고유 날짜 수**에 의존해(torch.randperm(len(uniq)))
    # 데이터가 늘면 같은 SEED 로도 검증셋 구성이 통째로 달라진다(실측:
    # 날짜가 1개만 늘어도 검증셋 겹침이 20%). 파일이 바뀌면 최소한 캐시를
    # 다시 만들게 해서, 아래 _assert_split_matches_checkpoint() 의 대조가
    # 반드시 한 번은 실행되도록 한다.
    try:
        dst = os.stat(DATA_CACHE)
        sig["data_mtime"] = float(dst.st_mtime)
        sig["data_size"] = int(dst.st_size)
    except OSError:
        sig["data_mtime"] = 0.0
        sig["data_size"] = 0
    return sig


def _assert_split_matches_checkpoint(ds, val_idx, ckpt) -> None:
    """재구성한 검증셋이 학습 당시의 그것과 같은지 기준선으로 대조한다.

    make_split 의 그룹 분할은 고유 날짜 수에 의존하므로, 학습 이후 데이터가
    누적되면 같은 SEED·같은 split_mode 여도 다른 검증셋이 나온다. 그러면
    "학습 때 본 적 없는 표본"이라는 전제가 깨진 채 지표가 산출되는데,
    지표 자체는 멀쩡해 보여서 눈치채기 어렵다.

    체크포인트에 저장된 val_temp_naive_mae / val_precip_naive_mae 는 학습
    당시 검증셋에서 계산한 값이다 — 지금 재구성한 검증셋에서 같은 값이
    나오지 않으면 표본이 달라졌다는 뜻이므로 즉시 멈춘다. CLAUDE.md 2절이
    사람에게 수동으로 요구하던 대조를 자동화한 것이다(2026-08-29 추가).
    """
    checks = (
        ("val_temp_naive_mae", float(ds.temp_persist_abs[val_idx].mean()), "기온 퍼시스턴스"),
        ("val_precip_naive_mae", float(ds.y[val_idx, 1].abs().mean()), "강수 상시 0"),
    )
    for key, got, ko in checks:
        want = ckpt.get(key)
        if want is None:
            continue                      # 구버전 체크포인트 — 대조 불가
        if abs(float(want) - got) > 1e-4:
            raise RuntimeError(
                f"검증셋이 학습 당시와 다르다 — {ko} 기준선 저장값 {float(want):.6f} "
                f"vs 재구성값 {got:.6f}.\n"
                f"  원인: make_split 의 그룹 분할이 고유 날짜 수에 의존하는데 "
                f"{DATA_CACHE} 가 학습 이후 변경됐을 가능성이 높다.\n"
                f"  이 상태의 지표는 '학습에 쓰인 표본'을 검증셋에 섞어 산출한 "
                f"값일 수 있으므로 신뢰할 수 없다.\n"
                f"  조치: 이 체크포인트를 학습할 때 쓴 데이터 파일로 되돌리거나, "
                f"현재 데이터로 재학습할 것.")


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
    island_collector = IslandPrecipCollector() if ckpt.get("use_island", False) else None
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(
            records, STATION_COORDS, n_bands=ckpt.get("re_channels", 4)),
        txt_collector=txt_collector, island_collector=island_collector,
        lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=True)
    val_idx = np.array(val_ds.indices)
    # 재구성한 검증셋이 학습 당시와 같은지 먼저 대조한다 — 다르면 여기서
    # 멈춘다. 이 확인 없이 추론을 돌리면 누수된 지표가 나와도 알 수 없다.
    _assert_split_matches_checkpoint(ds, val_idx, ckpt)
    print(f"검증 표본 {len(val_idx):,}개 · 배치 {batch:,} · 장치 {DEVICE}"
          f" · 검증셋 대조 통과")

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
