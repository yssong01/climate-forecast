"""
perf_profile.py — 학습 파이프라인의 실제 병목을 구간별로 잰다.

배경(2026-08-15): 13년치 학습이 1회 약 4시간 20분인데 nvidia-smi 기준 GPU
사용률이 31%, VRAM 747MiB/8GB에 그치고 CPU는 110%(32코어 중 1.1코어)였다.
GPU가 아니라 CPU 단일 스레드가 병목이라는 뜻이다. 앞으로 실험을 여러 번
돌려야 하므로(시간분할 기준선, crop 사전학습 2단계, 대조군들) 어디를 고쳐야
하는지부터 실측한다.

구간
  1. Re축 공간보간장 생성 — get_batch()가 표본마다 32×32 격자에 이웃 11곳을
     IDW 보간한다. 표본 138만 × (meshgrid + 32×32×11 축약 5회)라 후보 1순위.
  2. Im축 경향벡터 생성 — 딕셔너리 조회 3회뿐이라 쌀 것으로 예상.
  3. 학습 스텝 — DataLoader(num_workers=0) 수집 + forward/backward.
     batch_size=32에 표본 110만이면 에폭당 34,524스텝이라 파이썬 오버헤드가
     GPU 연산을 압도할 수 있다.

실행: python perf_profile.py
"""
import json
import time

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from train import DATA_CACHE, record_to_vec, BATCH_SIZE, DEVICE
from pipeline_model import TriCHEFPipeline

N_PROBE = 20_000        # 이만큼만 재고 138만으로 환산
FULL_N = 1_380_966


def main():
    with open(DATA_CACHE, encoding="utf-8") as f:
        records = json.load(f)
    print(f"레코드 {len(records):,}개 로드\n")

    probe = records[:N_PROBE]

    print("=" * 88)
    print(f"구간별 소요 — 표본 {N_PROBE:,}개로 재고 전체 {FULL_N:,}개로 환산")
    print("=" * 88)

    t0 = time.perf_counter()
    sat = InterpolatedFieldCollector(records, STATION_COORDS)
    t_idx_re = time.perf_counter() - t0

    t0 = time.perf_counter()
    txt = TendencyCollector(records)
    t_idx_im = time.perf_counter() - t0
    print(f"  인덱스 구성 — Re {t_idx_re:6.1f}초 · Im {t_idx_im:6.1f}초  (1회성)")

    t0 = time.perf_counter()
    _ = sat.get_batch(probe)
    t_re = time.perf_counter() - t0
    print(f"  Re축 보간장 생성 : {t_re:7.2f}초 / {N_PROBE:,}개 "
          f"→ 전체 환산 {t_re * FULL_N / N_PROBE / 60:6.1f}분")

    t0 = time.perf_counter()
    _ = txt.get_batch(probe)
    t_im = time.perf_counter() - t0
    print(f"  Im축 경향벡터    : {t_im:7.2f}초 / {N_PROBE:,}개 "
          f"→ 전체 환산 {t_im * FULL_N / N_PROBE / 60:6.1f}분")

    t0 = time.perf_counter()
    _ = np.array([record_to_vec(r) for r in probe], dtype=np.float32)
    t_num = time.perf_counter() - t0
    print(f"  Z축 수치벡터     : {t_num:7.2f}초 / {N_PROBE:,}개 "
          f"→ 전체 환산 {t_num * FULL_N / N_PROBE / 60:6.1f}분")

    # ── 학습 스텝 처리량 — DataLoader 설정별 ────────────────────────
    print()
    print("=" * 88)
    print("학습 스텝 처리량 — DataLoader 설정별 (모델·손실은 실제와 동일 구조)")
    print("=" * 88)

    n_bench = 60_000
    X_num = torch.randn(n_bench, 12)
    X_img = torch.randn(n_bench, 4, 32, 32, dtype=torch.float16)
    X_txt = torch.randn(n_bench, 12)
    Y = torch.randn(n_bench, 2)
    extra = torch.zeros(n_bench, 6)
    ds = TensorDataset(X_num, X_img, X_txt, Y, extra)

    model = TriCHEFPipeline(
        embed_dim=64, num_features=12, alpha_init=0.4, phi_init=0.2,
        orthogonalize=False, temp_mean=15.0, precip_mean=0.1,
        persistence_residual=True,
        feat_mean=[0.0] * 12, feat_std=[1.0] * 12,
        dynamic_gate=True, compact_satellite=True,
        wet_prior=0.06, heatwave_prior=0.06, coldwave_prior=0.015,
        dust_prior=0.028, im_dim=12,
    ).to(DEVICE)
    opt = torch.optim.Adam(model.parameters(), lr=1e-3)
    mse = nn.MSELoss()

    def bench(bs, workers, pin, max_steps=300):
        loader = DataLoader(ds, batch_size=bs, shuffle=True,
                            num_workers=workers, pin_memory=pin,
                            persistent_workers=(workers > 0))
        it = iter(loader)
        # 워커 기동·CUDA 워밍업은 측정에서 뺀다
        for _ in range(5):
            try:
                b = next(it)
            except StopIteration:
                it = iter(loader); b = next(it)
        torch.cuda.synchronize() if DEVICE == "cuda" else None

        t0 = time.perf_counter()
        n_samples = 0
        for step in range(max_steps):
            try:
                xn, xi, xt, y, _ = next(it)
            except StopIteration:
                it = iter(loader)
                xn, xi, xt, y, _ = next(it)
            xn = xn.to(DEVICE, non_blocking=pin)
            xi = xi.to(DEVICE, non_blocking=pin).float()
            xt = xt.to(DEVICE, non_blocking=pin)
            y = y.to(DEVICE, non_blocking=pin)
            opt.zero_grad()
            pred = model(num_x=xn, img_x=xi, txt_x=xt)
            loss = mse(pred, y)
            loss.backward()
            opt.step()
            n_samples += xn.shape[0]
        torch.cuda.synchronize() if DEVICE == "cuda" else None
        dt = time.perf_counter() - t0
        del loader, it
        return n_samples / dt

    configs = [
        (32,   0, False, "현행 (batch 32, workers 0)"),
        (32,   8, True,  "batch 32,  workers 8 + pin"),
        (128,  8, True,  "batch 128, workers 8 + pin"),
        (512,  8, True,  "batch 512, workers 8 + pin"),
        (1024, 8, True,  "batch 1024, workers 8 + pin"),
        (1024, 12, True, "batch 1024, workers 12 + pin"),
    ]
    base = None
    print(f"{'설정':<34} | {'표본/초':>12} | {'배속':>7} | {'에폭 환산':>10}")
    print("-" * 88)
    for bs, w, pin, name in configs:
        try:
            tp = bench(bs, w, pin)
        except Exception as e:
            print(f"{name:<34} | 실패: {e}")
            continue
        if base is None:
            base = tp
        epoch_min = (FULL_N * 0.8) / tp / 60
        print(f"{name:<34} | {tp:>12,.0f} | {tp/base:>6.2f}x | {epoch_min:>9.2f}분")


if __name__ == "__main__":
    main()
