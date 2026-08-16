"""
verify_parallel_reaxis.py — interp_field_collector.get_batch() 병렬화 검증.

1) 직렬(n_workers=1) 결과와 병렬(n_workers=N) 결과가 정확히 같은지(정확도
   손해 없음 확인).
2) 표본 규모별 처리 시간 — 병렬화가 실제로 빨라지는지, 어느 표본 수부터
   이득이 나는지.

실행: python verify_parallel_reaxis.py
"""
import json
import os
import time

import numpy as np

from train import DATA_CACHE
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector


def main():
    with open(DATA_CACHE, encoding="utf-8") as f:
        records = json.load(f)
    print(f"레코드 {len(records):,}개 로드, CPU {os.cpu_count()}코어\n")

    sat = InterpolatedFieldCollector(records, STATION_COORDS)

    # ── 1) 정확도 검증 ────────────────────────────────────────────
    probe = records[:5000]
    serial = sat.get_batch(probe, n_workers=1)
    parallel = sat.get_batch(probe, n_workers=8)
    same = np.array_equal(serial, parallel)
    print(f"정확도 검증(5,000개): 직렬==병렬 → {same}")
    if not same:
        diff = np.abs(serial.astype(np.float32) - parallel.astype(np.float32))
        print(f"  [실패] 최대 절대오차 {diff.max():.6f}")
        raise SystemExit(1)

    # ── 2) 속도 비교 ─────────────────────────────────────────────
    print(f"\n{'표본 수':>10} | {'직렬(초)':>10} | {'8워커(초)':>10} | {'배속':>6}")
    print("-" * 50)
    for n in (5_000, 20_000, 100_000):
        sub = records[:n]
        t0 = time.perf_counter()
        sat.get_batch(sub, n_workers=1)
        t_serial = time.perf_counter() - t0

        t0 = time.perf_counter()
        sat.get_batch(sub, n_workers=8)
        t_parallel = time.perf_counter() - t0

        print(f"{n:>10,} | {t_serial:>10.2f} | {t_parallel:>10.2f} | "
              f"{t_serial/t_parallel:>5.2f}x")

    # 전체 규모(138만개) 환산
    full_n = len(records)
    t0 = time.perf_counter()
    sat.get_batch(records[:100_000], n_workers=8)
    t_100k = time.perf_counter() - t0
    print(f"\n전체({full_n:,}개) 환산 예상: {t_100k * full_n / 100_000 / 60:.1f}분 "
          f"(기존 실측 약 8분)")


if __name__ == "__main__":
    main()
