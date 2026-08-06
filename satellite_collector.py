"""
satellite_collector.py — Phase 3-2: 위성 이미지 수집기

Phase 3-2 (현재): 시뮬레이션 — 기상 조건 기반 합성 이미지
Phase 3-실제 (B안): Copernicus Sentinel-2 API 연결

시뮬레이션 채널 설계 (Sentinel-2 밴드 근사):
  채널 0 (Red)   : 기온  → 지표 열복사 패턴
  채널 1 (Green) : 습도  → 식생·수분 반응
  채널 2 (Blue)  : 기압  → 대기 산란·안정도
  채널 3 (NIR)   : 풍속  → 운동 에너지 분포
  + 강수량 > 0   : 전체 밝기 +0.3 (구름 반사율 근사)

재현성: 타임스탬프 SHA-256 시드 → 같은 레코드 = 항상 같은 이미지
"""
import hashlib
import numpy as np

IMG_SIZE = 32   # 32×32 픽셀 (Sentinel-2 실데이터 교체 시 64~128로 확장)
N_BANDS  = 4    # Red, Green, Blue, NIR


class SimulatedSatelliteCollector:
    """
    ASOS 관측 레코드 → 합성 위성 이미지 (4채널, 32×32).
    타임스탬프 기반 SHA-256 시드로 재현 가능한 공간 패턴 생성.
    """

    def get_image(self, record: dict) -> np.ndarray:
        """
        record : weather_collector 레코드 (timestamp, temperature 등 포함)
        반환   : float32 ndarray, shape (4, 32, 32), 값 [0, 1]
        """
        seed_str = record.get("timestamp", "00000000000")
        seed = int(hashlib.sha256(seed_str.encode()).hexdigest()[:8], 16) % (2**31)
        rng = np.random.RandomState(seed)

        # 기상 특성 → [0, 1] 정규화
        temp  = float(np.clip((record.get("temperature",   20.0) + 10) / 60, 0, 1))
        humid = float(np.clip( record.get("humidity",      50.0) / 100, 0, 1))
        pres  = float(np.clip((record.get("pressure",    1013.0) - 980) / 60, 0, 1))
        wind  = float(np.clip( record.get("wind_speed",    1.5) / 20,  0, 1))
        rain  = float(np.clip( record.get("precipitation", 0.0) / 10,  0, 1))

        signals = [temp, humid, pres, wind]
        channels = []
        for sig in signals:
            # 저주파 공간 패턴 (Perlin 근사): 8×8 블록을 업샘플링
            coarse = rng.rand(8, 8).astype(np.float32)
            fine   = rng.rand(IMG_SIZE, IMG_SIZE).astype(np.float32) * 0.2

            # 단순 bilinear 업샘플 (numpy)
            row_idx = np.linspace(0, 7, IMG_SIZE)
            col_idx = np.linspace(0, 7, IMG_SIZE)
            ri, ci  = np.floor(row_idx).astype(int), np.floor(col_idx).astype(int)
            ri = np.clip(ri, 0, 6); ci = np.clip(ci, 0, 6)
            rf, cf  = row_idx - ri, col_idx - ci
            spatial = (coarse[ri, :][:, ci] * (1 - rf[:, None]) * (1 - cf[None, :])
                     + coarse[ri+1, :][:, ci] * rf[:, None] * (1 - cf[None, :])
                     + coarse[ri, :][:, ci+1] * (1 - rf[:, None]) * cf[None, :]
                     + coarse[ri+1, :][:, ci+1] * rf[:, None] * cf[None, :])

            ch = np.clip(sig * spatial + fine + rain * 0.3, 0, 1).astype(np.float32)
            channels.append(ch)

        return np.stack(channels, axis=0)   # (4, 32, 32)

    def get_batch(self, records: list) -> np.ndarray:
        """레코드 리스트 → 배치 이미지 (N, 4, 32, 32)."""
        return np.stack([self.get_image(r) for r in records], axis=0)
