"""
interp_field_collector.py — Re축을 "가짜 위성"에서 "12개 관측소 실측
공간보간장(場)"으로 교체.

왜: SimulatedSatelliteCollector 는 대상 관측소 자신의 스칼라(기온·습도·
기압·풍속)를 랜덤 노이즈로 흩뿌린 것뿐이라 원리적으로 새 정보가 없다
(deploy_ablation.py 실측, 2026-08-09: Re축 제거해도 결과가 소수점까지
완전히 동일). 이 컬렉터는 같은 시각 "다른" 11개 관측소의 실측값을
역거리가중(IDW)으로 보간해 격자에 얹는다 — 대상 관측소 자신의 값은 절대
넣지 않는다(넣으면 Z축과 다시 중복정보가 되어 같은 문제가 재발한다).

한계(정직하게 명시): 관측소가 전국 12개뿐이라 보간 격자가 성기다. 진짜
위성(천리안 GK2A)만큼 정밀하진 않지만, 순수 노이즈보다는 확실한 정보량을
가진다 — 이웃 관측소의 기압강하·풍향변화 같은 "주변에서 다가오는" 신호를
담을 수 있다.

채널 설계(SimulatedSatelliteCollector와 동일 4채널, 물리량만 다름):
  채널 0: 기온   채널 1: 습도   채널 2: 기압   채널 3: 풍속
"""
import numpy as np

IMG_SIZE = 32
N_BANDS = 4
IDW_POWER = 2.0
GRID_HALF_SPAN_DEG = 3.0   # 대상 관측소 중심 ±3도(≈300km) 격자


def _norm_temp(v):  return float(np.clip((v + 10) / 60, 0, 1))
def _norm_humid(v): return float(np.clip(v / 100, 0, 1))
def _norm_pres(v):  return float(np.clip((v - 980) / 60, 0, 1))
def _norm_wind(v):  return float(np.clip(v / 20, 0, 1))

_CHANNEL_SPEC = [
    ("temperature", _norm_temp),
    ("humidity",    _norm_humid),
    ("pressure",    _norm_pres),
    ("wind_speed",  _norm_wind),
]


class InterpolatedFieldCollector:
    """
    records 전체(모든 관측소·모든 시각)로 (시각)→{관측소: 레코드} 인덱스를
    미리 만들어두고, get_image() 호출 시 대상 관측소를 뺀 나머지 관측소들의
    같은 시각 값을 IDW 로 격자에 보간한다.
    """

    def __init__(self, records: list[dict], station_coords: dict):
        self.coords = station_coords
        self.by_time: dict[str, dict[str, dict]] = {}
        for r in records:
            ts = str(r["timestamp"])[:12]
            self.by_time.setdefault(ts, {})[str(r.get("stn"))] = r

    def get_image(self, record: dict) -> np.ndarray:
        stn = str(record.get("stn"))
        ts = str(record["timestamp"])[:12]
        snapshot = self.by_time.get(ts, {})

        neighbor_stns = [s for s in snapshot
                         if s != stn and s in self.coords]

        center = self.coords.get(stn, (36.5, 127.5))
        lons = np.linspace(center[1] - GRID_HALF_SPAN_DEG,
                           center[1] + GRID_HALF_SPAN_DEG, IMG_SIZE)
        lats = np.linspace(center[0] + GRID_HALF_SPAN_DEG,
                           center[0] - GRID_HALF_SPAN_DEG, IMG_SIZE)
        lon_grid, lat_grid = np.meshgrid(lons, lats)   # (32,32) 각각

        if not neighbor_stns:
            # 같은 시각 이웃 관측 자체가 없는 드문 경우 — 노이즈가 아니라
            # "정보 없음"을 뜻하는 중립값(0.5)으로 채운다.
            return np.full((N_BANDS, IMG_SIZE, IMG_SIZE), 0.5, dtype=np.float32)

        n_lats = np.array([self.coords[s][0] for s in neighbor_stns])
        n_lons = np.array([self.coords[s][1] for s in neighbor_stns])
        d2 = ((lat_grid[..., None] - n_lats[None, None, :]) ** 2
             + (lon_grid[..., None] - n_lons[None, None, :]) ** 2)
        w = 1.0 / (np.sqrt(d2) ** IDW_POWER + 1e-6)   # (32,32,n_neighbors)
        w_sum = w.sum(axis=-1)

        channels = []
        for key, norm_fn in _CHANNEL_SPEC:
            n_vals = np.array([
                norm_fn(snapshot[s].get(key, 0.0)) for s in neighbor_stns
            ])
            ch = (w * n_vals[None, None, :]).sum(axis=-1) / w_sum
            channels.append(ch.astype(np.float32))

        return np.stack(channels, axis=0)   # (4, 32, 32)

    def get_batch(self, records: list) -> np.ndarray:
        return np.stack([self.get_image(r) for r in records], axis=0)
