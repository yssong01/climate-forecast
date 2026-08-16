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
import os
import multiprocessing as mp

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


# get_batch() 의 프로세스 풀 워커용 — fork 자식 프로세스에서만 채워진다.
# 모듈 전역인 이유: Pool.imap 에 넘길 함수가 최상위(top-level)여야 pickling
# 없이 fork 로 전달된다(바운드 메서드를 직접 넘기면 인스턴스 전체를 매
# 태스크마다 pickle 하려 든다).
_WORKER_COLLECTOR = None


def _init_worker(collector):
    global _WORKER_COLLECTOR
    _WORKER_COLLECTOR = collector


def _get_image_worker(record):
    return _WORKER_COLLECTOR.get_image(record)


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

    def get_batch(self, records: list, n_workers: int = None) -> np.ndarray:
        """float16으로 직접 채운다 — records 개수가 많을 때(13년치, 130만+)
        'float32 리스트 전체 + np.stack 결과'가 동시에 메모리에 떠서 최종
        크기의 2배를 순간적으로 잡아먹는 걸 피하기 위함이다(2026-08-15
        OOM 실측 원인 중 하나). 학습 루프에서 모델에 넣기 직전 다시
        float32로 캐스팅한다.

        표본 138만 개 기준 단일 스레드로 약 8분 걸린다(perf_profile.py
        실측, 32코어 중 1개만 사용). 레코드별로 독립적인 계산(같은 시각
        스냅샷 조회 + IDW)이라 프로세스 풀로 병렬화한다. self.by_time
        (레코드 전체의 시각 인덱스)은 fork 시 COW로 워커에 그대로 공유되므로
        다시 만들거나 프로세스 간에 옮길 필요가 없고, 오가는 건 record
        하나(작은 dict)뿐이다. 표본이 적으면(스모크 테스트 등) 병렬화
        오버헤드가 이득보다 커 단일 스레드로 그대로 처리한다.
        """
        n = len(records)
        if n_workers is None:
            # 8/16/24/30 워커 실측(2026-08-16, 20만 표본): 8→3.9배, 16→4.3배,
            # 그 이상은 거의 안 늘었다(24·30도 16과 오차범위 내). 코어를
            # 더 얹어도 안 빨라지는 지점부터는 다른 프로세스(다른 실험 등)에
            # 코어를 남겨주는 게 낫다.
            n_workers = min(16, max(1, (os.cpu_count() or 4) - 2))
        out = np.empty((n, N_BANDS, IMG_SIZE, IMG_SIZE), dtype=np.float16)
        if n_workers <= 1 or n < 20_000:
            for i, r in enumerate(records):
                out[i] = self.get_image(r)
            return out
        chunksize = max(1, n // (n_workers * 8))
        ctx = mp.get_context("fork")
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(self,)) as pool:
            for i, img in enumerate(pool.imap(_get_image_worker, records, chunksize=chunksize)):
                out[i] = img
        return out
