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

채널 설계(SimulatedSatelliteCollector와 동일 물리량 순서 + 강수 채널 추가):
  채널 0: 기온   채널 1: 습도   채널 2: 기압   채널 3: 풍속   채널 4: 강수(선택)

n_bands=4(기본값)면 강수 채널 없이 구버전과 완전히 동일하게 동작한다.
강수 채널(n_bands=5)은 시도했다가 기각했다(2026-08-30) — 아래 참조.

**강수 채널을 시도했다가 기각한 경위.** data_expansion_probe.py 의 절제
실험에서 "이웃 현재값"(강수 포함) 단계의 증분이 ΔAUC +0.0254 로 국지
경향 다음으로 컸는데, 그 신호가 이 4채널 격자에는 전혀 반영되지 않고
있었다 — 강수는 채널 스펙에 없었다. 이 공백을 메우려 5채널로 늘려
대조 실험(동일 SEED=42, group split, RE_CHANNELS 만 변경)을 돌렸으나,
전체 파이프라인에서는 기온 MAE 만 개선(1.3219→1.2644°C)되고 강수·폭염·
한파·황사 4개 지표가 전부 악화됐다(강수 -16.0%→-17.4%, 폭염 F1
0.803→0.792, 한파 0.426→0.410, 황사 0.176→0.151). 단순 프로브의 ΔAUC
가 예측한 이득은 실제로는 기온 축에만 나타났다. 이득 1개 대 손해 4개라
기각. train.py 의 RE_CHANNELS 기본값은 4로 되돌렸다 — n_bands=5 경로는
재현 가능하도록 코드에 남겨둔다.
"""
import os
import multiprocessing as mp

import numpy as np

IMG_SIZE = 32
N_BANDS = 4   # 구버전 기본값 — 하위 호환을 위해 그대로 둔다
IDW_POWER = 2.0
GRID_HALF_SPAN_DEG = 3.0   # 대상 관측소 중심 ±3도(≈300km) 격자


def _norm_temp(v):  return float(np.clip((v + 10) / 60, 0, 1))
def _norm_humid(v): return float(np.clip(v / 100, 0, 1))
def _norm_pres(v):  return float(np.clip((v - 980) / 60, 0, 1))
def _norm_wind(v):  return float(np.clip(v / 20, 0, 1))
def _norm_precip(v):
    # 선형 정규화는 대부분 0 근처에 몰리는 강수 분포에서 약한 비를 뭉갠다
    # (historical_data_1y.json 실측: p99=4.0mm, p99.9=17.0mm, p99.99=39.9mm,
    # 최대 86.2mm — 오른쪽으로 크게 치우침). sqrt 압축으로 저·중강도 구간의
    # 해상도를 넓히고, 50mm/h 이상(상위 0.1% 미만)만 포화시킨다.
    return float(np.clip(np.sqrt(max(v, 0.0)) / np.sqrt(50.0), 0, 1))

# 새 채널은 끝에 붙인다(기존 배열 인덱스가 안 밀리도록) — n_bands 로 앞에서부터
# 몇 개를 쓸지 고른다.
_CHANNEL_SPEC_FULL = [
    ("temperature",   _norm_temp),
    ("humidity",      _norm_humid),
    ("pressure",      _norm_pres),
    ("wind_speed",    _norm_wind),
    ("precipitation", _norm_precip),
]


# get_batch() 의 프로세스 풀 워커용 — fork 자식 프로세스에서만 채워진다.
# 모듈 전역인 이유: Pool.imap 에 넘길 함수가 최상위(top-level)여야 pickling
# 없이 fork 로 전달된다(바운드 메서드를 직접 넘기면 인스턴스 전체를 매
# 태스크마다 pickle 하려 든다).
_WORKER_COLLECTOR = None


def _init_worker(collector):
    global _WORKER_COLLECTOR
    _WORKER_COLLECTOR = collector


def _get_image_chunk_worker(keys):
    """
    (관측소, 시각) 키 묶음 하나를 받아 float16 블록으로 돌려준다.

    레코드 dict가 아니라 키 튜플을 받는 이유 — 워커는 fork로 by_time 전체를
    이미 갖고 있어 dict를 다시 받을 필요가 없다. 태스크 인자는 fork와 달리
    매번 pickle되므로, 9개 키짜리 dict 대신 짧은 문자열 2개만 넘긴다.

    낱개가 아니라 묶음으로 주고받는 이유 — 반환값이 진짜 병목이었다.
    (4,32,32) float32는 16KB이고 138만 개면 22GB가 파이프를 오간다.
    float16 블록으로 묶으면 전송량이 1/2이 되고 전송 횟수도 청크 크기만큼
    줄어든다(2026-08-16 실측: 16워커인데 단일 대비 4배에 그치던 원인).
    """
    c = _WORKER_COLLECTOR
    out = np.empty((len(keys), c.n_bands, IMG_SIZE, IMG_SIZE), dtype=np.float16)
    for i, (stn, ts) in enumerate(keys):
        out[i] = c.get_image_by_key(stn, ts)
    return out


class InterpolatedFieldCollector:
    """
    records 전체(모든 관측소·모든 시각)로 (시각)→{관측소: 레코드} 인덱스를
    미리 만들어두고, get_image() 호출 시 대상 관측소를 뺀 나머지 관측소들의
    같은 시각 값을 IDW 로 격자에 보간한다.
    """

    # 가중치 캐시 상한 — 중심 관측소 12개 × 이웃 조합이라 정상 데이터에서는
    # 수십 개면 충분하다. 결측 패턴이 심해 조합이 폭증하는 비정상 입력에서
    # 메모리를 무한정 먹지 않도록 막아둔다(엔트리당 약 90KB).
    _W_CACHE_MAX = 512

    def __init__(self, records: list[dict], station_coords: dict, n_bands: int = N_BANDS):
        self.coords = station_coords
        self.n_bands = n_bands
        self.channel_spec = _CHANNEL_SPEC_FULL[:n_bands]
        self.by_time: dict[str, dict[str, dict]] = {}
        for r in records:
            ts = str(r["timestamp"])[:12]
            self.by_time.setdefault(ts, {})[str(r.get("stn"))] = r
        # (중심 관측소, 이웃 집합) → (가중치, 가중치합). 아래 _weights 참고.
        self._w_cache: dict[tuple, tuple] = {}

    def _weights(self, stn: str, neighbor_stns: tuple):
        """
        IDW 가중치와 그 합을 (중심 관측소, 이웃 집합)별로 한 번만 계산한다.

        격자 좌표·이웃까지의 거리·가중치는 관측값에 전혀 의존하지 않는다 —
        오직 중심 관측소가 어디이고 이웃이 누구인지에만 달려 있다. 그런데
        종전 구현은 이걸 레코드마다 다시 계산했다. 138만 표본이면 실제로
        필요한 계산은 수십 번인데 138만 번을 한 셈이다(2026-08-16 실측:
        단일 스레드 8.1분 중 대부분).

        이웃 집합을 키에 포함하는 이유 — 어떤 시각에 일부 관측소가 결측이면
        이웃 목록이 달라지고 가중치도 달라진다. 중심 관측소만으로 키를 잡으면
        그 경우에 틀린 가중치를 재사용하게 된다.
        """
        key = (stn, neighbor_stns)
        hit = self._w_cache.get(key)
        if hit is not None:
            return hit

        center = self.coords.get(stn, (36.5, 127.5))
        lons = np.linspace(center[1] - GRID_HALF_SPAN_DEG,
                           center[1] + GRID_HALF_SPAN_DEG, IMG_SIZE)
        lats = np.linspace(center[0] + GRID_HALF_SPAN_DEG,
                           center[0] - GRID_HALF_SPAN_DEG, IMG_SIZE)
        lon_grid, lat_grid = np.meshgrid(lons, lats)   # (32,32) 각각

        n_lats = np.array([self.coords[s][0] for s in neighbor_stns])
        n_lons = np.array([self.coords[s][1] for s in neighbor_stns])
        d2 = ((lat_grid[..., None] - n_lats[None, None, :]) ** 2
             + (lon_grid[..., None] - n_lons[None, None, :]) ** 2)
        w = 1.0 / (np.sqrt(d2) ** IDW_POWER + 1e-6)   # (32,32,n_neighbors)
        w_sum = w.sum(axis=-1)

        if len(self._w_cache) < self._W_CACHE_MAX:
            self._w_cache[key] = (w, w_sum)
        return w, w_sum

    def get_image(self, record: dict) -> np.ndarray:
        return self.get_image_by_key(str(record.get("stn")),
                                     str(record["timestamp"])[:12])

    def get_image_by_key(self, stn: str, ts: str) -> np.ndarray:
        """레코드 dict 없이 (관측소, 시각)만으로 보간장을 만든다 — 계산에
        실제로 필요한 건 이 둘뿐이다. 프로세스 풀에 dict를 pickle해 넘기는
        비용을 없애기 위해 분리했다(`_get_image_chunk_worker` 참고)."""
        snapshot = self.by_time.get(ts, {})

        # 캐시 키로 쓰므로 튜플(해시 가능)로 만든다. dict 순회 순서는 삽입
        # 순서로 고정되어 있어 같은 이웃 집합이면 같은 튜플이 나온다.
        neighbor_stns = tuple(s for s in snapshot
                              if s != stn and s in self.coords)

        if not neighbor_stns:
            # 같은 시각 이웃 관측 자체가 없는 드문 경우 — 노이즈가 아니라
            # "정보 없음"을 뜻하는 중립값(0.5)으로 채운다.
            return np.full((self.n_bands, IMG_SIZE, IMG_SIZE), 0.5, dtype=np.float32)

        w, w_sum = self._weights(stn, neighbor_stns)

        channels = []
        for key, norm_fn in self.channel_spec:
            n_vals = np.array([
                norm_fn(snapshot[s].get(key, 0.0)) for s in neighbor_stns
            ])
            ch = (w * n_vals[None, None, :]).sum(axis=-1) / w_sum
            channels.append(ch.astype(np.float32))

        return np.stack(channels, axis=0)   # (n_bands, 32, 32)

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
        out = np.empty((n, self.n_bands, IMG_SIZE, IMG_SIZE), dtype=np.float16)
        if n_workers <= 1 or n < 20_000:
            for i, r in enumerate(records):
                out[i] = self.get_image(r)
            return out
        # 워커에는 레코드 dict가 아니라 (관측소, 시각) 키만 넘긴다 — 계산에
        # 필요한 건 이 둘뿐이고, 태스크 인자는 매번 pickle되기 때문이다.
        keys = [(str(r.get("stn")), str(r["timestamp"])[:12]) for r in records]
        # 청크 하나가 float16 블록 하나로 돌아온다. 2048개면 블록당 약 16MB로,
        # 전송 횟수를 줄이면서도 워커 간 부하 불균형이 눈에 띄지 않는 크기다.
        chunk = 2048
        chunks = [keys[i:i + chunk] for i in range(0, n, chunk)]
        ctx = mp.get_context("fork")
        pos = 0
        with ctx.Pool(n_workers, initializer=_init_worker, initargs=(self,)) as pool:
            # imap은 입력 순서를 보존한다 — out의 행 순서가 records와 일치해야
            # 라벨·타깃과 어긋나지 않는다.
            for block in pool.imap(_get_image_chunk_worker, chunks):
                out[pos:pos + len(block)] = block
                pos += len(block)
        return out
