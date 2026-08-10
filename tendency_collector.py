"""
tendency_collector.py — Im축을 "MiniLM 텍스트"에서 "시간 경향 벡터"로 교체.

왜: 기존 SimulatedTextCollector 는 대상 레코드의 스칼라(기온·습도·풍속·기압·
강수)를 이산 버킷("매우 더운 날씨", "강한 비")으로 뭉갠 뒤 문장으로 만들어
MiniLM 에 태운 것이다. 즉 Im축의 정보는 Z축 스칼라에서 결정론적으로 유도되며,
정보이론적으로 Z를 넘을 수 없다(오히려 버킷팅으로 정밀도가 깎인다). 실측에서도
Im축 제거 시 손해가 나긴 했지만 그건 "새 정보"가 아니라 별도 학습경로가 주는
추가 비선형 용량(암묵적 앙상블) 때문으로 해석된다.

대신 단일 스냅샷에서 **유도 불가능한** 정보를 넣는다: 시간 미분(경향).
  ∂s/∂t 는 s(x₀,t) 하나만으로는 절대 계산할 수 없다 → Z와 진짜로 독립.

이게 왜 예보에 중요한가:
  · 기압 하강률은 저기압 접근의 고전적 선행지표(호우·강풍)
  · 기온 하강률은 복사냉각 → 이슬점 접근(안개) 과정 자체
  · 풍속·습도 변화는 기단 교체 신호
현재 모델은 이 정보를 **아예 못 본다** — 단일 시점 t 스냅샷만 입력이라서.

부수 효과: sentence-transformers(약 450MB) 의존성이 서빙 경로에서 사라진다.
Streamlit Community Cloud RAM 1GB 제약의 지배적 원인이었다(실측 1.06GiB).

출력 차원: 4개 변수 × 3개 시차(1h/3h/6h) = 12
결측 처리: 해당 시차의 과거 관측이 없으면 0(= "변화 정보 없음"). 실측상
  323,249개 중 페어 결손이 404개뿐이라 이 경우는 드물다.
"""
import numpy as np
from datetime import datetime, timedelta

LAGS_HOURS = (1, 3, 6)
TENDENCY_VARS = ("temperature", "pressure", "humidity", "wind_speed")
TENDENCY_DIM = len(TENDENCY_VARS) * len(LAGS_HOURS)   # 12

# 변화량 정규화 스케일 — 6시간 내 통상 변동폭 기준(단위별 크기 차이를 없애
# 인코더가 특정 변수에 끌려가지 않게 한다). 고정 상수지만 물리 단위의 성질에서
# 오는 값이라 데이터 의존 파라미터가 아니다.
_SCALE = {
    "temperature": 10.0,   # °C
    "pressure":    10.0,   # hPa
    "humidity":    30.0,   # %
    "wind_speed":   5.0,   # m/s
}


def _parse_ts(ts) -> datetime:
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


class TendencyCollector:
    """
    (관측소, 시각) 인덱스를 미리 만들어두고, 각 레코드마다 과거 1/3/6시간 전
    관측과의 차분을 계산해 12차원 경향 벡터를 만든다.

    SimulatedTextCollector 와 동일한 인터페이스(get_batch/encode_single)를
    제공해 WeatherDataset·predict.py 를 그대로 재사용한다.
    """

    def __init__(self, records: list[dict]):
        self.by_key: dict[tuple[str, str], dict] = {}
        for r in records:
            key = (str(r.get("stn")), str(r["timestamp"])[:12])
            self.by_key[key] = r

    def _vector(self, record: dict) -> np.ndarray:
        stn = str(record.get("stn"))
        try:
            t = _parse_ts(record["timestamp"])
        except (ValueError, KeyError):
            return np.zeros(TENDENCY_DIM, dtype=np.float32)

        out = []
        for lag in LAGS_HOURS:
            past_ts = (t - timedelta(hours=lag)).strftime("%Y%m%d%H%M")
            past = self.by_key.get((stn, past_ts))
            for var in TENDENCY_VARS:
                if past is None:
                    out.append(0.0)
                    continue
                now_v = record.get(var)
                past_v = past.get(var)
                if now_v is None or past_v is None:
                    out.append(0.0)
                    continue
                out.append(float(now_v - past_v) / _SCALE[var])
        return np.array(out, dtype=np.float32)

    def encode_single(self, record: dict) -> np.ndarray:
        """단일 레코드 → (12,). predict.py 실시간 추론 경로."""
        return self._vector(record)

    def get_batch(self, records: list) -> np.ndarray:
        """레코드 리스트 → (N, 12). 캐시 파일을 쓰지 않는다 — 계산이 이미 싸다."""
        return np.stack([self._vector(r) for r in records], axis=0)
