"""
text_collector.py — Phase 3-3: 기상 텍스트 인코더

Phase 3-3 (현재): 시뮬레이션 — ASOS 레코드 → 기상청 스타일 한국어 예보문 → MiniLM 임베딩
Phase 3-실제    : 기상청 RSS/예보 API 연결 시 record_to_text() 만 교체하면 됨

MiniLM 모델: paraphrase-multilingual-MiniLM-L12-v2
  - 한국어 지원 (50개 언어)
  - 출력 384차원
  - 첫 실행 시 ~450MB 자동 다운로드

캐싱: 타임스탬프 목록이 동일하면 NPZ 파일 재사용 (MiniLM 추론 생략)
"""
import os
import json
import numpy as np

from weather_collector import STATIONS

_STATION_NAMES = {v: k for k, v in STATIONS.items()}

CACHE_FILE = "./cache/text_embeddings.npz"
CACHE_META = "./cache/text_embeddings_meta.json"
MODEL_NAME = "sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2"
TEXT_DIM   = 384


class SimulatedTextCollector:
    """
    ASOS 관측 레코드 → 한국어 기상 예보문 → MiniLM 384차원 임베딩.
    """

    def __init__(self, model_name: str = MODEL_NAME):
        self.model_name = model_name
        self._model = None   # lazy load: 실제 필요 시점에만 로드

    @property
    def model(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            print(f"  MiniLM 모델 로드: {self.model_name}")
            print(f"  (첫 실행 시 ~450MB 다운로드, 이후 캐시 사용)")
            self._model = SentenceTransformer(self.model_name)
        return self._model

    def record_to_text(self, record: dict) -> str:
        """
        ASOS 레코드 → 기상청 스타일 한국어 예보문.
        관측소 이름을 포함해 Im 축이 지역을 구분할 수 있게 한다
        (다중 관측소 확장 이후 필요 — 그 전엔 항상 서울 단일 관측소였다).
        """
        stn_name = _STATION_NAMES.get(str(record.get("stn", "")), "관측소")
        temp  = record.get("temperature",   20.0)
        humid = record.get("humidity",      50.0)
        ws    = record.get("wind_speed",     1.5)
        wd    = record.get("wind_dir",       0.0)
        pres  = record.get("pressure",    1013.0)
        rain  = record.get("precipitation",  0.0)

        # 기온 묘사
        if temp >= 35:
            t_desc = "폭염 경보 수준의 매우 더운 날씨"
        elif temp >= 30:
            t_desc = "매우 더운 날씨"
        elif temp >= 25:
            t_desc = "덥고 불쾌지수 높은 날씨"
        elif temp >= 20:
            t_desc = "따뜻한 날씨"
        elif temp >= 15:
            t_desc = "온화한 날씨"
        elif temp >= 10:
            t_desc = "서늘한 날씨"
        elif temp >= 0:
            t_desc = "쌀쌀한 날씨"
        else:
            t_desc = "영하의 매우 추운 날씨"

        # 강수 묘사
        if rain >= 30:
            r_desc = "매우 강한 비(폭우 수준)"
        elif rain >= 10:
            r_desc = "강한 비"
        elif rain >= 1:
            r_desc = "비"
        elif rain > 0:
            r_desc = "이슬비 또는 약한 비"
        else:
            r_desc = "강수 없음(맑음)"

        # 습도 묘사
        if humid >= 80:
            h_desc = "매우 습함"
        elif humid >= 60:
            h_desc = "습함"
        elif humid >= 40:
            h_desc = "보통"
        else:
            h_desc = "건조함"

        # 기압 상태
        if pres < 1000:
            p_desc = "저기압 접근으로 날씨 악화 예상"
        elif pres < 1008:
            p_desc = "기압 다소 낮음"
        elif pres > 1022:
            p_desc = "고기압 영향으로 맑은 날씨"
        else:
            p_desc = "기압 정상 범위"

        # 풍향 8방위 변환
        dirs = ["북", "북동", "동", "남동", "남", "남서", "서", "북서"]
        wd_name = dirs[int(((wd + 22.5) % 360) / 45)]

        return (
            f"{stn_name} 지역 현재 기온 {temp:.1f}°C로 {t_desc}. "
            f"습도 {humid:.0f}%({h_desc}), {wd_name}풍 {ws:.1f}m/s. "
            f"기압 {pres:.0f}hPa, {p_desc}. "
            f"강수 상황: {r_desc}."
        )

    def encode_single(self, record: dict) -> np.ndarray:
        """
        단일 레코드 → MiniLM 임베딩 (384,).

        get_batch() 와 달리 디스크 캐시(text_embeddings.npz)를 읽거나 쓰지
        않는다. 실시간 추론(predict.py)에서 매 호출마다 새 timestamp 를 쓰면
        캐시가 항상 미스되어, get_batch() 를 쓰면 학습에 쓴 8,546개짜리 캐시를
        1개짜리로 덮어써버린다.
        """
        text = self.record_to_text(record)
        return self.model.encode([text], convert_to_numpy=True)[0].astype(np.float32)

    def get_batch(self, records: list) -> np.ndarray:
        """
        레코드 리스트 → MiniLM 배치 임베딩 (N, 384).
        타임스탬프 목록이 캐시와 동일하면 NPZ 재사용.
        """
        timestamps = [r.get("timestamp", "") for r in records]

        # 캐시 확인
        if os.path.exists(CACHE_FILE) and os.path.exists(CACHE_META):
            with open(CACHE_META, "r", encoding="utf-8") as f:
                cached_ts = json.load(f)
            if cached_ts == timestamps:
                data = np.load(CACHE_FILE)
                print(f"  [텍스트 캐시] {len(records)}개 임베딩 재사용")
                return data["embeddings"].astype(np.float32)

        print(f"  MiniLM 임베딩 생성 중 ({len(records)}개)...")
        texts = [self.record_to_text(r) for r in records]
        embeddings = self.model.encode(
            texts,
            batch_size=64,
            show_progress_bar=True,
            convert_to_numpy=True,
        ).astype(np.float32)   # (N, 384)

        os.makedirs("./cache", exist_ok=True)
        np.savez_compressed(CACHE_FILE, embeddings=embeddings)
        with open(CACHE_META, "w", encoding="utf-8") as f:
            json.dump(timestamps, f)
        print(f"  임베딩 저장 완료: {CACHE_FILE}")

        return embeddings
