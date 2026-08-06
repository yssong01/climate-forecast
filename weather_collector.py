"""
weather_collector.py
기상청 API허브 (apihub.kma.go.kr) ASOS 지상시간관측 수집기.
API 장애 시 2단계 Fallback: 디스크 캐시 → Safe Default.
SHA-256 기반 중복 인코딩 방지 (논문 §V 캐시 설계 적용).
"""
import os
import json
import time
import hashlib
import requests
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

load_dotenv()

# apihub.kma.go.kr ASOS 지상시간관측 (매시 정각 관측, 약 10분 후 공개)
ASOS_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"

# 주요 관측소 번호 참고
STATIONS = {
    "서울": "108", "인천": "112", "수원": "119", "춘천": "101",
    "강릉": "105", "대전": "133", "청주": "131", "전주": "146",
    "광주": "156", "대구": "143", "부산": "159", "제주": "184",
}


class RobustWeatherCollector:

    def __init__(self, api_key: str = None, stn: str = None, cache_dir: str = "./cache"):
        self.api_key = api_key or os.getenv("KMA_API_KEY", "")
        # KMA_STN: 관측소 번호 (기본 108=서울)
        self.stn = str(stn or os.getenv("KMA_STN", "108"))
        self.cache_dir = cache_dir
        self.cache_file = os.path.join(cache_dir, "latest_weather.json")
        self.hash_file  = os.path.join(cache_dir, "content_hashes.json")
        self._memory_cache: dict | None = None
        self._content_hashes: dict = {}

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_disk_cache()
        self._load_hashes()

    # ── 캐시 관리 ─────────────────────────────────────────────────

    def _load_disk_cache(self):
        if os.path.exists(self.cache_file):
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    self._memory_cache = json.load(f)
            except Exception:
                self._memory_cache = None

    def _save_disk_cache(self, data: dict):
        self._memory_cache = data
        try:
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
        except Exception:
            pass

    def _load_hashes(self):
        if os.path.exists(self.hash_file):
            try:
                with open(self.hash_file, "r") as f:
                    self._content_hashes = json.load(f)
            except Exception:
                self._content_hashes = {}

    def _save_hashes(self):
        try:
            with open(self.hash_file, "w") as f:
                json.dump(self._content_hashes, f, indent=2)
        except Exception:
            pass

    def _is_duplicate(self, data: dict) -> bool:
        """SHA-256으로 동일 응답 중복 인코딩 방지 (논문 §V)."""
        key = data.get("timestamp", "")
        content = json.dumps(
            {k: v for k, v in data.items() if k != "status"}, sort_keys=True
        )
        sha = hashlib.sha256(content.encode()).hexdigest()
        if self._content_hashes.get(key) == sha:
            return True
        self._content_hashes[key] = sha
        if len(self._content_hashes) > 1000:
            del self._content_hashes[next(iter(self._content_hashes))]
        self._save_hashes()
        return False

    # ── 시간 계산 ──────────────────────────────────────────────────

    @staticmethod
    def _obs_time() -> str:
        """
        ASOS 기준 시각 (KST). 관측은 매시 정각, 약 10분 후 공개.
        현재 분이 10분 미만이면 전 시각 기준.
        반환 형식: YYYYMMDDHHmm (예: 202608062200)
        """
        now = datetime.now()
        if now.minute < 10:
            now -= timedelta(hours=1)
        return now.strftime("%Y%m%d%H00")

    # ── API 호출 ────────────────────────────────────────────────────

    def fetch(self, retries: int = 3, timeout: int = 10) -> dict:
        """
        ASOS 지상시간관측 수집.
        Returns dict with keys:
          timestamp, temperature, precipitation, humidity,
          wind_speed, wind_dir, pressure, precip_type, stn, status
        """
        if not self.api_key or self.api_key == "여기에_발급받은_API_키_입력":
            print("[WARN] KMA_API_KEY가 설정되지 않았습니다. .env 파일을 확인하세요.")
            return self._fallback()

        tm = self._obs_time()
        params = {
            "tm":      tm,
            "stn":     self.stn,
            "help":    "1",       # 헤더 라인 포함 → 열 이름으로 파싱
            "authKey": self.api_key,
        }

        for attempt in range(1, retries + 1):
            try:
                resp = requests.get(ASOS_URL, params=params, timeout=timeout)
                if resp.status_code == 200:
                    parsed = self._parse_asos(resp.text, tm)
                    if parsed:
                        self._save_disk_cache(parsed)
                        parsed["status"] = "SUCCESS_LIVE"
                        return parsed
                    # 데이터가 아직 없으면 1분 후 재시도
                    print(f"[WARN] 관측 데이터 미준비 (시도 {attempt}/{retries}), 60초 대기")
                    time.sleep(60)
                    continue
                else:
                    print(f"[WARN] HTTP {resp.status_code}: {resp.text[:150]}")
            except requests.exceptions.Timeout:
                print(f"[WARN] 타임아웃 (시도 {attempt}/{retries})")
            except Exception as e:
                print(f"[WARN] 오류 (시도 {attempt}/{retries}): {e}")
            if attempt < retries:
                time.sleep(2)

        return self._fallback(tm)

    # ── 응답 파싱 ──────────────────────────────────────────────────

    def _parse_asos(self, text: str, tm: str) -> dict | None:
        """
        apihub ASOS 텍스트 응답 파싱.
        응답 구조 (help=1):
          #START7777
          # YYYYMMDDHHMM STN WD WS ... TA TD HM ... RN ...
          202608062200  108  270  2.3  ...  25.4  ...  65  ...  0.0  ...
          #7777END
        """
        lines = [ln.strip() for ln in text.splitlines()]
        header_cols: list[str] | None = None
        data_line:   str | None       = None

        for line in lines:
            if not line or line in ("#START7777", "#7777END"):
                continue
            if line.startswith("#"):
                cols = line.lstrip("#").split()
                # 헤더 라인은 첫 열이 시각 형식(YYYY...)
                if cols and cols[0].startswith("YYYY"):
                    header_cols = cols
                continue
            data_line = line  # 마지막 데이터 라인 사용

        if data_line is None:
            # 403 / 활용신청 필요 등 에러 응답
            if "활용신청" in text or "403" in text:
                print("[ERROR] 활용신청이 필요합니다. apihub.kma.go.kr → 마이페이지 → 지상관측 → API활용신청")
            else:
                print(f"[WARN] 데이터 라인 없음. 응답: {text[:200]}")
            return None

        vals = data_line.split()

        # 헤더와 값 개수가 맞으면 이름으로 매핑, 아니면 위치 기반
        if header_cols and len(vals) >= len(header_cols):
            d = dict(zip(header_cols, vals))
        else:
            # 위치 기반 폴백 (kma_sfctm2 기본 열 순서)
            # 0:tm 1:stn 2:WD 3:WS 4:GST_WD 5:GST_WS 6:GST_TM
            # 7:PA 8:PS 9:PT 10:PR 11:TA 12:TD 13:HM 14:PV 15:RN
            if len(vals) < 16:
                print(f"[WARN] 열 개수 부족 ({len(vals)}): {data_line[:100]}")
                return None
            d = {
                "YYYYMMDDHHMM": vals[0], "STN": vals[1],
                "WD": vals[2],  "WS": vals[3],
                "PA": vals[7],  "PS": vals[8],
                "TA": vals[11], "HM": vals[13], "RN": vals[15],
            }

        def sf(key: str, default: float) -> float:
            """안전 float 변환 — -9/−999 등 결측 코드는 default로."""
            try:
                v = float(d.get(key, default))
                return default if v <= -8 else v
            except (ValueError, TypeError):
                return default

        return {
            "timestamp":     tm,
            "temperature":   sf("TA",  20.0),
            "precipitation": max(0.0, sf("RN", 0.0)),
            "humidity":      sf("HM",  50.0),
            "wind_speed":    sf("WS",   1.5),
            "wind_dir":      sf("WD",   0.0),
            "pressure":      sf("PS", 1013.0),
            "precip_type":   0,   # ASOS WW 코드는 추후 매핑
            "stn":           self.stn,
        }

    # ── Fallback ────────────────────────────────────────────────────

    def _fallback(self, tm: str = None) -> dict:
        if tm is None:
            tm = self._obs_time()

        if self._memory_cache is not None:
            fb = self._memory_cache.copy()
            fb["timestamp"] = tm
            fb["status"]    = "FALLBACK_CACHED"
            fb["temperature"] = round(
                fb["temperature"] + float(np.random.uniform(-0.05, 0.05)), 2
            )
            return fb

        return {
            "timestamp": tm, "temperature": 20.0, "precipitation": 0.0,
            "humidity": 50.0, "wind_speed": 1.5, "wind_dir": 0.0,
            "pressure": 1013.0, "precip_type": 0,
            "stn": self.stn, "status": "FALLBACK_DEFAULT",
        }

    # ── 모델 입력 벡터 ──────────────────────────────────────────────

    def to_model_vector(self, data: dict) -> list[float]:
        """
        8차원 수치 벡터 반환 — pipeline_model.py num_enc 입력과 일치.
        [temperature, precipitation, humidity, wind_speed,
         wind_dir_sin, wind_dir_cos, pressure, precip_type]
        """
        wd_rad = np.deg2rad(data.get("wind_dir", 0.0))
        return [
            data.get("temperature",   20.0),
            data.get("precipitation",  0.0),
            data.get("humidity",      50.0),
            data.get("wind_speed",     1.5),
            float(np.sin(wd_rad)),
            float(np.cos(wd_rad)),
            data.get("pressure",    1013.0),
            float(data.get("precip_type", 0)),
        ]


# ── 단독 실행 테스트 ─────────────────────────────────────────────────

if __name__ == "__main__":
    collector = RobustWeatherCollector()
    stn_name = {v: k for k, v in STATIONS.items()}.get(collector.stn, collector.stn)
    print(f"기상청 API허브 연결 테스트 — 관측소: {stn_name} ({collector.stn})")
    print(f"요청 시각: {collector._obs_time()}\n")

    data = collector.fetch()

    print(f"상태:     {data.get('status')}")
    print(f"시각:     {data.get('timestamp')}")
    print(f"기온:     {data.get('temperature')} °C")
    print(f"강수:     {data.get('precipitation')} mm")
    print(f"습도:     {data.get('humidity')} %")
    print(f"풍속:     {data.get('wind_speed')} m/s  풍향: {data.get('wind_dir')}°")
    print(f"기압:     {data.get('pressure')} hPa")

    vec = collector.to_model_vector(data)
    print(f"\n모델 입력 벡터 (8차원): {[round(v, 3) for v in vec]}")
