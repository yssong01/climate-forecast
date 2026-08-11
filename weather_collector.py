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
import threading
import requests
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# ASOS 관측 시각은 KST 기준이다. datetime.now()는 시스템 로컬 시간대를
# 따르는데, Docker 컨테이너는 기본값이 UTC라 이 저장소를 컨테이너에서
# 돌리면 실제 KST보다 9시간 뒤처진 시각을 "현재"로 계산해 API에 요청하게
# 된다 — 그 시각도 이미 지난 관측이라 API가 정상 응답을 주는 경우가 많아
# 조용히 9시간 묵은 데이터를 "실시간"으로 표시하는 버그가 된다. 컨테이너의
# TZ 설정에 의존하지 않도록 여기서 시간대를 명시한다.
KST = ZoneInfo("Asia/Seoul")

load_dotenv()

# apihub.kma.go.kr ASOS 지상시간관측 (매시 정각 관측, 약 10분 후 공개)
ASOS_URL = "https://apihub.kma.go.kr/api/typ01/url/kma_sfctm2.php"

# ── API 호출 예산 ──────────────────────────────────────────────────────
#
# 이 키는 누적 약 9,800건 시점에 차단된 이력이 있다(CLAUDE.md 4절). 로컬
# 실행에서는 호출량이 사람의 조작 빈도에 묶여 있어 문제가 안 됐지만, 공개
# URL로 배포하면 "접속자 수 × 페이지 갱신 주기"로 늘어나 규모가 달라진다.
# 그래서 프로세스 차원의 하루 상한을 둔다.
#
# 상한에 걸려도 예외를 던지지 않고 _fallback() 으로 내려간다 — 호출부가
# status(FALLBACK_CACHED / FALLBACK_DEFAULT)로 감지할 수 있고, app.py 는 그
# 상태를 화면 상단에 그대로 표시하므로 "묵은 값이 실측으로 위장"되지 않는다.
#
# 카운터는 프로세스 메모리에만 있다 — 재시작하면 0으로 돌아가고, 여러
# 프로세스가 같은 키를 쓰면 합산되지 않는다. 정확한 회계가 아니라 폭주
# 방지용 안전장치다. (정확한 누적치가 필요하면 발급처 콘솔이 원본이다.)
DAILY_CALL_BUDGET = int(os.getenv("KMA_DAILY_CALL_BUDGET", "3000"))

_budget_lock = threading.Lock()
_call_stats = {"date": None, "count": 0, "blocked": 0}


def _reserve_call() -> bool:
    """호출 1건을 예산에서 차감. 남아 있으면 True, 상한 초과면 False."""
    today = datetime.now(KST).strftime("%Y%m%d")
    with _budget_lock:
        if _call_stats["date"] != today:
            _call_stats.update(date=today, count=0, blocked=0)
        if DAILY_CALL_BUDGET > 0 and _call_stats["count"] >= DAILY_CALL_BUDGET:
            _call_stats["blocked"] += 1
            return False
        _call_stats["count"] += 1
        return True


def api_call_stats() -> dict:
    """
    오늘(KST) 이 프로세스가 쓴 API 호출 수. app.py 사이드바 표시용.
    반환: {"date","count","blocked","budget"}
    """
    with _budget_lock:
        return dict(_call_stats, budget=DAILY_CALL_BUDGET)


def _empty_retry_wait() -> float:
    """
    응답은 200인데 관측 레코드가 아직 비어 있을 때 재시도 전 대기 시간(초).

    수집 스크립트(collect_*.py)는 60초를 기다렸다 다시 받는 게 맞다 —
    배치라 지연을 감수할 수 있고, 한 번 못 받은 시각은 영영 구멍으로 남는다.
    반면 대시보드는 이 대기가 치명적이다: predict() 가 관측소 12곳을 순차
    조회하므로 최악의 경우 12 × 60초 × (재시도 2회) = 24분간 화면이 멈춘다.
    실제로 걸릴 수 있는 구간이 있다 — _obs_time() 은 매시 10분부터 그 시각
    관측을 요청하는데, 발표가 1~2분 늦으면 정확히 이 경로를 탄다.
    그래서 서빙(app.py)은 KMA_EMPTY_RETRY_WAIT=0 으로 대기를 없애고 즉시
    폴백시킨다. 호출 시점에 읽으므로 import 순서에 의존하지 않는다.
    """
    try:
        return max(0.0, float(os.getenv("KMA_EMPTY_RETRY_WAIT", "60")))
    except ValueError:
        return 60.0

# 주요 관측소 번호 참고
STATIONS = {
    "서울": "108", "인천": "112", "수원": "119", "춘천": "101",
    "강릉": "105", "대전": "133", "청주": "131", "전주": "146",
    "광주": "156", "대구": "143", "부산": "159", "제주": "184",
}

# 관측소별 위도·경도 — Z축 입력 특성(지역 구분)에 사용.
# 값의 정밀도는 특성 스케일 목적으로 충분하며 항법용이 아니다.
STATION_COORDS = {
    "108": (37.5714, 126.9658),  # 서울
    "112": (37.4776, 126.6249),  # 인천
    "119": (37.2569, 126.9827),  # 수원
    "101": (37.9026, 127.7357),  # 춘천
    "105": (37.7514, 128.8911),  # 강릉
    "133": (36.3721, 127.3720),  # 대전
    "131": (36.6392, 127.4407),  # 청주
    "146": (35.8410, 127.1191),  # 전주
    "156": (35.1729, 126.8916),  # 광주
    "143": (35.8858, 128.6531),  # 대구
    "159": (35.1047, 129.0320),  # 부산
    "184": (33.5141, 126.5297),  # 제주
}
_COORD_DEFAULT = (36.5, 127.5)   # 미등록 관측소 fallback (한반도 중심 근사)


class RobustWeatherCollector:

    def __init__(self, api_key: str = None, stn: str = None, cache_dir: str = "./cache"):
        self.api_key = api_key or os.getenv("KMA_API_KEY", "")
        # KMA_STN: 관측소 번호 (기본 108=서울)
        self.stn = str(stn or os.getenv("KMA_STN", "108"))
        self.cache_dir = cache_dir
        # 폴백 캐시는 관측소별로 나눈다 — 왜 그래야 하는지는
        # _load_disk_cache() 주석 참고. hash_file 도 같은 이유로 나눈다
        # (_is_duplicate 가 timestamp 만으로 키를 만들어, 파일을 공유하면
        # 같은 시각의 다른 관측소 응답이 서로 중복으로 판정된다. 현재
        # _is_duplicate 는 호출되는 곳이 없어 이 파일은 생성되지 않지만,
        # 나중에 연결할 때 같은 함정을 밟지 않도록 지금 갈라둔다).
        self.cache_file = os.path.join(cache_dir, f"latest_weather_{self.stn}.json")
        self.hash_file  = os.path.join(cache_dir, f"content_hashes_{self.stn}.json")
        # 관측소 구분이 없던 시절의 공유 파일. stn 이 일치할 때만 이어받는다.
        self.legacy_cache_file = os.path.join(cache_dir, "latest_weather.json")
        self._memory_cache: dict | None = None
        self._content_hashes: dict = {}

        os.makedirs(self.cache_dir, exist_ok=True)
        self._load_disk_cache()
        self._load_hashes()

    # ── 캐시 관리 ─────────────────────────────────────────────────

    def _load_disk_cache(self):
        """
        이 관측소의 마지막 성공 응답을 읽어 폴백용으로 들고 있는다.

        예전에는 관측소 구분 없이 cache/latest_weather.json 하나를 12개
        관측소가 공유했다. 그러면 A관측소 조회가 성공해 파일을 덮어쓴 뒤
        B관측소 조회가 실패했을 때, A의 관측값이 A의 stn 을 그대로 달고
        B의 폴백으로 돌아온다 — 화면은 "B관측소"라고 표시하는데
        record_to_vec(train.py:166)은 레코드의 stn 으로 위경도를 뽑으므로
        A의 좌표가 조용히 모델 입력에 들어간다(2026-08-11 확인).
        app.py 는 12개 관측소를 매 예보마다 순차 조회하므로 이 교차오염은
        드문 경우가 아니라 "일부 조회 실패 = 곧바로 발생"이었다.

        구 공유 파일은 stn 이 일치할 때만 이어받는다. 남의 관측값으로
        결측을 메우지 않는다는 원칙(CLAUDE.md 4절)의 같은 적용이다.
        """
        self._memory_cache = None
        for path in (self.cache_file, self.legacy_cache_file):
            if not os.path.exists(path):
                continue
            try:
                with open(path, "r", encoding="utf-8") as f:
                    data = json.load(f)
            except Exception:
                continue
            if str(data.get("stn")) != self.stn:
                continue
            self._memory_cache = data
            return

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

    def fetch_at(self, tm: str, retries: int = 2, timeout: int = 10) -> dict:
        """
        특정 시각의 ASOS 관측 데이터 수집 (과거 데이터 수집용).
        tm 형식: YYYYMMDDHHmm (예: 202608062200)
        """
        if not self.api_key or self.api_key == "여기에_발급받은_API_키_입력":
            return self._fallback(tm)

        params = {"tm": tm, "stn": self.stn, "help": "1", "authKey": self.api_key}

        for attempt in range(1, retries + 1):
            if not _reserve_call():
                print(f"[WARN] 일일 API 예산({DAILY_CALL_BUDGET}건) 초과 — "
                      f"fetch_at {tm} 은 폴백으로 처리")
                return self._fallback(tm)
            try:
                resp = requests.get(ASOS_URL, params=params, timeout=timeout)
                if resp.status_code == 200:
                    parsed = self._parse_asos(resp.text, tm)
                    if parsed:
                        parsed["status"] = "SUCCESS_LIVE"
                        return parsed
            except Exception as e:
                if attempt == retries:
                    print(f"[WARN] fetch_at {tm} 실패: {e}")
            if attempt < retries:
                time.sleep(0.5)

        return self._fallback(tm)

    @staticmethod
    def _obs_time() -> str:
        """
        ASOS 기준 시각 (KST). 관측은 매시 정각, 약 10분 후 공개.
        현재 분이 10분 미만이면 전 시각 기준.
        반환 형식: YYYYMMDDHHmm (예: 202608062200)
        """
        now = datetime.now(KST)
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
            if not _reserve_call():
                print(f"[WARN] 일일 API 예산({DAILY_CALL_BUDGET}건) 초과 — "
                      f"관측소 {self.stn} 은 폴백으로 처리")
                return self._fallback(tm)
            try:
                resp = requests.get(ASOS_URL, params=params, timeout=timeout)
                if resp.status_code == 200:
                    parsed = self._parse_asos(resp.text, tm)
                    if parsed:
                        self._save_disk_cache(parsed)
                        parsed["status"] = "SUCCESS_LIVE"
                        return parsed
                    # 데이터가 아직 없으면 잠시 후 재시도. 대기 시간은
                    # _empty_retry_wait() 참고 — 서빙에서는 0으로 두어
                    # 화면이 수 분간 멈추는 것을 막는다.
                    wait = _empty_retry_wait()
                    if wait <= 0:
                        print(f"[WARN] 관측 데이터 미준비 (시도 {attempt}/{retries}) "
                              f"— 대기 없이 폴백")
                        return self._fallback(tm)
                    print(f"[WARN] 관측 데이터 미준비 (시도 {attempt}/{retries}), "
                          f"{wait:.0f}초 대기")
                    time.sleep(wait)
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

        def sf_or_none(key: str):
            """결측(-9 이하 또는 파싱 불가)이면 None — 기본값으로 채워
            '그럴듯한 가짜 관측'을 만들지 않는다. import_kma_fileset.py
            의 _sf_or_none 과 같은 이유(2026-08-07 실측: 결측이 기본값
            20.0°C 등으로 위장돼 학습 데이터에 섞여 들어갔던 사고)."""
            try:
                v = float(d.get(key, "-9"))
                return None if v <= -8 else v
            except (ValueError, TypeError):
                return None

        temp, humid, wind, pres = (sf_or_none("TA"), sf_or_none("HM"),
                                   sf_or_none("WS"), sf_or_none("PS"))
        if None in (temp, humid, wind, pres):
            return None

        return {
            "timestamp":     tm,
            "temperature":   temp,
            "precipitation": max(0.0, sf("RN", 0.0)),
            "humidity":      humid,
            "wind_speed":    wind,
            "wind_dir":      sf("WD",   0.0),
            "pressure":      pres,
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
        12차원 수치 벡터 반환 — train.record_to_vec 와 순서가 반드시 같아야 한다.
        [기온, 강수량, 습도, 풍속, 풍향sin, 풍향cos, 기압, 강수형태,
         시각sin, 시각cos, 위도, 경도]
        """
        wd_rad = np.deg2rad(data.get("wind_dir", 0.0))

        ts = str(data.get("timestamp", ""))      # YYYYMMDDHHmm
        hour = int(ts[8:10]) if len(ts) >= 10 else 0
        h_rad = 2.0 * np.pi * hour / 24.0

        lat, lon = STATION_COORDS.get(str(data.get("stn", self.stn)), _COORD_DEFAULT)

        return [
            data.get("temperature",   20.0),
            data.get("precipitation",  0.0),
            data.get("humidity",      50.0),
            data.get("wind_speed",     1.5),
            float(np.sin(wd_rad)),
            float(np.cos(wd_rad)),
            data.get("pressure",    1013.0),
            float(data.get("precip_type", 0)),
            float(np.sin(h_rad)),
            float(np.cos(h_rad)),
            lat,
            lon,
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
    print(f"\n모델 입력 벡터 (12차원): {[round(v, 3) for v in vec]}")
