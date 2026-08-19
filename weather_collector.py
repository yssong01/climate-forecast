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
import functools
import threading
import requests
import numpy as np
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo
from dotenv import load_dotenv

# 이 모듈의 [WARN]/[ERROR] 진단 메시지는 컨테이너 배포 환경(Streamlit
# Community Cloud 등)에서 표준출력이 완전 버퍼링되면 로그 화면에 한참
# 지연되거나(버퍼가 다 찰 때까지) 아예 안 보일 수 있다 — 2026-08-11
# 실측: API 실패가 42건 쌓였는데도 로그 검색에 "WARN"이 전혀 안 잡힘.
# 이 모듈의 print() 만 즉시 flush 하도록 재정의해 진단 지연을 없앤다.
print = functools.partial(print, flush=True)

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


# ── 연결 차단 감지(circuit breaker) ─────────────────────────────────
#
# 2026-08-19 실측: Streamlit Cloud 발신 IP 에서만 apihub.kma.go.kr 로의 TCP
# 연결이 통째로 막혔다(ConnectTimeoutError, 같은 시각 GitHub Actions 와 로컬은
# 정상 응답). 그 상태에서 12개 관측소가 각자 재시도 한도까지 소진하면
# 관측소당 34초 × 12곳 + 과거 3시점 61초 ≈ 469초 — 조회마다 화면이 8분 멈춘다.
# 실패는 캐시되지 않으므로 재실행마다 이 시간을 다시 쓴다.
#
# 연결 자체가 안 되는 상황은 관측소를 바꿔도 재시도를 해도 결과가 같다.
# 한 번 확인되면 쿨다운 동안 나머지 호출을 즉시 폴백으로 내린다.
# ReadTimeout(서버에는 닿았는데 응답이 늦음)은 여기 넣지 않는다 — 그건
# 시각·관측소마다 결과가 달라 개별 재시도에 의미가 있다.
CONN_FAIL_COOLDOWN = float(os.getenv("KMA_CONN_FAIL_COOLDOWN", "120"))

_conn_lock = threading.Lock()
_conn_state = {"blocked_until": 0.0, "last_error": "", "failures": 0}


def _note_conn_failure(err: Exception) -> None:
    with _conn_lock:
        _conn_state["blocked_until"] = time.monotonic() + CONN_FAIL_COOLDOWN
        _conn_state["last_error"] = f"{type(err).__name__}: {str(err)[:200]}"
        _conn_state["failures"] += 1


def _note_conn_success() -> None:
    with _conn_lock:
        _conn_state["blocked_until"] = 0.0


def _conn_blocked() -> bool:
    with _conn_lock:
        return time.monotonic() < _conn_state["blocked_until"]


def connection_state() -> dict:
    """화면 진단용 — 연결이 차단 상태인지, 마지막 오류가 무엇이었는지."""
    with _conn_lock:
        remain = max(0.0, _conn_state["blocked_until"] - time.monotonic())
        return {
            "blocked": remain > 0,
            "cooldown_remaining": int(remain),
            "last_error": _conn_state["last_error"],
            "failures": _conn_state["failures"],
        }


def network_env_report() -> dict:
    """
    배포 환경이 프록시를 주입했는지 확인하는 진단(2026-08-19 추가).

    requests 는 HTTP_PROXY/HTTPS_PROXY 환경변수를 자동으로 신뢰한다(trust_env).
    호스팅 측이 이 변수를 넣어두면 요청이 엉뚱한 경로로 나가 connect timeout 이
    날 수 있는데, 코드만 읽어서는 배제할 수 없다 — 실제 배포 환경의 환경변수를
    봐야 판별된다. 이 저장소 코드는 프록시를 설정하지도 읽지도 않으므로,
    값이 잡히면 그건 전부 플랫폼이 주입한 것이다.

    값에 자격증명(user:pass@host)이 섞일 수 있어 호스트 부분만 남긴다.
    """
    keys = ("HTTP_PROXY", "HTTPS_PROXY", "ALL_PROXY", "NO_PROXY",
            "http_proxy", "https_proxy", "all_proxy", "no_proxy")
    found = {}
    for k in keys:
        v = os.environ.get(k)
        if not v:
            continue
        found[k] = v.split("@")[-1] if "@" in v else v
    return found


# ── 저장소 실측 폴백(offline fallback) ──────────────────────────────
#
# 연결이 막혔을 때 종전 _fallback() 은 20.0°C/50%/1013hPa 라는 **날조 상수**를
# 실측인 양 돌려줬다(FALLBACK_DEFAULT). 배포 환경은 컨테이너 파일시스템이
# 휘발성인 데다 cache/latest_weather_*.json 이 gitignore 대상이라 디스크 캐시가
# 아예 없어서, 실패하면 곧바로 이 날조값으로 떨어진다 — 2026-08-19 화면에 뜬
# "현재 기온 20.0°C · 습도 50% · 기압 1013hPa" 가 정확히 그것이었고, 그 값으로
# 계산한 출력값까지 표시됐다. 규약 5(가짜 데이터를 입력으로 쓰지 않는다)에
# 정면으로 어긋난다.
#
# 저장소에는 GitHub Actions(refresh-data.yml)가 갱신하는 recent_window.json 이
# 있고, 그 레코드는 이 컬렉터의 반환 형식과 키·단위가 동일한 실측이다
# (status=SUCCESS_LIVE). 날조 대신 그걸 쓴다. 12개 관측소가 모두 채워져 있어
# 대상 관측소(Z축)뿐 아니라 이웃 보간(Re축)까지 실측으로 복원된다.
#
# 컬렉터가 파일 경로를 알 필요가 없도록 호출부(app.py)가 주입한다.
_offline_lock = threading.Lock()
_offline_latest: dict = {}    # stn -> 가장 최근 실측 레코드
_offline_by_key: dict = {}    # (stn, timestamp) -> 실측 레코드


def set_offline_fallback(records) -> int:
    """
    저장소 창의 실측 레코드를 폴백 재료로 등록한다.

    관측소별 '가장 최근 1건'(fetch 용)과 (관측소, 시각) 색인(fetch_at 용)을
    함께 만든다. 시각 색인이 필요한 이유는 Im축(시간 경향)이 특정 과거 시각을
    콕 집어 요청하기 때문이다 — 그 시각의 실측이 아니라 '최근 값'을 돌려주면
    존재한 적 없는 변화량이 만들어진다.

    records: recent_window.json 형태의 list[dict].
    반환: 등록된 관측소 수.
    """
    latest: dict = {}
    by_key: dict = {}
    for r in (records or []):
        if not isinstance(r, dict):
            continue
        # 실측만 받는다 — 폴백으로 만들어진 레코드를 다시 폴백 재료로
        # 쓰면 날조가 세탁되어 되돌아온다.
        if r.get("status") != "SUCCESS_LIVE":
            continue
        stn = str(r.get("stn", ""))
        ts = str(r.get("timestamp", ""))[:12]
        if not stn or not ts:
            continue
        by_key[(stn, ts)] = r
        if stn not in latest or ts > str(latest[stn].get("timestamp", "")):
            latest[stn] = r
    with _offline_lock:
        _offline_latest.clear()
        _offline_latest.update(latest)
        _offline_by_key.clear()
        _offline_by_key.update(by_key)
    return len(latest)


def offline_fallback_state() -> dict:
    """등록된 저장소 폴백의 관측소 수와 가장 최근 관측 시각."""
    with _offline_lock:
        if not _offline_latest:
            return {"stations": 0, "latest": None, "records": 0}
        return {
            "stations": len(_offline_latest),
            "latest": max(str(r.get("timestamp", ""))
                          for r in _offline_latest.values()),
            "records": len(_offline_by_key),
        }


def _offline_record(stn: str, tm: str = None) -> dict | None:
    """tm 을 주면 그 시각의 실측만, 없으면 가장 최근 실측을 돌려준다."""
    with _offline_lock:
        if tm is not None:
            r = _offline_by_key.get((str(stn), str(tm)[:12]))
        else:
            r = _offline_latest.get(str(stn))
    return dict(r) if r else None


# 실측으로 인정하는 status 목록.
#
# FALLBACK_WINDOW 는 저장소 창에서 꺼낸 값이라 '관측 시각이 과거일 뿐 사람이
# 만든 값이 아니다'. FALLBACK_CACHED 도 이 프로세스가 앞서 실제로 받은 응답
# 이다. 둘 다 관측 시각(timestamp)을 조작하지 않고 원본 그대로 보존하므로,
# 보간·경향 계산에 넣어도 규약 5 를 어기지 않는다 — 다만 호출부는 '같은 시각
# 스냅샷'인지를 별도로 확인해야 한다(predict.py 참고).
#
# FALLBACK_DEFAULT 만 유일하게 날조값이며, 여기 포함되지 않는다.
REAL_STATUSES = ("SUCCESS_LIVE", "FALLBACK_WINDOW", "FALLBACK_CACHED")


def is_real_observation(status) -> bool:
    """이 레코드가 실제 관측에서 온 값인가(날조 기본값이 아닌가)."""
    return str(status) in REAL_STATUSES


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
            return self._fallback(tm, exact_time=True)

        # 연결 자체가 막힌 것으로 확인된 동안은 시도하지 않는다
        # (_note_conn_failure 주석 참고).
        if _conn_blocked():
            return self._fallback(tm, exact_time=True)

        params = {"tm": tm, "stn": self.stn, "help": "1", "authKey": self.api_key}

        for attempt in range(1, retries + 1):
            if not _reserve_call():
                print(f"[WARN] 일일 API 예산({DAILY_CALL_BUDGET}건) 초과 — "
                      f"fetch_at {tm} 은 폴백으로 처리")
                return self._fallback(tm, exact_time=True)
            try:
                resp = requests.get(ASOS_URL, params=params, timeout=timeout)
                if resp.status_code == 200:
                    parsed = self._parse_asos(resp.text, tm)
                    if parsed:
                        _note_conn_success()
                        parsed["status"] = "SUCCESS_LIVE"
                        return parsed
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as e:
                _note_conn_failure(e)
                print(f"[WARN] fetch_at {tm} 연결 실패 — 이후 "
                      f"{CONN_FAIL_COOLDOWN:.0f}초간 조회를 건너뛴다 "
                      f"({type(e).__name__})")
                break
            except Exception as e:
                if attempt == retries:
                    print(f"[WARN] fetch_at {tm} 실패: {e}")
            if attempt < retries:
                time.sleep(0.5)

        return self._fallback(tm, exact_time=True)

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

        # 연결 자체가 막힌 것으로 확인된 동안은 시도하지 않는다. 12개 관측소가
        # 각자 재시도 한도를 소진하면 조회 1회가 8분까지 늘어나기 때문이다
        # (_note_conn_failure 주석의 실측 계산 참고).
        if _conn_blocked():
            return self._fallback(tm)

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
                        _note_conn_success()
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
            except (requests.exceptions.ConnectTimeout,
                    requests.exceptions.ConnectionError) as e:
                # 연결이 안 되는 것은 관측소를 바꾸거나 재시도해도 결과가
                # 같다 — 이 관측소의 남은 재시도를 접고, 쿨다운 동안 다른
                # 관측소 조회도 건너뛴다.
                _note_conn_failure(e)
                print(f"[WARN] 연결 실패 (관측소 {self.stn}) — 이후 "
                      f"{CONN_FAIL_COOLDOWN:.0f}초간 조회를 건너뛴다 "
                      f"({type(e).__name__})")
                break
            except requests.exceptions.Timeout:
                # ReadTimeout — 서버에는 닿았으므로 재시도할 값어치가 있다.
                print(f"[WARN] 응답 지연 (시도 {attempt}/{retries})")
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

    def _fallback(self, tm: str = None, exact_time: bool = False) -> dict:
        """
        조회 실패 시 대체 레코드. 실측 → 실측 → (최후에만) 날조 순으로 내려간다.

        exact_time=True 면 요청한 시각(tm)의 실측만 인정한다. Im축(시간 경향)이
        특정 과거 시각을 집어 요청할 때 쓴다 — '가장 최근 값'을 대신 주면
        그 시각에 관측이 있었던 것처럼 보여 가짜 변화량이 만들어진다.

        2026-08-19 개편 전에는 두 가지 날조가 있었다.
          ① 캐시가 있으면 그 값의 **타임스탬프를 요청 시각으로 갈아끼우고**
             기온에 `np.random.uniform(-0.05, 0.05)` 노이즈를 더했다 — 묵은
             관측이 방금 받은 실측처럼 보이고, 실재하지 않는 변동까지 생겼다.
          ② 캐시가 없으면 20.0°C/50%/1013hPa 라는 상수를 돌려줬다. 배포
             환경은 디스크 캐시가 없어(휘발성 + gitignore) 항상 이 경로였다.
        둘 다 규약 5(가짜 데이터를 입력으로 쓰지 않는다) 위반이라 제거했다.

        이제 관측 시각을 **조작하지 않는다**. 폴백 레코드는 자기가 실제로
        관측된 시각을 그대로 달고 나가므로, 호출부는 그 값이 얼마나 묵었는지
        판단할 수 있고 record_to_vec 의 시각 특징(시각·연중 시각 sin/cos)도
        그 관측 시각과 맞아떨어진다.
        """
        if tm is None:
            tm = self._obs_time()

        # 후보 둘 다 실측이다. 이 프로세스가 앞서 받은 응답(캐시)이 저장소
        # 창보다 최신일 수도, 그 반대일 수도 있으므로 관측 시각이 더 새로운
        # 쪽을 고른다.
        candidates = []
        if self._memory_cache is not None:
            fb = dict(self._memory_cache)
            if not exact_time or str(fb.get("timestamp", ""))[:12] == str(tm)[:12]:
                fb["status"] = "FALLBACK_CACHED"
                candidates.append(fb)

        win = _offline_record(self.stn, tm if exact_time else None)
        if win is not None:
            win["status"] = "FALLBACK_WINDOW"
            candidates.append(win)

        if candidates:
            best = max(candidates, key=lambda r: str(r.get("timestamp", "")))
            # 관측소가 섞이면 record_to_vec 이 남의 위경도를 쓰게 된다.
            best["stn"] = self.stn
            return best

        # 여기까지 왔다는 건 실측이 하나도 없다는 뜻이다. 이 값은 관측이
        # 아니라 자리표시자이며, 호출부는 status 로 이를 구분해 화면 표시와
        # 적중률 기록에서 제외해야 한다(app.py·predict.py 참고).
        return {
            "timestamp": tm, "temperature": 20.0, "precipitation": 0.0,
            "humidity": 50.0, "wind_speed": 1.5, "wind_dir": 0.0,
            "pressure": 1013.0, "precip_type": 0,
            "stn": self.stn, "status": "FALLBACK_DEFAULT",
        }

    # ── 모델 입력 벡터 ──────────────────────────────────────────────

    def to_model_vector(self, data: dict) -> list[float]:
        """
        14차원 수치 벡터 반환 — train.record_to_vec 와 순서가 반드시 같아야 한다.
        [기온, 강수량, 습도, 풍속, 풍향sin, 풍향cos, 기압, 강수형태,
         시각sin, 시각cos, 위도, 경도, 연중시각sin, 연중시각cos]
        """
        wd_rad = np.deg2rad(data.get("wind_dir", 0.0))

        ts = str(data.get("timestamp", ""))      # YYYYMMDDHHmm
        hour = int(ts[8:10]) if len(ts) >= 10 else 0
        h_rad = 2.0 * np.pi * hour / 24.0

        if len(ts) >= 8:
            doy = datetime.strptime(ts[:8], "%Y%m%d").timetuple().tm_yday
        else:
            doy = 1
        y_rad = 2.0 * np.pi * doy / 365.25

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
            float(np.sin(y_rad)),
            float(np.cos(y_rad)),
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
