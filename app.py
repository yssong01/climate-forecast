"""
app.py — Tri-CHEF 기후 모델 출력 Streamlit 대시보드.

학습된 3축 파이프라인(predict.py)을 실시간 관측과 연결하여 표시한다.
모델 로드는 st.cache_resource로 1회만 수행한다 — Streamlit은 위젯 조작마다
스크립트 전체를 재실행하므로, 캐싱이 없으면 매번 체크포인트를 다시 읽는다.
(2026-08-09부터 Im축이 MiniLM 텍스트에서 시간 경향 벡터로 대체되면서
sentence-transformers 로드 자체가 사라졌다 — 캐싱 대상은 모델 1개뿐이다.)

화면은 탭 4개로 구성된다 — 출력값 추이 / 극한기상 / 성능 검증 / 모델 구조.
수치·그래프마다 산출 근거(무엇을 어떻게 측정했는지)를 캡션 또는
st.metric의 help 툴팁으로 함께 표시한다 — "이 값이 왜 이렇게 나왔는가"를
화면만 보고 답할 수 있어야 한다는 원칙(2026-08-09 확정)을 따른다.
"""
import json
import math
import os
import time
import pandas as pd
import requests
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

# 관측이 아직 발표되지 않았을 때 수집기가 기다렸다 재시도하는 기본 대기(60초)를
# 서빙에서는 없앤다. predict() 가 관측소 12곳을 순차 조회하므로 그대로 두면
# 최악의 경우 화면이 20분 넘게 멈춘다(weather_collector._empty_retry_wait
# 참고). 프로젝트 임포트보다 먼저 설정할 필요는 없지만 — 호출 시점에 읽는다 —
# 의도를 드러내기 위해 파일 맨 앞에 둔다. setdefault 라 배포 환경에서
# 명시적으로 다른 값을 주면 그쪽이 우선한다.
os.environ.setdefault("KMA_EMPTY_RETRY_WAIT", "0")

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from weather_collector import (
    STATIONS, STATION_COORDS, api_call_stats,
    set_offline_fallback, offline_fallback_state,
    connection_state, network_env_report, redact_secrets,
)
from predict import (
    load_model, predict, CHECKPOINT, event_threshold,
    PRECIP_PROB_GATE, PRECIP_PROB_GATE_BY_LEAD,
)
import accuracy

# +12h 2차 산출값(2026-08-18) — 기온·강수 성능 저하 곡선을 실측한 결과,
# +6h 대비 절대오차 악화가 완만해(기온 1.30→1.66°C) 병행 제공 가치가
# 있다고 판단해 채택했다(채택 당시 한파 F1은 0.458→0.453으로 사실상 그대로).
# +6h 는 이후(2026-08-19) 중첩 전이학습 승격으로 한파 F1 0.499 로 올랐고,
# +12h 도 같은 커리큘럼을 독립 2회 실행해 검증 후 적용했다(F1 0.453→0.509,
# 계절 오탐도 5.41% WARN → 1.21% PASS 로 개선). README '+12h 2차 산출값' 참고.
CHECKPOINT_12H = "./checkpoints/numerical_trichef_12h.pt"

# ASOS 타임스탬프는 tz 정보 없는 KST 벽시계 표기다. 컨테이너 기본 시간대는
# UTC라 datetime.now()를 그대로 쓰면 "최근 72시간" 커트라인이 실제로는 9시간
# 밀려 81시간이 된다(weather_collector.py가 이미 조심하는 것과 같은 문제 —
# 여기서도 놓쳤다가 실측으로 발견: 2026-08-10, 화면에 "최근 80시간"으로 표시됨).
KST = ZoneInfo("Asia/Seoul")


def _now_kst_naive() -> datetime:
    return datetime.now(KST).replace(tzinfo=None)

# 사이드바 스파크라인은 최근 72시간만 쓰는데, 예전엔 3.5년치 원본(64MB,
# 32만 레코드)을 매번 통째로 파싱해 RAM 1.55GiB 중 가장 큰 낭비였다
# (docker stats 실측, 2026-08-08). build_recent_window.py 가 미리 추려낸
# 파일만 읽는다 — 배포 저장소에는 이 파일만 포함시키고 원본은 제외한다.
HIST_FILES = ["./cache/recent_window.json"]

# recent_window.json 을 로컬 디스크가 아니라 GitHub raw 로 직접 읽는다
# (2026-08-20 — 재배포와 최신성 분리). refresh-data.yml 이 이제 main 이
# 아니라 data 브랜치에 커밋한다 — Streamlit Cloud 는 main 만 감시하므로
# data 브랜치에 15분마다 커밋해도 재배포가 안 걸린다. 대신 이 앱이 직접
# 그 브랜치의 최신 파일을 끌어와야 최신성이 반영된다.
#
# 실패(네트워크 문제, 저장소 구조 변경 등)하면 로컬 디스크(HIST_FILES)로
# 폴백한다 — 그건 이 앱이 마지막으로 재배포됐을 시점의 스냅숏이라 최신은
# 아니지만, 완전히 끊기지는 않는다.
RAW_WINDOW_URL = (
    "https://raw.githubusercontent.com/yssong01/climate-forecast/data/"
    "cache/recent_window.json"
)

st.set_page_config(
    page_title="Tri-CHEF 기후 모델 출력값",
    page_icon="🌦️",
    layout="centered",
)


# ── 캐시된 리소스 — 세션당 한 번만 로드 ──────────────────────────────

def ckpt_fingerprint(path: str = CHECKPOINT) -> str:
    """
    체크포인트 파일의 신원(수정시각·크기). 캐시 키에 넣어 파일이 바뀌면
    자동으로 새로 로드되게 한다.

    왜 필요한가 — get_model() 을 인수 없이 @st.cache_resource 로 감싸면
    캐시 키가 항상 같아서, 체크포인트를 새로 학습해 배포해도 살아 있는
    프로세스는 옛 모델을 계속 재사용한다. 2026-08-11 재학습 배포에서
    실제로 발생했다: 저장소에는 새 체크포인트가 올라갔는데 화면은 옛
    수치(기온 1.27 / 폭염 F1 0.822)를 그대로 보여줬고, 기준선 증감도
    "저장돼 있지 않다"로 표시됐다. cache_resource 는 스크립트 재실행은
    물론 코드 갱신 후에도 같은 프로세스면 살아남기 때문이다.
    """
    try:
        st_ = os.stat(path)
        return f"{st_.st_mtime_ns}:{st_.st_size}"
    except OSError:
        return "missing"


@st.cache_resource(show_spinner="모델 로드 중...")
def get_model_at(path: str, fingerprint: str):
    """fingerprint 는 캐시 무효화 전용 — 값 자체는 쓰지 않는다.

    path 를 캐시 키에 포함시켜(+6h/+12h 처럼) 서로 다른 체크포인트를
    동시에 로드해도 한쪽이 다른 쪽을 덮어쓰지 않는다(2026-08-18, +12h
    2차 산출값 추가 시 일반화).
    """
    return load_model(path)


def get_model(fingerprint: str):
    """기존 호출부 호환용 — +6h 배포 체크포인트를 로드한다."""
    return get_model_at(CHECKPOINT, fingerprint)


def obs_hour_key() -> str:
    """
    지금 조회하면 받게 될 ASOS 관측 시각(KST, YYYYMMDDHH00).

    weather_collector.RobustWeatherCollector._obs_time() 과 같은 규칙이다
    (매시 정각 관측, 약 10분 후 공개 → 10분 전이면 직전 시각). 예측 캐시의
    키로 쓰기 위해 여기서도 계산한다.
    """
    now = datetime.now(KST)
    if now.minute < 10:
        now -= timedelta(hours=1)
    return now.strftime("%Y%m%d%H00")


# 출력값 1건을 만드는 데 API 호출 15회가 든다 — 대상 관측소 + 이웃 11곳(Re축
# 공간보간장) + 대상 관측소의 1·3·6시간 전(Im축 경향벡터). 캐시가 없으면
# 이 비용이 "접속자 수 × 페이지 갱신 횟수"로 곱해지는데, 이 인증키는 누적
# 약 9,800건에서 차단된 이력이 있다(CLAUDE.md 4절).
#
# 캐시 키에 관측 시각을 넣어 호출량을 시계에 묶는다 — 같은 정시 안에서는
# 몇 명이 몇 번을 새로고침하든 결과를 재사용하므로 추가 호출이 0이다.
# 애초에 ASOS 는 매시 정각 관측이라 같은 시각을 다시 조회해도 같은 값이다.
#
# 상한은 관측소당 시간당 30회다(2026-08-25 실측 정정) — 화면이 +6h·+12h 를
# 함께 표시하므로 산출값이 2건이고, 두 리드타임이 서로 다른 캐시 항목이라
# 같은 15건을 각자 조회한다. 15 회는 산출값 1건 기준이다. 첫 조회 결과를
# 두 체크포인트가 공유하게 하면 30 을 15 로 되돌릴 수 있으나, 관측 수집과
# 추론을 분리하도록 predict() 시그니처를 바꿔야 해서 지금은 두지 않는다
# (실사용 호출량이 차단 지점의 10~17% 수준 — README 'API 트래픽' 참고).
#
# model/ckpt 는 언더스코어 접두어로 넘긴다 — Streamlit 이 해시 대상에서
# 제외하는 규약이다(torch 모듈은 해시할 수 없다).
#
# lead_hours 는 캐시 키에 반드시 넣어야 한다(2026-08-18, +12h 추가 시
# 확인) — model/ckpt 가 해시 대상에서 빠지므로, 이 인자가 없으면 캐시 키가
# (stn, obs_hour) 뿐이라 +6h 호출과 +12h 호출이 같은 항목을 공유해 버린다.
# 먼저 계산된 쪽의 결과를 나중 호출이 그대로 돌려받는 사고가 난다.
#
# ckpt_fp(체크포인트 지문)도 같은 이유로 반드시 넣는다(2026-08-29 총점검에서
# 발견). 위 lead_hours 와 get_model_at() 의 fingerprint 는 각각 고쳤으면서
# 이 함수의 모델 신원만 빠져 있었다 — 프로세스가 살아 있는 동안 체크포인트가
# 교체되면(promote_checkpoint.py 는 배포 경로를 제자리에서 덮어쓴다)
# get_model_at() 은 지문 변화를 감지해 새 모델을 로드하지만, 이 캐시는
# (stn, obs_hour, lead_hours) 가 같아 **옛 모델이 만든 예측을 그대로**
# 돌려준다. 화면은 새 모델이라 믿고 옛 수치를 보여주게 되며, 이는 2026-08-11
# get_model() 사고와 정확히 같은 유형이다.
@st.cache_data(ttl=3600, show_spinner="관측 조회 중... (12개 관측소)")
def cached_predict(stn: str, obs_hour: str, lead_hours: int, ckpt_fp: str,
                   _model, _ckpt) -> dict:
    return predict(stn=stn, model=_model, ckpt=_ckpt)


@st.cache_resource(show_spinner=False)
def _history_last_good_store() -> dict:
    """
    이 프로세스가 최근에 성공한 raw 조회 결과를 담아두는 그릇.

    cache_resource 라 프로세스 생애 동안 하나의 dict 객체가 유지된다 —
    load_merged_history() 가 매 성공마다 이 안의 "data" 를 갈아 끼운다.
    로컬 디스크 스냅숏(HIST_FILES)은 main 브랜치가 더 이상 갱신하지 않아
    며칠씩 묵을 수 있는 반면, 이 값은 최악의 경우에도 "이 프로세스가 뜬
    이후 raw 조회가 한 번이라도 성공했던 시점"까지만 묵는다 — 컨테이너가
    오래 떠 있었다면 로컬 스냅숏보다 훨씬 최신일 가능성이 높다.
    """
    return {"data": None}


def _fetch_raw_window(retries: int = 1, backoff_sec: float = 1.0) -> dict | None:
    """RAW_WINDOW_URL 조회. 성공하면 dict, 모든 시도 실패 시 None.

    재시도를 넣는 이유(2026-08-23) — Streamlit Cloud 의 발신 네트워크는
    KMA API 쪽에서 이미 간헐적 타임아웃 이력이 있다(CLAUDE.md 4절).
    raw.githubusercontent.com 도 같은 공유 네트워크를 타므로 순간적인
    타임아웃 하나로 곧장 폴백까지 내려가지 않도록, 짧은 대기 후 한 번만
    더 시도한다. 여러 번 재시도하지 않는 이유는 이 함수가 페이지 렌더링을
    막고 있어(동기 호출) 과도한 재시도가 체감 지연으로 바로 이어지기
    때문이다.
    """
    for attempt in range(retries + 1):
        try:
            resp = requests.get(RAW_WINDOW_URL, timeout=8)
            if resp.status_code == 200:
                records = resp.json()
                return {(r["stn"], r["timestamp"]): r for r in records}
            print(f"[WARN] raw 창 조회 실패 HTTP {resp.status_code} "
                  f"(시도 {attempt + 1}/{retries + 1})")
        except Exception as e:
            print(f"[WARN] raw 창 조회 실패({type(e).__name__}) "
                  f"(시도 {attempt + 1}/{retries + 1})")
        if attempt < retries:
            time.sleep(backoff_sec)
    return None


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_history() -> tuple[dict, str]:
    """
    (관측소, 시각) → 레코드, 그리고 그 출처.
    "raw"(정상) / "last_good_fallback"(이 프로세스의 마지막 성공값 재사용)
    / "local_fallback"(그마저 없어 main 브랜치 스냅숏 사용) 중 하나다.
    5분 캐시라 data 브랜치가 그사이 더 갱신돼도 페이지가 곧 따라잡는다 —
    매 새로고침마다 다시 받지는 않는다.

    GitHub raw 를 먼저 시도하고(재시도 1회 포함), 실패하면 이 프로세스가
    마지막으로 성공했던 값으로, 그것도 없으면 로컬 디스크로 폴백한다(위
    RAW_WINDOW_URL 주석 참고). 이 함수 자체는 실측 여부를 판정하지 않는다
    — status 필드를 그대로 전달할 뿐이며, "실측인가"는 이 데이터를 쓰는
    쪽(weather_collector.is_real_observation 등)이 판단한다.

    출처를 함께 돌려주는 이유(2026-08-23) — 로컬 디스크(HIST_FILES)는
    main 브랜치의 스냅숏이라, refresh-data.yml 이 data 브랜치로 옮겨간
    이후로는 이 앱이 마지막 재배포됐을 시점에 영원히 멈춰 있다. raw 조회가
    타임아웃·일시 오류로 단 한 번만 실패해도 이 폴백으로 조용히 넘어가고,
    그 결과를 "data 브랜치 자체가 몇 시간 뒤처졌다"와 똑같은 문구로
    보여주면 원인을 구분할 수 없다 — 실제로는 며칠 묵은 값일 수 있는데도.
    호출부가 출처를 보고 경고 문구를 갈라 쓴다.
    """
    store = _history_last_good_store()

    fetched = _fetch_raw_window()
    if fetched is not None:
        store["data"] = fetched
        return fetched, "raw"

    if store["data"] is not None:
        print("[WARN] raw 창 조회 실패 — 이 프로세스의 마지막 성공값으로 대체")
        return store["data"], "last_good_fallback"

    print("[WARN] raw 창 조회 실패, 이 프로세스에 성공 기록 없음 — 로컬 폴백")
    merged = {}
    for path in HIST_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                merged[(r["stn"], r["timestamp"])] = r
    return merged, "local_fallback"


def recent_series(history: dict, stn: str, hours: int = 72) -> list:
    """지정 관측소의 최근 `hours`시간 관측치, 시각순 정렬."""
    cutoff = _now_kst_naive() - timedelta(hours=hours)
    rows = [r for (s, ts), r in history.items()
            if s == stn and datetime.strptime(ts[:12], "%Y%m%d%H%M") >= cutoff]
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def history_age_hours(history: dict) -> float | None:
    """
    창(recent_window.json)에서 가장 최근 관측이 몇 시간 전 것인지.

    배포판에서 이 값이 계속 커지면 갱신 경로가 끊겼다는 뜻이다 — 클라우드
    컨테이너는 파일을 유지하지 못하므로, 창을 채우는 주체는 앱이 아니라
    저장소를 갱신하는 GitHub Actions(refresh-data.yml)다. 화면에 그대로
    노출해 "묵은 창"을 알리지 않고 넘기지 않는다. 창이 비었으면 None.
    """
    if not history:
        return None
    latest = max(datetime.strptime(ts[:12], "%Y%m%d%H%M") for _, ts in history)
    return (_now_kst_naive() - latest).total_seconds() / 3600


def observed_age_hours(observed_at) -> float | None:
    """이 관측 시각이 지금으로부터 몇 시간 전인지. 파싱 불가면 None."""
    try:
        t = datetime.strptime(str(observed_at)[:12], "%Y%m%d%H%M")
    except Exception:
        return None
    return (_now_kst_naive() - t).total_seconds() / 3600


@st.cache_resource(show_spinner=False)
def accuracy_baseline() -> dict:
    """
    이 프로세스가 뜬 시점의 적중률 로그 상태 = 저장소에 커밋된 스냅샷.

    이후 앱이 추가하는 항목(record_prediction)은 컨테이너가 살아 있는
    동안만 존재한다 — 재시작하면 여기 담긴 상태로 되돌아간다. 그 차이를
    화면에서 구분해 보여주기 위해 시작 시점 값을 잡아둔다.
    cache_resource 라 세션이 아니라 프로세스 단위로 한 번만 계산된다.
    """
    return accuracy.log_summary()


# 지도 지형색 — 어느 테마에서도 그대로 쓰는 반투명 회색.
#
# 예전엔 어두운 테마 기준의 불투명 색(#242830 등)을 박아 넣었다. 그래서 Light
# 로 바꾸면 사이드바는 흰데 지도만 검은 판으로 남았다(2026-08-11 사용자 실측
# 스크린샷). st.context.theme 로 테마를 읽어 색을 고르는 방법도 재봤는데,
# 테마 전환은 rerun 을 일으키지 않아(실측: 전환 직후에도 st.context.theme.type
# 이 여전히 dark, 수동 rerun 후에야 light) 다음 자동 갱신까지 지도가 어긋난
# 채 남는다.
# 반투명이면 뒤의 사이드바 배경이 그대로 비쳐 어느 테마든 즉시 맞는다 —
# 파이썬이 테마를 알 필요가 없어진다.
_MAP_LAND = "rgba(128, 140, 158, 0.28)"
_MAP_OCEAN = "rgba(128, 140, 158, 0.08)"
_MAP_LINE = "rgba(128, 140, 158, 0.85)"
_MAP_MARKER_EDGE = "rgba(128, 140, 158, 0.9)"

# 리드타임 색(2026-08-20). '출력값 추이' 탭의 차트 별 심벌과 그 아래 표의
# 열 제목이 같은 색을 써야 어느 열이 어느 별인지 곧바로 이어진다 — 두 곳에
# 색을 따로 적어두면 한쪽만 바뀌어도 대응이 조용히 깨지므로 상수로 묶는다.
_LEAD6_COLOR = "#E2954F"    # 꽉 찬 별(★) — +6시간
_LEAD12_COLOR = "#B25FE0"   # 속 빈 별(☆) — +12시간


def render_station_map(stations: dict, selected_stn: str) -> str | None:
    """
    설정창(사이드바)에 넣는 간략화된 대한민국 지도 — 관측소 위치 표시 +
    클릭으로 선택. 예전엔 OpenStreetMap 타일(Scattermapbox)을 썼는데, 이게
    두 가지 문제가 있었다: ① 함수 안에서 bare st.plotly_chart()를 써서
    실제로는 사이드바가 아니라 메인 화면 맨 위, 넓은 폭 그대로 렌더링되고
    있었다(2026-08-10 스크린샷으로 발견) ② 실제 지도 타일이라 사이드바처럼
    좁고 긴 공간에 넣으면 왜곡되거나 잘려 보인다. Scattergeo는 Plotly.js에
    내장된 저해상도 지형 데이터를 쓰므로(외부 타일 서버 요청 없음) 오프라인
    렌더링이 가능하고, "간략화한 지도"라는 요청에도 더 맞는다.
    """
    codes = list(stations.values())
    lats = [STATION_COORDS[c][0] for c in codes]
    lons = [STATION_COORDS[c][1] for c in codes]
    names = list(stations.keys())
    colors = ["#E2954F" if c == selected_stn else "#4C78A8" for c in codes]
    sizes = [17 if c == selected_stn else 10 for c in codes]

    fig = go.Figure(go.Scattergeo(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=sizes, color=colors,
                    line=dict(width=0.5, color=_MAP_MARKER_EDGE)),
        text=names, textposition="middle right",
        # 관측소 이름의 글자색은 지정하지 않는다 — Streamlit 이 차트에 입히는
        # 테마 템플릿의 글자색을 그대로 물려받아야 Light/Dark 를 따라간다.
        textfont=dict(size=10),
        customdata=codes,
        hovertext=names, hoverinfo="text",
    ))
    fig.update_geos(
        resolution=50,
        lataxis_range=[33, 39], lonaxis_range=[124.2, 130.2],   # 한반도만 딱 맞게
        showland=True, landcolor=_MAP_LAND,
        showocean=True, oceancolor=_MAP_OCEAN,
        showcountries=True, countrycolor=_MAP_LINE,
        showcoastlines=True, coastlinecolor=_MAP_LINE,
        showframe=False, projection_type="mercator",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=340,   # 세로 폭 최소화(2026-08-10) — 좌우는 사이드바 폭 그대로
        paper_bgcolor="rgba(0,0,0,0)",
    )
    event = st.sidebar.plotly_chart(
        fig, width="stretch", key="station_geo_map",
        config={"scrollZoom": False, "displayModeBar": False},
        on_select="rerun", selection_mode="points",
    )
    points = (event or {}).get("selection", {}).get("points", [])
    if not points:
        return None
    clicked = points[0].get("customdata")
    return clicked[0] if isinstance(clicked, list) else clicked


def event_gauge(label: str, prob: float | None, thresh: float, color: str):
    """극한기상 확률 하나를 가로 막대 + 임계값 눈금으로 표시."""
    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=[prob * 100 if prob is not None else 0], y=[""], orientation="h",
        marker_color=color, width=0.55,
        text=[f"{prob:.1%}" if prob is not None else "N/A"],
        textposition="outside",
        cliponaxis=False,   # 확률이 100%에 가까우면 라벨이 플롯 경계에서 잘리는 것 방지
    ))
    fig.add_vline(x=thresh * 100, line_dash="dot", line_color="#B0413E",
                 annotation_text=f"판정 임계값 {thresh:.0%}", annotation_position="top")
    fig.update_layout(
        xaxis=dict(range=[0, 100], title=None, showticklabels=True, ticksuffix="%"),
        yaxis=dict(showticklabels=False),
        height=110, margin=dict(l=10, r=50, t=28, b=10),
        showlegend=False,
    )
    # 글자색은 지정하지 않는다 — 지정하면(예전엔 #FFFFFF 고정) Light 테마에서
    # 흰 배경에 흰 글씨가 된다. 상속받으면 Streamlit 테마 색을 그대로 쓴다.
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; '
        f'margin-bottom:0.2rem;">{label}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, width="stretch", config={"staticPlot": True})


# ── 사이드바 ──────────────────────────────────────────────────────

st.sidebar.title("설정")

station_options = dict(STATIONS)  # {"서울": "108", ...}
code_to_name = {v: k for k, v in station_options.items()}

# 관측소 선택은 셀렉트박스와 지도 클릭 두 입력이 같은 상태(st.session_state
# ["stn"])를 공유한다. 지도 클릭 처리를 셀렉트박스 위젯 인스턴스화보다 먼저
# (스크립트 실행 순서 기준 — 화면에 보이는 순서와는 별개) 끝내둬야 한다.
# Streamlit은 "이미 인스턴스화된 위젯의 session_state 키"를 그 위젯이
# 만들어진 이후에 코드로 덮어쓰면 예외를 던진다(StreamlitAPIException,
# 2026-08-10 실측: 지도 클릭 시 station_selectbox 키 직접 수정 시도로 발생).
# 그래서 지도 클릭은 이번 실행에서 즉시 반영하지 않고 _pending_stn 에
# 적어두고 rerun만 하며, 그 값을 실제로 적용하는 건 다음 실행의 맨 앞
# (셀렉트박스가 만들어지기 전)이다 — Streamlit 공식 문서가 권장하는
# "위젯 인스턴스화 전에 값 미리 반영" 패턴.
if "_pending_stn" in st.session_state:
    _pending = st.session_state.pop("_pending_stn")
    st.session_state.stn = _pending
    st.session_state.station_selectbox = code_to_name[_pending]
if "stn" not in st.session_state:
    st.session_state.stn = "108"
if "station_selectbox" not in st.session_state:
    st.session_state.station_selectbox = code_to_name[st.session_state.stn]

auto_refresh = st.sidebar.toggle("자동 갱신", value=True)
interval_min = st.sidebar.slider("갱신 주기 (분)", 1, 30, 5, disabled=not auto_refresh)

if auto_refresh:
    st_autorefresh(interval=interval_min * 60 * 1000, key="refresh")

st.sidebar.caption(
    "자동 갱신은 설정한 주기마다 페이지를 재실행하여 기상청 종관기상관측(ASOS) "
    "실시간 관측(API)에서 이 관측소와 인접 11개 관측소의 현재값, 그리고 이 "
    "관측소의 과거 1·3·6시간 전 값을 다시 조회한다. 모델을 재학습하는 것이 "
    "아니라 동일한 가중치로 추론만 다시 수행한다."
)
st.sidebar.caption(
    "단, 관측은 매시 정각에 1회만 이루어지며 약 10분 후 공개된다. 따라서 동일 "
    "정시 내 재실행 시에는 신규 조회 없이 직전 결과를 재사용한다 — 갱신 주기를 "
    "짧게 설정해도 API 호출 수는 증가하지 않으며, 신규 값은 다음 정시로 넘어가야 "
    "반영된다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("관측소 위치(지도에서 선택 가능)")


def _sync_stn_from_selectbox():
    st.session_state.stn = station_options[st.session_state.station_selectbox]


st.sidebar.selectbox(
    "관측소", list(station_options.keys()),
    key="station_selectbox", on_change=_sync_stn_from_selectbox,
)

clicked_code = render_station_map(station_options, st.session_state.stn)
# 지도의 클릭 선택 상태는 프론트엔드에 남아있어, 셀렉트박스로 다른 도시를
# 고른 뒤에도 다음 rerun에서 "예전 클릭"이 그대로 이벤트로 잡힌다. 그걸
# 한 번 처리했으면 같은 코드로는 다시 반응하지 않도록 _last_map_click으로
# 소비 여부를 기록한다 — 안 그러면 셀렉트박스 선택이 지도 클릭으로 도로
# 덮어써진다. (알려진 트레이드오프: 지도에서 같은 도시를 다시 클릭해도
# 그 사이 셀렉트박스로 딴 곳을 갔다오지 않았으면 무반응 — 데모 스코프에서는
# 셀렉트박스로 우회 가능해 허용.)
if clicked_code and clicked_code != st.session_state.get("_last_map_click"):
    st.session_state["_last_map_click"] = clicked_code
    if clicked_code != st.session_state.stn:
        st.session_state["_pending_stn"] = clicked_code
        st.rerun()

stn = st.session_state.stn

st.sidebar.markdown("---")
st.sidebar.caption(
    "출력값 1건마다 12개 관측소를 모두 조회한다. 선택한 관측소는 계산 입력으로, "
    "나머지 11개는 '모델 구조' 탭에서 설명하는 공간축(Re)의 보간에 사용된다."
)

# API 호출량을 숨기지 않는다. 이 인증키는 누적 약 9,800건에서 차단된 적이
# 있어(CLAUDE.md 4절) 사용량이 관리 대상이고, "얼마나 쓰고 있는지"를 추측이
# 아니라 실측으로 보여줘야 상한 설정이 근거를 갖는다. 카운터는 프로세스
# 메모리 기준이라 재시작하면 0으로 돌아간다 — 그 사실도 같이 적는다.
#
# 자리만 잡아두고 예측이 끝난 뒤에 채운다 — 사이드바는 스크립트 순서상
# 예측보다 먼저 실행되므로, 여기서 바로 읽으면 이번 실행의 호출분이 빠진
# "한 박자 늦은 값"이 표시된다(첫 실행에서 실시간 조회 15회를 하고도 0건으로
# 보였다 — 2026-08-11 실측).
st.sidebar.markdown("---")
api_usage_slot = st.sidebar.empty()
st.sidebar.caption(
    "이 앱 프로세스가 시작된 이후의 누적치다(재시작 시 0으로 초기화). 상한을 "
    "초과하면 신규 조회 대신 마지막 성공값을 사용하며, 화면 상단 배지가 이 "
    "상태를 표시한다."
)

# 인증키 유출 대응 — 사용량 이상 감시(2026-08-19 추가).
#
# 이 저장소가 "확실히 우리가 쓴 것"으로 아는 두 수치를 합쳐 보여준다.
#   · 이 프로세스(위 캡션) — 재시작마다 리셋되어 하루 전체를 담지 못한다.
#   · GitHub Actions(아래) — refresh_deploy_data.py 가 저장소에 커밋하므로
#     재시작과 무관하게 오늘 하루 누적이 보존된다.
# 이 합은 여전히 우리 시스템이 정상적으로 쓴 하한선일 뿐이다. 두 프로세스
# 바깥(다른 곳에서 이 키로 직접 호출)에서 쓴 건 여기 잡히지 않는다 — 그건
# API허브 마이페이지의 "API 호출 현황"만이 볼 수 있는 진짜 원본이다. 그래서
# 비교 대상을 명시해 안내한다.
#
# 로컬 디스크가 아니라 data 브랜치 raw 를 읽는다(2026-08-20) — 이 파일도
# refresh_deploy_data.py 가 이제 main 이 아니라 data 브랜치에 커밋하므로,
# 로컬 디스크를 읽으면 재배포가 끊긴 뒤로 영원히 얼어붙은 값을 보여주게
# 된다(recent_window.json 과 같은 이유, 위 RAW_WINDOW_URL 주석 참고).
# 캐시를 씌운다(2026-08-21) — Streamlit 은 위젯 조작·자동 갱신마다 스크립트
# 전체를 다시 돌리므로, 캐시 없이 모듈 최상단에서 호출하면 매 재실행이 최대
# 8초(timeout) 동안 이 요청에 묶인다. 이 값은 하루 단위 집계라 5분 캐시로
# 충분하다(같은 이유로 recent_window.json 도 ttl=300 을 쓴다).
@st.cache_data(ttl=300, show_spinner=False)
def _fetch_usage_log() -> dict:
    resp = requests.get(
        "https://raw.githubusercontent.com/yssong01/climate-forecast/"
        "data/cache/api_usage_daily.json",
        timeout=8,
    )
    return resp.json() if resp.status_code == 200 else {}


try:
    _usage_log = _fetch_usage_log()
    _today_kst = datetime.now(KST).strftime("%Y%m%d")
    if _usage_log.get("date") == _today_kst:
        st.sidebar.caption(
            f"GitHub Actions(저장소 기준) 오늘 {_usage_log.get('total', 0)}건 "
            "— 재시작과 무관하게 보존됨."
        )
    else:
        st.sidebar.caption("GitHub Actions 사용량 기록 없음(오늘 아직 미실행이거나 오래됨).")
except Exception:
    pass
st.sidebar.caption(
    "⚠️ 위 두 수치의 합보다 API허브 마이페이지의 실제 호출 현황이 크게 높다면 "
    "이 키가 다른 곳에서도 쓰이고 있다는 뜻이다 — 그쪽 숫자가 최종 근거다."
)


# ── 메인 ──────────────────────────────────────────────────────────

if not os.path.exists(CHECKPOINT):
    st.error(
        f"체크포인트가 없다: `{CHECKPOINT}`.\n\n"
        f"터미널에서 `python train.py`를 실행하여 모델을 먼저 학습해야 한다."
    )
    st.stop()

# API 연결이 막혔을 때 쓸 폴백 재료를 컬렉터에 등록한다(2026-08-19 추가).
#
# 저장소 창(recent_window.json)은 GitHub Actions(refresh-data.yml)가 15분마다
# 갱신하는 **실측**이고, 레코드 형식이 컬렉터 반환값과 동일하다. 등록해두면
# 조회가 실패해도 컬렉터가 날조 상수(20.0°C/50%/1013hPa) 대신 이 실측으로
# 내려간다 — 2026-08-19 Streamlit Cloud 발신 IP 가 막혔을 때 화면에 날조값이
# 실측처럼 표시된 사고의 재발 방지책이다. 12개 관측소가 모두 채워져 있으므로
# 대상 관측소(Z축)뿐 아니라 이웃 보간(Re축)까지 실측으로 복원된다.
#
# load_merged_history() 는 5분 캐시라 여기서 먼저 불러도 추가 비용이 없다.
set_offline_fallback(list(load_merged_history()[0].values()))

try:
    _fp6 = ckpt_fingerprint()
    model, ckpt = get_model(_fp6)
    result = cached_predict(stn, obs_hour_key(), ckpt["lead_hours"], _fp6, model, ckpt)
except Exception as e:
    st.error(f"예측 실패: {redact_secrets(str(e))}")
    st.stop()

# 이번 실행의 조회분까지 반영해 사이드바 사용량을 채운다(위 api_usage_slot 참고).
_calls = api_call_stats()
api_usage_slot.caption(
    f"API 호출 — 오늘 {_calls['count']}건"
    + (f" / 상한 {_calls['budget']}건" if _calls["budget"] > 0 else " (상한 없음)")
    + (f" · 상한 초과로 폴백 {_calls['blocked']}건" if _calls["blocked"] else "")
)

# 제목부터 현재 관측값(기온~기압)까지는 스크롤해도 상단에 고정된다 —
# 탭 내용은 길어질 수 있는데, "지금 어디를 보고 있는지"(관측소·현재값)는
# 계속 보이는 게 맞다는 요청(2026-08-10)에 따른 것.
#
# 처음엔 CSS position:sticky만으로 시도했는데 실제로는 안 먹혔다(사용자
# 실측 재현, 2026-08-10). Playwright로 직접 렌더링해 원인을 추적한 결과 —
# 조상 요소 어디에도 overflow·transform·contain 등 sticky를 깨는 흔한
# 원인이 없는데도 스크롤에 그대로 딸려 올라갔다. 반면 같은 요소에
# position:fixed를 걸었더니 정확히 고정됐다(재현 확인). 그래서 sticky
# 대신 fixed + JS로 구현한다: components.html의 iframe(같은 출처라 부모
# document에 접근 가능 — sandbox에 allow-same-origin 확인됨)에서 메인
# 콘텐츠 영역의 위치/폭을 측정해 헤더에 그대로 입혀 고정하고, 헤더가
# 빠지며 생기는 빈 공간은 ResizeObserver로 원래 높이를 재서 부모 wrapper
# 에 되돌려준다.
st.markdown(
    """
    <style>
    /* Streamlit 기본값은 툴바 아래 여백이 커서(약 6rem) "기후 모델 출력값" 위로
       빈 공간이 크게 남는다 — 줄인다. 헤더가 이 컨테이너의 자연스러운
       위치를 기준으로 접히므로, 여기를 줄이면 스크롤 유무와 상관없이
       같이 줄어든다. */
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
    }
    /* 배경색은 여기서 정하지 않는다. 예전엔 var(--background-color, #0e1117)
       였는데 Streamlit 이 그 변수를 내주지 않아 항상 폴백인 어두운 색이
       쓰였고, Light 테마로 바꿔도 헤더만 검게 남았다(2026-08-11 실측).
       아래 JS 가 실제 앱 배경(getComputedStyle)을 읽어 그대로 입힌다 —
       테마 전환이 rerun 없이 일어나도 따라간다. 헤더가 아직 정상 흐름에
       있는 동안(=JS 실행 전)에는 투명이 오히려 맞다. */
    .st-key-sticky_header {
        background-color: transparent;
        padding-top: 0.3rem;
        padding-bottom: 0.3rem;
    }
    /* 고정 헤더 바로 뒤(스페이서 자리)에 오는 컴포넌트 iframe의 기본
       여백을 없앤다 — 안 그러면 기온~기압 수치와 탭 사이가 벌어져 보인다. */
    div[data-testid="stElementContainer"]:has(> iframe) {
        margin: 0 !important;
        height: 0 !important;
    }
    /* 창이 아주 좁아졌을 때 제목이 뷰포트 밖으로 흘러넘치는 안전망. */
    .st-key-sticky_header h1 {
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
    }

    /* ── 모바일·태블릿(≤900px) 대응 ─────────────────────────────
       휴대폰에서 화면이 헤더로만 채워지는 문제를 고친다(2026-08-17 제보).
       원인은 두 가지였다.
         1) 고정 헤더가 제목·설명·관측소명·지표 4개를 모두 담는데, 좁은
            화면에서는 st.columns 가 세로로 쌓여 헤더 높이가 뷰포트만큼
            커진다. 헤더가 fixed 라 그만큼의 스페이서가 본문을 아래로
            밀어내므로, 처음 화면에는 헤더밖에 안 보인다. 아래 JS 가
            이 폭에서는 고정을 해제하여 일반 흐름으로 되돌린다.
         2) 제목에 걸어둔 말줄임표 규칙이 좁은 화면에서 항상 발동해
            "기후 모델 ..." 로 잘렸다.
       ------------------------------------------------------------ */
    @media (max-width: 900px) {
        /* 제목은 자르지 말고 줄바꿈한다 — 좁을수록 잘라선 안 된다. */
        .st-key-sticky_header h1 {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
            font-size: 1.6rem !important;
            line-height: 1.25 !important;
            word-break: keep-all;
        }
        /* 헤더의 지표 4개를 2×2 로 접는다. Streamlit 은 좁은 화면에서
           컬럼을 세로로 쌓는데, 그러면 네 줄이 화면을 다 먹는다. */
        .st-key-sticky_header [data-testid="stHorizontalBlock"] {
            flex-direction: row !important;
            flex-wrap: wrap !important;
            gap: 0.25rem 0.5rem !important;
        }
        .st-key-sticky_header [data-testid="stColumn"] {
            flex: 1 1 44% !important;
            min-width: 44% !important;
            width: auto !important;
        }
        .st-key-sticky_header [data-testid="stMetricValue"] {
            font-size: 1.35rem !important;
        }
        .st-key-sticky_header [data-testid="stMetricLabel"] p {
            font-size: 0.78rem !important;
        }
        /* 상태 배지는 툴바로 옮기지 않고 본문에 그대로 둔다(JS 참조).
           잘리지 않도록 말줄임표 규칙도 해제한다. */
        .st-key-status_badge [data-testid="stAlertContainer"] {
            white-space: normal;
            overflow: visible;
            text-overflow: clip;
        }
        /* 표가 넘칠 때 가로 스크롤을 허용한다 — 글자가 겹치는 것보다 낫다. */
        [data-testid="stMarkdownContainer"] table {
            display: block;
            overflow-x: auto;
            white-space: nowrap;
        }
        /* 탭 라벨이 네 개라 좁은 화면에서 넘친다 — 가로 스크롤을 허용하고
           글자를 줄여 최대한 한 줄에 담는다. */
        [data-testid="stTabs"] [role="tablist"] {
            overflow-x: auto;
            scrollbar-width: none;
        }
        [data-testid="stTabs"] [role="tablist"]::-webkit-scrollbar { display: none; }
        [data-testid="stTabs"] [role="tab"] p { font-size: 0.86rem !important; }
        /* Plotly 툴바(카메라·확대 아이콘)는 데스크톱에서는 hover 시에만 뜨지만
           터치 기기에서는 계속 떠 있어 차트 상단을 가린다. 좁은 화면에서는
           숨긴다 — 확대·저장은 터치로 대체된다. */
        .js-plotly-plot .modebar { display: none !important; }
    }
    /* 상태 배지 — Deploy 툴바 줄로 옮겨 붙는다(JS). 그 줄이 60px로 좁으니
       내용이 넘치면 말줄임표로 정리한다. */
    .st-key-status_badge [data-testid="stAlertContainer"] {
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0 !important;
    }
    /* '각 확률의 정의와 판정 기준' 표 — 가운데 '정의' 열이 넓어서 양쪽
       '사건'·'판정선' 열이 좁게 밀려 "폭염"·"판정선" 같은 짧은 텍스트까지
       줄바꿈됐다(2026-08-16 스크린샷으로 발견). 양쪽 열만 줄바꿈을 막고
       나머지 폭은 가운데 열이 가져가게 한다. */
    .st-key-event_label_table table th:first-child,
    .st-key-event_label_table table td:first-child,
    .st-key-event_label_table table th:last-child,
    .st-key-event_label_table table td:last-child {
        white-space: nowrap;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# 상태 배지 — Deploy 버튼이 있는 Streamlit 상단 툴바 줄에 왼쪽 정렬로
# 옮겨 붙인다(2026-08-10 요청). 제목 줄과 분리해 별도 컨테이너로 두고,
# 아래 컴포넌트의 JS가 이 요소를 실제로 그 줄로 재배치한다.
with st.container(key="status_badge"):
    status = result["data_status"]
    # '실측이 아닌 것'과 '지금 것이 아닌 것'을 구분해 표시한다(2026-08-19).
    # 종전에는 폴백을 한 덩어리로 묶어 "수신 실패"라고만 했는데, 그 안에는
    # 실측(묵은 관측)과 날조 상수가 섞여 있었다 — 후자가 훨씬 심각하므로
    # 경고 수준을 달리한다.
    _age = observed_age_hours(result["observed_at"])
    _age_txt = f" · {_age:.0f}시간 전 관측" if _age is not None and _age >= 1 else ""
    if status == "SUCCESS_LIVE":
        st.success(f"실시간 관측 데이터 · {result['observed_at']}", icon="✅")
    elif status in ("FALLBACK_WINDOW", "FALLBACK_CACHED"):
        _src = "저장소 갱신분" if status == "FALLBACK_WINDOW" else "직전 조회분"
        st.warning(
            f"실시간 조회 실패 — {_src} 실측으로 대체 · "
            f"관측 {result['observed_at']}{_age_txt}",
            icon="⚠️",
        )
    elif status == "FALLBACK_DEFAULT":
        st.error(
            "실측 데이터가 없다 — 아래 '현재' 수치는 관측이 아니라 자리표시자이며, "
            "그것으로 계산한 출력값도 근거가 없다.",
            icon="🚫",
        )
    else:
        st.info(f"{status} · {result['observed_at']}")

# 예측 결과는 관측 시각 단위로 캐시된다(cached_predict). 그래서 조회가
# 실패해 폴백으로 내려간 결과도 그 정시 동안 그대로 남는다 — 자동으로
# 재시도하게 만들면 API 가 계속 죽어 있을 때 갱신 주기마다 15회씩 호출이
# 급증하므로, 재시도는 사람이 누를 때만 일어나게 한다.
if str(result.get("data_status", "")).startswith("FALLBACK"):
    if st.button("🔄 실시간 조회 다시 시도", help="캐시된 결과를 폐기하고 API를 재호출한다."):
        cached_predict.clear()
        st.rerun()

    # 연결 진단 — 왜 실패했는지를 화면에서 바로 판별하기 위한 정보다
    # (2026-08-19 추가). 이 저장소 코드는 프록시를 설정하지도 읽지도 않으므로,
    # 프록시 항목에 값이 잡히면 그건 전부 호스팅 플랫폼이 주입한 것이다 —
    # requests 가 그 변수를 자동으로 신뢰하므로 연결이 엉뚱한 경로로 나갔을
    # 수 있고, 그 가설은 배포 환경의 환경변수를 직접 봐야만 판별된다.
    with st.expander("연결 진단 정보"):
        _conn = connection_state()
        _off = offline_fallback_state()
        _proxy = network_env_report()
        st.write(
            {
                "연결 차단 감지": _conn["blocked"],
                "쿨다운 잔여(초)": _conn["cooldown_remaining"],
                "연결 실패 횟수": _conn["failures"],
                "마지막 오류": _conn["last_error"] or "(없음)",
                "저장소 폴백 관측소 수": _off["stations"],
                "저장소 폴백 최신 관측": _off["latest"] or "(없음)",
                "프록시 환경변수": _proxy or "(주입된 값 없음)",
            }
        )
        st.caption(
            "연결 차단 감지가 참이면 컬렉터가 쿨다운 동안 조회를 건너뛴다 — "
            "12개 관측소가 각자 재시도 한도를 소진해 화면이 8분간 멈추던 것을 "
            "막기 위한 장치다. 프록시 환경변수는 이 저장소가 설정하지 않으므로, "
            "값이 보이면 호스팅 플랫폼이 주입한 것이다."
        )

# 대상 관측소 조회가 성공해도 이웃(Re축)·과거(Im축) 조회가 실패하면 출력값의
# 근거가 제한된다 — 그 축 입력이 중립값으로 채워지기 때문이다. 배지 하나로는
# 드러나지 않으므로 별도로 알린다.
_cov = result.get("input_coverage") or {}
_cov_msgs = []
if _cov.get("neighbors_total") and _cov["neighbors_live"] < _cov["neighbors_total"]:
    _cov_msgs.append(
        f"공간보간장(Re축): 이웃 관측소 {_cov['neighbors_live']}/{_cov['neighbors_total']}곳만 실측"
        + (" — 보간 격자를 중립값(0.5)으로 대체했다"
           if _cov["neighbors_live"] == 0 else "")
    )
if _cov.get("lags_total") and _cov["lags_live"] < _cov["lags_total"]:
    _cov_msgs.append(
        f"시간경향벡터(Im축): 과거 시점 {_cov['lags_live']}/{_cov['lags_total']}개만 실측"
        + (" — 경향값을 전부 0(변화 없음)으로 처리했다"
           if _cov["lags_live"] == 0 else "")
    )
if _cov_msgs:
    st.warning(
        "보조 입력 일부가 실측값으로 채워지지 않았다. "
        + " · ".join(_cov_msgs)
        + "  \n결측을 추정값으로 대체하지 않고 중립값으로 두므로, 이 출력값은 "
          "그만큼 근거가 제한적이다.",
        icon="⚠️",
    )

with st.container(key="sticky_header"):
    st.title("🌦️ 기후 모델 출력값")
    # 반응형으로 줄바꿈되어 창 폭에 관계없이 전체 내용이 보이도록 한다
    # (2026-08-10) — 말줄임표로 자르지 않는다.
    st.caption("지상 관측값 3축(주변·흐름·현재 상태)을 융합하여 +6시간 후 기온·강수를 산출한 값이다 — 기상청 공식 예보가 아니다.")

    st.subheader(f"{result['station_name']} ({result['station_code']})")

    c = result["current"]
    lead = result["forecast_lead_hours"]
    f = result["forecast"]
    col1, col2, col3, col4 = st.columns(4)
    col1.metric("현재 기온", f"{c['temperature']:.1f} °C")
    col2.metric("현재 강수", f"{c['precipitation']:.1f} mm")
    col3.metric("습도", f"{c['humidity']:.0f} %")
    col4.metric("기압", f"{c['pressure']:.0f} hPa")

components.html(
    """
    <script>
    (function poll() {
        const doc = window.parent.document;
        const header = doc.querySelector('.st-key-sticky_header');
        const mainEl = doc.querySelector('[data-testid="stMain"]');
        const blockEl = doc.querySelector('[data-testid="stMainBlockContainer"]');
        const badge = doc.querySelector('.st-key-status_badge');
        const toolbarEl = doc.querySelector('[data-testid="stAppToolbar"]') ||
                           doc.querySelector('[data-testid="stHeader"]');
        if (!header || !mainEl || !blockEl) { setTimeout(poll, 300); return; }

        const wrapper = header.parentElement;   // fixed로 빠지며 생기는 공백을 되돌려줄 대상
        const appEl = doc.querySelector('[data-testid="stApp"]') || doc.body;

        // 좁은 화면(휴대폰·세로 태블릿)에서는 고정 헤더를 쓰지 않는다.
        // 헤더가 담는 제목·설명·관측소명·지표 4개는 좁아지면 세로로 쌓여
        // 높이가 뷰포트만큼 커지고, fixed 로 빠진 만큼의 스페이서가 본문을
        // 아래로 밀어내 첫 화면에 헤더밖에 안 보이게 된다(2026-08-17 제보).
        // CSS 미디어 쿼리(max-width:900px)와 같은 경계를 쓴다.
        const NARROW_MAX = 900;
        function isNarrow() {
            return (window.parent.innerWidth || doc.documentElement.clientWidth)
                   <= NARROW_MAX;
        }

        // 고정을 풀고 원래 흐름으로 되돌린다 — 인라인으로 박아둔 값을 모두
        // 지워야 CSS 가 다시 주도권을 갖는다.
        function releaseHeader() {
            header.style.position = '';
            header.style.top = '';
            header.style.left = '';
            header.style.width = '';
            header.style.zIndex = '';
            header.style.backgroundColor = '';
            wrapper.style.height = '';
        }
        function releaseBadge() {
            if (!badge) return;
            badge.style.position = '';
            badge.style.top = '';
            badge.style.left = '';
            badge.style.height = '';
            badge.style.maxWidth = '';
            badge.style.display = '';
            badge.style.alignItems = '';
            badge.style.zIndex = '';
            const bWrapper = badge.parentElement;
            bWrapper.style.height = '';
            bWrapper.style.margin = '';
            bWrapper.style.overflow = '';
        }

        function toolbarBottom() {
            return toolbarEl ? toolbarEl.getBoundingClientRect().bottom : 0;
        }

        // 고정된 헤더는 본문 위에 떠 있으므로 불투명한 배경이 있어야 아래
        // 내용이 비쳐 보이지 않는다. 그 색을 상수로 박으면 테마를 바꿨을 때
        // 헤더만 반대 색으로 남는다(Light 전환 시 검은 띠 — 2026-08-11 실측).
        // 앱 배경을 실제로 읽어서 그대로 쓰면 Light/Dark/System 어느 쪽이든
        // 자동으로 맞고, 테마 전환이 rerun 없이 일어나도 따라간다.
        function syncBackground() {
            const bg = getComputedStyle(appEl).backgroundColor;
            // 투명(rgba(...,0))이면 덮어쓰지 않는다 — 그대로 두면 그 아래
            // 실제 배경이 비치므로 검은 띠가 생기는 것보다 낫다.
            if (bg && !/,\\s*0\\s*\\)$/.test(bg) && bg !== 'transparent') {
                header.style.backgroundColor = bg;
            }
        }

        // 상태 배지("실시간 관측 데이터")를 Deploy 버튼이 있는 툴바 줄로
        // 옮겨 붙인다 — 왼쪽 정렬, 세로로는 그 줄 가운데(2026-08-10 요청).
        function positionBadge() {
            if (!badge || !toolbarEl) return;
            if (isNarrow()) { releaseBadge(); return; }
            const r = toolbarEl.getBoundingClientRect();
            badge.style.position = 'fixed';
            badge.style.top = r.top + 'px';
            badge.style.left = (r.left + 16) + 'px';
            badge.style.height = r.height + 'px';
            badge.style.maxWidth = (r.width * 0.55) + 'px';   // Deploy·메뉴와 안 겹치게
            badge.style.display = 'flex';
            badge.style.alignItems = 'center';
            badge.style.zIndex = String(999990 + 1);   // 툴바(999990) 바로 위
            badge.style.boxSizing = 'border-box';
            const bWrapper = badge.parentElement;
            bWrapper.style.height = '0px';
            bWrapper.style.margin = '0px';
            bWrapper.style.overflow = 'hidden';
        }

        function reposition() {
            if (isNarrow()) { releaseHeader(); return; }
            // 좌우 폭은 stMainBlockContainer(실제 콘텐츠 폭) 기준으로 맞추되,
            // 그 컨테이너 자체의 좌우 padding까지 더해야 탭·본문과 정확히
            // 일치한다 — 헤더는 fixed로 빠져나와 있어 blockEl의 padding을
            // 물려받지 못하고, 탭 등 아직 정상 흐름에 있는 형제들은 그 padding
            // 만큼 안쪽으로 밀려 렌더링되기 때문에 그대로 두면 16px 안팎으로
            // 어긋난다(실측: 2026-08-10).
            const blockRect = blockEl.getBoundingClientRect();
            const blockStyle = getComputedStyle(blockEl);
            const padL = parseFloat(blockStyle.paddingLeft) || 0;
            const padR = parseFloat(blockStyle.paddingRight) || 0;
            // 세로 위치는 "원래 있던 자리(wrapper의 현재 위치)"를 그대로 따라가다가,
            // 툴바 밑보다 위로 올라가려는 순간부터만 툴바 바로 밑에서 멈춘다 —
            // 진짜 position:sticky가 하는 일을 좌표로 직접 계산해 흉내낸다.
            const wrapperRect = wrapper.getBoundingClientRect();
            const top = Math.max(toolbarBottom(), wrapperRect.top);
            header.style.position = 'fixed';
            header.style.top = top + 'px';
            header.style.left = (blockRect.left + padL) + 'px';
            header.style.width = (blockRect.width - padL - padR) + 'px';
            header.style.zIndex = '999';
            header.style.boxSizing = 'border-box';
            syncBackground();
        }

        // header.dataset 플래그로 중복 초기화를 막는다 — Streamlit이 rerun마다
        // 이 컴포넌트를 다시 실행해도, 실제 헤더 DOM 노드는 그대로 재사용되는
        // 경우가 많아 리스너/옵저버가 중복으로 쌓이는 걸 막아야 한다.
        if (!header.dataset.stickyInit) {
            header.dataset.stickyInit = '1';
            const ro = new ResizeObserver((entries) => {
                for (const entry of entries) {
                    const h = entry.borderBoxSize
                        ? entry.borderBoxSize[0].blockSize
                        : entry.contentRect.height;
                    wrapper.style.height = isNarrow() ? '' : (h + 'px');
                }
                reposition();
            });
            ro.observe(header);
            // blockEl(콘텐츠 폭 기준)도 직접 관찰한다 — 사이드바 접기/펼치기는
            // 창 크기 자체는 안 바뀌므로 window 'resize' 이벤트가 안 뜬다.
            // 그 경우를 1초 간격 안전망에만 맡기면 그 사이 잠깐씩 폭이
            // 어긋나 보이므로(실측: 2026-08-10), 폭 변화를 직접 감지한다.
            const roBlock = new ResizeObserver(() => { reposition(); positionBadge(); });
            roBlock.observe(blockEl);
            mainEl.addEventListener('scroll', reposition);
            window.parent.addEventListener('resize', () => { reposition(); positionBadge(); });
            setInterval(() => { reposition(); positionBadge(); }, 1000);   // 안전망
        }
        reposition();
        positionBadge();
    })();
    </script>
    """,
    height=0,
)

tab_trend, tab_extreme, tab_perf, tab_model = st.tabs(
    ["📈 출력값 추이", "🚨 극한 기상", "✅ 성능 검증", "🧭 모델 구조"]
)

# ── 탭 1: 출력값 추이 ──────────────────────────────────────────────

with tab_trend:
    # +6h/+12h 동시 표시(2026-08-20 개편). 예전에는 라디오로 하나만 골라
    # 보여줬는데, 둘을 한 화면에서 바로 비교할 수 있어야 한다는 요청에 따라
    # 토글을 없애고 항상 둘 다 계산해 차트·표에 함께 그린다.
    #
    # 이 탭 안에서만 쓰는 지역 변수(result_12h 등)로 분리한다 — 전역
    # f/c/lead 를 여기서 재할당하면 '성능 검증' 탭이 뒤에서 그 값을 다시
    # 읽으므로(lead, f["temperature"] 등) +12h 계산이 다른 탭까지 조용히
    # 물들인다.
    result_12h = None
    try:
        _fp12 = ckpt_fingerprint(CHECKPOINT_12H)
        model_12h, ckpt_12h = get_model_at(CHECKPOINT_12H, _fp12)
        result_12h = cached_predict(
            stn, obs_hour_key(), ckpt_12h["lead_hours"], _fp12, model_12h, ckpt_12h)
    except Exception as e:
        # +6h(배포 필수 경로)와 달리 +12h 는 2차 산출값이라 없어도 앱
        # 전체가 멈출 이유는 없다 — 실패하면 그 부분만 빠진 채 표시한다.
        st.caption(
            f"⚠️ +12시간 출력값을 불러오지 못했다: {redact_secrets(str(e))}"
        )
        ckpt_12h = None
    if result_12h is not None:
        # 이 문구는 한 번 낡았다(2026-08-25 정정). 채택 당시(2026-08-18)에는
        # "폭염·강수는 비슷하게 유지되나 한파·황사가 크게 흔들린다"고 적었는데,
        # 그 뒤 +12h 한파 헤드에 중첩 전이학습을 적용해(2026-08-19, F1
        # 0.453→0.509) 한파가 오히려 +6h 를 앞질렀다. 두 체크포인트를 다시
        # 재서 실제 방향으로 고쳤다 — 리드타임별 저하 폭은 지표마다 다르며,
        # 가장 크게 무너지는 것은 강수다.
        st.caption(
            "+12시간은 2차 산출값이다 — 절대오차가 +6시간보다 크다(기온 MAE "
            "1.30→1.66°C, 검증셋 평균). 저하 폭은 지표마다 다르다: 강수가 "
            "가장 크게 무너지고(기준선 대비 −9.9%→−25.1%), 폭염이 그다음이며"
            "(F1 0.814→0.776), 한파·황사는 사실상 유지된다(0.499→0.509, "
            "0.191→0.192 — 한파는 중첩 전이학습을 따로 적용한 결과다). 계절 "
            "오탐은 1.21%로 차단 기준(10%) 아래다. '성능 검증' 탭 수치는 "
            "+6시간 기준이며 이 값의 영향을 받지 않는다."
        )

    history, history_source = load_merged_history()
    series = recent_series(history, stn, hours=72)

    # 관측 이력 창의 최신성을 명시한다. 배포 환경에서는 앱이 이 파일을
    # 유지할 수 없고(컨테이너 파일시스템 휘발), 저장소를 갱신하는
    # GitHub Actions(refresh-data.yml)가 유일한 공급자다. 그 경로가 끊기면
    # 차트가 원인 표시 없이 비어가는 대신 왜 비는지 화면에서 말하게 한다.
    #
    # 경고를 두 갈래로 나눈다(2026-08-23) — history_source 가
    # "local_fallback"이면 원인이 "data 브랜치 자체가 뒤처짐"이 아니라
    # "raw.githubusercontent.com 조회가 실패해 로컬 디스크로 대체됨"이다.
    # 이 둘을 같은 문구로 보여주면 사람이 매번 원인을 추측해야 한다 —
    # 로컬 폴백은 main 브랜치의 스냅숏이라 2026-08-20 이후로는 갱신되지
    # 않고 재배포 시점에 멈춰 있으므로, "몇 시간 지연"이 아니라 "조회가
    # 순간적으로 실패했다"는 훨씬 더 흔하고 훨씬 덜 걱정할 원인일 수 있다.
    _age = history_age_hours(history)
    if _age is None:
        st.warning(
            f"관측 이력 창(`{HIST_FILES[0]}`)이 비어 있다. 로컬 환경에서는 "
            "`python refresh_deploy_data.py`, 배포 환경에서는 GitHub Actions의 "
            "Refresh deploy data 워크플로(workflow)가 이 파일을 채운다.",
            icon="⚠️",
        )
    elif _age > 2 and history_source == "local_fallback":
        st.warning(
            f"GitHub raw(`data` 브랜치) 관측 이력 창 조회가 재시도 후에도 실패해 "
            f"로컬 디스크 스냅숏(`{HIST_FILES[0]}`)으로 대체했다 — 이 스냅숏의 "
            f"최신 데이터는 {_age:.1f}시간 전 값이며, 이 프로세스가 뜬 이후 raw "
            "조회가 한 번도 성공하지 못했다는 뜻이다. 배포 환경에서는 이 파일을 "
            "refresh-data.yml이 더 이상 갱신하지 않는다(2026-08-20부터 data "
            "브랜치 전용) — 마지막 재배포 시점에 멈춰 있으므로 갱신 경로 지연이 "
            "아니라 이 조회 자체가 실패하고 있다는 뜻이다. 다음 자동 새로고침(5분 "
            "캐시)에서 raw 조회가 다시 성공하면 정상화된다 — 반복되면 Streamlit "
            "Cloud 에서 raw.githubusercontent.com 접근 자체를 점검해야 한다.",
            icon="⚠️",
        )
    elif _age > 2 and history_source == "last_good_fallback":
        st.warning(
            f"GitHub raw 관측 이력 창 조회가 재시도 후에도 실패해 이 프로세스가 "
            f"마지막으로 성공했던 값을 대신 쓰고 있다 — 그 값도 이제 {_age:.1f}시간 "
            "전 것이다. 최근 성공 기록조차 이만큼 묵었다는 것은 순간적인 blip이 "
            "아니라 raw 조회가 상당 시간 계속 실패 중이라는 뜻이다 — Streamlit "
            "Cloud 에서 raw.githubusercontent.com 접근을 점검해야 한다.",
            icon="⚠️",
        )
    elif _age > 2:
        st.warning(
            f"관측 이력 창의 최신 데이터가 {_age:.1f}시간 전 값이다 — 갱신 경로가 "
            "중단되었을 가능성이 있다(갱신 주기는 15분이며, ASOS 공개 지연과 "
            "스케줄 지연을 감안해 2시간을 넘을 때만 이 경고를 띄운다). 차트의 "
            "'현재'는 상단 헤더의 실시간 관측값이며, 아래 곡선은 이 창의 데이터로 "
            "그린다. GitHub Actions의 Refresh deploy data 실행 이력(data 브랜치 "
            "커밋 여부)을 확인해야 한다.",
            icon="⚠️",
        )

    # 차트를 그리는 else 절 안이 아니라 **여기서** 꺼낸다(2026-08-19 수정).
    # 아래 출력값 카드는 series 길이와 무관하게 항상 그리는데, 이 세 개를
    # else 안에서 할당하면 관측 창이 비었을 때(len(series)<2) NameError 로
    # 앱 전체가 죽는다 — 갱신 경로가 72시간 넘게 끊기면 실제로 발생하는
    # 조건이고(a1329e7 참고), 그때야말로 화면이 살아 있어야 한다.
    # +6h 는 항상 전역 f/c/lead(위쪽에서 이미 계산됨)를 그대로 쓴다 — 배포
    # 필수 경로라 실패 시 이미 st.stop() 으로 앱이 멈춘 뒤이므로 여기 도달한
    # 시점엔 반드시 존재한다.
    f6, c6, lead6 = f, c, lead
    # 아래 표(모델 출력값)는 series 길이와 무관하게 항상 그리므로, f12 등도
    # 차트를 그리는 else 절 안이 아니라 여기서 미리 꺼낸다 — 관측 창이 비어
    # 차트를 건너뛰어도(len(series)<2) 표는 여전히 +12h 값을 참조한다.
    if result_12h is not None:
        f12, c12, lead12 = (result_12h["forecast"], result_12h["current"],
                            result_12h["forecast_lead_hours"])
    else:
        f12 = c12 = lead12 = None

    if len(series) < 2:
        st.caption("최근 추이를 표시하기에 로컬 캐시의 데이터가 부족하다"
                   "(수집이 진행 중이면 곧 채워진다).")
    else:
        xs = [datetime.strptime(r["timestamp"][:12], "%Y%m%d%H%M") for r in series]
        temps = [r["temperature"] for r in series]
        precs = [r["precipitation"] for r in series]
        hums  = [r["humidity"] for r in series]
        press = [r["pressure"] for r in series]
        tgt_x6 = datetime.strptime(result["target_time"][:12], "%Y%m%d%H%M")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("기온 (°C)", "강수 (mm)", "습도 (%)", "기압 (hPa)"),
            vertical_spacing=0.24, horizontal_spacing=0.14,   # 그래프 사이 여백 — 텍스트 한 줄 정도
        )

        fig.add_trace(go.Scatter(x=xs, y=temps, mode="lines", name="실측 기온",
                                 line=dict(color="#4C78A8", width=2)), row=1, col=1)
        # +6h(기본)는 꽉 찬 별, +12h(2차 산출값)는 속 빈 별이다(2026-08-20) —
        # 둘을 동시에 그리므로 심벌로 바로 구분돼야 한다. 목표 시각(tgt_x6 vs
        # tgt_x12)도 서로 달라 리드타임 차이가 x축 위치로도 드러난다.
        fig.add_trace(go.Scatter(x=[xs[-1], tgt_x6], y=[temps[-1], f6["temperature"]],
                                 mode="lines+markers", name="+6h 출력값",
                                 line=dict(color=_LEAD6_COLOR, width=1.5, dash="dot"),
                                 marker=dict(size=[0, 10], symbol="star")),
                     row=1, col=1)

        fig.add_trace(go.Bar(x=xs, y=precs, name="실측 강수",
                             marker_color="#4C78A8", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[tgt_x6], y=[f6["precipitation"]], mode="markers",
                                 name="+6h 출력값(강수)", marker=dict(size=10, symbol="star",
                                 color=_LEAD6_COLOR), showlegend=False), row=1, col=2)

        _prec_vals = list(precs) + [f6["precipitation"]]

        if result_12h is not None:
            tgt_x12 = datetime.strptime(result_12h["target_time"][:12], "%Y%m%d%H%M")
            # 보라 점선은 마지막 실측점(xs[-1])에서 +12h 별표로 잇는다(2026-08-20,
            # 원복) — +12h는 +6h 예측을 이어받아 계산하는 게 아니라 같은 현재
            # 관측값을 입력받는 독립된 체크포인트의 출력이다. 별표끼리 이으면
            # 연쇄 계산처럼 보여 오해를 준다 — +6h 선과 마찬가지로 '현재'에서
            # 부채꼴로 뻗어나가야 두 예측이 각자 독립적이라는 게 정확히 전달된다.
            fig.add_trace(go.Scatter(x=[xs[-1], tgt_x12], y=[temps[-1], f12["temperature"]],
                                     mode="lines+markers", name="+12h 출력값",
                                     line=dict(color=_LEAD12_COLOR, width=1.5, dash="dot"),
                                     marker=dict(size=[0, 10], symbol="star-open")),
                         row=1, col=1)
            fig.add_trace(go.Scatter(x=[tgt_x12], y=[f12["precipitation"]], mode="markers",
                                     name="+12h 출력값(강수)", marker=dict(size=10,
                                     symbol="star-open", color=_LEAD12_COLOR), showlegend=False),
                         row=1, col=2)
            _prec_vals.append(f12["precipitation"])

        fig.add_trace(go.Scatter(x=xs, y=hums, mode="lines", name="습도",
                                 line=dict(color="#54A24B", width=2),
                                 showlegend=False), row=2, col=1)
        fig.add_trace(go.Scatter(x=xs, y=press, mode="lines", name="기압",
                                 line=dict(color="#9C8060", width=2),
                                 showlegend=False), row=2, col=2)

        fig.update_layout(
            height=480, margin=dict(l=24, r=24, t=40, b=24),   # 바깥 여백도 텍스트 크기만큼
            showlegend=True, legend=dict(orientation="h", y=1.14, x=0),
            paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        )
        # 습도·기압은 음수가 있을 수 없는 값이다 — 자동범위 여백 때문에
        # 세로축이 0 밑으로 살짝 내려가 보이던 것을 막는다. 기온만 음수(영하)가
        # 실제로 가능하므로 그대로 자동범위를 둔다.
        fig.update_yaxes(rangemode="nonnegative", row=2, col=1)   # 습도
        fig.update_yaxes(rangemode="nonnegative", row=2, col=2)   # 기압

        # 세로축 눈금 간격을 표시 범위에 맞춰 계산한다(2026-08-25) — 예전에는
        # 넷 다(강수는 명시적으로, 나머지는 Plotly 자동선택이 우연히) 0.5
        # 근처에 머물러, 값 변동폭이 커지면 눈금이 촘촘히 밀집됐다. 범위가
        # 좁으면 축별 최소 간격(min_step)을 쓰고, 넓어지면 1·2·5 배수로 키운다.
        def _nice_dtick(vmin, vmax, min_step, target_ticks=6):
            raw_step = max(vmax - vmin, 0.0) / target_ticks
            if raw_step <= min_step:
                return min_step
            magnitude = 10 ** math.floor(math.log10(raw_step))
            for mult in (1, 2, 5, 10):
                step = mult * magnitude
                if step >= raw_step:
                    return step
            return 10 * magnitude

        _temp_vals = list(temps) + [f6["temperature"]]
        if result_12h is not None:
            _temp_vals.append(f12["temperature"])
        fig.update_yaxes(
            dtick=_nice_dtick(min(_temp_vals), max(_temp_vals), 0.5), row=1, col=1)
        fig.update_yaxes(
            dtick=_nice_dtick(min(hums, default=0.0), max(hums, default=0.0), 5),
            row=2, col=1)
        fig.update_yaxes(
            dtick=_nice_dtick(min(press, default=0.0), max(press, default=0.0), 1),
            row=2, col=2)

        # 강수 축은 하한을 -0.1로 살짝 내려 막대·별 심벌이 0선에 바로 붙지
        # 않게 한다(2026-08-20). 상한은 실제 표시 데이터(실측 막대 + 예측 별,
        # +12h 포함)에 맞춰 0.5 배수로 자동 계산하고, 눈금 간격도 그 범위에
        # 맞춰 커진다 — 큰 비가 와도 눈금이 촘촘해지지 않는다.
        _prec_max = max(_prec_vals) if _prec_vals else 0.0
        _prec_upper = max(0.5, math.ceil((_prec_max + 0.05) / 0.5) * 0.5)
        fig.update_yaxes(
            range=[-0.1, _prec_upper], dtick=_nice_dtick(0.0, _prec_upper, 0.5),
            row=1, col=2)
        st.plotly_chart(fig, width="stretch")
        _star_caption = (
            f"선과 막대는 실측값이다 — 최근 72시간 중 관측이 있는 {len(series)}개 "
            f"시점. 꽉 찬 ★ 는 +{lead6}시간 후 예측값이다."
        )
        if result_12h is not None:
            _star_caption += f" 속 빈 ☆ 는 +{lead12}시간 후 예측값(2차 산출값)이다."
        _star_caption += " 예측 대상은 기온·강수 두 항목이며, 습도·기압은 참고용 실측값으로 예측 대상이 아니다."
        st.caption(_star_caption)

    st.markdown("#### 모델 출력값")
    _val_temp_mae6 = ckpt.get("val_temp_mae")
    _val_precip_mae6 = ckpt.get("val_precip_mae")
    _d_temp6 = f6["temperature"] - c6["temperature"]
    _d_precip6 = f6["precipitation"] - c6["precipitation"]
    _rows = [
        {
            "항목": "기온 출력값",
            "+6시간(기본)": f"{f6['temperature']:.1f} °C ({_d_temp6:+.1f} 현재 대비)"
                          + (f" ±{_val_temp_mae6:.2f}" if _val_temp_mae6 is not None else ""),
        },
        {
            "항목": "강수 출력값",
            "+6시간(기본)": f"{f6['precipitation']:.1f} mm ({_d_precip6:+.1f} 현재 대비)"
                          + (f" ±{_val_precip_mae6:.3f}" if _val_precip_mae6 is not None else ""),
        },
    ]
    # 상승/하강 방향에 따른 글자색(2026-08-20) — 현재 대비 증가는 빨강,
    # 감소는 초록. 표 셀 문자열에서 부호를 다시 파싱하지 않도록, 델타 부호를
    # 같은 모양의 별도 표에 담아 pandas Styler 로 입힌다.
    _deltas = [{"항목": "기온 출력값", "+6시간(기본)": _d_temp6},
               {"항목": "강수 출력값", "+6시간(기본)": _d_precip6}]
    if result_12h is not None:
        _val_temp_mae12 = ckpt_12h.get("val_temp_mae")
        _val_precip_mae12 = ckpt_12h.get("val_precip_mae")
        _d_temp12 = f12["temperature"] - c12["temperature"]
        _d_precip12 = f12["precipitation"] - c12["precipitation"]
        _rows[0]["+12시간(2차 산출값)"] = (
            f"{f12['temperature']:.1f} °C ({_d_temp12:+.1f} 현재 대비)"
            + (f" ±{_val_temp_mae12:.2f}" if _val_temp_mae12 is not None else "")
        )
        _rows[1]["+12시간(2차 산출값)"] = (
            f"{f12['precipitation']:.1f} mm ({_d_precip12:+.1f} 현재 대비)"
            + (f" ±{_val_precip_mae12:.3f}" if _val_precip_mae12 is not None else "")
        )
        _deltas[0]["+12시간(2차 산출값)"] = _d_temp12
        _deltas[1]["+12시간(2차 산출값)"] = _d_precip12
    else:
        _rows[0]["+12시간(2차 산출값)"] = "불러오기 실패"
        _rows[1]["+12시간(2차 산출값)"] = "불러오기 실패"
        _deltas[0]["+12시간(2차 산출값)"] = 0.0
        _deltas[1]["+12시간(2차 산출값)"] = 0.0
    _out_df = pd.DataFrame(_rows).set_index("항목")
    _delta_df = pd.DataFrame(_deltas).set_index("항목")

    def _color_by_delta(_ignored):
        return _delta_df.map(
            lambda d: "color:#C0392B" if d > 0 else ("color:#1E8449" if d < 0 else "")
        )

    # st.dataframe 이 아니라 st.table 을 쓴다(2026-08-20) — 전자는 캔버스로
    # 그려서 글자 크기도 열 제목 색도 지정할 수 없다. 열 제목을 차트의 별
    # 심벌과 같은 색으로 맞추려면 Styler 의 set_table_styles 가 필요하고,
    # 그건 HTML 로 렌더되는 st.table 에서만 먹는다.
    _tbl_styles = [
        {"selector": "th, td", "props": [("font-size", "1.05rem")]},
        {"selector": "th.col_heading.level0.col0",
         "props": [("color", _LEAD6_COLOR)]},
        {"selector": "th.col_heading.level0.col1",
         "props": [("color", _LEAD12_COLOR)]},
    ]
    st.table(
        _out_df.style
        .apply(_color_by_delta, axis=None)
        .set_table_styles(_tbl_styles)
    )
    st.caption(
        "기온은 '현재 기온 대비 변화량(Δ)'을 더한 값이고, 강수는 강수 확률과 "
        "강수량 추정치를 곱한 값이다(허들(Hurdle) 구조 — '모델 구조' 탭에 상세 "
        f"설명). 강수 확률이 +6시간은 {PRECIP_PROB_GATE_BY_LEAD[6]:.0%}, "
        f"+12시간은 {PRECIP_PROB_GATE_BY_LEAD[12]:.0%} 미만이면 강수량을 0으로 "
        "반올림한다(리드타임마다 확률 분포가 달라 임계값을 따로 골랐다). "
        "'±' 뒤 숫자는 "
        "검증셋 평균절대오차(MAE)이며 이 예측 1건의 신뢰구간이 아니다. 빨간 "
        "글자는 현재 대비 상승, 초록 글자는 하강 전망이다. 열 제목 색은 위 "
        "차트의 별 색과 같다 — 주황이 꽉 찬 별(+6시간), 보라가 속 빈 "
        "별(+12시간)이다."
    )
    st.caption(
        "⚠️ 강수량 출력값은 본 프로젝트에서 가장 취약한 부분이다 — '성능 검증' "
        "탭의 강수 MAE 설명을 함께 참조한다."
    )

# ── 탭 2: 극한기상 ───────────────────────────────────────────────

with tab_extreme:
    ev = result.get("extreme_event_probs") or {}
    if not any(ev.get(k) is not None for k in ("heatwave", "coldwave", "dust")):
        st.caption("이 체크포인트는 극한기상 헤드가 없다(구버전) — "
                   "재학습된 체크포인트를 사용하면 표시된다.")
    else:
        st.caption(
            "이 막대는 확률이며, 발령된 경보가 아니다. 회귀(기온·강수)와 별개로 "
            "학습된 이진분류(binary classification) 헤드의 출력이고, 빨간 점선은 "
            "사건 판정 임계값(threshold)이다. 대부분의 시각은 사건이 아니므로 "
            "확률이 낮게 나오는 것이 정상이며, 실제 폭염·한파가 임박할 때만 "
            "임계값을 초과한다."
        )
        st.caption(
            "이 값은 실제 발생 빈도에 맞춰 보정(calibration)한 확률이다 — "
            "\"30%\"로 표시된 상황들을 모으면 실제로 약 30%에서 사건이 발생했다는 "
            "의미로 해석할 수 있다. 보정 전 원본 출력은 체계적으로 과신했다(한파 "
            "60~70%로 표시되던 구간의 실제 빈도가 27%). 희소 사건이 '항상 아님'"
            "으로 수렴하는 것을 방지하기 위해 학습 시 양성 표본에 가중치를 준 것이 "
            "확률을 위로 밀어올린 결과이며, 검증셋에서 등온 회귀(isotonic "
            "regression)로 되돌렸다. 판정 임계값도 보정 후 공간의 값을 쓰므로 "
            "판정 성능(F1)은 유지되거나 소폭 개선된다. 상세 내용은 '성능 검증' "
            "탭을 참조한다."
        )
        if ev.get("heatwave") is not None:
            event_gauge("🔥 폭염 확률", ev["heatwave"],
                        event_threshold("heatwave", stn, ckpt), "#E2954F")
        if ev.get("coldwave") is not None:
            event_gauge("🥶 한파 확률", ev["coldwave"],
                        event_threshold("coldwave", stn, ckpt), "#4C78A8")

        # 정밀도는 체크포인트에서 읽는다 — 상수로 적어두면 재학습·미세조정 때마다
        # 화면만 옛 값에 머문다(2026-08-17에 실제로 10.7%로 굳어 있었다).
        _dust_p = ((ckpt.get("extreme_metrics") or {}).get("dust") or {}).get("precision")
        _dust_txt = (f"{_dust_p:.1%}(황사로 판정한 것 중 {1 - _dust_p:.1%}가 오탐)"
                     if _dust_p else "매우 낮은 수준")
        st.caption(
            f"⚠️ 황사 확률은 이 화면에서 제외했다. 재보정 후에도 정밀도(precision)가 "
            f"{_dust_txt}에 불과하여 실용 수준에 미달한다. "
            "PM10 농도는 국외(주로 중국) 장거리 수송의 영향이 지배적이나, 본 모델의 "
            "입력은 국내 12개 지점의 기온·습도·기압·풍속에 한정되어 구조적으로 "
            "판별 단서가 부족하다. 헤드 자체는 유지되어 값은 계산되나, 신뢰도가 "
            "낮아 화면에는 표시하지 않는다."
        )

        st.markdown("##### 각 확률의 정의와 판정 기준")
        with st.container(key="event_label_table"):
            st.markdown(
                "| 사건 | 정답 라벨(label)의 정의 | 판정 임계값 |\n"
                "|---|---|---|\n"
                "| 🔥 폭염 | 기상청이 발표한 폭염주의보 기록(해당 날짜·관측소). "
                "기록이 없는 표본은 학습·평가에서 모두 제외 | "
                f"{event_threshold('heatwave', stn, ckpt):.0%} |\n"
                "| 🥶 한파 | 발표된 한파주의보 기록. 동일 기준으로 기록이 없는 "
                "표본은 제외 | "
                f"{event_threshold('coldwave', stn, ckpt):.0%} |\n"
            )
        st.caption(
            "이전에는 공식 기록이 없는 표본을 순간 기온 임계값(폭염 33°C 이상·"
            "한파 −12°C 이하)으로 근사하여 채웠다. 그러나 공식 특보는 일 단위로 "
            "발효되는 반면 근사값은 해당 시각의 기온에만 반응하여, 동일 현상에 "
            "정의가 상이한 두 라벨이 혼재하는 문제가 있었다. 공식 기록이 있는 "
            "표본만 사용하도록 수정했다(2026-08-16 재학습)."
        )
        st.caption(
            "판정 임계값이 0.5가 아닌 이유: 사건이 드물어 0.5가 최선이 아닐 수 "
            "있으므로, 검증셋을 둘로 나누어 재보정을 시도한다(절차는 '성능 검증' "
            "탭 참조). 전체 관측소를 합산하면 폭염·한파는 재보정 이득이 통계적으로 "
            "유의하지 않아(순이득 각각 0.004·0.004 — 채택 기준 0.01 미달) 원본 "
            "확률 기준 0.5를 유지했다. 강수만 재산출값(0.83)을 채택했다. 위에 "
            "표시되는 값이 0.5보다 낮은 것은 확률 보정을 거친 뒤의 값이기 "
            "때문이다. 판정선은 보정 후 공간에서 따로 정한다 — 폭염은 원본 "
            "임계값을 보정 곡선으로 옮긴 값을 그대로 쓰고(판정 동일), 한파는 "
            "옮기기만 하면 곡선의 평탄 구간 때문에 판정이 어긋나 보정 공간에서 "
            "다시 골랐다(순이득 +0.0125). 근거는 '성능 검증' 탭 참조.  \n"
            "이전 모델에서는 부산 폭염에만 전용 임계값(0.33)을 적용했다. 현행 "
            "모델에서 재검증한 결과 부산 전용값의 순이득이 −0.0033으로 오히려 전역 "
            "0.5보다 나빠(부산 F1 0.574) 제거했다 — 검증을 통과하지 못한 예외는 "
            "두지 않는다. 반대로 전주(146) 한파는 같은 분리 검증에서 순이득 "
            "+0.0689로 기준을 넘어 전용 임계값(0.33)을 적용 중이다(2026-08-19)."
        )
        st.caption(
            "호우(강수)는 이 탭에 게이지가 없다 — '출력값 추이' 탭의 강수량 값이 "
            "이미 강수 확률을 곱하여 산출된 값이기 때문이다('모델 구조' 탭의 "
            "허들(Hurdle) 구조 설명 참조)."
        )

# ── 탭 3: 성능 검증 ──────────────────────────────────────────────

with tab_perf:
    val_temp = ckpt.get("val_temp_mae")
    val_precip = ckpt.get("val_precip_mae")
    if val_temp is not None:
        st.markdown("#### 회귀(regression) 성능 — 기온·강수 오차")
        st.caption(
            "평균절대오차(MAE) = |예측값 − 실제값|의 평균이다. 0에 가까울수록 "
            "우수하며, 단위는 예측 대상과 동일하다(기온 °C, 강수 mm). 아래 값은 "
            "학습에 사용하지 않고 별도로 분리한 검증셋(전체의 20%)에서만 측정했다."
        )
        # MAE 는 기준선 없이는 우열을 말할 수 없는 지표다. 특히 강수는 대부분의
        # 시각이 0mm 라 "항상 0mm 라고 답하는" 자명한 기준선의 MAE 도 아주 낮게
        # 나오고, 이 프로젝트의 모델은 실제로 그 기준선보다 나쁘다. 그 사실을
        # 가리는 표현을 쓰지 않는다는 것이 프로젝트 규칙이다.
        _naive_temp   = ckpt.get("val_temp_naive_mae")
        _naive_precip = ckpt.get("val_precip_naive_mae")

        def _baseline_delta(model_mae, naive_mae):
            """
            기준선 대비 증감 문구. 부호만 붙이면 "-0.9% vs 기준선"이 "0.9%
            더 좋다"로 읽힐 수 있다 — MAE 는 낮을수록 좋은 지표라 방향이
            직관과 반대이기 때문이다. 그래서 숫자 뒤에 판정을 문구로 병기한다.
            부호는 유지해 Streamlit 이 색(초록/빨강)으로도 구분하게 둔다.
            """
            if not naive_mae:
                return None
            r = (naive_mae - model_mae) / naive_mae
            return f"{r:+.1%} · " + ("기준선보다 정확" if r > 0 else "기준선 미달")

        vc1, vc2 = st.columns(2)
        vc1.metric(
            "기온 MAE", f"{val_temp:.2f} °C",
            delta=_baseline_delta(val_temp, _naive_temp),
            help=f"평균적으로 실제 기온과 이 값만큼 차이가 난다. 기준선은 "
                 f"\"{lead}시간 후에도 현재와 동일한 기온\"으로 답하는 퍼시스턴스"
                 f"(persistence) 방식"
                 + (f"이며, {_naive_temp:.2f}°C다." if _naive_temp else "이다."),
        )
        vc2.metric(
            # 강수는 0.17 vs 0.168 처럼 소수점 둘째 자리에서 같아 보인다 —
            # 기준선과의 차이가 작을수록 자릿수를 늘려야 비교가 성립한다.
            "강수 MAE", f"{val_precip:.3f} mm",
            delta=_baseline_delta(val_precip, _naive_precip),
            help="기준선은 상시 '0mm'로 답하는 방식이다"
                 + (f"({_naive_precip:.3f}mm)." if _naive_precip else ".")
                 + " 이 값만으로는 판단할 수 없다 — 아래 설명을 참조한다.",
        )
        st.caption(
            "기준선(baseline)은 학습을 거치지 않은 자명한 방법이다. MAE는 그 "
            "자체로는 성능의 좋고 나쁨을 판단할 수 없으며, 기준선과의 비교를 "
            "통해서만 의미를 가진다. 위 기준선 값은 모델과 동일한 검증 표본에서 "
            "계산했다(표본이 다르면 표본 차이만으로 개선처럼 보일 수 있다)."
        )

        if _naive_precip and val_precip > _naive_precip:
            _gap = (val_precip - _naive_precip) / _naive_precip
            st.warning(
                f"강수는 아직 기준선에 미달한다. 모델 {val_precip:.4f}mm vs 상시 "
                f"무강수 예측 {_naive_precip:.4f}mm — 격차는 {_gap:.1%}다. 즉 이 "
                f"모델의 강수량 출력은 아무것도 예측하지 않는 방법보다 평균 오차가 "
                f"(근소하게) 크다. 이 사실을 은폐하지 않고 그대로 표시한다.\n\n"
                f"원인은 구조적으로 규명되었다 — 극한기상 헤드가 3개로 증가하면서 "
                f"헤드 간에 공유 표현(shared representation)을 두고 경쟁이 발생해 "
                f"무강수 억제력이 약화되었다(초기 −25.6%). 학습 데이터를 3.6년에서 "
                f"13.6년으로 확장해도 격차가 좁혀지지 않아, 데이터양이 아니라 경쟁 "
                f"구조 자체가 원인임을 확인했다. 2026-08-17 극한기상 헤드에 별도 "
                f"입력을 주면서(아래 '계절 오탐' 참조) 격차가 −17.9%에서 현재 "
                f"수준으로 좁혀졌다 — 헤드들이 필요한 정보를 공유 표현 밖에서 얻게 "
                f"되어 경쟁이 완화된 것으로 해석한다. 다만 기준선은 아직 넘지 "
                f"못했다. 강수는 MAE보다 '강수 발생 여부의 적중 여부'로 판단하는 "
                f"것이 실용적이며, 해당 지표는 아래 '출력값 적중률'의 강수 항목을 "
                f"참조한다.\n\n"
                f"구간별로 쪼개면 격차의 출처가 하나로 좁혀진다(2026-08-17 실측). "
                f"1mm 이상 전 구간에서는 모델이 기준선을 이긴다. 지는 이유는 표본의 "
                f"93.8%인 무강수 구간에서 softplus가 정확한 0을 내지 못해 표본마다 "
                f"평균 0.032mm를 깔기 때문이며, 이 구간 하나가 전체 격차의 201%를 "
                f"만든다. 위 수치는 후처리 전 모델 출력 기준이고, 화면에 실제로 "
                f"나가는 값은 0.2mm 미만을 0으로 반올림한 뒤라 격차가 약 6%로 "
                f"줄어든다.\n\n"
                f"그래서 강수의 주 지표를 바꿨다. **비가 오는 구간의 조건부 MAE는 "
                f"2.240mm로 기준선(2.458mm) 대비 8.9% 개선**이고, 강수 발생 판정 "
                f"F1은 0.427이다. 전체 MAE는 두 질문 — '비가 올 때 얼마나 맞히는가'와 "
                f"'비가 안 올 때 0을 얼마나 잘 내는가' — 을 하나로 통합하므로 어느 "
                f"쪽도 알려주지 않는다. 정직성 차원에서 계속 표시하되 판단 근거로는 "
                f"위 두 지표를 먼저 본다.",
                icon="⚠️",
            )
        elif _naive_precip is None:
            st.warning(
                f"강수 MAE {val_precip:.2f}mm는 '정확히 예측한다'는 의미가 아니다. "
                "대부분의 시각은 강수가 없으므로, 아무 계산 없이 상시 0mm로 답해도 "
                "MAE는 유사하게 낮게 나온다. 학습 로그 기준으로 현재 모델의 강수 "
                "MAE는 이 기준선보다 나쁘다. 강수는 '강수 발생 여부의 적중 여부'로 "
                "판단해야 하며, 해당 지표는 아래 '출력값 적중률'의 강수 항목을 "
                "참조한다.",
                icon="⚠️",
            )
            st.caption(
                "이 체크포인트에는 기준선 수치가 저장되어 있지 않아 화면에서 직접 "
                "비교할 수 없다(학습 시 계산은 하나 저장하지 않던 버전이다). 다음 "
                "재학습부터는 저장되어 위 지표 옆에 증감으로 표시된다."
            )

    # 극한기상 헤드의 분류 성능 — 체크포인트에 저장돼 있는데도 화면에 없었다
    # (2026-08-11 점검에서 발견: 위 문단이 "아래 극한기상 F1을 함께 봐야
    # 합니다"라고 안내하는데 정작 그 표가 없었다).
    _em = ckpt.get("extreme_metrics") or {}
    if _em:
        st.markdown("---")
        st.markdown("#### 극한기상 분류 성능")
        st.caption(
            "드문 사건은 정확도(accuracy)로 평가해서는 안 된다. 예를 들어 폭염 "
            "발생일이 전체의 1%라면, 아무 계산 없이 '항상 폭염 아님'으로 답해도 "
            "정확도는 99%에 달한다 — 폭염을 한 번도 맞히지 못했음에도 그렇다. "
            "따라서 아래 세 지표로 평가한다."
        )
        st.markdown(
            "- 정밀도(precision) — 모델이 \"발생\"으로 판정한 것 중 실제로 발생한 "
            "비율. 낮을수록 오탐(false positive)이 많다.\n"
            "- 재현율(recall) — 실제로 발생한 것 중 모델이 판별한 비율. 낮을수록 "
            "미탐지(false negative)가 많다.\n"
            "- F1 — 정밀도와 재현율의 조화평균(harmonic mean)이다. 한쪽만 우수하고 "
            "다른 쪽이 저조하면 함께 낮아지므로, 두 지표를 하나로 요약할 때 "
            "사용한다. 범위는 0~1이며 1이 최댓값이다."
        )
        _label_ko = {"heatwave": "🔥 폭염", "coldwave": "🥶 한파"}
        _rows = ["| 사건 | 정밀도 | 재현율 | F1 | 검증셋 실제 발생 |",
                 "|---|---|---|---|---|"]
        for _k in ("heatwave", "coldwave"):
            _m = _em.get(_k)
            if not _m:
                continue
            _rows.append(
                f"| {_label_ko[_k]} | {_m['precision']:.0%} | {_m['recall']:.0%} | "
                f"{_m['f1']:.3f} | {_m['n_pos']:,}건 |"
            )
        st.markdown("\n".join(_rows))
        st.caption(
            "주의 — 이 표의 값은 보정 전 원본 확률 0.5를 기준으로 계산했으므로, "
            "'극한 기상' 탭의 판정 임계값(폭염 "
            f"{event_threshold('heatwave', stn, ckpt):.0%} · 한파 {event_threshold('coldwave', stn, ckpt):.0%})"
            "을 적용했을 때의 성능과는 다르다. 임계값을 높이면 오탐(정밀도↑)은 "
            "감소하고 미탐지(재현율↓)는 증가한다."
        )
        _dust_m = _em.get("dust")
        if _dust_m:
            st.caption(
                f"🌫️ 황사는 이 표에서 제외했다 — 정밀도 {_dust_m['precision']:.0%}"
                f"(F1 {_dust_m['f1']:.3f})로 재보정해도 실용 수준에 미달한다. "
                "상세 사유는 '극한 기상' 탭을 참조한다."
            )

        st.markdown("---")
        st.markdown("#### 관측소별 성능 — 신뢰도 비교")
        st.caption(
            "위 표는 12개 관측소를 합산한 평균이다. 관측소별로 분해하면 양상이 "
            "다르다. 가로축은 재현율, 세로축은 정밀도이며 관측소마다 점 하나로 "
            "표시하고, 십자 표시는 95% 신뢰구간(confidence interval, Wilson score)"
            "이다. 한파·폭염은 수일간 지속되는 사건이므로 동일 사건 내 연속 시간 "
            "표본은 서로 강하게 상관되어 있다(독립이 아님) — 이에 '24시간 이내로 "
            "연속된 표본을 하나의 사건으로 묶은 수'를 유효 표본 크기(effective "
            "sample size)로 사용한다. 사건 수가 적을수록 신뢰구간이 커진다(신뢰도가 "
            "낮음을 그대로 반영한다). 정확도 대신 재현율·정밀도를 사용하는 이유는 "
            "위와 동일하다(양성 사례 희박)."
        )
        _calib_path = "./docs/images/calibration_plot.png"
        if os.path.exists(_calib_path):
            st.image(_calib_path, width="stretch")
            st.caption(
                "폭염은 부산(159)이 여전히 가장 낮으나(재현율 63.4%·정밀도 "
                "54.0%) 이전 모델(71%·47%)보다 정밀도가 개선되었다. 부산만 공식 "
                "판정 기준이 다른데 모델은 관측소 구분 없이 공통 기준으로 "
                "학습된다는 구조는 그대로이므로, 격차 자체는 남아 있다. 이전 "
                "모델에서는 이를 보완하려 부산 전용 임계값을 적용했으나 현행 "
                "모델에서는 검증을 통과하지 못해 제거했다(위 '판정 임계값' 참조).  \n"
                "한파는 관측소별 편차가 크며 원인은 표본 부족이다. 사건 수로 세면 "
                "춘천 40건인 반면 광주 4건·부산 2건·제주 1건에 불과하다 — 시간 "
                "단위로는 5,300건 안팎으로 충분해 보이지만, 한파는 며칠씩 이어져 "
                "연속 표본이 서로 독립이 아니다. 사건이 적은 관측소일수록 신뢰구간"
                "(십자)이 크게 그려지는 것이 이 사실을 그대로 보여준다."
            )
        else:
            st.caption(
                "관측소별 플롯이 아직 생성되지 않았다 — `calibration_plot_diagnose.py`"
                "를 실행하면 생성된다(재학습 시마다 갱신이 필요하다)."
            )

    # ── 확률 보정 ───────────────────────────────────────────────
    _cal = (ckpt.get("prob_calibration") or {}).get("heads") or {}
    if _cal:
        st.markdown("---")
        st.markdown("#### 확률 보정 — 표시된 확률을 액면 그대로 읽어도 되는가")
        st.caption(
            "위 지표는 '얼마나 잘 맞히는가'를 재지만, '모델이 말한 확률이 실제 "
            "빈도와 일치하는가'는 별개의 문제다. 후자를 측정한 결과 네 헤드 모두 "
            "체계적으로 과신했다 — 한파를 60~70%로 표시하던 구간의 실제 발생 "
            "빈도는 27%에 불과했다."
        )
        st.caption(
            "원인은 결함이 아니라 설계상의 대가다. 한파처럼 양성이 6%뿐인 사건은 "
            "모델이 '항상 아님'으로 답하는 자명해로 수렴하기 쉬워, 학습 시 양성 "
            "표본의 손실에 가중치(pos_weight)를 준다. 이것이 재현율을 확보하는 "
            "대신 확률을 위로 밀어올린다. 실제로 보정 오차의 크기 순서가 이 "
            "가중치의 크기 순서와 정확히 일치했다(폭염 3.1배 → 한파 13.6배 → "
            "강수 15.6배 → 황사 36.0배)."
        )
        st.caption(
            "재학습 없이 등온 회귀(isotonic regression)로 되돌렸다. 검증셋을 "
            "보정용·평가용으로 나누어 보정용에서만 곡선을 적합하고, 평가용에서 "
            "개선을 확인한 헤드만 채택했다(임계값 산출과 동일한 절차)."
        )
        st.caption(
            "보정은 단조 증가 변환이지만 순증가는 아니다 — 등온 회귀 결과는 계단 "
            "모양이라 평탄 구간에서 서로 다른 확률이 같은 값으로 접힌다. 따라서 "
            "원본 임계값을 곡선으로 옮기는 것만으로는 판정이 보존되지 않는다. "
            "폭염은 보존되어 옮긴 값을 그대로 쓰고, 한파는 F1이 0.5054에서 "
            "0.4948로 떨어져 판정선을 보정 공간에서 다시 골랐다(0.5071, 순이득 "
            "+0.0124). 재선정도 고르는 표본과 채점하는 표본을 분리한다."
        )
        # 보정 전 열은 검증셋 전체(probability_calibration_check.py), 보정 후 열과
        # F1 은 평가용 절반(probability_calibration_fit.py)에서 잰 값이다. 두 열의
        # 표본이 다르므로 그 사실을 캡션에 밝힌다 — 밝히지 않으면 README 와
        # 소수점 셋째 자리가 어긋나 보인다(2026-08-17에 실제로 폭염이 0.0338/0.0340
        # 으로 갈렸다. 헤드마다 다른 표본을 섞어 적은 것이 원인이었다).
        _rows = ["| 사건 | 보정 전 ECE | 보정 후 ECE | F1 변화 |", "|---|---|---|---|"]
        for _k, _ko, _e0, _e1, _f in (
            # 승격할 때마다 사람이 갱신해야 하는 값이다 —
            # `probability_calibration_fit.py`(--apply 없이 실행)의 요약표에서
            # 그대로 옮긴다. 2026-08-21 한파 헤드 승격 후 재측정본.
            ("heatwave", "🔥 폭염", 0.0340, 0.0048, "0.8144 → 0.8144"),
            ("coldwave", "🥶 한파", 0.0397, 0.0015, "0.5054 → 0.5071"),
        ):
            if _k in _cal:
                _rows.append(f"| {_ko} | {_e0:.4f} | **{_e1:.4f}** | {_f} |")
        st.markdown("\n".join(_rows))
        st.caption(
            "ECE(Expected Calibration Error)는 구간별 '표시 확률 − 실제 빈도'의 "
            "차이를 표본 수로 가중평균한 값이다. 0에 가까울수록 "
            "표시값을 그대로 신뢰할 수 있다. 보정 전 열은 검증셋 전체에서, 보정 후 "
            "열과 F1은 곡선 적합에 쓰지 않은 평가용 절반에서 측정했다 — 표본이 "
            "달라 소수점 셋째 자리에서 차이가 날 수 있다."
        )
        _cal_png = "./docs/images/probability_calibration.png"
        if os.path.exists(_cal_png):
            with st.expander("보정 전 신뢰도 곡선 보기"):
                st.image(_cal_png, width="stretch")
                st.caption(
                    "네 곡선 모두 대각선 아래에 있다 — 모델이 말한 확률이 실제 "
                    "빈도보다 일관되게 높다는 의미다. 이 그림은 보정을 적용하기 "
                    "전 상태이며, 현재 화면의 확률은 이 어긋남을 되돌린 값이다."
                )

    st.markdown("---")
    st.markdown("#### 판정 임계값 산출 절차 — 과적합 방지")
    st.caption(
        "'극한 기상' 탭의 판정 임계값은 다음 절차로 산출했다. ① 검증 데이터를 "
        "보정용(calibration)·평가용(test)으로 무작위 절반씩 분리한다(학습·검증 "
        "분할과는 다른 난수를 사용한다). ② 보정용에서 F1이 최대가 되는 값을 "
        "탐색한다. ③ 그 값을 한 번도 참조하지 않은 평가용에 적용하여 성능이 "
        "유지되는지 확인한다."
    )
    st.caption(
        "이 절차가 필요한 이유: 동일 데이터에서 값을 선정하고 같은 데이터로 "
        "채점하면 해당 데이터에만 최적화된 값이 우수해 보인다(과적합). 실제로 "
        "강수만 이 검증을 통과해 재산출값(0.83)을 채택했고, 폭염·한파·황사는 "
        "통과하지 못해(순이득 각각 0.004·0.004·0.002 — 채택 기준 0.01 미달) "
        "원본 확률 기준 0.5를 유지했다."
    )

    st.markdown("---")
    st.markdown("#### 출력값 적중률(실측 대조)")
    st.caption(
        f"기온 — 오차 ±{accuracy.HIT_TEMP_TOL}°C 이내면 적중. "
        f"강수 — {accuracy.PRECIP_THRESH}mm 기준 강수/무강수 판정 일치 시 적중."
    )

    if status == "SUCCESS_LIVE" and result.get("target_time"):
        accuracy.record_prediction(
            station=stn, made_at=result["observed_at"],
            target_time=result["target_time"],
            pred_temp=f["temperature"], pred_precip=f["precipitation"],
            source="live", model_id=ckpt_fingerprint(),
        )
    accuracy.resolve_pending(history)

    # model_id 로 필터링 — 체크포인트를 바꾼 뒤에는 옛 모델이 쌓아둔 기록과
    # 섞이지 않고 지금 배포된 모델의 예측만 집계한다(accuracy.stats 참고).
    acc = accuracy.stats(station=stn, model_id=ckpt_fingerprint())
    if acc["cum_n"] == 0:
        # 아래 '로그 구성' 줄은 로그 전체(전 관측소·전 모델)를 세는데 이 지표는
        # 현재 관측소 + 현재 체크포인트로만 좁혀 센다. 그래서 "대조된 것이
        # 없다"와 "스냅샷 N건"이 한 화면에 동시에 뜰 수 있다 — 모순이 아니라
        # 집계 범위가 다른 것이므로, 왜 0인지를 문구에 밝힌다(2026-08-20 수정.
        # 그 전에는 이유 없이 "없다"고만 해서 바로 아래 줄과 어긋나 보였다).
        st.caption(
            f"현재 배포 중인 체크포인트로 {result['station_name']}에서 대조 완료된 "
            "출력값이 아직 없다 — 이 지표는 체크포인트를 교체하면 0에서 다시 "
            "시작한다(옛 모델의 기록과 섞으면 어느 모델의 성능도 아닌 값이 "
            "되기 때문이다). 아래 '로그 구성'의 건수는 전 관측소·전 모델을 "
            "합친 값이라 이 숫자와 범위가 다르다."
        )
    else:
        a1, a2, a3, a4 = st.columns(4)
        a1.metric("누적 기온 적중률",
                 f"{acc['cum_temp']:.0%}" if acc["cum_temp"] is not None else "—")
        a2.metric("누적 강수 적중률",
                 f"{acc['cum_precip']:.0%}" if acc["cum_precip"] is not None else "—")
        a3.metric(f"최근 {acc['recent_n']}건 기온",
                 f"{acc['recent_temp']:.0%}" if acc["recent_temp"] is not None else "—")
        a4.metric(f"최근 {acc['recent_n']}건 강수",
                 f"{acc['recent_precip']:.0%}" if acc["recent_precip"] is not None else "—")
        st.caption(f"누적 표본 {acc['cum_n']}건({result['station_name']} 기준). "
                   "이 지표는 검증셋 MAE와 다르다 — 실제 운영 시각에 조회 가능했던 "
                   "데이터로만 산출한 출력값을, 이후 확정된 실측값과 대조한 것이다.")

    # 로그가 어디까지 영구인지 밝힌다. 배포 컨테이너의 파일시스템은
    # 휘발성이라 앱이 방금 추가한 항목은 재시작하면 사라지고, 저장소에
    # 커밋된 스냅샷만 남는다. 두 숫자를 합쳐 하나로 보여주면 "쌓이고 있다"는
    # 인상을 주는데 실제로는 아니므로 나눠서 표시한다.
    _base = accuracy_baseline()
    _now_log = accuracy.log_summary()
    _added = _now_log["total"] - _base["total"]
    _snapshot_at = _base["latest_target"]
    st.caption(
        f"로그 구성 — 저장소 스냅샷 {_base['total']}건"
        + (f"(최근 대조 {_snapshot_at[:4]}-{_snapshot_at[4:6]}-{_snapshot_at[6:8]} "
           f"{_snapshot_at[8:10]}시)" if _snapshot_at else "")
        + f" + 이 앱 실행 중 추가 {_added}건 · 대조 대기 {_now_log['pending']}건"
    )
    st.caption(
        "실행 중 추가된 항목은 이 프로세스가 유지되는 동안만 존재한다 — 배포 "
        "환경의 컨테이너 파일시스템은 재시작 시 저장소 상태로 복원된다. 영구 "
        "누적은 GitHub Actions의 Refresh deploy data 워크플로가 담당한다. 신규 "
        "관측을 수신하여 대기 항목을 실측과 대조한 후, 결과를 저장소에 "
        "재커밋한다."
    )

# ── 탭 4: 모델 구조 ──────────────────────────────────────────────

with tab_model:
    gw = result["gate_weights"]
    if gw:
        st.markdown("#### 축 기여도 — 이번 출력값에 대한 3축 배분")
        st.caption(
            "3축의 기여도는 고정값이 아니라 출력마다 입력을 근거로 재계산된다. "
            "소형 신경망이 세 값을 산출하고 소프트맥스(softmax, 합이 1이 되도록 "
            "정규화하는 함수)를 거치므로 합은 항상 100%다. 아래 '3축 융합 수식'의 "
            "`w`가 이 값이다."
        )
        axes = ["수치 센서 (Z)", "시간경향벡터 (Im)", "공간보간장 (Re)"]
        values = [gw.get("w_z", 0) * 100, gw.get("w_im", 0) * 100, gw.get("w_re", 0) * 100]
        colors = ["#54A24B", "#F58518", "#4C78A8"]
        fig = go.Figure(go.Bar(
            x=values, y=axes, orientation="h", marker_color=colors,
            text=[f"{v:.1f}%" for v in values], textposition="outside",
            cliponaxis=False,
        ))
        fig.update_layout(
            xaxis=dict(range=[0, 100], title="배분 비율 (%)"),
            height=210, margin=dict(l=10, r=40, t=10, b=10),
        )
        st.plotly_chart(fig, width="stretch")
    else:
        st.caption(f"정적 축 가중치 — {gw}")

    st.markdown("---")
    st.markdown("#### 3축이 각각 포착하는 정보")
    st.markdown(
        "본 모델은 출력값 하나를 산출할 때 서로 다른 세 질문에 답하고, 그 답을 "
        "하나로 융합한다. 세 질문은 다음과 같다."
    )
    st.markdown(
        "| 축 | 질문 | 입력 데이터 |\n"
        "|---|---|---|\n"
        "| Z축(현재 상태) | 이 관측소는 현재 어떤 상태인가 | 이번 시각의 "
        "관측값 — 기온·강수·습도·풍속·풍향·기압·위도·경도·연중 시각 등 수치 14개 |\n"
        "| Re축(주변 상태) | 주변 지역은 현재 어떤 상태인가 | 나머지 11개 "
        "관측소의 동일 시각 관측값을, 거리에 따라 가중치를 두어 지도 형태로 "
        "보간(거리가중 보간) |\n"
        "| Im축(변화 추세) | 어느 방향으로 변화하는 중인가 | 동일 관측소의 "
        "1·3·6시간 전 대비 변화량 — 기압 상승/하강 등의 추세 정보 |\n"
    )
    st.caption(
        "Re·Im·Z라는 명칭의 유래 — 아래에서 설명할 원 논문이 세 축을 복소수"
        "(complex number)에 빗대어 명명했으며, 그 명칭을 그대로 차용했다. Re는 "
        "실수부(real), Im은 허수부(imaginary)에서 유래한다. 복소수에서 이 둘은 "
        "서로 직교하므로, \"세 축이 서로 겹치지 않는 정보를 담는다\"는 설계 "
        "의도를 명칭으로 표현한 것이다. 명칭일 뿐 실제로 복소수 연산을 수행하지는 "
        "않는다 — 아래 수식은 전부 실수 연산이다."
    )
    st.caption(
        "세 질문이 서로 독립적이라는 점이 핵심이다. 주변 관측소의 값(Re)과 과거 "
        "대비 변화량(Im)은 이번 시각의 관측값(Z)만으로는 산출할 수 없는 "
        "정보다. 따라서 축을 추가한 만큼 실제로 새로운 정보가 유입된다."
    )
    st.caption(
        "Re·Im 두 축은 원 논문에서 위성 이미지·기상 문서가 위치하던 자리다. "
        "본 프로젝트의 관측망(지상 관측소 12곳, 위성·텍스트 없음)에는 그대로 "
        "이전할 수 없어 초기에는 대체 데이터를 사용했으나, 해당 축을 제거해도 "
        "출력이 전혀 변화하지 않아 정보량이 0임을 확인했다. 이후 '단일 시점 "
        "스냅샷만으로는 유도할 수 없는 정보만 신규 축에 포함한다'는 기준으로 "
        "재설계한 것이 현재의 공간·시간 축이다 — 주변 관측소의 값과 과거 대비 "
        "변화량은 원리적으로 현재 시각의 스냅샷만으로는 산출할 수 없다."
    )
    _diag = ckpt.get("diagnostics") or {}
    if _diag.get("cos_re_z_post") is not None:
        st.caption(
            "다만 '입력이 독립'인 것과 '학습된 표현이 독립'인 것은 별개다. 학습 "
            "후 세 축 벡터가 이루는 각도를 코사인 유사도(cosine similarity)로 "
            f"측정하면 Re–Z {_diag['cos_re_z_post']:.2f} · "
            f"Re–Im {_diag['cos_re_im_post']:.2f} · "
            f"Im–Z {_diag['cos_im_z_post']:.2f}다(0이면 완전히 다른 방향, 1이면 "
            "동일 방향). 0은 아니지만 낮은 편으로, 각 축이 상당히 다른 정보를 "
            "포착하고 있음을 시사한다."
        )

    st.markdown("---")
    st.markdown("#### 3축 융합 수식(논문 Eq.1)")
    st.caption(
        "3축은 각각 벡터(수치가 나열된 목록)로 변환된 후 아래 수식으로 융합된다. "
        "기호가 생소해 보이나 연산 자체는 단순하다 — 아래에서 단계별로 설명한다."
    )
    st.latex(
        r"s_i \;=\; \sqrt{\;(w_{Re}\,\cdot\,Re_i)^2 \;+\; (w_{Im}\,\cdot\,Im_i)^2"
        r"\;+\; (w_{Z}\,\cdot\,Z_i)^2\;}"
    )

    _embed_dim = ckpt.get("embed_dim", 64)
    _w_re, _w_im, _w_z = gw.get("w_re"), gw.get("w_im"), gw.get("w_z")
    _fmt = (lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—")
    st.markdown(
        "기호 정리\n\n"
        "| 기호 | 설명 | 값 |\n"
        "|---|---|---|\n"
        f"| `Re`, `Im`, `Z` | 3축을 각각 인공신경망에 통과시켜 얻은 "
        f"{_embed_dim}차원 벡터 | 축당 {_embed_dim}개 |\n"
        f"| `i` | 벡터 내 위치를 가리키는 인덱스 | 1~{_embed_dim} |\n"
        f"| `w` | 각 축의 가중치(합은 항상 1). 이번 출력값 실시간 계산값 — "
        f"예측마다 입력에 따라 달라진다(검증셋 전체 평균과는 다른 수치) | "
        f"Re {_fmt(_w_re)} · Im {_fmt(_w_im)} · Z {_fmt(_w_z)} |\n"
        f"| `s` | 융합 결과 벡터({_embed_dim}차원). 이 벡터 하나를 5개 예측 "
        f"헤드가 공유한다 | {_embed_dim}개 |\n"
        "| √ | 제곱근(제곱의 역연산) | — |\n"
    )
    st.markdown(
        "연산 절차\n\n"
        "1. 곱셈 — 각 축의 값에 해당 축의 가중치를 곱한다. 가중치가 낮게 배분된 "
        "축은 이 단계에서 값이 작아진다.\n"
        "2. 제곱 — 부호(+/−)를 제거하고 크기만 남긴다. 두 축이 반대 부호일 때 "
        "합산 과정에서 상쇄되어 소실되는 것을 방지한다.\n"
        "3. 합산 후 제곱근 — 직각삼각형의 빗변 길이를 구하는 연산과 동일하다 "
        "(밑변 3, 높이 4면 빗변 5). 3축을 서로 직교하는 방향으로 두고 그 합성 "
        "크기를 측정하는 방식이므로, 한 축이 커지면 결과도 커지되 다른 축이 이를 "
        "완전히 상쇄하지는 못한다.\n\n"
        f"이 3단계는 벡터의 각 원소(element)마다 독립적으로 적용된다. 따라서 "
        f"결과 `s`도 스칼라가 아니라 {_embed_dim}차원 벡터다."
    )
    st.caption(
        "각 축 벡터는 융합 전에 노름(norm)이 1이 되도록 정규화된다. 따라서 "
        "축별 원본 값의 크기는 결과에 영향을 주지 않으며, 패턴의 방향과 "
        "가중치만으로 결과가 결정된다."
    )

    with st.expander("이 수식의 출처 — 원 논문과의 관계"):
        st.markdown(
            "본 프로젝트는 아래 논문의 융합 방식만 차용하여 상이한 분야에 적용한 "
            "실험이다.\n\n"
            "> Tri-CHEF: Complex-Hermitian Embedding Fusion for Korean Multimodal "
            "Retrieval  \n"
            "> Zenodo 프리프린트(2026년 5월) · DOI [10.5281/zenodo.20034370]"
            "(https://doi.org/10.5281/zenodo.20034370) · CC BY 4.0"
        )
        st.markdown(
            "원 논문은 기후 예측과 무관하며, 한국어 멀티모달 검색(retrieval) "
            "시스템에 관한 연구다 — 문서·이미지·영상·음성을 통합 검색할 때, "
            "상이한 세 인코더의 출력을 어떻게 하나의 점수로 융합할 것인가를 "
            "다룬다.\n\n"
            "가장 흔한 방법은 가중합(각 점수에 가중치를 곱해 합산)이나, 이 경우 "
            "한 채널의 점수가 특히 크면 해당 채널이 최종 점수를 실질적으로 "
            "독점한다. 논문이 제안한 대안이 위의 제곱→합산→제곱근 형태이며, "
            "이것이 논문의 Eq.1이다. 각 축을 제곱하여 합산하므로 한 축이 다른 "
            "축을 상쇄하여 소거하지 못하고, 3축의 근거가 모두 최종값에 반영된다."
        )
        st.markdown(
            "본 프로젝트가 변경한 사항\n\n"
            "| | 원 논문 | 본 프로젝트 |\n"
            "|---|---|---|\n"
            "| 과제 | 검색(질의에 부합하는 문서·이미지 탐색) | 기후 예측(+"
            f"{lead}시간 후 기온·강수) |\n"
            "| 3축 입력 | 사전학습 인코더 3종의 출력 | 현재 상태(Z)·주변(Re)·"
            "추세(Im) 관측값 |\n"
            "| 축 가중치 | 학습 후 고정 | 출력마다 입력 조건부로 재계산 |\n"
            "| 수식의 출력 | 검색 순위 점수 | 5개 예측 헤드의 공통 입력 |\n"
        )
        st.markdown(
            "논문의 원 표기는 `s = √(A² + (αB)² + (φC)²)`다. A·B·C는 각각 "
            "Re·Im·Z축이며, α(알파)·φ(파이)는 A(Re축)를 1로 두었을 때 나머지 두 "
            "축의 상대적 비중이다. 위 수식은 3축을 대등하게 `w`로 표기한 동일한 "
            "식이며, `α = w(Im) ÷ w(Re)`, `φ = w(Z) ÷ w(Re)`로 상호 환산된다"
            + (f" — 이번 출력에서는 α={gw['alpha']:.3f}, φ={gw['phi']:.3f}다."
               if isinstance(gw.get("alpha"), (int, float)) else ".")
        )
        st.caption(
            "원 논문에서 α·φ는 학습 완료 후 고정되는 상수다. 본 프로젝트에서는 "
            "이를 그대로 적용했더니 정보가 없는 축의 가중치가 오히려 증가하는 "
            "문제가 발생하여, 출력마다 재계산하는 방식으로 변경했다(위 '축 "
            "기여도' 그래프가 그 결과다). 원 논문과 다르게 변경한 지점이며, "
            "변경 근거는 실측에서 도출했다."
        )

    st.markdown("---")
    st.markdown("#### 융합 벡터에서 분기하는 5개 예측")
    st.markdown(
        f"위에서 산출한 {_embed_dim}차원 벡터 `s`까지가 공통 과정이다. 이제 이 "
        "벡터 하나를 6개의 소형 신경망이 각자 읽어 5가지 예측을 산출한다. "
        "공통 부분 뒤에 연결되는 이러한 소형 신경망을 헤드(head)라 부른다 — "
        "몸통 하나에 여러 개의 머리가 달린 구조에서 유래한 명칭이다.\n\n"
        f"- 기온 — 현재 기온 대비 변화량(Δ)을 산출해 현재 기온에 더한다. 기온 "
        f"자체를 직접 예측하지 않는 이유는, +{lead}시간 정도에서는 \"현재와 "
        f"유사할 것\"이라는 답이 이미 상당히 정확하기 때문이다. 그 위에 변화량만 "
        f"학습하는 편이 유리하다.\n"
        "- 강수 — 헤드 2개의 출력을 곱하여 산출한다(아래 상세 설명).\n"
        "- 폭염·한파·황사 — 각각 발생 확률 1개씩 산출한다. 폭염·한파는 "
        "'극한 기상' 탭의 막대가 이 값이며, 황사는 신뢰도가 낮아 화면에서 "
        "제외했다."
    )
    st.caption(
        "기온·강수처럼 수치를 예측하는 과제를 회귀(regression), 폭염·한파·황사"
        "처럼 발생 여부를 판별하는 과제를 이진분류(binary classification)라 "
        "부른다. 둘은 학습 방식이 상이하여 헤드를 분리한다."
    )
    if ckpt.get("signed_head_input"):
        st.caption(
            "극한기상 헤드 3종만 `s` 외에 Z축 벡터를 함께 입력받는다(2026-08-17). "
            "위 융합 수식은 제곱을 거치므로 부호가 사라지고, 헤드는 '얼마나 "
            "벗어났는가'만 알 뿐 '위로 벗어났는지 아래로 벗어났는지'를 알 수 "
            "없다. 그 결과 이전 모델은 한겨울에 폭염 확률을 올렸다 — 실제 −15°C "
            "이하 관측의 85.7%에서 폭염 확률이 판정선을 넘었다. Z축 벡터는 제곱을 "
            "거치지 않아 부호가 살아 있으므로, 이를 함께 넣어 두 극단을 구분하게 "
            "했다(오탐 85.7% → 0.08%). 기온 헤드가 절대값 대신 변화량을 학습하는 "
            "것과 같은 이유의 조치이며, 상세 내용은 '성능 검증' 탭을 참조한다."
        )
    st.markdown("강수의 허들(Hurdle) 구조 — 두 헤드를 곱하는 이유")
    st.latex(r"\hat{y}_{precip} \;=\; \sigma(z_1)\;\times\;\mathrm{softplus}(z_2)")
    st.markdown(
        "- ŷ(`y_precip`) — 최종 강수량 예측값이다. 모자(^) 표기는 \"실제값이 "
        "아니라 모델의 추정값\"이라는 관례다.\n"
        "- z₁, z₂ — 위에서 산출한 벡터 `s`를 각각 별도의 소형 신경망에 "
        "통과시켜 얻은 중간값이다. 아직 확률도 mm도 아닌 미가공 값이므로, "
        "아래 두 함수로 각각 의미 있는 범위로 변환한다.\n"
        "- σ(시그모이드, sigmoid) — 임의의 실수를 0~1 범위로 압축하는 함수다. "
        "따라서 `σ(z₁)`은 강수 확률로 해석한다.\n"
        "- softplus — 결과를 항상 0보다 크게 만드는 함수다. 강수량은 음수가 "
        "될 수 없어 사용한다. 더 단순한 방법(음수를 0으로 절단)은 절단 구간에서 "
        "학습 신호가 끊겨 사용하지 않았다.\n"
        "- 곱셈의 의미 — 강수 확률이 0에 가까우면 추정량이 커도 최종값은 0에 "
        "수렴한다. \"강수 발생 여부\"와 \"발생 시 강수량\"을 하나의 수식에 담는 "
        "구조이며, 무강수가 대부분인 데이터에서 정확한 0을 산출하기 위한 "
        "장치다.\n"
        f"- 마지막으로 강수 확률이 "
        f"{PRECIP_PROB_GATE_BY_LEAD.get(ckpt.get('lead_hours'), PRECIP_PROB_GATE):.0%} "
        "미만이면 강수량을 0으로 반올림한다(리드타임마다 확률 분포가 달라 "
        "체크포인트별로 임계값을 따로 골랐다 — 위 표 캡션 참고) — 매그니튜드"
        "(mm) 임계값 대신 확률 자체로 자르면 강수 발생 판정 F1을 해치지 "
        "않으면서 무강수 구간의 잔여 오차를 줄일 수 있음을 실측했다"
        "(precip_prob_gate_sweep.py, 2026-08-29)."
    )

    st.markdown("---")
    st.markdown("#### 실시간 관측의 용도와 재학습의 관계")
    st.markdown(
        f"아니다. 화면의 모델은 고정된 체크포인트이며, 실시간 관측은 두 가지 "
        f"용도로만 사용된다.\n\n"
        f"1. 추론 입력 — 현재 시점의 관측값(인접 11개 관측소 + 과거 1h/3h/6h "
        f"포함)을 모델에 입력하여 +{lead}시간 후를 예측한다. 가중치는 변경되지 "
        f"않는다.\n"
        f"2. 적중률 기록(모니터링) — 출력값을 로그에 기록하고 +{lead}시간 후 "
        f"실제값이 확정되면 대조하여 위 '성능 검증' 탭의 누적 적중률에 반영한다. "
        f"이 대조 결과는 모델을 변경하지 않는다 — 순수한 사후 기록이다.\n\n"
        f"모델을 갱신(파인튜닝)하려면 `train.py`를 다시 실행하여 축적된 "
        f"데이터로 배치 재학습을 수행해야 한다. 관측이 유입될 때마다 즉시 "
        f"가중치를 갱신하는 온라인 학습은 의도적으로 포함하지 않았다."
    )
    with st.expander("온라인 학습(즉시 파인튜닝)을 배제한 근거"):
        st.markdown(
            "- 표본 밀도 — 관측소당 신규 표본이 시간당 1건에 불과하다. 배치 1 "
            "에폭이 처리하는 수만 건 규모의 그레디언트 평균과 달리, 1건 단위 "
            "업데이트는 노이즈가 학습 신호를 초과할 위험이 크다.\n"
            "- 재앙적 망각(catastrophic forgetting) — 최근 데이터에 즉시 맞춰 "
            "갱신하면 이전에 학습한 계절 패턴을 망각할 수 있다. 이를 방지할 "
            "리플레이 버퍼나 정규화 장치가 아직 없어, 배제하는 것이 더 안전한 "
            "선택이라 판단했다.\n"
            "- 재현성(reproducibility) — 배치 재학습은 '이 체크포인트는 이 "
            "데이터와 이 코드로 학습되었다'를 그대로 재현·감사할 수 있다. "
            "온라인 학습은 매 순간 가중치가 변화하여 동일 상황을 재현하기 "
            "어렵다.\n\n"
            "즉, 본 프로젝트에서 '실시간'은 실시간 추론을 의미하며 실시간 "
            "학습을 의미하지 않는다 — 두 개념을 구분하여 표기하는 것 자체가 "
            "본 대시보드가 지키는 원칙 중 하나다."
        )

st.markdown("---")
st.caption(
    "Tri-CHEF — 공간(Re)·시간(Im)·현재 상태(Z) 3축을 논문 Eq.1로 융합하여 "
    "예측하는 파이프라인이다(수식과 기호 설명은 '모델 구조' 탭 참조). Re축은 "
    "12개 관측소의 실측 공간보간장, Im축은 동일 관측소의 시간 변화율이며, 둘 "
    "다 실측에서 산출한 값이다 — 시뮬레이션이나 합성 데이터가 아니다."
)
st.caption(
    "⚠️ 이 화면의 수치는 모델 출력값이며 기상청의 공식 예보·특보가 아니다. "
    "연구·포트폴리오 목적으로 제작했으며, 실제 방재 판단의 근거로 사용해서는 "
    "안 된다. 공식 정보는 기상청(weather.go.kr)에서 확인한다."
)
