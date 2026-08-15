"""
app.py — Tri-CHEF 기후 모델 출력 Streamlit 대시보드

학습된 3축 파이프라인(predict.py)을 실시간 관측과 연결해 보여준다.
모델 로드는 st.cache_resource 로 한 번만 수행 — Streamlit 은 위젯 조작마다
스크립트 전체를 재실행하므로 캐싱 없이는 매번 체크포인트를 다시 읽는다.
(2026-08-09부터 Im축이 MiniLM 텍스트에서 시간 경향 벡터로 바뀌면서
sentence-transformers 로드 자체가 없어졌다 — 캐싱할 무거운 리소스는
모델 하나뿐이다.)

화면 구성은 탭 4개로 나뉜다 — 출력값 추이 / 극한기상 / 성능 검증 / 모델 구조.
숫자·그래프마다 그 근거(무엇을 어떻게 측정했는지)를 캡션이나 st.metric의
help 툴팁으로 바로 옆에 붙인다 — "이 값이 왜 이렇게 나왔는지"를 화면만
보고 답할 수 있어야 한다는 원칙(2026-08-09 논의)을 따른다.
"""
import json
import os
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

from weather_collector import STATIONS, STATION_COORDS, api_call_stats
from predict import (
    load_model, predict, CHECKPOINT, EXTREME_EVENT_THRESH, PRECIP_CLIP_THRESH,
)
import accuracy

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

st.set_page_config(
    page_title="Tri-CHEF 기후 모델 출력값",
    page_icon="🌦️",
    layout="centered",
)


# ── 캐시된 리소스 — 세션당 한 번만 로드 ──────────────────────────────

def ckpt_fingerprint() -> str:
    """
    체크포인트 파일의 신원(수정시각·크기). 캐시 키에 넣어 파일이 바뀌면
    자동으로 새로 로드되게 한다.

    왜 필요한가 — get_model() 을 인수 없이 @st.cache_resource 로 감싸면
    캐시 키가 항상 같아서, 체크포인트를 새로 학습해 배포해도 **살아 있는
    프로세스는 옛 모델을 계속 재사용**한다. 2026-08-11 재학습 배포에서
    실제로 발생했다: 저장소에는 새 체크포인트가 올라갔는데 화면은 옛
    수치(기온 1.27 / 폭염 F1 0.822)를 그대로 보여줬고, 기준선 증감도
    "저장돼 있지 않다"로 표시됐다. cache_resource 는 스크립트 재실행은
    물론 코드 갱신 후에도 같은 프로세스면 살아남기 때문이다.
    """
    try:
        st_ = os.stat(CHECKPOINT)
        return f"{st_.st_mtime_ns}:{st_.st_size}"
    except OSError:
        return "missing"


@st.cache_resource(show_spinner="모델 로드 중...")
def get_model(fingerprint: str):
    """fingerprint 는 캐시 무효화 전용 — 값 자체는 쓰지 않는다."""
    return load_model(CHECKPOINT)


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
# 관측소당 시간당 15회가 상한이고, 이는 자동 갱신 주기와 무관하다.
# 애초에 ASOS 는 매시 정각 관측이라 같은 시각을 다시 조회해도 같은 값이다.
#
# model/ckpt 는 언더스코어 접두어로 넘긴다 — Streamlit 이 해시 대상에서
# 제외하는 규약이다(torch 모듈은 해시할 수 없다).
@st.cache_data(ttl=3600, show_spinner="관측 조회 중... (12개 관측소)")
def cached_predict(stn: str, obs_hour: str, _model, _ckpt) -> dict:
    return predict(stn=stn, model=_model, ckpt=_ckpt)


@st.cache_data(ttl=300, show_spinner=False)
def load_merged_history() -> dict:
    """
    (관측소, 시각) → 레코드. collect_year.py/collect_incremental.py 가 만드는
    두 캐시 파일을 합친다. 5분 캐시라 백그라운드 수집이 새 데이터를 채워도
    페이지가 곧 따라잡는다 — 매 새로고침마다 두 JSON을 다시 읽지는 않는다.
    """
    merged = {}
    for path in HIST_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            for r in json.load(f):
                merged[(r["stn"], r["timestamp"])] = r
    return merged


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
    노출해 "묵은 창"을 조용히 넘기지 않는다. 창이 비었으면 None.
    """
    if not history:
        return None
    latest = max(datetime.strptime(ts[:12], "%Y%m%d%H%M") for _, ts in history)
    return (_now_kst_naive() - latest).total_seconds() / 3600


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
    "🔄 자동 갱신이 하는 일: 설정한 주기마다 페이지를 다시 실행해, 기상청 "
    "ASOS 실시간관측(API)에서 이 관측소 + 이웃 11개 관측소의 현재값과 "
    "이 관측소의 과거 1·3·6시간 전 값을 새로 조회합니다. **모델을 다시 "
    "학습시키는 게 아니라, 그 값으로 추론만 새로 수행**합니다."
)
st.sidebar.caption(
    "단, 관측은 매시 정각에 한 번뿐이고 약 10분 뒤 공개됩니다. 그래서 **같은 정시 "
    "안에서 다시 실행하면 새로 조회하지 않고 직전 결과를 재사용**합니다 — 갱신 주기를 "
    "짧게 잡아도 API 호출이 늘지 않고, 대신 새 값도 정시가 바뀌어야 나옵니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**관측소 위치** — 지도에서 도시 선택 가능")


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
    "출력값 한 건마다 12개 관측소를 모두 조회합니다 — 선택한 관측소는 계산 입력으로, "
    "나머지 11개는 '모델 구조' 탭에서 설명하는 공간 축(Re) 계산에 쓰입니다."
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
    "이 앱 프로세스가 시작된 뒤의 누적치입니다(재시작 시 0으로 초기화). "
    "상한을 넘기면 새 조회 대신 마지막 성공값으로 내려가고, 화면 상단 배지가 "
    "그 상태를 표시합니다."
)


# ── 메인 ──────────────────────────────────────────────────────────

if not os.path.exists(CHECKPOINT):
    st.error(
        f"체크포인트가 없습니다: `{CHECKPOINT}`\n\n"
        f"먼저 터미널에서 `python train.py` 를 실행해 모델을 학습하세요."
    )
    st.stop()

try:
    model, ckpt = get_model(ckpt_fingerprint())
    result = cached_predict(stn, obs_hour_key(), model, ckpt)
except Exception as e:
    st.error(f"예측 실패: {e}")
    st.stop()

# 이번 실행의 조회분까지 반영해 사이드바 사용량을 채운다(위 api_usage_slot 참고).
_calls = api_call_stats()
api_usage_slot.caption(
    f"**API 호출** — 오늘 {_calls['count']}건"
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
    /* 상태 배지 — Deploy 툴바 줄로 옮겨 붙는다(JS). 그 줄이 60px로 좁으니
       내용이 넘치면 말줄임표로 정리한다. */
    .st-key-status_badge [data-testid="stAlertContainer"] {
        min-width: 0;
        white-space: nowrap;
        overflow: hidden;
        text-overflow: ellipsis;
        margin: 0 !important;
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
    if status == "SUCCESS_LIVE":
        st.success(f"실시간 관측 데이터 · {result['observed_at']}", icon="✅")
    elif status.startswith("FALLBACK"):
        st.warning(
            f"실시간 데이터 수신 실패 — 대체 데이터 사용 중 ({status}) · "
            f"{result['observed_at']}",
            icon="⚠️",
        )
    else:
        st.info(f"{status} · {result['observed_at']}")

# 예측 결과는 관측 시각 단위로 캐시된다(cached_predict). 그래서 조회가
# 실패해 폴백으로 내려간 결과도 그 정시 동안 그대로 남는다 — 자동으로
# 재시도하게 만들면 API 가 계속 죽어 있을 때 갱신 주기마다 15회씩 호출을
# 쏟아내므로, 재시도는 사람이 누를 때만 일어나게 한다.
if str(result.get("data_status", "")).startswith("FALLBACK"):
    if st.button("🔄 실시간 조회 다시 시도", help="캐시된 결과를 버리고 API를 다시 호출합니다"):
        cached_predict.clear()
        st.rerun()

# 대상 관측소 조회가 성공해도 이웃(Re축)·과거(Im축) 조회가 실패하면 출력값의
# 근거가 얇아진다 — 그 축 입력이 중립값으로 채워지기 때문이다. 배지 하나로는
# 드러나지 않으므로 별도로 알린다.
_cov = result.get("input_coverage") or {}
_cov_msgs = []
if _cov.get("neighbors_total") and _cov["neighbors_live"] < _cov["neighbors_total"]:
    _cov_msgs.append(
        f"공간보간장(Re축): 이웃 관측소 {_cov['neighbors_live']}/{_cov['neighbors_total']}곳만 실측"
        + (" — 보간 격자가 중립값(0.5)으로 채워졌습니다"
           if _cov["neighbors_live"] == 0 else "")
    )
if _cov.get("lags_total") and _cov["lags_live"] < _cov["lags_total"]:
    _cov_msgs.append(
        f"시간경향벡터(Im축): 과거 시점 {_cov['lags_live']}/{_cov['lags_total']}개만 실측"
        + (" — 경향이 전부 0(변화 정보 없음)으로 처리됐습니다"
           if _cov["lags_live"] == 0 else "")
    )
if _cov_msgs:
    st.warning(
        "**보조 입력 일부가 실측으로 채워지지 않았습니다.** "
        + " · ".join(_cov_msgs)
        + "  \n결측을 추정값으로 메우지 않고 중립값으로 두기 때문에, 이 출력값은 "
          "그만큼 근거가 얇습니다.",
        icon="⚠️",
    )

with st.container(key="sticky_header"):
    st.title("🌦️ 기후 모델 출력값")
    # 반응형으로 줄바꿈되어 창 폭에 관계없이 전체 내용이 보이도록 한다
    # (2026-08-10) — 말줄임표로 자르지 않는다.
    st.caption("지상 관측값 3축(주변·흐름·지금)을 합쳐 +6시간 뒤 기온·강수를 계산한 값입니다 — 기상청 예보가 아닙니다")

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
                    wrapper.style.height = h + 'px';
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
    history = load_merged_history()
    series = recent_series(history, stn, hours=72)

    # 관측 이력 창의 신선도를 명시한다. 배포 환경에서는 앱이 이 파일을
    # 유지할 수 없고(컨테이너 파일시스템 휘발), 저장소를 갱신하는
    # GitHub Actions(refresh-data.yml)가 유일한 공급자다. 그 경로가 끊기면
    # 차트가 조용히 비어가는 대신 왜 비는지 화면에서 말하게 한다.
    _age = history_age_hours(history)
    if _age is None:
        st.warning(
            f"관측 이력 창(`{HIST_FILES[0]}`)이 비어 있습니다. "
            "로컬에서는 `python refresh_deploy_data.py`, 배포판에서는 "
            "GitHub Actions의 **Refresh deploy data** 워크플로가 이 파일을 채웁니다.",
            icon="⚠️",
        )
    elif _age > 12:
        st.warning(
            f"관측 이력 창의 최신 데이터가 **{_age:.0f}시간 전** 것입니다 — "
            "갱신 경로가 멈췄을 수 있습니다(정상은 6시간 이내). "
            "차트의 '현재'는 위 헤더의 실시간 관측값이고, 아래 곡선은 이 창에서 "
            "그립니다. GitHub Actions의 **Refresh deploy data** 실행 이력을 확인하세요.",
            icon="⚠️",
        )

    if len(series) < 2:
        st.caption("최근 추이를 그리기엔 로컬 캐시에 데이터가 부족합니다 "
                   "(수집이 진행 중이면 곧 채워집니다).")
    else:
        xs = [datetime.strptime(r["timestamp"][:12], "%Y%m%d%H%M") for r in series]
        temps = [r["temperature"] for r in series]
        precs = [r["precipitation"] for r in series]
        hums  = [r["humidity"] for r in series]
        press = [r["pressure"] for r in series]
        tgt_x = datetime.strptime(result["target_time"][:12], "%Y%m%d%H%M")

        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=("기온 (°C)", "강수 (mm)", "습도 (%)", "기압 (hPa)"),
            vertical_spacing=0.24, horizontal_spacing=0.14,   # 그래프 사이 여백 — 텍스트 한 줄 정도
        )

        fig.add_trace(go.Scatter(x=xs, y=temps, mode="lines", name="실측 기온",
                                 line=dict(color="#4C78A8", width=2)), row=1, col=1)
        fig.add_trace(go.Scatter(x=[xs[-1], tgt_x], y=[temps[-1], f["temperature"]],
                                 mode="lines+markers", name="모델 출력값",
                                 line=dict(color="#E2954F", width=1.5, dash="dot"),
                                 marker=dict(size=[0, 10], symbol="star")),
                     row=1, col=1)

        fig.add_trace(go.Bar(x=xs, y=precs, name="실측 강수",
                             marker_color="#4C78A8", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[tgt_x], y=[f["precipitation"]], mode="markers",
                                 name="모델 출력값(강수)", marker=dict(size=10, symbol="star",
                                 color="#E2954F"), showlegend=False), row=1, col=2)

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
        # 강수·습도·기압은 음수가 있을 수 없는 값이다 — 자동범위 여백 때문에
        # (예: 강수가 거의 0이면) 세로축이 0 밑으로 살짝 내려가 보이던 것을
        # 막는다. 기온만 음수(영하)가 실제로 가능하므로 그대로 자동범위를 둔다.
        fig.update_yaxes(rangemode="nonnegative", row=1, col=2)   # 강수
        fig.update_yaxes(rangemode="nonnegative", row=2, col=1)   # 습도
        fig.update_yaxes(rangemode="nonnegative", row=2, col=2)   # 기압
        st.plotly_chart(fig, width="stretch")
        st.caption(
            f"선과 막대는 **실제 관측값**입니다 — 최근 72시간 중 관측이 있는 "
            f"{len(series)}개 시점. ★ 별표가 이번 +{lead}시간 예측값입니다. "
            f"예측 대상은 기온·강수 둘뿐이고, 습도·기압은 상황을 읽는 데 참고하시라고 "
            f"함께 그린 관측값입니다(예측하지 않습니다)."
        )

    st.markdown("#### +%d시간 뒤 모델 출력값" % lead)
    fc1, fc2 = st.columns(2)
    fc1.metric(
        "기온 출력값", f"{f['temperature']:.1f} °C",
        delta=f"{f['temperature'] - c['temperature']:+.1f} °C (현재 대비)",
        help="모델은 절대 기온을 바로 맞히지 않고 '현재 기온에서 얼마나 변할지'를 "
             "계산해 더합니다. 평균적으로 '성능 검증' 탭의 기온 MAE만큼 오차가 "
             "예상됩니다.",
    )
    fc2.metric(
        "강수 출력값", f"{f['precipitation']:.1f} mm",
        delta=f"{f['precipitation'] - c['precipitation']:+.1f} mm (현재 대비)",
        delta_color="inverse",
        help=f"'비가 올 확률' × '온다면 몇 mm'를 곱해서 나온 값입니다(Hurdle 구조 — "
             f"'모델 구조' 탭에 그림과 설명). 확률이 낮으면 양이 커도 최종값이 0에 "
             f"가까워집니다. {PRECIP_CLIP_THRESH}mm 미만은 0으로 반올림합니다.",
    )
    st.caption(
        "⚠️ 강수량은 이 프로젝트에서 가장 약한 부분입니다 — '성능 검증' 탭의 "
        "강수 MAE 설명을 함께 읽어주세요."
    )

# ── 탭 2: 극한기상 ───────────────────────────────────────────────

with tab_extreme:
    ev = result.get("extreme_event_probs") or {}
    if not any(ev.get(k) is not None for k in ("heatwave", "coldwave", "dust")):
        st.caption("이 체크포인트는 극한기상 헤드가 없습니다(구버전) — "
                   "재학습된 체크포인트를 사용하면 표시됩니다.")
    else:
        st.caption(
            "**이 막대는 확률입니다 — 발령된 경보가 아닙니다.** 회귀(기온·강수)와 "
            "별개로 학습된 이진분류 헤드의 출력이고, 빨간 점선은 \"이 값을 넘으면 "
            "사건으로 친다\"는 판정선입니다."
        )
        if ev.get("heatwave") is not None:
            event_gauge("🔥 폭염 확률", ev["heatwave"],
                        EXTREME_EVENT_THRESH["heatwave"], "#E2954F")
        if ev.get("coldwave") is not None:
            event_gauge("🥶 한파 확률", ev["coldwave"],
                        EXTREME_EVENT_THRESH["coldwave"], "#4C78A8")
        if ev.get("dust") is not None:
            event_gauge("🌫️ 황사 확률", ev["dust"],
                        EXTREME_EVENT_THRESH["dust"], "#9C8060")

        st.markdown("##### 각 확률이 무엇을 배운 것인가")
        st.markdown(
            "| 사건 | 학습에 쓴 정답(라벨)의 정의 | 판정선 |\n"
            "|---|---|---|\n"
            "| 🔥 폭염 | **기상청이 실제로 발표한 폭염주의보 기록**(해당 날짜·관측소). "
            "기록을 구하지 못한 구간만 예외적으로, 발표기준 수치인 33°C를 "
            "계산 대상 시각의 기온에 적용해 근사 | "
            f"{EXTREME_EVENT_THRESH['heatwave']:.0%} |\n"
            "| 🥶 한파 | **실제 발표된 한파주의보 기록**. 같은 방식으로, 기록이 없으면 "
            "발표기준 수치인 −12°C를 대상 시각 기온에 적용해 근사 | "
            f"{EXTREME_EVENT_THRESH['coldwave']:.0%} |\n"
            "| 🌫️ 황사 | **PM10 1시간 평균 150㎍/㎥ 이상**(대기환경지수 '매우나쁨' "
            "등급). 공식 황사주의보 기준(400㎍/㎥ 2시간 지속)은 보유 데이터에 사례가 "
            "거의 없어 학습이 불가능했습니다 | "
            f"{EXTREME_EVENT_THRESH['dust']:.0%} |\n"
        )
        st.caption(
            "근사 폴백에 대한 정직한 단서: 기상청 발표기준은 원래 **일 최고기온**(폭염)· "
            "**아침 최저기온**(한파)에 적용되는 값인데, 폴백 구간에서는 같은 수치를 "
            "시간별 기온에 그대로 적용합니다. 즉 폴백으로 만들어진 라벨은 공식 특보와 "
            "정확히 같은 정의가 아닙니다 — 발표 기록이 있는 구간에서는 이 근사를 쓰지 "
            "않습니다."
        )
        st.caption(
            "⚠️ **황사는 12개 관측소 중 6곳(서울·수원·춘천·대구·전주·광주)에만 PM10 "
            "관측이 있습니다.** 나머지 6곳(인천·강릉·청주·대전·부산·제주)은 정답 자체가 "
            "없어 학습에서 제외됐습니다 — 그 지역의 황사 확률은 다른 지역에서 배운 "
            "패턴을 옮긴 것이므로 폭염·한파보다 신뢰도가 낮다고 보는 것이 정확합니다."
        )
        st.caption(
            "판정선이 0.5가 아닌 이유: 이 사건들은 '일어나지 않음'이 압도적으로 많아 "
            "(검증셋 양성 건수는 '성능 검증' 탭 표 참고) 0.5로 자르면 균형이 맞지 "
            "않습니다. 그래서 검증 데이터를 둘로 갈라 한쪽에서 F1이 가장 높아지는 값을 "
            "찾고, **한 번도 보지 않은** 다른 쪽에서도 성능이 유지되는지 확인한 뒤 "
            "채택했습니다(자세한 절차는 '성능 검증' 탭)."
        )
        st.caption(
            "호우(강수)는 이 탭에 게이지가 없습니다 — '출력값 추이' 탭의 강수량 값이 "
            "이미 비 올 확률을 곱해서 나온 값이기 때문입니다(아래 '모델 구조' 탭의 "
            "Hurdle 설명 참고)."
        )

# ── 탭 3: 성능 검증 ──────────────────────────────────────────────

with tab_perf:
    val_temp = ckpt.get("val_temp_mae")
    val_precip = ckpt.get("val_precip_mae")
    if val_temp is not None:
        st.markdown("#### 회귀 성능 — 기온·강수를 숫자로 얼마나 맞히나")
        st.caption(
            "**MAE(평균절대오차)** = |예측값 − 실제값|의 평균. 0에 가까울수록 좋고, "
            "단위는 예측 대상과 같습니다(기온 °C, 강수 mm). 아래 값은 학습에 쓰지 않고 "
            "따로 떼어둔 검증셋(전체의 20%)에서만 측정한 것입니다."
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
            직관과 반대이기 때문이다. 그래서 숫자 뒤에 판정을 말로 붙인다.
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
            help=f"평균적으로 실제 기온과 이만큼 차이가 난다는 뜻입니다. 기준선은 "
                 f"\"{lead}시간 뒤에도 지금과 같은 기온\"이라고 답하는 방식(퍼시스턴스)"
                 + (f"으로, {_naive_temp:.2f}°C 입니다." if _naive_temp else "입니다."),
        )
        vc2.metric(
            # 강수는 0.17 vs 0.168 처럼 소수점 둘째 자리에서 같아 보인다 —
            # 기준선과의 차이가 작을수록 자릿수를 늘려야 비교가 성립한다.
            "강수 MAE", f"{val_precip:.3f} mm",
            delta=_baseline_delta(val_precip, _naive_precip),
            help="기준선은 '항상 0mm'라고 답하는 방식입니다"
                 + (f" ({_naive_precip:.3f}mm)." if _naive_precip else ".")
                 + " 이 숫자만으로 판단하면 안 됩니다 — 아래 설명 참고.",
        )
        st.caption(
            "**기준선(baseline)** 이란 학습을 전혀 하지 않은 자명한 방법입니다. "
            "MAE는 그 자체로는 좋고 나쁨을 말해주지 않고, 기준선보다 나은지로 "
            "판단해야 의미가 생깁니다. 위 기준선 값은 모델과 **똑같은 검증 표본**에서 "
            "계산한 것입니다(다른 표본과 비교하면 표본 차이만으로 개선처럼 보입니다)."
        )

        if _naive_precip and val_precip > _naive_precip:
            _gap = (val_precip - _naive_precip) / _naive_precip
            st.warning(
                f"**강수는 아직 기준선을 넘지 못했습니다.** 모델 {val_precip:.4f}mm vs "
                f"'항상 0mm' {_naive_precip:.4f}mm — 차이는 {_gap:.1%}입니다. 즉 이 모델의 "
                f"강수량 출력은 아무것도 예측하지 않는 것보다 평균 오차가 (근소하게) 큽니다. "
                f"숨기지 않고 그대로 표시합니다.\n\n"
                f"원인은 구조적으로 추적됐습니다 — 극한기상 헤드가 3개로 늘면서 헤드들이 공유 "
                f"표현을 두고 경쟁해 무강수 억제력이 약해졌습니다(초기 −25.6%). 가중치 하향과 "
                f"데이터 확장(3.6년)으로 현재 수준까지 좁혔습니다. 강수는 MAE가 아니라 "
                f"'비가 올지 안 올지 맞혔나'로 보는 것이 실용적이며, 그 지표는 아래 "
                f"'출력값 적중률'의 강수 항목입니다.",
                icon="⚠️",
            )
        elif _naive_precip is None:
            st.warning(
                f"**강수 MAE {val_precip:.2f}mm는 '잘 맞힌다'는 뜻이 아닙니다.** "
                "대부분의 시각은 비가 오지 않으므로, 아무 계산 없이 **항상 0mm라고 "
                "답하기만 해도** MAE는 비슷하게 낮게 나옵니다. 이 프로젝트의 학습 로그 "
                "기준으로 현재 모델의 강수 MAE는 그 기준선보다 **나쁩니다**. "
                "강수는 '비가 올지 안 올지 맞혔나'로 봐야 하며, 그 지표는 아래 "
                "'출력값 적중률'의 강수 항목입니다.",
                icon="⚠️",
            )
            st.caption(
                "이 체크포인트에는 기준선 수치가 저장돼 있지 않아 화면에서 직접 "
                "비교하지 못합니다(학습 시 계산은 하지만 저장하지 않던 버전입니다). "
                "다음 재학습부터는 저장되어 위 지표 옆에 증감으로 표시됩니다."
            )

    # 극한기상 헤드의 분류 성능 — 체크포인트에 저장돼 있는데도 화면에 없었다
    # (2026-08-11 점검에서 발견: 위 문단이 "아래 극한기상 F1을 함께 봐야
    # 합니다"라고 안내하는데 정작 그 표가 없었다).
    _em = ckpt.get("extreme_metrics") or {}
    if _em:
        st.markdown("---")
        st.markdown("#### 극한기상 분류 성능")
        st.caption(
            # 닫는 ** 뒤에 조사가 바로 붙으면(…99%**가) 마크다운이 강조를
            # 닫지 못해 별표가 화면에 그대로 노출된다(2026-08-11 배포본에서
            # '***항상'으로 깨져 보임). 강조 구간을 문장 끝까지 늘려 닫는
            # ** 뒤에 공백이 오게 한다.
            "드문 사건은 정확도(accuracy)로 재면 안 됩니다. 예를 들어 폭염인 날이 전체의 "
            "1%뿐이라면, 아무 계산 없이 **'항상 폭염 아님'이라고만 답해도 정확도가 99%가 "
            "나옵니다** — 폭염을 한 번도 맞히지 못했는데도 말입니다. "
            "그래서 아래 세 지표로 봅니다."
        )
        st.markdown(
            "- **정밀도(Precision)** — 모델이 \"일어난다\"고 한 것 중 실제로 일어난 비율. "
            "낮으면 잘못 잡아낸 것(오탐)이 많다는 뜻입니다.\n"
            "- **재현율(Recall)** — 실제로 일어난 것 중 모델이 잡아낸 비율. "
            "낮으면 놓친 사건이 많다는 뜻입니다.\n"
            "- **F1** — 위 둘의 조화평균. 한쪽만 좋고 다른 쪽이 나쁘면 함께 낮아지므로, "
            "두 지표를 한 숫자로 요약할 때 씁니다. 0~1이고 1이 최고입니다."
        )
        _label_ko = {"heatwave": "🔥 폭염", "coldwave": "🥶 한파", "dust": "🌫️ 황사"}
        _rows = ["| 사건 | 정밀도 | 재현율 | F1 | 검증셋 실제 발생 |",
                 "|---|---|---|---|---|"]
        for _k in ("heatwave", "coldwave", "dust"):
            _m = _em.get(_k)
            if not _m:
                continue
            _rows.append(
                f"| {_label_ko[_k]} | {_m['precision']:.0%} | {_m['recall']:.0%} | "
                f"{_m['f1']:.3f} | {_m['n_pos']:,}건 |"
            )
        st.markdown("\n".join(_rows))
        st.caption(
            "**주의 — 이 표의 값은 확률 0.5를 기준으로 계산한 것**이라, '극한 기상' 탭의 "
            "빨간 판정선(폭염 "
            f"{EXTREME_EVENT_THRESH['heatwave']:.0%} · 한파 {EXTREME_EVENT_THRESH['coldwave']:.0%} · "
            f"황사 {EXTREME_EVENT_THRESH['dust']:.0%})을 적용했을 때의 성능과는 다릅니다. "
            "판정선을 올리면 오탐(정밀도↑)은 줄고 놓침(재현율↓)은 늘어납니다. "
            "표본이 적은 사건일수록 F1의 상한이 낮다는 점도 함께 봐야 합니다 — "
            "황사가 폭염보다 낮은 것은 모델이 특별히 못해서라기보다 배울 사례 자체가 "
            "적기 때문입니다."
        )

    st.markdown("---")
    st.markdown("#### 판정선은 어떻게 정했나 — 과적합을 피하는 절차")
    st.caption(
        "'극한 기상' 탭의 빨간 판정선은 다음 순서로 정했습니다. ① 검증 데이터를 "
        "**보정용 / 평가용**으로 무작위 절반씩 나눕니다(학습·검증을 나눌 때와는 다른 "
        "난수를 씁니다). ② 보정용에서만 F1이 가장 높아지는 값을 찾습니다. ③ 그 값을 "
        "**한 번도 보지 않은** 평가용에 적용해 성능이 유지되는지 확인합니다."
    )
    st.caption(
        "이 절차가 필요한 이유: 같은 데이터에서 값을 고르고 그 데이터로 채점하면 "
        "그 데이터에만 맞춘 값이 좋아 보이게 됩니다(과적합). 실제로 **강수** 판정선 "
        "재보정은 이 검증을 통과하지 못해(개선폭이 0.01에 못 미침) 기존의 보수적인 "
        "값을 유지했고, 폭염·한파·황사만 통과해 새 값을 채택했습니다."
    )

    st.markdown("---")
    st.markdown("#### 출력값 적중률 (실측과 사후 대조)")
    st.caption(
        f"기온: 오차 ±{accuracy.HIT_TEMP_TOL}°C 이내면 적중 · "
        f"강수: {accuracy.PRECIP_THRESH}mm 기준 비/무비 판정 일치 시 적중"
    )

    if status == "SUCCESS_LIVE" and result.get("target_time"):
        accuracy.record_prediction(
            station=stn, made_at=result["observed_at"],
            target_time=result["target_time"],
            pred_temp=f["temperature"], pred_precip=f["precipitation"],
            source="live",
        )
    accuracy.resolve_pending(history)

    acc = accuracy.stats(station=stn)
    if acc["cum_n"] == 0:
        st.caption("아직 실측과 대조된 출력값이 없습니다 — "
                   "`python backtest_accuracy.py`로 과거 데이터를 먼저 채울 수 있습니다.")
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
        st.caption(f"누적 표본 {acc['cum_n']}건 ({result['station_name']} 기준). "
                   "이 지표는 검증셋 MAE와 다릅니다 — 실제 운영 시각에 실제로 조회 가능했던 "
                   "데이터로만 만든 출력값을, 그 시점 이후 확정된 실측과 대조한 것입니다.")

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
        "실행 중 추가분은 이 프로세스가 살아 있는 동안만 남습니다 — 배포 환경의 "
        "컨테이너 파일시스템은 재시작 시 저장소 상태로 되돌아갑니다. 영구 누적은 "
        "GitHub Actions의 **Refresh deploy data** 워크플로가 담당합니다: 새 관측을 "
        "받아 대기 항목을 실측과 대조한 뒤 결과를 저장소에 되커밋합니다."
    )

# ── 탭 4: 모델 구조 ──────────────────────────────────────────────

with tab_model:
    gw = result["gate_weights"]
    if gw:
        st.markdown("#### 축 배분 — 이번 출력값에 대한 각 축의 기여도")
        st.caption(
            "세 축의 기여도는 고정값이 아니라 **값을 낼 때마다 입력을 보고 다시 계산**됩니다. "
            "작은 신경망이 세 숫자를 내놓고 softmax(합이 1이 되도록 만드는 함수)를 "
            "거치므로 합은 항상 100%입니다. 아래 '세 축을 하나로 합치는 식'에 나오는 "
            "`w`가 바로 이 값입니다."
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
    st.markdown("#### 3축이 각각 무엇을 보는가")
    st.markdown(
        "이 모델은 값을 하나 낼 때 **서로 다른 세 가지 질문**을 던지고, 그 답 셋을 "
        "하나로 합칩니다. 세 질문은 이렇습니다."
    )
    st.markdown(
        "| 축 | 던지는 질문 | 실제로 넣는 데이터 |\n"
        "|---|---|---|\n"
        "| **Z축** — 지금 여기 | *\"이 관측소는 지금 어떤 상태인가?\"* | "
        "이번 시각 이 관측소의 관측값 그대로 — 기온·강수·습도·풍속·풍향·기압 등 "
        "숫자 12개 |\n"
        "| **Re축** — 주변 | *\"주변 지역은 지금 어떤가?\"* | 나머지 11개 관측소의 "
        "**같은 시각** 관측값. 가까운 관측소일수록 크게 반영해 주변 상황을 지도처럼 "
        "펼칩니다(거리가중 보간) |\n"
        "| **Im축** — 흐름 | *\"어느 쪽으로 움직이는 중인가?\"* | 같은 관측소의 "
        "**1·3·6시간 전과 비교한 변화량**. 기압이 떨어지는 중인지 오르는 중인지 같은 "
        "정보 |\n"
    )
    st.caption(
        "**왜 Re·Im·Z라는 이름인가** — 아래에서 설명할 원 논문이 세 축을 복소수"
        "(complex number)에 빗대어 배치했고, 그 이름을 그대로 물려받았습니다. "
        "Re는 실수부(**Re**al), Im은 허수부(**Im**aginary)에서 온 말입니다. 복소수에서 "
        "이 둘은 서로 **직각 방향**이라, \"세 축이 서로 겹치지 않는 정보를 담는다\"는 "
        "설계 의도를 이름으로 드러낸 것입니다. 이름일 뿐이고 실제로 복소수 연산을 하지는 "
        "않습니다 — 아래 식을 보면 전부 실수 계산입니다."
    )
    st.caption(
        "세 질문이 서로 다른 것이 핵심입니다. 주변 관측소의 값(Re)과 몇 시간 전 대비 "
        "변화량(Im)은 **이번 시각 이 관측소의 관측값(Z)만 봐서는 절대 계산해낼 수 "
        "없는 정보**입니다. 그래서 축을 추가한 만큼 실제로 새 정보가 들어옵니다."
    )
    st.caption(
        "Re·Im 두 축은 원래 논문에서 위성 이미지·기상 문서가 들어가던 자리입니다. "
        "이 프로젝트의 관측망(지상 관측소 12곳, 위성·텍스트 없음)에는 그대로 옮길 수 "
        "없어 대체 데이터를 넣었는데, 축 하나를 꺼도 출력이 전혀 바뀌지 않는 것을 "
        "확인하고 **그 데이터가 아무 정보도 주지 않았다**는 결론을 냈습니다. 이후 "
        "'이번 시각 하나만 봐서는 알아낼 수 없는 정보만 새 축에 담는다'는 기준으로 "
        "다시 설계한 것이 지금의 공간·시간 축입니다 — 주변 관측소의 값과 몇 시간 전 "
        "대비 변화량은 원리적으로 현재 시각의 스냅샷에서 계산해낼 수 없습니다."
    )
    _diag = ckpt.get("diagnostics") or {}
    if _diag.get("cos_re_z_post") is not None:
        st.caption(
            "다만 '입력이 독립'인 것과 '모델이 배운 표현이 독립'인 것은 다릅니다. "
            "학습 후 세 축 벡터가 이루는 각도를 코사인 유사도로 재보면 "
            f"Re–Z {_diag['cos_re_z_post']:.2f} · Re–Im {_diag['cos_re_im_post']:.2f} · "
            f"Im–Z {_diag['cos_im_z_post']:.2f} 입니다(0이면 완전히 다른 방향, 1이면 "
            "같은 방향). 0은 아니지만 낮은 편으로, 축들이 서로 상당히 다른 것을 보고 "
            "있다는 뜻입니다."
        )

    st.markdown("---")
    st.markdown("#### 세 축을 하나로 합치는 식 (논문 Eq.1)")
    st.caption(
        "세 축은 각각 숫자 여러 개가 늘어선 **벡터**로 바뀐 뒤 아래 식으로 합쳐집니다. "
        "기호가 낯설어 보이지만 하는 일은 단순합니다 — 아래에서 하나씩 풉니다."
    )
    st.latex(
        r"s_i \;=\; \sqrt{\;(w_{Re}\,\cdot\,Re_i)^2 \;+\; (w_{Im}\,\cdot\,Im_i)^2"
        r"\;+\; (w_{Z}\,\cdot\,Z_i)^2\;}"
    )

    _embed_dim = ckpt.get("embed_dim", 64)
    _w_re, _w_im, _w_z = gw.get("w_re"), gw.get("w_im"), gw.get("w_z")
    _fmt = (lambda v: f"{v:.3f}" if isinstance(v, (int, float)) else "—")
    st.markdown(
        "**기호 정리표**\n\n"
        "| 기호 | 무엇인가 | 이번 출력의 값 |\n"
        "|---|---|---|\n"
        f"| `Re`, `Im`, `Z` | 세 축을 각각 인공신경망에 통과시켜 "
        f"얻은 **숫자 {_embed_dim}개짜리 목록**(벡터) | 축마다 {_embed_dim}개 |\n"
        f"| `i` | 그 목록에서 몇 번째 숫자인지 가리키는 번호 | "
        f"1 ~ {_embed_dim} |\n"
        f"| `w` | 각 축을 **얼마나 믿을지**. 셋을 더하면 "
        f"1이 됩니다 | Re {_fmt(_w_re)} · Im {_fmt(_w_im)} · Z {_fmt(_w_z)} |\n"
        f"| `s` | 합쳐진 결과. 이것도 숫자 {_embed_dim}개짜리 목록이고, "
        f"이 목록 하나를 5개 예측이 나눠 씁니다 | {_embed_dim}개 |\n"
        "| √ | 제곱을 되돌리는 연산(제곱근) | — |\n"
    )
    st.markdown(
        "**식이 하는 일을 순서대로**\n\n"
        "1. **곱하기** — 각 축의 값에 그 축의 가중치를 곱합니다. 이번에 덜 믿기로 한 "
        "축은 여기서 작아집니다.\n"
        "2. **제곱** — 부호(＋/−)를 없애고 크기만 남깁니다. 두 축이 서로 반대 부호라서 "
        "더할 때 상쇄돼 사라지는 일을 막습니다.\n"
        "3. **더한 뒤 제곱근** — 직각삼각형의 빗변 길이를 구하는 계산과 같습니다 "
        "(가로 3, 세로 4면 빗변 5). 세 축을 서로 직각인 방향으로 놓고 그 합성 크기를 "
        "재는 것이라, 한 축이 커지면 결과도 커지되 다른 축이 이를 완전히 지우지는 "
        "못합니다.\n\n"
        f"이 3단계는 목록의 **각 자리마다 따로** 적용됩니다. 그래서 결과 `s`도 숫자 "
        f"하나가 아니라 {_embed_dim}개짜리 목록입니다."
    )
    st.caption(
        "각 축 벡터는 합쳐지기 전에 길이가 1이 되도록 맞춰집니다(정규화). 그래서 "
        "어느 축이 원래 큰 숫자를 갖고 있었는지는 영향을 주지 않고, **패턴의 방향과 "
        "가중치만으로** 결과가 정해집니다."
    )

    with st.expander("이 식은 어디서 왔나 — 원 논문과의 관계"):
        st.markdown(
            "이 프로젝트는 아래 논문의 **합치는 방식만** 빌려와 완전히 다른 분야에 "
            "적용해 본 것입니다.\n\n"
            "> **Tri-CHEF: Complex-Hermitian Embedding Fusion for Korean Multimodal "
            "Retrieval**  \n"
            "> Zenodo 프리프린트 (2026년 5월) · DOI [10.5281/zenodo.20034370]"
            "(https://doi.org/10.5281/zenodo.20034370) · CC BY 4.0"
        )
        st.markdown(
            "**원 논문은 날씨와 무관합니다.** 한국어 **검색(retrieval)** 시스템에 "
            "관한 논문입니다 — 문서·이미지·영상·음성을 한 번에 검색할 때, 종류가 다른 "
            "세 개의 AI 인코더가 내놓은 결과를 어떻게 하나의 점수로 합칠 것인가를 "
            "다룹니다.\n\n"
            "가장 흔한 방법은 **가중합**(각 점수에 가중치를 곱해 그냥 더하기)인데, "
            "이러면 한 채널의 점수가 유난히 크면 그 채널이 최종 점수를 사실상 "
            "독점해버립니다. 논문이 제안한 대안이 위에서 본 **제곱→합→제곱근** 형태이고, "
            "이것이 논문에서 **식 1번(Eq.1)**입니다. 각 축을 제곱해서 더하므로 한 축이 "
            "다른 축을 상쇄해 지워버리지 못하고, 세 축의 근거가 모두 최종 값에 남습니다."
        )
        st.markdown(
            "**이 프로젝트가 바꾼 것**\n\n"
            "| | 원 논문 | 이 프로젝트 |\n"
            "|---|---|---|\n"
            "| 하는 일 | 검색 (질의에 맞는 문서·이미지 찾기) | 기후 예측 (+"
            f"{lead}시간 뒤 기온·강수) |\n"
            "| 세 축에 넣는 것 | 사전학습 인코더 3종의 출력 | 지금 여기(Z) · 주변(Re) · "
            "흐름(Im) 관측값 |\n"
            "| 축 가중치 | 학습으로 정해 고정 | 값을 낼 때마다 입력을 보고 다시 계산 |\n"
            "| 식의 출력 | 검색 순위를 매기는 점수 | 5가지 예측의 공통 입력 |\n"
        )
        st.markdown(
            "논문의 원래 표기는 `s = √(A² + (αB)² + (φC)²)` 입니다. A·B·C가 각각 "
            "Re·Im·Z축이고, α(알파)·φ(파이)는 **A(Re축)를 1로 놓았을 때 나머지 두 축의 "
            "상대적 비중**입니다. 위쪽 식은 셋을 대등하게 `w`로 적었을 뿐 같은 식이며, "
            "`α = w(Im) ÷ w(Re)`, `φ = w(Z) ÷ w(Re)` 로 서로 환산됩니다"
            + (f" — 이번 출력에서는 α={gw['alpha']:.3f}, φ={gw['phi']:.3f} 입니다."
               if isinstance(gw.get("alpha"), (int, float)) else ".")
        )
        st.caption(
            "원 논문에서 α·φ는 학습이 끝나면 고정되는 상수입니다. 이 프로젝트에서는 "
            "그렇게 뒀더니 **정보가 없는 축의 가중치가 오히려 커지는** 문제가 있어, "
            "값을 낼 때마다 다시 계산하는 방식으로 바꿨습니다(위 '축 배분' 그래프가 그 "
            "결과입니다). 논문을 그대로 따르지 않고 바꾼 지점이며, 바꾼 이유는 실측에서 "
            "나왔습니다."
        )

    st.markdown("---")
    st.markdown("#### 합쳐진 값에서 5가지 예측이 갈라져 나옵니다")
    st.markdown(
        f"위에서 만든 숫자 {_embed_dim}개짜리 목록 `s`는 **여기까지가 공통 과정**입니다. "
        "이제 이 목록 하나를 6개의 작은 신경망이 각자 읽어서 5가지 예측을 만듭니다. "
        "이렇게 공통 부분 뒤에 붙는 작은 신경망을 흔히 **헤드(head)** 라고 부릅니다 — "
        "몸통 하나에 머리 여럿이 달린 모양이라 붙은 이름입니다.\n\n"
        f"- **기온** — \"지금 기온에서 얼마나 변할지\"(변화량)를 구해 현재 기온에 "
        f"더합니다. 기온 자체를 통째로 맞히려 하지 않는 이유는, +{lead}시간 정도면 "
        f"\"지금과 비슷할 것\"이라는 답이 이미 상당히 정확하기 때문입니다. 그 위에 "
        f"**얼마나 달라질지만** 학습하는 편이 유리합니다.\n"
        "- **강수** — 헤드 두 개를 곱해서 만듭니다(바로 아래 설명).\n"
        "- **폭염 · 한파 · 황사** — 각각 \"일어날 확률\" 하나씩. '극한 기상' 탭의 "
        "막대가 이 값입니다."
    )
    st.caption(
        "기온·강수처럼 **숫자 값**을 맞히는 것을 회귀(regression), 폭염·한파·황사처럼 "
        "**일어난다/아니다**를 맞히는 것을 이진분류(binary classification)라고 부릅니다. "
        "둘은 학습 방식이 달라서 헤드를 따로 둡니다."
    )
    st.markdown("**강수의 Hurdle 구조 — 왜 두 갈래를 곱하나**")
    st.latex(r"\hat{y}_{precip} \;=\; \sigma(z_1)\;\times\;\mathrm{softplus}(z_2)")
    st.markdown(
        "- **ŷ**(와이햇, `y_precip`) — 최종 강수량 예측값. 모자(^)는 \"실제값이 아니라 "
        "모델이 추정한 값\"이라는 관례 표기입니다.\n"
        "- **z₁, z₂**(제트) — 위에서 만든 `s` 목록을 각각 다른 작은 신경망에 통과시켜 "
        "나온 중간 숫자입니다. 아직 확률도 mm도 아닌 날것의 값이라, 아래 두 함수로 "
        "각각 의미 있는 범위로 바꿔줍니다.\n"
        "- **σ**(시그마 / 시그모이드) — 어떤 실수든 **0~1 사이**로 눌러 담는 함수. "
        "그래서 `σ(z₁)`은 \"비가 올 확률\"로 읽습니다.\n"
        "- **softplus**(소프트플러스) — 결과를 **항상 0보다 크게** 만드는 함수. "
        "강수량은 음수가 될 수 없어서 씁니다. 더 단순한 방법(음수를 그냥 0으로 자르기)은 "
        "잘린 구간에서 학습 신호가 끊겨버려 쓰지 않았습니다.\n"
        "- **곱하는 이유** — 비가 올 확률이 0에 가까우면 예상량이 아무리 커도 최종값이 "
        "0에 가까워집니다. \"올까 안 올까\"와 \"온다면 얼마나\"를 한 식에 담는 구조이고, "
        "무강수가 대부분인 데이터에서 정확한 0을 내기 위한 장치입니다.\n"
        f"- 마지막으로 {PRECIP_CLIP_THRESH}mm 미만은 0으로 반올림합니다(미세 노이즈 억제)."
    )

    st.markdown("---")
    st.markdown("#### 실시간 데이터는 모델을 다시 학습시키나요?")
    st.markdown(
        f"**아니요.** 화면의 모델은 고정된 체크포인트이고, 실시간 관측은 두 가지 "
        f"용도로만 쓰입니다.\n\n"
        f"1. **추론 입력** — 지금 이 순간의 관측값(+ 이웃 11개 관측소 + 과거 "
        f"1h/3h/6h)을 모델에 넣어 +{lead}시간 후를 예측합니다. 가중치는 건드리지 "
        f"않습니다.\n"
        f"2. **적중률 기록(모니터링)** — 출력값을 로그에 남겨두고 +{lead}시간 뒤 "
        f"실제값이 들어오면 대조해 위 '성능 검증' 탭의 누적 적중률에 반영합니다. "
        f"**이 대조 결과는 모델을 바꾸지 않습니다** — 순수 사후 기록입니다.\n\n"
        f"모델을 갱신하려면(파인튜닝) 사람이 `train.py`를 다시 실행해 그동안 "
        f"쌓인 데이터를 처음부터 배치로 재학습해야 합니다. 관측이 들어올 때마다 "
        f"즉시 가중치를 업데이트하는 온라인 학습은 의도적으로 넣지 않았습니다."
    )
    with st.expander("왜 온라인 학습(즉시 파인튜닝)을 넣지 않았나 — 판단 근거"):
        st.markdown(
            "- **표본 밀도**: 관측소당 신규 표본이 시간당 1건뿐입니다. 배치 학습 "
            "1 에폭이 보는 수만 건 규모의 그레디언트 평균과 달리, 관측 1건짜리 "
            "업데이트는 노이즈가 학습 신호보다 클 위험이 큽니다.\n"
            "- **재앙적 망각(catastrophic forgetting)**: 최근 데이터에 맞춰 즉시 "
            "업데이트하면 예전에 배운 계절적 패턴을 잊을 수 있습니다. 이를 막는 "
            "리플레이 버퍼나 정규화 장치가 이 프로젝트엔 아직 없어, 넣지 않은 것이 "
            "더 안전한 선택이라고 판단했습니다.\n"
            "- **재현성**: 배치 재학습은 '이 체크포인트가 이 데이터·이 코드로 이렇게 "
            "학습됐다'를 그대로 재현·감사할 수 있습니다. 온라인 학습은 매 순간 "
            "가중치가 달라 같은 상황을 재현하기 어렵습니다.\n\n"
            "즉, 이 프로젝트에서 '실시간'이 의미하는 건 실시간 추론이지 실시간 "
            "학습이 아닙니다 — 두 개념을 구분해서 표기하는 것 자체가 이 대시보드가 "
            "지키려는 원칙 중 하나입니다."
        )

st.markdown("---")
st.caption(
    "Tri-CHEF — 공간(Re)·시간(Im)·현재상태(Z) 세 축을 논문 Eq.1로 합쳐 예측하는 "
    "파이프라인입니다(식과 기호 설명은 '모델 구조' 탭). Re축은 12개 관측소의 실측 "
    "공간보간장, Im축은 같은 관측소의 시간 변화율이며, 둘 다 실제 관측에서 나온 "
    "값입니다 — 시뮬레이션이나 합성 데이터가 아닙니다."
)
st.caption(
    "⚠️ **이 화면의 숫자는 모델 출력값이며 기상청의 공식 예보·특보가 아닙니다.** "
    "연구·포트폴리오 목적으로 만든 것으로, 실제 방재 판단의 근거로 사용해서는 "
    "안 됩니다. 공식 정보는 기상청(weather.go.kr)을 확인하세요."
)
