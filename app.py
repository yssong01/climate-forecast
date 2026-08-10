"""
app.py — Tri-CHEF 기후 예보 Streamlit 대시보드

학습된 3축 파이프라인(predict.py)을 실시간 관측과 연결해 보여준다.
모델 로드는 st.cache_resource 로 한 번만 수행 — Streamlit 은 위젯 조작마다
스크립트 전체를 재실행하므로 캐싱 없이는 매번 체크포인트를 다시 읽는다.
(2026-08-09부터 Im축이 MiniLM 텍스트에서 시간 경향 벡터로 바뀌면서
sentence-transformers 로드 자체가 없어졌다 — 캐싱할 무거운 리소스는
모델 하나뿐이다.)

화면 구성은 탭 4개로 나뉜다 — 예보 추이 / 극한기상 / 성능 검증 / 모델 구조.
숫자·그래프마다 그 근거(무엇을 어떻게 측정했는지)를 캡션이나 st.metric의
help 툴팁으로 바로 옆에 붙인다 — "이 값이 왜 이렇게 나왔는지"를 화면만
보고 답할 수 있어야 한다는 원칙(2026-08-09 논의)을 따른다.
"""
import json
import os
from datetime import datetime, timedelta
from zoneinfo import ZoneInfo

import streamlit as st
import streamlit.components.v1 as components
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from weather_collector import STATIONS, STATION_COORDS
from predict import load_model, predict, CHECKPOINT, EXTREME_EVENT_THRESH
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
    page_title="Tri-CHEF 기후 예보",
    page_icon="🌦️",
    layout="centered",
)


# ── 캐시된 리소스 — 세션당 한 번만 로드 ──────────────────────────────

@st.cache_resource(show_spinner="모델 로드 중...")
def get_model():
    return load_model(CHECKPOINT)


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
        marker=dict(size=sizes, color=colors, line=dict(width=0.5, color="#0E1117")),
        text=names, textposition="middle right",
        textfont=dict(size=10, color="#C9D1D9"),
        customdata=codes,
        hovertext=names, hoverinfo="text",
    ))
    fig.update_geos(
        resolution=50,
        lataxis_range=[33, 39], lonaxis_range=[124.2, 130.2],   # 한반도만 딱 맞게
        showland=True, landcolor="#242830",
        showocean=True, oceancolor="#12151b",
        showcountries=True, countrycolor="#3a4048",
        showcoastlines=True, coastlinecolor="#3a4048",
        showframe=False, projection_type="mercator",
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        margin=dict(l=0, r=0, t=0, b=0), height=340,   # 세로 폭 최소화(2026-08-10) — 좌우는 사이드바 폭 그대로
        paper_bgcolor="rgba(0,0,0,0)",
    )
    event = st.sidebar.plotly_chart(
        fig, use_container_width=True, key="station_geo_map",
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
                 annotation_text=f"경보 임계값 {thresh:.0%}", annotation_position="top")
    fig.update_layout(
        xaxis=dict(range=[0, 100], title=None, showticklabels=True, ticksuffix="%"),
        yaxis=dict(showticklabels=False),
        height=110, margin=dict(l=10, r=50, t=28, b=10),
        showlegend=False,
    )
    st.markdown(
        f'<div style="font-size:1.05rem; font-weight:700; color:#FFFFFF; '
        f'margin-bottom:0.2rem;">{label}</div>',
        unsafe_allow_html=True,
    )
    st.plotly_chart(fig, use_container_width=True, config={"staticPlot": True})


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

st.sidebar.markdown("---")
st.sidebar.markdown("**관측소 위치** — 지도에서 도시를 클릭해도 선택됩니다")


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
    "ASOS 지상관측은 매시 정각 관측 후 약 10분 뒤 공개됩니다. "
    "갱신 주기를 너무 짧게 두어도 새 데이터가 없을 수 있습니다."
)
st.sidebar.caption(
    "12개 관측소 모두 매 예보마다 조회됩니다 — 대상 관측소는 예측 입력으로, "
    "나머지 11개는 아래 '모델 구조' 탭에서 설명하는 공간보간장(Re축) 계산에 쓰입니다."
)


# ── 메인 ──────────────────────────────────────────────────────────

if not os.path.exists(CHECKPOINT):
    st.error(
        f"체크포인트가 없습니다: `{CHECKPOINT}`\n\n"
        f"먼저 터미널에서 `python train.py` 를 실행해 모델을 학습하세요."
    )
    st.stop()

try:
    model, ckpt = get_model()
    result = predict(stn=stn, model=model, ckpt=ckpt)
except Exception as e:
    st.error(f"예측 실패: {e}")
    st.stop()

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
    /* Streamlit 기본값은 툴바 아래 여백이 커서(약 6rem) "환경 예보" 위로
       빈 공간이 크게 남는다 — 줄인다. 헤더가 이 컨테이너의 자연스러운
       위치를 기준으로 접히므로, 여기를 줄이면 스크롤 유무와 상관없이
       같이 줄어든다. */
    [data-testid="stMainBlockContainer"] {
        padding-top: 1rem !important;
    }
    .st-key-sticky_header {
        background-color: var(--background-color, #0e1117);
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

with st.container(key="sticky_header"):
    st.title("🌦️ 환경 예보")
    # 반응형으로 줄바꿈되어 창 폭에 관계없이 전체 내용이 보이도록 한다
    # (2026-08-10) — 말줄임표로 자르지 않는다.
    st.caption("3축 Tri-CHEF 파이프라인(공간보간·시간경향·수치센서) 기반 +6시간 예보")

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

        function toolbarBottom() {
            return toolbarEl ? toolbarEl.getBoundingClientRect().bottom : 0;
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
    ["📈 예보 추이", "🚨 극한 기상", "✅ 성능 검증", "🧭 모델 구조"]
)

# ── 탭 1: 예보 추이 ──────────────────────────────────────────────

with tab_trend:
    history = load_merged_history()
    series = recent_series(history, stn, hours=72)

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
                                 mode="lines+markers", name="예보",
                                 line=dict(color="#E2954F", width=1.5, dash="dot"),
                                 marker=dict(size=[0, 10], symbol="star")),
                     row=1, col=1)

        fig.add_trace(go.Bar(x=xs, y=precs, name="실측 강수",
                             marker_color="#4C78A8", showlegend=False), row=1, col=2)
        fig.add_trace(go.Scatter(x=[tgt_x], y=[f["precipitation"]], mode="markers",
                                 name="예보 강수", marker=dict(size=10, symbol="star",
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
        st.plotly_chart(fig, use_container_width=True)
        st.caption(f"최근 {len(series)}시간 실측(로컬 캐시) + 별표는 이번 +{lead}시간 예보. "
                   "기온·강수만 예보 대상 — 습도·기압은 참고용 실측치입니다.")

    st.markdown("#### +%d시간 후 예보" % lead)
    fc1, fc2 = st.columns(2)
    fc1.metric(
        "예상 기온", f"{f['temperature']:.1f} °C",
        delta=f"{f['temperature'] - c['temperature']:+.1f} °C (현재 대비)",
        help="회귀 헤드(MLP) 출력. 학습 시 정규화한 값을 역변환한 것으로, "
             "아래 '성능 검증' 탭의 검증 MAE만큼의 오차가 통계적으로 기대됩니다.",
    )
    fc2.metric(
        "예상 강수", f"{f['precipitation']:.1f} mm",
        delta=f"{f['precipitation'] - c['precipitation']:+.1f} mm (현재 대비)",
        delta_color="inverse",
        help="Hurdle 구조: sigmoid(강수여부) × softplus(강수량)의 곱. "
             "강수 확률이 낮으면 양이 커도 최종값이 0에 가까워지고, "
             "0.2mm 미만은 후처리로 0에 반올림합니다(미세한 노이즈 억제).",
    )

# ── 탭 2: 극한기상 ───────────────────────────────────────────────

with tab_extreme:
    ev = result.get("extreme_event_probs") or {}
    if not any(ev.get(k) is not None for k in ("heatwave", "coldwave", "dust")):
        st.caption("이 체크포인트는 극한기상 헤드가 없습니다(구버전) — "
                   "재학습된 체크포인트를 사용하면 표시됩니다.")
    else:
        st.caption(
            "각 확률은 회귀와 별개로 학습된 이진분류 헤드(공유 임베딩 → 전용 MLP → "
            "sigmoid)의 출력입니다. 빨간 점선은 경보 판정 임계값 — 0.5가 아니라 "
            "검증셋을 보정용/평가용으로 나눠 F1을 최대화하는 값을 고르고, 그 값이 "
            "평가용에서도 유지되는지 확인한 뒤 정한 것입니다(과적합 임계값이 아님, "
            "'모델 구조' 탭에 방법론 설명)."
        )
        if ev.get("heatwave") is not None:
            event_gauge("🔥 폭염 확률 (공식 특보 기준 라벨로 학습)",
                       ev["heatwave"], EXTREME_EVENT_THRESH["heatwave"], "#E2954F")
        if ev.get("coldwave") is not None:
            event_gauge("🥶 한파 확률 (공식 특보 기준 라벨로 학습)",
                       ev["coldwave"], EXTREME_EVENT_THRESH["coldwave"], "#4C78A8")
        if ev.get("dust") is not None:
            event_gauge("🌫️ 황사 확률 (PM10≥150㎍/㎥ 라벨로 학습, 6개 관측소만 커버)",
                       ev["dust"], EXTREME_EVENT_THRESH["dust"], "#9C8060")
            st.caption(
                "⚠️ 황사는 PM10 실측이 있는 관측소(서울·인천·수원 등 6곳)만 라벨이 "
                "있어 나머지 관측소는 학습 표본이 상대적으로 적습니다 — 다른 이벤트보다 "
                "확률의 신뢰 구간이 넓다고 해석하는 것이 정확합니다."
            )
        st.caption(
            "강수(호우)는 이 탭이 아니라 위 '예보 추이' 탭의 강수량 예보 자체가 "
            "Hurdle 분류 확률을 이미 곱해 반영하고 있어 별도 게이지를 두지 않았습니다."
        )

# ── 탭 3: 성능 검증 ──────────────────────────────────────────────

with tab_perf:
    val_temp = ckpt.get("val_temp_mae")
    val_precip = ckpt.get("val_precip_mae")
    if val_temp is not None:
        st.markdown("#### 회귀 성능 (학습 시 검증셋 측정)")
        vc1, vc2 = st.columns(2)
        vc1.metric("기온 MAE", f"{val_temp:.2f} °C",
                  help="평균절대오차 — 검증셋(학습에 안 쓴 20%)에서 |예측−실측|의 평균. "
                       "학습 손실이 아니라 별도로 떼어둔 데이터로만 측정합니다.")
        vc2.metric("강수 MAE", f"{val_precip:.2f} mm",
                  help="같은 방식, 강수량 기준. 대부분의 시간이 무강수(0mm)라 "
                       "이 값만으로는 강수 예측력을 온전히 판단하기 어렵고 "
                       "아래 극한기상 F1을 함께 봐야 합니다.")
        st.caption(
            "MAE는 이상치(집중호우 한 건)에 민감하지 않은 대신, '항상 무강수를 "
            "예측'해도 낮게 나올 수 있는 지표입니다 — 그래서 강수는 회귀(MAE)와 "
            "별개로 분류 헤드(F1)를 따로 두고 검증합니다."
        )

    st.markdown("---")
    st.markdown("#### 임계값 재보정 — 과적합 검증 방법론")
    st.caption(
        "극한기상 확률을 0/1로 나누는 임계값(위 '극한 기상' 탭의 빨간 점선)은 이렇게 "
        "정했습니다: ① 검증셋을 보정용/평가용으로 무작위 50/50 분할(학습/검증 분할과 "
        "별개의 시드 사용) ② 보정용에서 F1을 최대화하는 임계값 탐색 ③ 그 값을 "
        "**한 번도 보지 않은** 평가용에 적용해 F1이 유지되는지 확인. 보정용에서만 좋고 "
        "평가용에서 무너지면 과적합으로 판정해 폐기합니다 — 실제로 강수 임계값 "
        "재보정은 이 검증을 통과하지 못해(순이득 &lt;0.01) 보수적인 값을 유지했고, "
        "폭염·한파·황사는 통과해 채택했습니다."
    )

    st.markdown("---")
    st.markdown("#### 예보 적중률 (실측과 사후 대조)")
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
        st.caption("아직 실측과 대조된 예보가 없습니다 — "
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
                   "데이터로만 만든 예보를, 그 시점 이후 확정된 실측과 대조한 것입니다.")

# ── 탭 4: 모델 구조 ──────────────────────────────────────────────

with tab_model:
    gw = result["gate_weights"]
    if gw:
        st.markdown("#### 축 배분 (동적 게이팅)")
        st.caption(
            "입력 조건에 따라 세 모달리티의 기여도를 매 예보마다 새로 계산합니다 "
            "(고정 가중치가 아님). 합은 항상 100%입니다."
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
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.caption(f"정적 축 가중치 — {gw}")

    st.markdown("---")
    st.markdown("#### 3축이 각각 무엇을 보는가")
    st.markdown(
        "- **Re축 (공간)** — 이 관측소를 뺀 나머지 11개 관측소의 실측값을 "
        "거리가중(IDW)으로 보간한 공간장. *\"주변 지역은 지금 어떠한가?\"*\n"
        "- **Im축 (시간)** — 같은 관측소의 1·3·6시간 전 대비 변화율(경향벡터). "
        "*\"이 지점은 최근 어느 방향으로 움직이고 있는가?\"*\n"
        "- **Z축 (현재)** — 이번 시각의 수치 스냅샷(기온·습도·기압·풍속 등). "
        "*\"지금 이 순간 상태는 어떤가?\"*\n\n"
        "세 축을 `s = √(A² + (αB)² + (φC)²)`로 융합해(논문 Eq.1) 회귀 헤드"
        "(기온·강수)와 이진분류 헤드(폭염·한파·황사) 4갈래로 분기합니다."
    )
    st.caption(
        "Re·Im 두 축은 원래 논문의 위성 이미지·기상 문서 자리였지만, 이 프로젝트의 "
        "관측망(12개 지상 관측소, 위성·텍스트 데이터 없음)에는 그대로 적용할 수 "
        "없었습니다. 실측으로 검증해보니(축 하나를 꺼서 출력이 그대로인지 비교) "
        "원래 자리를 채웠던 대체 데이터가 정보량이 0이었다는 걸 확인했고, 이후 "
        "'단일 시점 스냅샷(Z축)에서 유도할 수 없는 정보만 새 축에 담는다'는 "
        "기준으로 위와 같이 재설계했습니다 — 공간(주변)과 시간(경향) 모두 "
        "그 시점 하나만 봐서는 알 수 없는 정보라는 점이 Z축과의 수학적 독립성 근거입니다."
    )

    st.markdown("---")
    st.markdown("#### 실시간 데이터는 모델을 다시 학습시키나요?")
    st.markdown(
        f"**아니요.** 화면의 모델은 고정된 체크포인트이고, 실시간 관측은 두 가지 "
        f"용도로만 쓰입니다.\n\n"
        f"1. **추론 입력** — 지금 이 순간의 관측값(+ 이웃 11개 관측소 + 과거 "
        f"1h/3h/6h)을 모델에 넣어 +{lead}시간 후를 예측합니다. 가중치는 건드리지 "
        f"않습니다.\n"
        f"2. **적중률 기록(모니터링)** — 예보를 로그에 남겨두고 +{lead}시간 뒤 "
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
    "Tri-CHEF: 논문 Eq.1 s=√(A²+(αB)²+(φC)²) 기반 3축 기후 예보 파이프라인. "
    "Re축은 12관측소 실측 공간보간장, Im축은 같은 관측소의 시간경향벡터 — "
    "둘 다 실측 데이터에서 유도되며 시뮬레이션/합성 데이터가 아닙니다."
)
