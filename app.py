"""
app.py — Tri-CHEF 기후 예보 Streamlit 대시보드

학습된 3축 파이프라인(predict.py)을 실시간 관측과 연결해 보여준다.
모델·MiniLM 로드는 st.cache_resource 로 한 번만 수행 — Streamlit 은
위젯 조작마다 스크립트 전체를 재실행하므로 캐싱 없이는 매번 체크포인트를
다시 읽고 MiniLM(약 450MB)을 다시 로드하게 된다.
"""
import json
import os
from datetime import datetime, timedelta

import streamlit as st
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from streamlit_autorefresh import st_autorefresh

from weather_collector import STATIONS, STATION_COORDS
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from predict import load_model, predict, CHECKPOINT
import accuracy

HIST_FILES = ["./cache/historical_data_1y.json", "./cache/historical_data.json"]

st.set_page_config(
    page_title="Tri-CHEF 기후 예보",
    page_icon="🌦️",
    layout="centered",
)


# ── 캐시된 리소스 — 세션당 한 번만 로드 ──────────────────────────────

@st.cache_resource(show_spinner="모델 로드 중...")
def get_model():
    return load_model(CHECKPOINT)


@st.cache_resource(show_spinner="위성/텍스트 인코더 준비 중 (MiniLM 첫 로드 시 다소 소요)...")
def get_collectors():
    return SimulatedSatelliteCollector(), SimulatedTextCollector()


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
    cutoff = datetime.now() - timedelta(hours=hours)
    rows = [r for (s, ts), r in history.items()
            if s == stn and datetime.strptime(ts[:12], "%Y%m%d%H%M") >= cutoff]
    rows.sort(key=lambda r: r["timestamp"])
    return rows


def render_station_map(stations: dict, selected_stn: str) -> None:
    """사이드바용 관측소 위치 간이 지도 — OpenStreetMap 무료 타일, API 키 불필요."""
    codes = list(stations.values())
    lats = [STATION_COORDS[c][0] for c in codes]
    lons = [STATION_COORDS[c][1] for c in codes]
    names = list(stations.keys())
    colors = ["#E2954F" if c == selected_stn else "#4C78A8" for c in codes]
    sizes = [15 if c == selected_stn else 9 for c in codes]

    fig = go.Figure(go.Scattermapbox(
        lat=lats, lon=lons, mode="markers+text",
        marker=dict(size=sizes, color=colors),
        text=names, textposition="top center",
        textfont=dict(size=10, color="#AAB4B6"),
        hovertext=names, hoverinfo="text",
    ))
    fig.update_layout(
        mapbox=dict(style="open-street-map", zoom=5.4,
                    center=dict(lat=36.4, lon=127.9)),
        margin=dict(l=0, r=0, t=0, b=0), height=260,
        paper_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True, config={"scrollZoom": False})


# ── 사이드바 ──────────────────────────────────────────────────────

st.sidebar.title("설정")

station_options = dict(STATIONS)  # {"서울": "108", ...}
station_name = st.sidebar.selectbox(
    "관측소", list(station_options.keys()), index=0
)
stn = station_options[station_name]

auto_refresh = st.sidebar.toggle("자동 갱신", value=True)
interval_min = st.sidebar.slider("갱신 주기 (분)", 1, 30, 5, disabled=not auto_refresh)

if auto_refresh:
    st_autorefresh(interval=interval_min * 60 * 1000, key="refresh")

st.sidebar.markdown("---")
st.sidebar.caption(
    "ASOS 지상관측은 매시 정각 관측 후 약 10분 뒤 공개됩니다. "
    "갱신 주기를 너무 짧게 두어도 새 데이터가 없을 수 있습니다."
)

st.sidebar.markdown("---")
st.sidebar.markdown("**관측소 위치**")
render_station_map(station_options, stn)
st.sidebar.caption(
    "실제 위성·레이더·특보 영상은 아직 연결 전입니다 — "
    "이 프로젝트의 위성 데이터는 현재 시뮬레이션이라(README 한계 1) "
    "실제 영상인 것처럼 표시하지 않습니다."
)


# ── 메인 ──────────────────────────────────────────────────────────

st.title("🌦️ Tri-CHEF 기후 예보")
st.caption("3축 Tri-CHEF 파이프라인 (위성·기상문서·수치센서) 기반 실시간 예보")

if not os.path.exists(CHECKPOINT):
    st.error(
        f"체크포인트가 없습니다: `{CHECKPOINT}`\n\n"
        f"먼저 터미널에서 `python train.py` 를 실행해 모델을 학습하세요."
    )
    st.stop()

try:
    model, ckpt = get_model()
    sat_collector, txt_collector = get_collectors()
    result = predict(stn=stn, model=model, ckpt=ckpt,
                     sat_collector=sat_collector, txt_collector=txt_collector)
except Exception as e:
    st.error(f"예측 실패: {e}")
    st.stop()

# 데이터 상태 배지
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

st.subheader(f"{result['station_name']} ({result['station_code']})")

# ── 현재 관측값 ───────────────────────────────────────────────────

c = result["current"]
lead = result["forecast_lead_hours"]
f = result["forecast"]
col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 기온", f"{c['temperature']:.1f} °C")
col2.metric("현재 강수", f"{c['precipitation']:.1f} mm")
col3.metric("습도", f"{c['humidity']:.0f} %")
col4.metric("기압", f"{c['pressure']:.0f} hPa")

# ── 최근 추이 + 예보 ────────────────────────────────────────────────

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
        vertical_spacing=0.16, horizontal_spacing=0.08,
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
        height=460, margin=dict(l=10, r=10, t=36, b=10),
        showlegend=True, legend=dict(orientation="h", y=1.12, x=0),
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
    )
    st.plotly_chart(fig, use_container_width=True)
    st.caption(f"최근 {len(series)}시간 실측(로컬 캐시) + 별표는 이번 +{lead}시간 예보. "
               "기온·강수만 예보 대상 — 습도·기압은 참고용 실측치입니다.")

st.markdown("---")

# ── 예보 ─────────────────────────────────────────────────────────

st.markdown(f"### +{lead}시간 후 예보")

fc1, fc2 = st.columns(2)
fc1.metric(
    "예상 기온", f"{f['temperature']:.1f} °C",
    delta=f"{f['temperature'] - c['temperature']:+.1f} °C (현재 대비)",
)
fc2.metric(
    "예상 강수", f"{f['precipitation']:.1f} mm",
    delta=f"{f['precipitation'] - c['precipitation']:+.1f} mm (현재 대비)",
    delta_color="inverse",
)

val_temp = ckpt.get("val_temp_mae")
val_precip = ckpt.get("val_precip_mae")
if val_temp is not None:
    st.caption(
        f"모델 검증 성능(학습 시 측정) — 기온 MAE {val_temp:.2f}°C, "
        f"강수 MAE {val_precip:.2f}mm"
    )

st.markdown("---")

# ── 예보 적중률 ──────────────────────────────────────────────────
#
# 실시간 예보를 로그에 남기고(같은 목표시각은 중복 기록 안 함), 로컬
# 캐시에서 이미 확인된 실측만으로 대기 중인 항목을 채운다 — 여기서
# apihub 를 새로 호출하지 않는다(모듈 accuracy.py 주석 참고). 백테스트로
# 시딩된 이력이 있으면 페이지를 처음 열자마자 누적 적중률이 바로 보인다.

if status == "SUCCESS_LIVE" and result.get("target_time"):
    accuracy.record_prediction(
        station=stn, made_at=result["observed_at"],
        target_time=result["target_time"],
        pred_temp=f["temperature"], pred_precip=f["precipitation"],
        source="live",
    )
lookup = {k: v for k, v in history.items()}
accuracy.resolve_pending(lookup)

st.markdown("### 예보 적중률")
st.caption(
    f"기온: 오차 ±{accuracy.HIT_TEMP_TOL}°C 이내면 적중 · "
    f"강수: {accuracy.PRECIP_THRESH}mm 기준 비/무비 판정 일치 시 적중"
)
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
    st.caption(f"누적 표본 {acc['cum_n']}건 ({result['station_name']} 기준)")

st.markdown("---")

# ── 축 배분 (Phase 3-5 동적 게이팅) ─────────────────────────────────

gw = result["gate_weights"]
if gw:
    st.markdown("### 축 배분 (동적 게이팅)")
    st.caption(
        "입력 조건에 따라 세 모달리티의 기여도를 실시간으로 계산합니다. "
        "합은 항상 100%입니다."
    )

    axes = ["위성 이미지 (Re)", "기상 문서 (Im)", "수치 센서 (Z)"]
    values = [gw.get("w_re", 0) * 100, gw.get("w_im", 0) * 100, gw.get("w_z", 0) * 100]
    colors = ["#4C78A8", "#F58518", "#54A24B"]

    fig = go.Figure(go.Bar(
        x=values, y=axes, orientation="h",
        marker_color=colors,
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        cliponaxis=False,   # 값이 100%에 가까운 막대의 라벨이 플롯 경계에서 잘리는 것 방지
    ))
    fig.update_layout(
        xaxis=dict(range=[0, 100], title="배분 비율 (%)"),
        height=220,
        margin=dict(l=10, r=40, t=10, b=10),   # 오른쪽 여백 확보 — 라벨 공간
    )
    st.plotly_chart(fig, use_container_width=True)
else:
    st.caption(
        f"정적 축 가중치 — α(텍스트) {result.get('gate_weights', {})}"
    )

st.markdown("---")
st.caption(
    "Tri-CHEF: 논문 Eq.1 s=√(A²+(αB)²+(φC)²) 기반 3축 기후 예보 파이프라인. "
    "위성·문서 축은 현재 시뮬레이션 데이터로 학습되었습니다."
)
