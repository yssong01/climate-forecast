"""
app.py — Tri-CHEF 기후 예보 Streamlit 대시보드

학습된 3축 파이프라인(predict.py)을 실시간 관측과 연결해 보여준다.
모델·MiniLM 로드는 st.cache_resource 로 한 번만 수행 — Streamlit 은
위젯 조작마다 스크립트 전체를 재실행하므로 캐싱 없이는 매번 체크포인트를
다시 읽고 MiniLM(약 450MB)을 다시 로드하게 된다.
"""
import os

import streamlit as st
import plotly.graph_objects as go
from streamlit_autorefresh import st_autorefresh

from weather_collector import STATIONS
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from predict import load_model, predict, CHECKPOINT

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
col1, col2, col3, col4 = st.columns(4)
col1.metric("현재 기온", f"{c['temperature']:.1f} °C")
col2.metric("현재 강수", f"{c['precipitation']:.1f} mm")
col3.metric("습도", f"{c['humidity']:.0f} %")
col4.metric("기압", f"{c['pressure']:.0f} hPa")

st.markdown("---")

# ── 예보 ─────────────────────────────────────────────────────────

lead = result["forecast_lead_hours"]
f = result["forecast"]
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
