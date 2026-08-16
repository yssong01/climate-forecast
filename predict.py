"""
predict.py — 학습된 Tri-CHEF 체크포인트로 실시간 +N시간 예보 생성.

사용법:
  python predict.py                    # 서울(108) 예보, 사람이 읽는 형식
  python predict.py --stn 159          # 부산 예보
  python predict.py --stn 159 --json   # JSON 출력 (app.py·API 연동용)
  python predict.py --list-stations    # 관측소 코드 목록

학습(train.py)과 추론이 동일한 전처리 경로를 쓰도록, 수치 벡터 변환은
train.record_to_vec() 을 그대로 재사용한다 — 별도 구현하면 두 경로가
조용히 어긋나는 것이 흔한 버그 원인이기 때문이다.
"""
import argparse
import json
import sys
from datetime import datetime, timedelta

import numpy as np
import torch

from weather_collector import RobustWeatherCollector, STATIONS, STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector, LAGS_HOURS
from text_collector import SimulatedTextCollector
from pipeline_model import TriCHEFPipeline
from train import record_to_vec, STATION_NAMES

CHECKPOINT = "./checkpoints/numerical_trichef.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 강수 후처리 클리핑 — precip_breakdown.py 실측: 극한기상 헤드가 늘어날수록
# (폭염→한파→황사) 공유 magnitude 표현을 두고 헤드들이 경쟁하면서 Hurdle의
# 무강수 억제력이 약해졌다. EXTREME_BCE_WEIGHT 하향(1.0→0.3, train.py)으로
# 근본 원인을 완화했지만(-25.6%→-19.3%), 남은 손해를 이 클리핑으로 마저
# 줄인다.
PRECIP_CLIP_THRESH = 0.2  # mm — 이 미만 예측은 0으로 반올림

# 극한기상 헤드 판정 임계값 — threshold_validation.py 재실측(2026-08-16,
# v9 체크포인트 — 13년·그룹분할·12차원·한파 디커플링, A안 롤백 후 기준).
#
# 왜 A안(14차원, 연중 시각 특징)에서 롤백했는가: A안 재학습 직후 한파 헤드가
# 기온에 거꾸로 반응하는 버그가 발견됐다(15°C→33°C로 올릴수록 한파확률이
# 0.14→0.97로 상승, coldwave_sensitivity_check.py 실측). 디커플링 전(A안
# 재학습 직후)부터 이미 나타난 문제라 원인은 디커플링이 아니라 A안 재학습
# 자체다. 근본 원인 미파악 상태라 우선 A안 이전 체크포인트(v9)로 롤백했다
# (README '정직한 한계' 참고). 임계값도 v9 기준으로 다시 쟀다.
#
# 채택 기준: 보정용/평가용 분할에서 t=0.5 대비 순이득이 0.01을 넘는 것만
# 재보정값을 쓴다(threshold_validation.py 자체 기준).
#   강수    — 순이득 +0.099 → 재보정값(0.81) 채택, F1 0.500 (t=0.5는 0.402)
#   폭염    — 순이득 +0.005 → 기준 미달, t=0.5 유지 (F1 0.812, 이미 근접 최적)
#   한파    — 순이득 +0.009 → 기준 미달(0.01 문턱 근소 미달), t=0.5 유지
#   황사    — 순이득 +0.028 → 재보정값(0.81) 채택. 다만 재보정해도 정밀도가
#            낮아 화면에서는 제외했다. 헤드 자체는 유지한다(내부적으로는
#            여전히 계산됨) — EXTREME_EVENT_THRESH 에 값을 남겨 predict()가
#            확률은 계속 반환하되, app.py 가 표시하지 않는다.
EXTREME_EVENT_THRESH = {
    "rain":     0.81,
    "heatwave": 0.50,
    "coldwave": 0.50,
    "dust":     0.81,
}

# 관측소별 임계값 재보정(2026-08-16, station_threshold_check.py) — 부산은
# calibration_plot_diagnose.py에서 폭염 정밀도만 확연히 낮게(47%) 나왔고,
# station_anomaly_investigate.py로 원인도 확인했다(부산은 오탐일 때 기온이
# 부산 자신의 실제 폭염일 평균보다 높다 — 공식 기준 자체가 다른데 모델은
# 전체 관측소 공통 임계값을 씀). 보정용/평가용 분리 검증에서 순이득 +0.039로
# 채택 기준(0.01)을 크게 넘어 채택. 다른 관측소는 순이득이 기준 미달이거나
# 아직 검증하지 않아 전역 임계값을 그대로 쓴다 — 검증 없이 관측소마다
# 따로 고르면 과적합이라는 규약(threshold_validation.py)은 관측소 단위에도
# 똑같이 적용된다.
STATION_EVENT_THRESH_OVERRIDES = {
    ("159", "heatwave"): 0.33,   # 부산 — 순이득 +0.039, t±0.02 F1 폭 0.0051(안정)
}


def event_threshold(event: str, stn: str) -> float:
    """관측소별 재보정값이 있으면 그걸, 없으면 전역 기본값을 반환한다."""
    return STATION_EVENT_THRESH_OVERRIDES.get((stn, event), EXTREME_EVENT_THRESH[event])


def load_model(checkpoint_path: str = CHECKPOINT, device: str = DEVICE):
    """체크포인트에서 모델 아키텍처와 가중치를 복원한다."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = TriCHEFPipeline(
        embed_dim=ckpt["embed_dim"],
        num_features=ckpt["num_features"],
        alpha_init=ckpt["alpha_init"], phi_init=ckpt["phi_init"],
        orthogonalize=ckpt["orthogonalize"],
        temp_mean=ckpt["temp_mean"], precip_mean=ckpt["precip_mean"],
        persistence_residual=ckpt["persistence_residual"],
        feat_mean=ckpt["mean"], feat_std=ckpt["std"],
        # 구버전 체크포인트(이 키들 추가 전) 호환 — 당시 기본값과 동일
        dynamic_gate=ckpt.get("dynamic_gate", True),
        compact_satellite=ckpt.get("compact_satellite", True),
        im_dim=ckpt.get("im_dim", 384),   # 384=MiniLM(구버전), 12=경향벡터
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(stn: str = "108",
            checkpoint_path: str = CHECKPOINT,
            device: str = DEVICE,
            model=None, ckpt: dict = None) -> dict:
    """
    지정 관측소의 현재 관측값으로 +lead_hours 시간 후 기온·강수를 예측.

    Re축은 다시 켠다 — interp_field_collector.py 실측(2026-08-09): 가짜
    위성(정보량 0)을 12관측소 실측 IDW 공간보간장으로 바꾸자 게이트가 Re축에
    51.6%를 배분했고 폭염·한파·황사 F1 이 전부 크게 올랐다(황사 0.257→0.671).
    이전엔 img_x=None 으로 꺼서 배포 RAM 을 아꼈지만, 이제 Re축이 가장 큰
    비중을 차지하므로 끄면 방금 얻은 개선을 그대로 잃는다.

    Im축은 체크포인트에 따라 자동 분기한다(model.im_dim) — 신버전(12차원
    시간 경향 벡터, tendency_collector.py, Phase 3-11)이면 sentence-
    transformers 로드가 아예 없고, 구버전(384차원 MiniLM 텍스트)이면 기존
    방식으로 폴백한다. 전환기 호환을 위한 분기이며, 재학습이 끝나 구버전
    체크포인트를 더 안 쓰게 되면 단순화할 수 있다.

    두 축 모두 "다른 시각·다른 관측소"의 실측이 필요해 API 조회가 늘어난다:
      · Re축(공간): 같은 시각 다른 11개 관측소 → 12회
      · Im축(시간): 대상 관측소의 1h/3h/6h 전 관측 → 3회
    (호출당 총 15회 — 이전 1회 대비 응답 지연 증가는 감수). 일부 조회가
    실패해도 InterpolatedFieldCollector/TendencyCollector 가 남은 것만으로
    동작하거나(전무하면 중립값 0.5 또는 0) 정상 동작한다.

    model/ckpt 를 미리 만들어 넘기면 재사용한다. CLI(1회 실행)에서는 None 으로
    두면 내부에서 새로 만든다. app.py 처럼 반복 호출되는 환경에서는 호출자가
    st.cache_resource 로 캐싱해 넘기는 게 좋다(체크포인트 재로드 방지).
    """
    if model is None or ckpt is None:
        model, ckpt = load_model(checkpoint_path, device)
    lead_hours = ckpt["lead_hours"]

    # 대상 관측소 + 보간용 이웃 11개 — 전부 같은 시각(현재) 스냅샷이 필요하다.
    #
    # 이웃은 SUCCESS_LIVE 만 보간장에 넣는다. 폴백 레코드를 그대로 쓰면
    # FALLBACK_CACHED 는 "직전 값의 복제(+미세 노이즈)"를, FALLBACK_DEFAULT 는
    # 합성 기본값(20.0°C / 1013hPa)을 실측 관측인 양 IDW 에 태우게 된다 —
    # 가짜 데이터를 입력으로 쓰지 않는다는 규칙(CLAUDE.md 1절 5항, Re축이
    # 정보량 0이었던 바로 그 원인)에 정면으로 어긋난다. 빼면 컬렉터가
    # 설계대로 "정보 없음"으로 처리한다(이웃이 하나도 없으면 중립값 0.5).
    #
    # 대상 관측소 자신의 레코드는 폴백이어도 그대로 쓴다 — Z축 입력이 아예
    # 없으면 예보를 만들 수 없고, 그 상태는 data_status 로 호출부에 전달돼
    # 화면 상단 배지에 경고로 표시된다.
    record = None
    neighbor_records = []
    for other_stn in STATIONS.values():
        try:
            r = RobustWeatherCollector(stn=other_stn).fetch()
        except Exception:
            continue
        if other_stn == stn:
            record = r
        if r.get("status") == "SUCCESS_LIVE":
            neighbor_records.append(r)
    if record is None:
        record = RobustWeatherCollector(stn=stn).fetch()
        if record.get("status") == "SUCCESS_LIVE":
            neighbor_records.append(record)
    # get_image() 는 대상 관측소를 스스로 제외하므로, 실제 보간에 쓰이는 건
    # 여기서 자신을 뺀 수다.
    live_neighbors = sum(1 for r in neighbor_records if str(r.get("stn")) != stn)

    field_collector = InterpolatedFieldCollector(neighbor_records, STATION_COORDS)

    # Im축 인코더 종류에 따라 분기 — im_dim=384 면 구버전(MiniLM 텍스트),
    # 12 면 신버전(시간 경향 벡터). 재학습 전환기 동안 두 체크포인트 모두
    # 이 함수 하나로 서빙하기 위해 필요하다.
    live_lags = None   # MiniLM 분기에는 시차 조회 자체가 없다
    if getattr(model, "im_dim", 384) >= 128:
        txt_vec = SimulatedTextCollector().encode_single(record)
    else:
        tendency_records = [record]
        target_collector = RobustWeatherCollector(stn=stn)
        now_ts = datetime.strptime(str(record["timestamp"])[:12], "%Y%m%d%H%M")
        for lag in LAGS_HOURS:
            try:
                past = target_collector.fetch_at(
                    (now_ts - timedelta(hours=lag)).strftime("%Y%m%d%H00"))
            except Exception:
                continue
            # 과거 시각도 SUCCESS_LIVE 만 받는다. 폴백은 요청한 시각의
            # 타임스탬프를 달고 오므로 그대로 넣으면 TendencyCollector 가
            # "그 시각에 실제 관측이 있었다"고 믿고 차분을 계산한다 —
            # FALLBACK_CACHED 는 변화량 ≈ 0 을, FALLBACK_DEFAULT 는 거대한
            # 가짜 변화량을 만들어낸다. 빼면 설계대로 0(변화 정보 없음)이 된다.
            if past.get("status") == "SUCCESS_LIVE":
                tendency_records.append(past)
        live_lags = len(tendency_records) - 1
        txt_vec = TendencyCollector(tendency_records).encode_single(record)

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)
    # record_to_vec() 은 14차원(연중 시각 sin/cos, 2026-08-16 추가)인데 구버전
    # 체크포인트는 12차원으로 학습됐다. 새 축은 끝에 덧붙었으므로 그 체크포인트의
    # 차원만큼만 앞에서 잘라 쓰면 구버전 입력을 그대로 복원한다.
    nf = ckpt.get("num_features", len(mean))
    num_vec  = np.array(record_to_vec(record), dtype=np.float32)[:nf]
    num_norm = (num_vec - mean) / std

    x_num = torch.tensor(num_norm, dtype=torch.float32).unsqueeze(0).to(device)
    x_img = torch.tensor(field_collector.get_image(record),
                         dtype=torch.float32).unsqueeze(0).to(device)
    x_txt = torch.tensor(txt_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)

    temp_pred   = pred[0, 0].item()
    precip_pred = max(0.0, pred[0, 1].item())
    if precip_pred < PRECIP_CLIP_THRESH:
        precip_pred = 0.0
    gate = model.axis_weights()   # dynamic_gate=False 면 {"alpha":..,"phi":..}

    # Phase 3-8/3-9 극한기상 확률 — 구버전 체크포인트(헤드 추가 전)는
    # extreme_event_probs() 가 전부 None 을 반환하므로 결과에서 빠진다.
    extreme = model.extreme_event_probs()
    heatwave_prob = extreme["heatwave"][0].item() if extreme["heatwave"] is not None else None
    coldwave_prob = extreme["coldwave"][0].item() if extreme["coldwave"] is not None else None
    dust_prob     = extreme["dust"][0].item()     if extreme["dust"]     is not None else None

    observed_at = record.get("timestamp")
    # accuracy.py 가 (관측소, 목표시각)으로 로그를 남기려면 이 시각이 필요하다
    # — 시간대는 관측시각 문자열이 이미 KST 벽시계 표기라 그대로 산술한다.
    target_time = (
        datetime.strptime(str(observed_at)[:12], "%Y%m%d%H%M")
        + timedelta(hours=lead_hours)
    ).strftime("%Y%m%d%H00") if observed_at else None

    return {
        "station_code": stn,
        "station_name": STATION_NAMES.get(stn, stn),
        "observed_at":  observed_at,
        "target_time":  target_time,
        "data_status":  record.get("status"),   # SUCCESS_LIVE / FALLBACK_*
        # 실측으로 채워진 보조 입력의 개수. 대상 관측소의 status 만으로는
        # Re·Im축이 얼마나 채워졌는지 알 수 없다 — 자기 조회는 성공했는데
        # 이웃·과거 조회가 예산 상한이나 API 장애로 폴백된 상태가 가능하다.
        # 그 경우 보간장은 중립값(0.5), 경향벡터는 0으로 채워지므로 예보의
        # 근거가 실제로 얇아진다. 화면이 그 사실을 말할 수 있게 내보낸다.
        "input_coverage": {
            "neighbors_live":  live_neighbors,
            "neighbors_total": len(STATIONS) - 1,
            "lags_live":       live_lags,
            "lags_total":      len(LAGS_HOURS) if live_lags is not None else None,
        },
        "current": {
            "temperature":   record.get("temperature"),
            "precipitation": record.get("precipitation"),
            "humidity":      record.get("humidity"),
            "wind_speed":    record.get("wind_speed"),
            "pressure":      record.get("pressure"),
        },
        "forecast_lead_hours": lead_hours,
        "forecast": {
            "temperature":   round(temp_pred, 2),
            "precipitation": round(precip_pred, 2),
        },
        "extreme_event_probs": {
            "heatwave": round(heatwave_prob, 4) if heatwave_prob is not None else None,
            "coldwave": round(coldwave_prob, 4) if coldwave_prob is not None else None,
            "dust":     round(dust_prob, 4)     if dust_prob     is not None else None,
        },
        "gate_weights": (
            {k: round(v, 4) for k, v in gate.items()} if "w_re" in gate else None
        ),
    }


def _print_human(result: dict) -> None:
    print(f"{'='*52}")
    print(f" Tri-CHEF 기후 예보 — {result['station_name']}({result['station_code']})")
    print(f"{'='*52}")
    status_note = "" if result["data_status"] == "SUCCESS_LIVE" else \
        f"  ⚠ {result['data_status']} (실시간 데이터 아님)"
    print(f" 관측 시각 : {result['observed_at']}{status_note}")

    c = result["current"]
    print(f" 현재 기온 : {c['temperature']:.1f} °C")
    print(f" 현재 강수 : {c['precipitation']:.1f} mm")
    print(f" 현재 습도 : {c['humidity']:.0f} %")
    print(f"{'-'*52}")

    f = result["forecast"]
    print(f" +{result['forecast_lead_hours']}시간 후 예보")
    print(f"   기온 : {f['temperature']:.1f} °C")
    print(f"   강수 : {f['precipitation']:.1f} mm")

    ev = result.get("extreme_event_probs") or {}
    if any(ev.get(k) is not None for k in ("heatwave", "coldwave", "dust")):
        print(f"{'-'*52}")
        if ev.get("heatwave") is not None:
            print(f" 폭염 확률 : {ev['heatwave']:.1%}")
        if ev.get("coldwave") is not None:
            print(f" 한파 확률 : {ev['coldwave']:.1%}")
        if ev.get("dust") is not None:
            print(f" 황사 확률 : {ev['dust']:.1%}")

    if result["gate_weights"]:
        g = result["gate_weights"]
        print(f"{'-'*52}")
        print(f" 축 배분 — 위성 {g.get('w_re', 0):.1%} | "
              f"텍스트 {g.get('w_im', 0):.1%} | 수치 {g.get('w_z', 0):.1%}")
    print(f"{'='*52}")


def main():
    parser = argparse.ArgumentParser(description="Tri-CHEF 기후 예보 추론")
    parser.add_argument("--stn", default="108",
                        help="관측소 코드 (기본: 108=서울)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력 (app.py·API 연동용)")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help=f"체크포인트 경로 (기본: {CHECKPOINT})")
    parser.add_argument("--list-stations", action="store_true",
                        help="사용 가능한 관측소 코드 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list_stations:
        for name, code in STATIONS.items():
            print(f"  {code}  {name}")
        return

    if args.stn not in STATIONS.values():
        print(f"[ERROR] 알 수 없는 관측소 코드: {args.stn}", file=sys.stderr)
        print(f"  사용 가능: {dict(STATIONS)}", file=sys.stderr)
        sys.exit(1)

    result = predict(stn=args.stn, checkpoint_path=args.checkpoint)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
