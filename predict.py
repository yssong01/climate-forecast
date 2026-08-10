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

# 극한기상 헤드 판정 임계값 — threshold_validation.py 실측(2026-08-10, Re·Im축
# 모두 실측 데이터인 체크포인트 기준, 보정용/평가용 분할로 과적합 아님 확인).
# 재보정 후 F1: 강수 0.571 / 폭염 0.862 / 한파 0.730 / 황사 0.602. Im축을
# MiniLM(시뮬레이션 텍스트)에서 시간경향벡터(실측)로 바꾼 뒤로는 이전 체크포인트
# (Im=MiniLM, F1: 0.585/0.880/0.765/0.671)보다 4개 지표 전부 낮다 — "가짜 데이터
# 금지" 원칙을 성능보다 우선해 그대로 채택했다(CLAUDE.md 5절 참고). 화면엔
# 확률을 그대로 보여주는 게 기본이지만("높은 신뢰도"의 조작적 정의 —
# 구조점검 결론), 이진 경보가 필요한 호출부를 위해 여기 상수로 노출한다.
EXTREME_EVENT_THRESH = {
    "rain":     0.78,
    "heatwave": 0.84,
    "coldwave": 0.85,
    "dust":     0.91,
}


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
    record = None
    neighbor_records = []
    for other_stn in STATIONS.values():
        try:
            r = RobustWeatherCollector(stn=other_stn).fetch()
        except Exception:
            continue
        if other_stn == stn:
            record = r
        neighbor_records.append(r)
    if record is None:
        record = RobustWeatherCollector(stn=stn).fetch()
        neighbor_records.append(record)

    field_collector = InterpolatedFieldCollector(neighbor_records, STATION_COORDS)

    # Im축 인코더 종류에 따라 분기 — im_dim=384 면 구버전(MiniLM 텍스트),
    # 12 면 신버전(시간 경향 벡터). 재학습 전환기 동안 두 체크포인트 모두
    # 이 함수 하나로 서빙하기 위해 필요하다.
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
                tendency_records.append(past)
            except Exception:
                continue
        txt_vec = TendencyCollector(tendency_records).encode_single(record)

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)
    num_vec  = np.array(record_to_vec(record), dtype=np.float32)
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
