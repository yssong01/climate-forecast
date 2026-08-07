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

from weather_collector import RobustWeatherCollector, STATIONS
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from pipeline_model import TriCHEFPipeline
from train import record_to_vec, STATION_NAMES

CHECKPOINT = "./checkpoints/numerical_trichef.pt"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


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
        # 구버전 체크포인트(이 두 키 추가 전) 호환 — 당시 기본값과 동일
        dynamic_gate=ckpt.get("dynamic_gate", True),
        compact_satellite=ckpt.get("compact_satellite", True),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(stn: str = "108",
            checkpoint_path: str = CHECKPOINT,
            device: str = DEVICE,
            model=None, ckpt: dict = None,
            sat_collector: SimulatedSatelliteCollector = None,
            txt_collector: SimulatedTextCollector = None) -> dict:
    """
    지정 관측소의 현재 관측값으로 +lead_hours 시간 후 기온·강수를 예측.

    model/ckpt, sat_collector/txt_collector 를 미리 만들어 넘기면 재사용한다.
    CLI(1회 실행)에서는 None 으로 두면 내부에서 새로 만든다. app.py 처럼
    반복 호출되는 환경(위젯 조작마다 재실행되는 Streamlit)에서는 호출자가
    st.cache_resource 로 캐싱해 넘겨야 한다 — 특히 txt_collector 는 첫 접근
    시 MiniLM(약 450MB)을 로드하므로, 매 호출마다 새로 만들면 그때마다
    모델을 다시 로드하게 된다.
    """
    if model is None or ckpt is None:
        model, ckpt = load_model(checkpoint_path, device)
    lead_hours = ckpt["lead_hours"]

    collector = RobustWeatherCollector(stn=stn)
    record = collector.fetch()

    sat_collector = sat_collector or SimulatedSatelliteCollector()
    txt_collector = txt_collector or SimulatedTextCollector()

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)
    num_vec  = np.array(record_to_vec(record), dtype=np.float32)
    num_norm = (num_vec - mean) / std

    x_num = torch.tensor(num_norm, dtype=torch.float32).unsqueeze(0).to(device)
    x_img = torch.tensor(sat_collector.get_image(record),
                         dtype=torch.float32).unsqueeze(0).to(device)
    x_txt = torch.tensor(txt_collector.encode_single(record),
                         dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)

    temp_pred   = pred[0, 0].item()
    precip_pred = max(0.0, pred[0, 1].item())
    gate = model.axis_weights()   # dynamic_gate=False 면 {"alpha":..,"phi":..}

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
