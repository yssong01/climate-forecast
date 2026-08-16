"""
coldwave_sensitivity_check.py — 8월 한낮인데 한파 확률이 서울 82%까지
튀는 게 실측됐다(2026-08-16, 사용자 스크린샷). 어떤 입력에 민감한지
한 축씩 흔들어 확인한다. 1회성 진단 스크립트.

실행: python coldwave_sensitivity_check.py
"""
import sys

import numpy as np
import torch

from predict import load_model, CHECKPOINT
from train import record_to_vec
from weather_collector import RobustWeatherCollector
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS, STATIONS

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def cold_prob(model, ckpt, record, img=None, txt=None):
    mean = np.array(ckpt["mean"], dtype=np.float32)
    std = np.array(ckpt["std"], dtype=np.float32)
    nf = ckpt.get("num_features", len(mean))
    vec = np.array(record_to_vec(record), dtype=np.float32)[:nf]
    x_num = torch.tensor((vec - mean) / std, dtype=torch.float32).unsqueeze(0).to(DEVICE)
    x_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(DEVICE) if img is not None \
        else torch.zeros(1, 4, 32, 32, dtype=torch.float32).to(DEVICE)
    x_txt = torch.tensor(txt, dtype=torch.float32).unsqueeze(0).to(DEVICE) if txt is not None \
        else torch.zeros(1, 12, dtype=torch.float32).to(DEVICE)
    with torch.no_grad():
        model(num_x=x_num, img_x=x_img, txt_x=x_txt)
        p = torch.sigmoid(model._last_coldwave_logit).item()
    return p


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    model, ckpt = load_model(ckpt_path)
    print(f"체크포인트: {ckpt_path} (num_features={ckpt.get('num_features')}, "
          f"decoupled={ckpt.get('head_decouple_heads')})")
    base = RobustWeatherCollector(stn="108").fetch()
    print("서울 현재 레코드:", {k: v for k, v in base.items() if k != "status"})
    print(f"기준 한파확률(영벡터 Re/Im) = {cold_prob(model, ckpt, base):.4f}\n")

    # ── 실제 Re/Im 축까지 넣어서 재확인 (영벡터가 아닌 진짜 입력) ──
    all_records = [RobustWeatherCollector(stn=s).fetch() for s in STATIONS.values()]
    sat = InterpolatedFieldCollector(all_records, STATION_COORDS)
    txt_collector = TendencyCollector(all_records)
    img = sat.get_image(base)
    txt = txt_collector.encode_single(base)
    print(f"실제 Re/Im 축 입력 시 한파확률 = {cold_prob(model, ckpt, base, img, txt):.4f}\n")

    print("=" * 70)
    print("한 축씩 흔들기 (실제 Re/Im 유지, Z축 한 항목만 변경)")
    print("=" * 70)

    for field, values in [
        ("temperature", [15, 20, 24.8, 28, 33]),
        ("pressure", [995, 1000, 1004.1, 1010, 1020]),
        ("humidity", [30, 50, 74, 85, 95]),
        ("wind_speed", [0.5, 2, 3.4, 6, 10]),
    ]:
        print(f"\n[{field}]")
        for v in values:
            r2 = dict(base)
            r2[field] = v
            p = cold_prob(model, ckpt, r2, img, txt)
            print(f"  {field}={v:>7} -> 한파확률 {p:.4f}")


if __name__ == "__main__":
    main()
