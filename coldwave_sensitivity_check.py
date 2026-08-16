"""
coldwave_sensitivity_check.py — 한파 헤드의 물리적 단조성(monotonicity)을
직접 흔들어 검증한다(CLAUDE.md 규칙 8).

최초 작성 계기: 8월 한낮에 한파 확률이 서울 82%까지 상승하는 현상이
실측됐다(2026-08-16). 어떤 입력에 민감한지 한 축씩 흔들어 확인한다.

계절 축을 함께 흔드는 이유(2026-08-16 추가) — A안(14차원)은 입력에 연중
시각(day-of-year) sin/cos이 포함되므로, 실시간 관측(8월)을 기준으로만
기온을 흔들면 모델이 "지금은 여름"을 근거로 한파 확률을 0 근처에 고정한다.
이 구간에서는 헤드가 역전됐는지 정상인지 구분할 수 없다 — 테스트가 눈이
먼다. 한파 헤드가 실제로 동작하는 겨울 구간에서 기온을 흔들어야 단조성
(기온 상승 → 한파 확률 하강)을 판정할 수 있다.

12차원 체크포인트는 연중 시각 특징이 잘려 나가므로(vec[:num_features])
계절을 바꿔도 입력이 동일하다 — 여름·겨울 결과가 같게 나오는 것이 정상이며,
이는 테스트가 의도대로 동작함을 보여주는 대조군 역할을 한다.

실행: python coldwave_sensitivity_check.py [체크포인트경로]
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

    # ── 계절 × 기온 격자 — 단조성 판정의 본 시험 ──────────────────
    # 시각(시)은 고정하고 날짜만 바꾼다. 연중 시각 sin/cos만 달라지도록
    # 통제한 대조 실험이다(Re/Im축은 실측 그대로 유지 — 한 번에 하나만
    # 흔든다는 원칙). 절대 확률값은 Re/Im이 여름 실측이라 인위적일 수
    # 있으나, 판정 기준은 절대값이 아니라 기온에 대한 '변화 방향'이다.
    hh = str(base.get("timestamp", "202601011200"))[8:12] or "1200"
    print("\n" + "=" * 70)
    print("계절 × 기온 격자 — 한파 헤드 단조성 판정")
    print("(기온이 오르면 한파 확률은 내려가야 정상)")
    print("=" * 70)

    temps = [-10, -5, 0, 5, 10, 15, 20, 25, 30]
    print(f"\n{'계절':<14}" + "".join(f"{t:>8}°C" for t in temps) + "   판정")
    for label, date in [("한겨울(1/15)", "20260115"),
                        ("봄(4/15)", "20260415"),
                        ("한여름(8/15)", "20260815"),
                        ("가을(10/15)", "20261015")]:
        probs = []
        for t in temps:
            r2 = dict(base)
            r2["temperature"] = t
            r2["timestamp"] = date + hh
            probs.append(cold_prob(model, ckpt, r2, img, txt))
        # 단조성 판정: 기온이 오를 때 확률이 유의하게 상승하는 구간만 위반으로
        # 센다. 확률이 0 근처인 구간의 미세한 흔들림(예: 0.0040→0.0052)까지
        # 위반으로 잡으면 물리적으로 무의미한 노이즈에 판정이 좌우된다.
        RISE_TOL = 0.01     # 이보다 큰 상승만 위반으로 인정
        rises = [b - a for a, b in zip(probs, probs[1:]) if b - a > RISE_TOL]
        span = max(probs) - min(probs)
        if span < RISE_TOL:
            verdict = "무반응(이 계절엔 헤드가 잠잠 — 판정 불가)"
        elif not rises:
            verdict = "정상(기온↑ → 한파확률↓)"
        elif max(rises) > 0.1:
            verdict = f"★역전★ (최대 +{max(rises):.3f} 상승)"
        else:
            verdict = f"경미한 역행 (최대 +{max(rises):.3f})"
        print(f"{label:<14}" + "".join(f"{p:>10.4f}" for p in probs) + f"   {verdict}")


if __name__ == "__main__":
    main()
