"""
seasonal_falsealarm_check.py — 실제 관측 전 구간에서 극한기상 헤드의 계절
오탐을 점검한다.

발견 경위(2026-08-16) — 배포 중이던 v9(12차원)의 폭염 헤드가 한겨울에
작동했다. 실제 −15°C 이하 관측 809건 중 85.7%에서 폭염 확률이 0.5를 넘었다.
원인은 융합 수식 `s=√((w·v)²)`가 부호를 없애 '극단적 저온'의 큰 편차가
'극단적 고온'과 같은 크기로 보이기 때문이며, 12차원 입력에는 이를 구분할
계절 단서가 없다.

**집계 지표로는 이 결함이 잡히지 않는다.** 폭염 라벨은 공식 특보 기록이
있는 표본(여름)에서만 채점하므로, 겨울 오탐은 애초에 채점 대상 밖이다.
F1이 0.813이어도 겨울에 폭염을 외칠 수 있다.

그래서 라벨과 무관하게 **실제 관측 기온대별로** 두 헤드의 확률을 직접 본다.
합성 입력이 아니라 그 시각의 실제 이웃 관측(Re축)·실제 경향(Im축)을 쓰므로,
물리적으로 불가능한 조합을 만들어 놓고 오작동이라 오판할 위험이 없다.

판정 기준
  폭염 헤드 — 0°C 이하 구간에서 확률이 높으면 오탐이다.
  한파 헤드 — 25°C 이상 구간에서 확률이 높으면 오탐이다.

실행: python seasonal_falsealarm_check.py [체크포인트 ...]
"""
import sys

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import WeatherDataset, collect_historical
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096

# (하한, 상한, 이 구간에서 오탐으로 봐야 할 헤드)
BANDS = [
    (-30, -15, "heatwave"),
    (-15, -10, "heatwave"),
    (-10,  -5, "heatwave"),
    (-5,    0, "heatwave"),
    (0,    10, None),
    (10,   20, None),
    (20,   25, None),
    (25,   30, "coldwave"),
    (30,   45, "coldwave"),
]


def main():
    paths = sys.argv[1:] or [CHECKPOINT]
    records = collect_historical()
    sat = InterpolatedFieldCollector(records, STATION_COORDS)
    tnd = TendencyCollector(records)

    for path in paths:
        model, ckpt = load_model(path, DEVICE)
        model.eval()
        ds = WeatherDataset(records, sat_collector=sat, txt_collector=tnd,
                            lead_hours=ckpt["lead_hours"],
                            mean=np.array(ckpt["mean"], dtype=np.float32),
                            std=np.array(ckpt["std"], dtype=np.float32))
        # 표준화된 값에서 실제 기온을 복원한다(인덱스 0 = 기온).
        temps = ds.X_num[:, 0].numpy() * ckpt["std"][0] + ckpt["mean"][0]

        def probs(idx):
            hs, cs = [], []
            for b in range(0, len(idx), BATCH):
                s = idx[b:b + BATCH].tolist()
                with torch.no_grad():
                    model(num_x=ds.X_num[s].to(DEVICE),
                          img_x=ds.X_img[s].to(DEVICE).float(),
                          txt_x=ds.X_txt[s].to(DEVICE))
                    hs.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
                    cs.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            return np.concatenate(hs), np.concatenate(cs)

        print(f"\n{'='*84}")
        print(f"{path.split('/')[-1]}  (num_features={ckpt.get('num_features')})")
        print(f"{'실제 기온대':<13}{'표본':>10}{'폭염평균':>10}{'폭염>.5':>9}"
              f"{'한파평균':>10}{'한파>.5':>9}   판정")
        worst = 0.0
        for lo, hi, watch in BANDS:
            idx = np.flatnonzero((temps >= lo) & (temps < hi))
            if len(idx) == 0:
                continue
            h, c = probs(idx)
            rate = (h > 0.5).mean() if watch == "heatwave" else (
                   (c > 0.5).mean() if watch == "coldwave" else 0.0)
            if watch:
                worst = max(worst, rate)
                mark = "★오탐★" if rate > 0.10 else ("주의" if rate > 0.02 else "정상")
                mark = f"{mark}({'폭염' if watch=='heatwave' else '한파'})"
            else:
                mark = "—"
            print(f"{f'{lo}~{hi}°C':<13}{len(idx):>10,}{h.mean():>10.4f}"
                  f"{(h > 0.5).mean():>9.2%}{c.mean():>10.4f}{(c > 0.5).mean():>9.2%}   {mark}")
        verdict = "FAIL" if worst > 0.10 else ("WARN" if worst > 0.02 else "PASS")
        print(f"\n  계절 오탐 최악값: {worst:.2%}  →  "
              f"{'배포 부적합' if verdict=='FAIL' else ('검토 필요' if verdict=='WARN' else '통과')}")
        # promote_checkpoint.py 가 이 줄을 파싱한다 — 형식을 바꾸지 말 것.
        print(f"VERDICT seasonal_false_alarm {verdict} worst={worst:.6f}")


if __name__ == "__main__":
    main()
