"""
station_coverage_check.py — 관측소 축에서 라벨 사각지대를 점검한다.

**왜 필요한가.** 극한기상 라벨은 공식 기록이 있는 표본에서만 채점한다. 기록이
없는 표본은 마스크로 손실·평가에서 제외되는데, 이 제외가 특정 구간에 몰리면
그 구간은 **지표가 영원히 보지 못한다.** 2026-08-17 겨울 폭염 오탐이 정확히
그런 사고였다 — 폭염 라벨이 여름에만 있어 겨울 오탐이 F1에 반영되지 않았고,
F1 0.813인 모델이 한겨울에 폭염을 외쳤다.

그 사고는 계절 축이었다. 이 스크립트는 **같은 구조가 관측소 축에도 있는지**
확인한다. 라벨이 아예 없는 관측소가 있다면 그 관측소의 출력은 한 번도 채점된
적이 없다는 뜻이므로, 실제로 경보를 얼마나 내는지 함께 본다.

실측 결과(2026-08-17, 현행 배포본)
  폭염 — 12개 관측소 전부 라벨 보유(각 24%), 사각지대 없음
  한파 — 12개 전부 보유. 제주만 양성 24건으로 희박하나 채점은 된다
  황사 — **6개 관측소(강릉·인천·청주·대전·부산·제주)가 라벨 0건.** PM10
         관측망이 절반에만 있기 때문이다. 그런데 이 관측소들이 임계값을
         넘는 비율은 2~23%로 낮지 않다(인천 23.35%) — 검증된 적 없는
         경보다. 황사는 이미 화면에서 제외했으므로 현재 실무 위험은 없으나,
         되살릴 경우 이 사실을 먼저 해결해야 한다.

판정
  ★채점 불가★ — 라벨 0건. 임계 초과율이 유의하면 위험하다.
  ★양성 0★    — 라벨은 있으나 양성이 없어 재현율을 잴 수 없다.
  양성 희박    — 양성 30건 미만. 지표를 신뢰하기 어렵다.

실행: python station_coverage_check.py [체크포인트]
"""
import sys
from collections import defaultdict

import numpy as np
import torch

from predict import CHECKPOINT, load_model, raw_event_threshold
from train import WeatherDataset, collect_historical, make_split
from weather_collector import STATION_COORDS, STATIONS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096
# 라벨이 없는 관측소가 이 비율 넘게 경보를 내면 경고한다.
UNSCORED_ALERT_WARN = 0.02


def main():
    path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    model, ckpt = load_model(path, DEVICE)
    model.eval()

    records = collect_historical()
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=TendencyCollector(records), lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32))
    _, val = make_split(ds, ckpt.get("split_mode", "group"), verbose=False, ckpt=ckpt)
    vi = np.array(val.indices)
    stns = np.array(ds.stns)[vi]
    code2name = {v: k for k, v in STATIONS.items()}

    acc = defaultdict(list)
    for b in range(0, len(vi), BATCH):
        s = vi[b:b + BATCH].tolist()
        with torch.no_grad():
            model(num_x=ds.X_num[s].to(DEVICE),
                  img_x=ds.X_img[s].to(DEVICE).float(),
                  txt_x=ds.X_txt[s].to(DEVICE))
            acc["heatwave"].append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
            acc["coldwave"].append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            acc["dust"].append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu().numpy())
    P = {k: np.concatenate(v) for k, v in acc.items()}
    M = {"heatwave": ds.heat_mask[vi].numpy().astype(bool),
         "coldwave": ds.cold_mask[vi].numpy().astype(bool),
         "dust":     ds.dust_mask[vi].numpy().astype(bool)}
    Y = {"heatwave": ds.y_heatwave[vi].numpy(),
         "coldwave": ds.y_coldwave[vi].numpy(),
         "dust":     ds.y_dust[vi].numpy()}

    worst = {}
    for ev in ("heatwave", "coldwave", "dust"):
        t = raw_event_threshold(ev, "108")
        print(f"\n{'='*90}\n[{ev}] 판정 임계값(보정 전) {t}")
        print(f"{'관측소':<8}{'검증표본':>9}{'라벨보유':>9}{'라벨비율':>9}{'양성':>7}"
              f"{'임계초과율':>11}   상태")
        unscored_alert = 0.0
        for code in sorted(set(stns.tolist())):
            sel = stns == code
            n, nl = int(sel.sum()), int((sel & M[ev]).sum())
            pos = int(Y[ev][sel & M[ev]].sum()) if nl else 0
            over = float((P[ev][sel] > t).mean())
            if nl == 0:
                state = "★채점 불가★"
                unscored_alert = max(unscored_alert, over)
            elif pos == 0:
                state = "★양성 0★"
            elif pos < 30:
                state = "양성 희박"
            else:
                state = "정상"
            print(f"{code2name.get(code, code):<8}{n:>9,}{nl:>9,}{nl/n:>8.0%}"
                  f"{pos:>7}{over:>11.2%}   {state}")
        worst[ev] = unscored_alert
        if unscored_alert > 0:
            print(f"  → 채점되지 않는 관측소가 최대 {unscored_alert:.2%} 비율로 "
                  f"임계값을 넘는다. 이 출력은 검증된 적이 없다.")

    # promote_checkpoint.py 가 이 줄을 파싱한다 — 형식을 바꾸지 말 것.
    for ev, w in worst.items():
        code = "WARN" if w > UNSCORED_ALERT_WARN else "PASS"
        print(f"VERDICT station_coverage_{ev} {code} unscored_alert={w:.6f}")


if __name__ == "__main__":
    main()
