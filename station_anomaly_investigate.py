"""
station_anomaly_investigate.py — 관측소별 캘리브레이션 플롯에서 발견된
두 이상 현상의 원인을 실측으로 파고든다(2026-08-16).

  (a) 부산 폭염: 재현율 71%·정밀도 47%로 다른 관측소(80%대)보다 확연히 낮음.
      가설 — 해풍 냉각으로 부산의 실제 온도 분포가 다른가, 아니면 표본
      편향(부산 양성비율이 낮아 학습 신호가 적음)인가.
  (b) 대구·강릉·대전 한파: 표본(216~384)이 적지 않은데 재현율이 27~39%로
      낮음. 가설 — 사건 단위로 보면 실제 독립 표본 수가 훨씬 적은가
      (연속 시간 표본이 사건 하나를 여러 번 세는 것), 아니면 진짜 어려운
      경계 사례가 많은가.

실행: python station_anomaly_investigate.py
"""
import numpy as np
import torch

from train import collect_historical, WeatherDataset, make_split, STATION_NAMES, _parse_ts
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from predict import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024


def count_events(timestamps, stations, gap_hours=24):
    """같은 관측소·gap_hours 이내로 이어지는 표본을 사건 하나로 묶어 센다.
    호출자가 이미 양성 표본만 걸러서 넘긴다고 가정한다."""
    items = sorted(zip(stations, (_parse_ts(t) for t in timestamps)))
    if not items:
        return 0
    n = 1
    prev_s, prev_t = items[0]
    for s, t in items[1:]:
        if s != prev_s or (t - prev_t).total_seconds() > gap_hours * 3600:
            n += 1
        prev_s, prev_t = s, t
    return n


def main():
    model, ckpt = load_model()
    records = collect_historical()
    txt = TendencyCollector(records)
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False)
    idx = np.array(val_ds.indices)
    stns = np.array([ds.stns[i] for i in idx])
    tgt_ts = np.array([ds.tgt_timestamps[i] for i in idx])
    tgt_temp = ds.y[idx, 0].numpy()

    heat_p, cold_p = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            sl = idx[s:s + BATCH]
            model(num_x=ds.X_num[sl].to(DEVICE), img_x=ds.X_img[sl].to(DEVICE).float(),
                  txt_x=ds.X_txt[sl].to(DEVICE))
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).cpu().squeeze(-1))
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).cpu().squeeze(-1))
    heat_p = torch.cat(heat_p).numpy()
    cold_p = torch.cat(cold_p).numpy()
    heat_y = ds.y_heatwave[idx].numpy().astype(bool)
    cold_y = ds.y_coldwave[idx].numpy().astype(bool)
    heat_m = ds.heat_mask[idx].numpy().astype(bool)
    cold_m = ds.cold_mask[idx].numpy().astype(bool)

    # ── (a) 부산 폭염 vs 대구(대조군) ────────────────────────────
    print("=" * 90)
    print(" (a) 부산 폭염 vs 대구(대조군) — 실제 기온 분포·오탐/놓침 비교")
    print("=" * 90)
    for stn_name, stn_code in (("부산", "159"), ("대구", "143")):
        sel = heat_m & (stns == stn_code)
        pos = sel & heat_y
        neg = sel & ~heat_y
        fp = neg & (heat_p >= 0.5)          # 실제 무사건인데 폭염으로 예측
        fn = pos & (heat_p < 0.5)           # 실제 폭염인데 못 잡음

        print(f"\n[{stn_name}] 양성(공식 폭염) {int(pos.sum())}건 / 음성 {int(neg.sum())}건")
        print(f"  양성일 기온 — 평균 {tgt_temp[pos].mean():.2f}°C, "
              f"5%ile {np.percentile(tgt_temp[pos], 5):.2f}°C, "
              f"중앙 {np.median(tgt_temp[pos]):.2f}°C")
        print(f"  음성일 기온 — 평균 {tgt_temp[neg].mean():.2f}°C, "
              f"95%ile {np.percentile(tgt_temp[neg], 95):.2f}°C")
        if fp.sum():
            print(f"  오탐(무사건인데 폭염 예측) {int(fp.sum())}건 — "
                  f"그 날 실제 기온 평균 {tgt_temp[fp].mean():.2f}°C "
                  f"(음성일 평균보다 {tgt_temp[fp].mean() - tgt_temp[neg].mean():+.2f}°C)")
        else:
            print("  오탐 0건")
        if fn.sum():
            print(f"  놓침(폭염인데 미탐) {int(fn.sum())}건 — "
                  f"그 날 실제 기온 평균 {tgt_temp[fn].mean():.2f}°C "
                  f"(양성일 평균보다 {tgt_temp[fn].mean() - tgt_temp[pos].mean():+.2f}°C)")
        else:
            print("  놓침 0건")

    # ── (b) 대구·강릉·대전 한파 vs 춘천(대조군) — 사건 단위 개수 ──
    print("\n" + "=" * 90)
    print(" (b) 한파 — 시간 표본 수 vs 실제 사건(연속 발효 기간) 수")
    print("=" * 90)
    for stn_name, stn_code in (("춘천", "101"), ("서울", "108"), ("대전", "133"),
                               ("대구", "143"), ("강릉", "105")):
        sel = cold_m & (stns == stn_code)
        pos = sel & cold_y
        n_hours = int(pos.sum())
        n_events = count_events(tgt_ts[pos], stns[pos])
        avg_len = n_hours / n_events if n_events else 0
        pos_temp = tgt_temp[pos]
        recall_p = cold_p[pos]
        print(f"\n[{stn_name}] 시간표본 {n_hours}건 → 사건(24h 이내 연속) {n_events}개 "
              f"(평균 길이 {avg_len:.1f}시간)")
        print(f"  양성일 기온 — 평균 {pos_temp.mean():.2f}°C, "
              f"최고 {pos_temp.max():.2f}°C, 최저 {pos_temp.min():.2f}°C")
        print(f"  양성일 예측확률 — 평균 {recall_p.mean():.3f}, "
              f"중앙 {np.median(recall_p):.3f}, "
              f"0.5 미만 비율(놓침) {(recall_p < 0.5).mean():.1%}")


if __name__ == "__main__":
    main()
