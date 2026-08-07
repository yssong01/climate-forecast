"""
backtest_accuracy.py — 기존 데이터로 정확도 로그를 즉시 시딩.

accuracy.py 의 live 경로는 예보를 만들고 +6시간을 기다려야 1건씩 쌓인다.
그러면 "누적 적중률"이 몇 주가 지나야 의미 있는 표본을 갖는다. 이미
확보된 과거 데이터에는 (t, t+6h) 쌍이 이미 완결돼 있으므로, 학습 때와
동일한 전처리 경로(train.WeatherDataset)로 모델을 한 번 더 통과시켜
"만약 이 시점에 이 모델로 예보했다면"을 그대로 재구성한다.

새 API 호출은 하지 않는다 — train.collect_historical() 은 캐시가 이미
조건을 만족하면 그대로 반환한다.

실행:
    python backtest_accuracy.py
"""
import numpy as np
import torch

from train import collect_historical, WeatherDataset
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from predict import load_model, CHECKPOINT
import accuracy

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 512


def main():
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    print(f"체크포인트 로드 — lead={ckpt['lead_hours']}h, "
          f"검증 기온MAE {ckpt['val_temp_mae']:.3f}°C, "
          f"강수MAE {ckpt['val_precip_mae']:.3f}mm")

    records = collect_historical()
    ds = WeatherDataset(
        records,
        sat_collector=SimulatedSatelliteCollector(),
        txt_collector=SimulatedTextCollector(),
        lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    n = len(ds)
    print(f"백테스트 대상: {n}개 (t, t+{ckpt['lead_hours']}h) 쌍")

    preds = []
    with torch.no_grad():
        for b in range(0, n, BATCH):
            sl = slice(b, b + BATCH)
            pred = model(
                num_x=ds.X_num[sl].to(DEVICE),
                img_x=ds.X_img[sl].to(DEVICE) if ds.X_img is not None else None,
                txt_x=ds.X_txt[sl].to(DEVICE) if ds.X_txt is not None else None,
            )
            preds.append(pred.cpu().numpy())
    preds = np.concatenate(preds, axis=0)

    entries = []
    for i in range(n):
        entries.append({
            "station": ds.stns[i],
            "made_at": ds.src_timestamps[i],
            "target_time": ds.tgt_timestamps[i],
            "pred_temp": round(float(preds[i, 0]), 3),
            "pred_precip": round(float(max(0.0, preds[i, 1])), 3),
            "actual_temp": round(float(ds.y[i, 0].item()), 3),
            "actual_precip": round(float(ds.y[i, 1].item()), 3),
            "source": "backtest",
        })

    # accuracy.py 의 스키마·적중 판정과 일치시켜 하나로 합쳐 쓸 수 있게 한다.
    for e in entries:
        e["hit_temp"], e["hit_precip"] = accuracy._hit(
            e["pred_temp"], e["pred_precip"], e["actual_temp"], e["actual_precip"]
        )

    # live 로그와 병합 — 같은 (station, target_time, source="backtest") 는
    # 덮어써 재실행 시 중복되지 않게 한다. live 항목(source="live")은 보존.
    existing = accuracy._load()
    live_only = [e for e in existing if e["source"] != "backtest"]
    accuracy._save(live_only + entries)

    s = accuracy.stats()
    print(f"\n시딩 완료 — 누적 {s['cum_n']}건 | "
          f"기온 적중률 {s['cum_temp']:.1%} | 강수 적중률 {s['cum_precip']:.1%}")
    print(f"(기온 허용오차 ±{accuracy.HIT_TEMP_TOL}°C, "
          f"강수 판정기준 {accuracy.PRECIP_THRESH}mm 이진 일치)")


if __name__ == "__main__":
    main()
