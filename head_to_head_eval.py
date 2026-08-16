"""
head_to_head_eval.py — 배포 모델(v8)과 13년 재학습 모델을 같은 잣대로 맞대결시킨다.

배경(2026-08-16): 13년치 재학습 후 극한기상 F1 이 배포 모델보다 낮게 나왔다
(황사 0.588 → 0.150). 그런데 두 값은 서로 다른 검증 방식에서 나왔다.
  · v8(배포 모델) 0.588 — 시각 단위 무작위 분할. 판정 가능한 검증 표본의
    100.00% 가 학습셋과 같은 날짜를 공유한다(split_leakage_check.py 실측).
    극한기상 라벨은 (관측소, 날짜) 단위라, 학습 중에 이미 그 날의 정답을 봤다.
  · 신규 모델 0.150 — 날짜 단위 그룹 분할. 공유 날짜 0개.
자를 바꿔놓고 길이를 비교한 셈이라 "나빠졌다"는 판정이 성립하지 않는다.

공정한 비교가 되려면 두 모델 모두 학습에 쓰지 않은 동일한 표본이 필요하다.
  · v8 은 2023-01-01 이후만 학습했다 → 2013~2022 전체가 미사용
  · 그룹 분할 모델은 검증셋 날짜가 미사용
따라서 "그룹 분할 검증셋 ∩ 2013~2022" 는 두 모델 모두에게 처음 보는 표본이다.

남는 비대칭(정직하게 명시): 그룹 분할 모델은 2013~2022 의 *다른 날짜* 로
학습했으므로 그 시기의 기후를 안다. v8 은 그 시기를 전혀 모른다. 이건 측정
오류가 아니라 데이터를 더 쓴 실제 이점이므로 그대로 둔다. 다만 v8 에 불리한
조건이라는 점은 결과 해석에 반영해야 한다.

실행: python head_to_head_eval.py [--n 60000]
"""
import argparse
import json
from datetime import timedelta

import numpy as np
import torch

from train import (
    collect_historical, record_to_vec, _parse_ts, make_split,
    LEAD_HOURS, DEVICE, WEATHER_ISSUE_LABELS, WeatherDataset,
)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from predict import load_model

CANDIDATES = [
    ("v8 (배포 중, 3.6년 학습)", "./checkpoints/numerical_trichef_v8_before_10y_data.pt"),
    ("신규 (13년, 그룹 분할)",   "./checkpoints/numerical_trichef_split_group.pt"),
]


def prf(probs, labels, t=0.5):
    pred = probs >= t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=60000)
    args = ap.parse_args()

    records = collect_historical()
    by_key = {(r.get("stn"), str(r["timestamp"])[:12]): i
              for i, r in enumerate(records)}
    src_idx, tgt_idx = [], []
    for i, r in enumerate(records):
        tgt_ts = (_parse_ts(r["timestamp"]) + timedelta(hours=LEAD_HOURS)) \
            .strftime("%Y%m%d%H%M")
        j = by_key.get((r.get("stn"), tgt_ts))
        if j is not None:
            src_idx.append(i); tgt_idx.append(j)

    # 그룹 분할의 검증 인덱스를 얻기 위해 최소 구성으로 데이터셋을 만든다
    # (Re축 이미지는 여기서 만들지 않는다 — 필요한 표본만 나중에 만든다).
    txt = TendencyCollector(records)
    ds_meta = WeatherDataset(records, sat_collector=None, txt_collector=txt,
                             lead_hours=LEAD_HOURS)
    _, val_ds = make_split(ds_meta, "group", verbose=False)
    val_pos = list(val_ds.indices)

    # v8 이 학습에 쓰지 않은 구간(2013~2022)으로 좁힌다
    years = np.array([int(str(records[tgt_idx[k]]["timestamp"])[:4]) for k in val_pos])
    keep = [k for k, y in zip(val_pos, years) if y <= 2022]
    print(f"그룹 분할 검증셋 {len(val_pos):,}개 중 2013~2022 구간 {len(keep):,}개")
    print("  → 이 표본은 v8 과 신규 모델 모두 학습에 쓰지 않았다\n")

    rng = np.random.default_rng(0)
    take = min(args.n, len(keep))
    sample = [keep[k] for k in rng.choice(len(keep), size=take, replace=False)]
    src_records = [records[src_idx[k]] for k in sample]
    tgt_records = [records[tgt_idx[k]] for k in sample]
    print(f"평가 표본 {take:,}개\n")

    with open(WEATHER_ISSUE_LABELS, encoding="utf-8") as f:
        issue_labels = json.load(f)
    lab = {k: ([], []) for k in ("heatwave_advisory", "coldwave_advisory", "dust_observed")}
    for r in tgt_records:
        ts = str(r["timestamp"])
        day = issue_labels.get(str(r.get("stn")), {}).get(
            f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}", {})
        for key in lab:
            y, m = lab[key]
            if key in day:
                y.append(day[key]); m.append(1)
            else:
                y.append(0); m.append(0)
    lab = {k: (np.array(v[0]), np.array(v[1]).astype(bool)) for k, v in lab.items()}

    temp_tgt = np.array([r["temperature"] for r in tgt_records], dtype=np.float32)
    temp_now = np.array([r["temperature"] for r in src_records], dtype=np.float32)
    precip_tgt = np.array([r["precipitation"] for r in tgt_records], dtype=np.float32)
    naive_temp = float(np.abs(temp_tgt - temp_now).mean())
    naive_precip = float(np.abs(precip_tgt).mean())

    sat = InterpolatedFieldCollector(records, STATION_COORDS)
    raw_num = np.array([record_to_vec(r) for r in src_records], dtype=np.float32)
    img = np.stack([sat.get_image(r) for r in src_records])
    txt_vec = np.stack([txt.encode_single(r) for r in src_records])

    print("=" * 92)
    print(f" 맞대결 — 두 모델 모두 학습에 쓰지 않은 동일 표본 {take:,}개 (2013~2022)")
    print(f" 기준선: 기온 퍼시스턴스 {naive_temp:.4f}°C · 강수 상시0 {naive_precip:.4f}mm")
    print("=" * 92)

    for name, path in CANDIDATES:
        model, ckpt = load_model(path)
        mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])
        xn_all = (raw_num - mean) / std

        tp_, pp_, hp_, cp_, dp_ = [], [], [], [], []
        B = 1024
        with torch.no_grad():
            for s in range(0, take, B):
                xn = torch.tensor(xn_all[s:s+B], dtype=torch.float32).to(DEVICE)
                xi = torch.tensor(img[s:s+B], dtype=torch.float32).to(DEVICE)
                xt = torch.tensor(txt_vec[s:s+B], dtype=torch.float32).to(DEVICE)
                pred = model(num_x=xn, img_x=xi, txt_x=xt)
                tp_.append(pred[:, 0].cpu().numpy())
                pp_.append(pred[:, 1].cpu().numpy())
                hp_.append(torch.sigmoid(model._last_heatwave_logit).cpu().numpy().ravel())
                cp_.append(torch.sigmoid(model._last_coldwave_logit).cpu().numpy().ravel())
                dp_.append(torch.sigmoid(model._last_dust_logit).cpu().numpy().ravel())

        t_mae = float(np.abs(np.concatenate(tp_) - temp_tgt).mean())
        p_mae = float(np.abs(np.concatenate(pp_) - precip_tgt).mean())
        print(f"\n[{name}]  split_mode={ckpt.get('split_mode', 'random(구버전)')}")
        print(f"  기온 MAE {t_mae:.4f}°C  (기준선 대비 {100*(naive_temp-t_mae)/naive_temp:+.1f}%)")
        print(f"  강수 MAE {p_mae:.4f}mm  (기준선 대비 {100*(naive_precip-p_mae)/naive_precip:+.1f}%)")
        for label, key, probs in (("폭염", "heatwave_advisory", hp_),
                                  ("한파", "coldwave_advisory", cp_),
                                  ("황사", "dust_observed", dp_)):
            y, m = lab[key]
            if m.sum() == 0:
                print(f"  {label}: 판정 가능 표본 없음"); continue
            pr = np.concatenate(probs)[m]
            p, r, f = prf(pr, y[m])
            print(f"  {label}: P={p:.3f} R={r:.3f} F1={f:.3f}  "
                  f"(표본 {int(m.sum()):,} · 실측양성 {int(y[m].sum()):,})")

    print(f"\n{'='*92}")


if __name__ == "__main__":
    main()
