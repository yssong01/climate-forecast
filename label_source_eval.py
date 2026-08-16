"""
label_source_eval.py — 13년치 재학습 모델을 "라벨 출처별"로 나눠 채점한다.

목적(2026-08-15): label_composition_diagnose.py 로 폭염·한파 라벨이 두 가지
정의로 섞여 있음을 확인했다(공식 특보 = 그 날 하루 전체 양성, 양성률 15~36%
/ 임계값 근사 = 그 시각 기온만, 양성률 0.25~2.02%). 이것이 정밀도 붕괴의
원인이라면, 같은 모델을 공식 라벨 표본에서만 채점했을 때는 이전 수준(폭염
F1 0.842)에 가까워야 하고 근사 라벨 표본에서만 채점하면 무너져야 한다.
확인되면 해법은 명확하다 — 황사가 이미 쓰는 마스킹(공식 라벨 없는 표본을
손실에서 제외)을 폭염·한파에도 적용한다.

재학습 없이 확인하려고 train.py 의 분할을 그대로 재현한다:
  · collect_historical() 은 캐시 JSON 을 순서 그대로 돌려준다(정렬·필터 없음)
  · WeatherDataset 의 (t, t+6h) 쌍 선별은 records 순회 순서에 종속적
  · random_split 의 순열은 길이와 SEED 에만 의존하므로, 이미지를 만들지
    않고도 torch.randperm 으로 검증 인덱스를 그대로 얻을 수 있다
쌍 개수가 학습 로그(1,380,966개)와 일치하는지 먼저 검증하고, 다르면
중단한다 — 분할이 어긋난 채 낸 수치는 비교 근거가 못 된다.

전체 검증셋(276,193개)에 Re축 이미지를 다 만들면 메모리가 크므로 무작위
부분표본만 평가한다(기본 40,000개, --n 으로 조절).

실행: python label_source_eval.py [--n 40000] [--ckpt ./checkpoints/...pt]
"""
import argparse
import json
import os
from datetime import datetime, timedelta

import numpy as np
import torch

from train import (
    collect_historical, record_to_vec, _parse_ts,
    SEED, VAL_RATIO, LEAD_HOURS, DEVICE,
    HEATWAVE_THRESH, COLDWAVE_THRESH, WEATHER_ISSUE_LABELS,
)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from predict import load_model

EXPECTED_PAIRS = 1_380_966   # 2026-08-15 학습 로그 기준 — 불일치 시 중단


def prf(pred_pos, true_pos, tp):
    p = tp / pred_pos if pred_pos else 0.0
    r = tp / true_pos if true_pos else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n", type=int, default=40000, help="평가할 검증 부분표본 수")
    ap.add_argument("--ckpt", default="./checkpoints/numerical_trichef.pt")
    args = ap.parse_args()

    records = collect_historical()

    # ── train.py WeatherDataset 과 동일한 쌍 선별 ────────────────────
    by_key = {(r.get("stn"), str(r["timestamp"])[:12]): i
              for i, r in enumerate(records)}
    src_idx, tgt_idx = [], []
    for i, r in enumerate(records):
        tgt_ts = (_parse_ts(r["timestamp"]) + timedelta(hours=LEAD_HOURS)) \
            .strftime("%Y%m%d%H%M")
        j = by_key.get((r.get("stn"), tgt_ts))
        if j is not None:
            src_idx.append(i)
            tgt_idx.append(j)

    n_total = len(src_idx)
    print(f"유효 쌍 {n_total:,}개 (학습 로그 기준 {EXPECTED_PAIRS:,}개)")
    if n_total != EXPECTED_PAIRS:
        raise SystemExit(
            f"[중단] 쌍 개수가 학습 시점과 다릅니다 — 분할을 재현할 수 없습니다."
        )

    # ── random_split 과 동일한 순열로 검증 인덱스 복원 ────────────────
    # torch.utils.data.random_split 은 randperm(sum(lengths), generator) 를
    # 만들어 앞에서부터 순서대로 잘라 준다. train.py 는 [n_train, n_val]
    # 순서로 넘기므로 뒤쪽 n_val 개가 검증셋이다.
    n_val = max(2, int(n_total * VAL_RATIO))
    n_train = n_total - n_val
    perm = torch.randperm(n_total,
                          generator=torch.Generator().manual_seed(SEED)).tolist()
    val_pos = perm[n_train:]
    print(f"학습 {n_train:,} / 검증 {n_val:,}  (검증 인덱스 복원 완료)")

    # 부분표본 — 평가 자체의 난수는 분할과 무관하므로 별도 시드를 쓴다.
    rng = np.random.default_rng(0)
    take = min(args.n, len(val_pos))
    sample = [val_pos[k] for k in rng.choice(len(val_pos), size=take, replace=False)]
    print(f"검증 부분표본 {take:,}개로 평가\n")

    src_records = [records[src_idx[k]] for k in sample]
    tgt_records = [records[tgt_idx[k]] for k in sample]

    # ── 라벨 + 출처(공식/근사) ──────────────────────────────────────
    with open(WEATHER_ISSUE_LABELS, encoding="utf-8") as f:
        issue_labels = json.load(f)

    heat_y, heat_official = [], []
    cold_y, cold_official = [], []
    dust_y, dust_has = [], []
    for r in tgt_records:
        ts = str(r["timestamp"])
        date_str = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
        day = issue_labels.get(str(r.get("stn")), {}).get(date_str, {})
        t = r["temperature"]

        if "heatwave_advisory" in day:
            heat_y.append(day["heatwave_advisory"]); heat_official.append(1)
        else:
            heat_y.append(int(t >= HEATWAVE_THRESH)); heat_official.append(0)

        if "coldwave_advisory" in day:
            cold_y.append(day["coldwave_advisory"]); cold_official.append(1)
        else:
            cold_y.append(int(t <= COLDWAVE_THRESH)); cold_official.append(0)

        if "dust_observed" in day:
            dust_y.append(day["dust_observed"]); dust_has.append(1)
        else:
            dust_y.append(0); dust_has.append(0)

    heat_y = np.array(heat_y); heat_official = np.array(heat_official)
    cold_y = np.array(cold_y); cold_official = np.array(cold_official)
    dust_y = np.array(dust_y); dust_has = np.array(dust_has)

    # ── 입력 구성 + 추론 ────────────────────────────────────────────
    print("Re축 공간보간장 / Im축 경향벡터 인덱스 구성 중...")
    sat = InterpolatedFieldCollector(records, STATION_COORDS)
    txt = TendencyCollector(records)

    model, ckpt = load_model(args.ckpt)
    mean, std = np.array(ckpt["mean"]), np.array(ckpt["std"])

    heat_p, cold_p, dust_p = [], [], []
    B = 512
    print("추론 중...")
    with torch.no_grad():
        for s in range(0, len(src_records), B):
            chunk = src_records[s:s + B]
            x_num = np.array([record_to_vec(r) for r in chunk], dtype=np.float32)
            x_num = torch.tensor((x_num - mean) / std, dtype=torch.float32).to(DEVICE)
            x_img = torch.tensor(np.stack([sat.get_image(r) for r in chunk]),
                                 dtype=torch.float32).to(DEVICE)
            x_txt = torch.tensor(np.stack([txt.encode_single(r) for r in chunk]),
                                 dtype=torch.float32).to(DEVICE)
            model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).cpu().numpy().ravel())
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).cpu().numpy().ravel())
            dust_p.append(torch.sigmoid(model._last_dust_logit).cpu().numpy().ravel())

    heat_p = np.concatenate(heat_p)
    cold_p = np.concatenate(cold_p)
    dust_p = np.concatenate(dust_p)

    # ── 라벨 출처별 채점 ────────────────────────────────────────────
    def report(name, probs, y, subset_mask, subset_name):
        m = subset_mask.astype(bool)
        if m.sum() == 0:
            print(f"  {subset_name:<22} 표본 없음")
            return
        pr, yy = probs[m], y[m]
        pred = (pr >= 0.5).astype(int)
        tp = int(((pred == 1) & (yy == 1)).sum())
        p, r, f1 = prf(int(pred.sum()), int(yy.sum()), tp)
        print(f"  {subset_name:<22} 표본 {m.sum():>7,} · 실측양성 {int(yy.sum()):>6,} "
              f"({100*yy.mean():5.2f}%) · P={p:.3f} R={r:.3f} F1={f1:.3f}")

    print(f"\n{'='*96}")
    print(f" 라벨 출처별 채점 — 체크포인트: {args.ckpt} (판정 임계 0.5)")
    print(f"{'='*96}")

    print("\n[폭염]")
    report("폭염", heat_p, heat_y, np.ones_like(heat_official), "전체(현행 학습 방식)")
    report("폭염", heat_p, heat_y, heat_official, "공식 특보 라벨만")
    report("폭염", heat_p, heat_y, 1 - heat_official, "임계값 근사 라벨만")

    print("\n[한파]")
    report("한파", cold_p, cold_y, np.ones_like(cold_official), "전체(현행 학습 방식)")
    report("한파", cold_p, cold_y, cold_official, "공식 특보 라벨만")
    report("한파", cold_p, cold_y, 1 - cold_official, "임계값 근사 라벨만")

    print("\n[황사] — 원래부터 공식 라벨만 쓰므로(마스킹) 대조군이 없다")
    report("황사", dust_p, dust_y, dust_has, "공식 라벨(PM10)만")

    # ── 시기별 채점 ────────────────────────────────────────────────
    # 라벨 정의가 동일한 표본만 놓고 시기를 갈라 본다. 확장분(2013~2022)에서만
    # 나쁘면 비정상성(같은 기상조건이 연도에 따라 다른 결과) 쪽이고, 시기와
    # 무관하게 고르게 나쁘면 모델 용량·공유표현 오염 쪽이다. 모델 입력에는
    # 연도가 없으므로(record_to_vec) 두 시기를 구분할 방법이 원리적으로 없다.
    years = np.array([int(str(r["timestamp"])[:4]) for r in tgt_records])
    periods = (("2013~2018", (years >= 2013) & (years <= 2018)),
               ("2019~2022", (years >= 2019) & (years <= 2022)),
               ("2023~2026", years >= 2023))

    print(f"\n{'='*96}")
    print(" 시기별 채점 — 비정상성 점검 (라벨 정의가 같은 표본끼리만 비교)")
    print(f"{'='*96}")

    print("\n[폭염 — 공식 특보 라벨 표본만]")
    for name, m in periods:
        report("폭염", heat_p, heat_y, heat_official.astype(bool) & m, name)

    print("\n[한파 — 공식 특보 라벨 표본만]")
    for name, m in periods:
        report("한파", cold_p, cold_y, cold_official.astype(bool) & m, name)

    print("\n[황사 — 전 구간 공식 라벨(PM10). 라벨 정의는 13년 내내 동일]")
    for name, m in periods:
        report("황사", dust_p, dust_y, dust_has.astype(bool) & m, name)

    print(f"\n{'='*96}")


if __name__ == "__main__":
    main()
