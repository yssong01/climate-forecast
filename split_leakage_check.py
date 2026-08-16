"""
split_leakage_check.py — 무작위 분할이 극한기상 라벨을 통해 누수시키는 양을 잰다.

배경(2026-08-15): 13년치로 늘린 뒤 황사 F1 이 0.588→0.316→0.190 으로 계속
떨어졌다. 라벨 마스킹으로도 회복되지 않았고(0.190→0.198), 시기별로도 고르게
나빴다. "데이터를 늘렸는데 성능이 떨어진다"는 것 자체가 이상해서 검증 설계를
의심하게 됐다.

의심 지점 — 폭염·한파·황사 라벨은 (관측소, 날짜) 단위다. 폭염특보가 발효된
날은 그날 24개 시각 표본이 전부 양성이 된다. 그런데 train.py 는 시각 단위
표본을 무작위로 80/20 분할한다. 그러면 같은 (관측소, 날짜)의 표본들이
학습셋과 검증셋에 쪼개져 들어간다 — 모델은 학습 중에 이미 그 관측소·그
날짜의 정답을 봤고, 검증에서 같은 날의 다른 시각을 묻는 셈이 된다.
하루 안에서는 기상 상태가 서로 비슷하므로(Z·Re·Im 입력이 강하게 상관),
"일반화"가 아니라 "같은 날 다른 시각 맞히기"를 채점하고 있을 수 있다.

이 스크립트는 학습을 돌리지 않고 그 겹침 정도를 센다. 회귀(기온·강수)는
시각마다 정답이 달라 이 문제에서 자유롭지만, 극한기상 3종은 전부 일 단위다.

실행: python split_leakage_check.py
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta

import torch

from train import (
    collect_historical, _parse_ts, SEED, VAL_RATIO, LEAD_HOURS,
    WEATHER_ISSUE_LABELS,
)

EXPECTED_PAIRS = 1_380_966


def main():
    records = collect_historical()
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
    n = len(src_idx)
    if n != EXPECTED_PAIRS:
        raise SystemExit(f"[중단] 쌍 개수 불일치: {n:,} vs {EXPECTED_PAIRS:,}")

    n_val = max(2, int(n * VAL_RATIO))
    n_train = n - n_val
    perm = torch.randperm(n, generator=torch.Generator().manual_seed(SEED)).tolist()
    train_pos = set(perm[:n_train])
    val_pos = perm[n_train:]
    print(f"학습 {n_train:,} / 검증 {n_val:,}\n")

    # 라벨 단위 키 = (관측소, 타깃 날짜)
    key_of = []
    for k in range(n):
        r = records[tgt_idx[k]]
        ts = str(r["timestamp"])
        key_of.append((str(r.get("stn")), f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"))

    train_keys = set()
    for k in train_pos:
        train_keys.add(key_of[k])

    with open(WEATHER_ISSUE_LABELS, encoding="utf-8") as f:
        issue_labels = json.load(f)

    print("=" * 88)
    print("검증 표본 중 '같은 (관측소, 날짜)'가 학습셋에도 있는 비율")
    print("=" * 88)

    for name, field in (("폭염", "heatwave_advisory"),
                        ("한파", "coldwave_advisory"),
                        ("황사", "dust_observed")):
        judgeable = shared = pos = pos_shared = 0
        for k in val_pos:
            stn, date = key_of[k]
            day = issue_labels.get(stn, {}).get(date, {})
            if field not in day:
                continue
            judgeable += 1
            in_train = (stn, date) in train_keys
            shared += in_train
            if day[field] == 1:
                pos += 1
                pos_shared += in_train
        print(f"  {name} — 판정가능 검증표본 {judgeable:>8,}개 중 "
              f"학습셋과 같은 날 공유 {shared:>8,}개 ({100*shared/max(judgeable,1):6.2f}%)")
        print(f"       그중 양성만  {pos:>8,}개 중 "
              f"                    {pos_shared:>8,}개 ({100*pos_shared/max(pos,1):6.2f}%)")

    # 하루당 표본 수 분포 — 왜 이렇게 겹치는지 보여준다
    per_day = defaultdict(int)
    for k in range(n):
        per_day[key_of[k]] += 1
    counts = list(per_day.values())
    print()
    print(f"  (관측소, 날짜) 조합 {len(per_day):,}개 · "
          f"조합당 평균 표본 {sum(counts)/len(counts):.1f}개")
    print(f"  → 하루 표본이 여러 개면 80/20 무작위 분할에서 양쪽에 쪼개질 확률이")
    print(f"     거의 1 이다(24개면 전부 한쪽에 몰릴 확률 0.8^24 ≈ 0.5%).")

    # 회귀 타깃은 시각마다 값이 달라 같은 문제가 없다는 점을 대조로 보인다
    print()
    print("=" * 88)
    print("대조 — 회귀 타깃(기온)은 시각마다 값이 달라 이 누수에서 자유롭다")
    print("=" * 88)
    same_day_temp_spread = []
    for key, cnt in list(per_day.items())[:5000]:
        pass
    temps_by_key = defaultdict(list)
    for k in range(n):
        temps_by_key[key_of[k]].append(records[tgt_idx[k]]["temperature"])
    spreads = [max(v) - min(v) for v in temps_by_key.values() if len(v) > 1]
    print(f"  같은 (관측소, 날짜) 안에서 기온이 벌어지는 폭: "
          f"평균 {sum(spreads)/len(spreads):.2f}°C")
    print(f"  → 극한기상 라벨은 하루 내내 같은 값이지만 기온 타깃은 그렇지 않다.")


if __name__ == "__main__":
    main()
