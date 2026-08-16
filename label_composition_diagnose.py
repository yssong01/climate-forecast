"""
label_composition_diagnose.py — 13년치 확장 후 극한기상 F1이 무너진 원인 추적.

배경(2026-08-15): 3.6년(2023~2026) → 13년(2013~2026)으로 데이터를 4배 늘려
재학습했더니 기온 MAE는 1.2554→1.2033으로 개선됐는데 극한기상 3종의 F1이
전부 무너졌다(폭염 0.842→0.670, 한파 0.667→0.442, 황사 0.588→0.316).
세 헤드 모두 재현율은 유지되고 정밀도만 붕괴한 것이 공통 패턴이다.

이 스크립트는 학습을 돌리지 않고 라벨만 집계해 두 가지를 확인한다.

1) 라벨 정의 불일치 — 폭염·한파는 기상청 공식 특보 기록이 있으면 그것을,
   없으면 임계값 근사(그 시각 기온 ≥33°C / ≤−12°C)를 쓴다. 공식 특보는
   "그 날 하루" 단위라 하루 24개 표본이 전부 양성이 되지만, 임계값 근사는
   오후 몇 시간만 양성이 된다. 즉 같은 날씨가 연도에 따라 다른 라벨을
   받는다. 공식 기록은 폭염 2019-05~, 한파 2020-11~ 뿐이라 확장분
   (2013~2018)은 전부 근사 쪽으로 떨어진다.

2) 비정상성(non-stationarity) — 황사는 라벨이 전부 공식(PM10 실측 임계)이라
   1)의 영향을 받지 않는데도 같이 무너졌다. 한국 PM10 농도는 2010년대에
   크게 낮아졌으므로, 같은 기상조건이 연도에 따라 다른 황사 결과로 이어졌을
   수 있다. 모델 입력에는 연도 정보가 없어(record_to_vec 참조) 두 시기를
   구분할 방법이 원리적으로 없다.

실행: python label_composition_diagnose.py
"""
import json
import os
from collections import defaultdict
from datetime import datetime, timedelta

DATA_CACHE = "./cache/historical_data_1y.json"
WEATHER_ISSUE_LABELS = "./cache/weather_issue_labels.json"
HEATWAVE_THRESH = 33.0
COLDWAVE_THRESH = -12.0
LEAD_HOURS = 6


def _parse_ts(ts) -> datetime:
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


def main():
    with open(DATA_CACHE, encoding="utf-8") as f:
        records = json.load(f)
    with open(WEATHER_ISSUE_LABELS, encoding="utf-8") as f:
        issue_labels = json.load(f)

    print(f"레코드 {len(records):,}개 로드")

    # train.py WeatherDataset 과 동일하게 (t, t+6h) 유효 쌍만 추린다.
    by_key = {(r.get("stn"), str(r["timestamp"])[:12]): i
              for i, r in enumerate(records)}
    pairs = []
    for i, r in enumerate(records):
        tgt_ts = (_parse_ts(r["timestamp"]) + timedelta(hours=LEAD_HOURS)) \
            .strftime("%Y%m%d%H%M")
        j = by_key.get((r.get("stn"), tgt_ts))
        if j is not None:
            pairs.append(j)          # 라벨은 타깃 시각 기준이므로 tgt 만 필요
    print(f"유효 (t, t+{LEAD_HOURS}h) 쌍 {len(pairs):,}개\n")

    # 연도별 집계 —
    #   heat_off_n  : 공식 라벨이 있는 표본 수 / heat_off_pos : 그중 양성
    #   heat_fb_n   : 근사 폴백으로 떨어진 표본 수 / heat_fb_pos : 그중 양성
    agg = defaultdict(lambda: defaultdict(int))
    for j in pairs:
        r = records[j]
        ts = str(r["timestamp"])
        year = ts[:4]
        date_str = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
        day = issue_labels.get(str(r.get("stn")), {}).get(date_str, {})
        temp = r["temperature"]

        a = agg[year]
        a["n"] += 1

        if "heatwave_advisory" in day:
            a["heat_off_n"] += 1
            a["heat_off_pos"] += int(day["heatwave_advisory"])
        else:
            a["heat_fb_n"] += 1
            a["heat_fb_pos"] += int(temp >= HEATWAVE_THRESH)

        if "coldwave_advisory" in day:
            a["cold_off_n"] += 1
            a["cold_off_pos"] += int(day["coldwave_advisory"])
        else:
            a["cold_fb_n"] += 1
            a["cold_fb_pos"] += int(temp <= COLDWAVE_THRESH)

        if "dust_observed" in day:
            a["dust_n"] += 1
            a["dust_pos"] += int(day["dust_observed"])

    def pct(pos, n):
        return f"{100.0*pos/n:6.2f}%" if n else "     —"

    print("=" * 100)
    print("폭염 라벨 — 공식 특보 기록 vs 임계값 근사(그 시각 기온 ≥33°C)")
    print("=" * 100)
    print(f"{'연도':>6} | {'전체표본':>10} | {'공식 표본':>10} {'양성률':>8} | "
          f"{'근사 표본':>10} {'양성률':>8}")
    print("-" * 100)
    for year in sorted(agg):
        a = agg[year]
        print(f"{year:>6} | {a['n']:>10,} | {a['heat_off_n']:>10,} "
              f"{pct(a['heat_off_pos'], a['heat_off_n']):>8} | "
              f"{a['heat_fb_n']:>10,} {pct(a['heat_fb_pos'], a['heat_fb_n']):>8}")

    print()
    print("=" * 100)
    print("한파 라벨 — 공식 특보 기록 vs 임계값 근사(그 시각 기온 ≤−12°C)")
    print("=" * 100)
    print(f"{'연도':>6} | {'전체표본':>10} | {'공식 표본':>10} {'양성률':>8} | "
          f"{'근사 표본':>10} {'양성률':>8}")
    print("-" * 100)
    for year in sorted(agg):
        a = agg[year]
        print(f"{year:>6} | {a['n']:>10,} | {a['cold_off_n']:>10,} "
              f"{pct(a['cold_off_pos'], a['cold_off_n']):>8} | "
              f"{a['cold_fb_n']:>10,} {pct(a['cold_fb_pos'], a['cold_fb_n']):>8}")

    print()
    print("=" * 100)
    print("황사 라벨 — 전부 공식(PM10 ≥150㎍/㎥). 연도별 양성률 = 비정상성 점검")
    print("=" * 100)
    print(f"{'연도':>6} | {'판정가능 표본':>14} {'양성':>10} {'양성률':>8}")
    print("-" * 100)
    for year in sorted(agg):
        a = agg[year]
        print(f"{year:>6} | {a['dust_n']:>14,} {a['dust_pos']:>10,} "
              f"{pct(a['dust_pos'], a['dust_n']):>8}")

    # 구간 요약 — 확장분(2013~2022)과 이전 학습 구간(2023~)을 갈라 본다.
    print()
    print("=" * 100)
    print("구간 요약 — 확장분(2013~2022) vs 직전 학습 구간(2023~2026)")
    print("=" * 100)
    for lo, hi, name in ((2013, 2022, "확장분 2013~2022"),
                         (2023, 2026, "직전 구간 2023~2026")):
        s = defaultdict(int)
        for year in sorted(agg):
            if lo <= int(year) <= hi:
                for k, v in agg[year].items():
                    s[k] += v
        heat_pos = s["heat_off_pos"] + s["heat_fb_pos"]
        cold_pos = s["cold_off_pos"] + s["cold_fb_pos"]
        print(f"\n[{name}]  표본 {s['n']:,}개")
        print(f"  폭염 — 공식 비중 {pct(s['heat_off_n'], s['n'])} · "
              f"공식 양성률 {pct(s['heat_off_pos'], s['heat_off_n'])} · "
              f"근사 양성률 {pct(s['heat_fb_pos'], s['heat_fb_n'])} · "
              f"합산 양성률 {pct(heat_pos, s['n'])}")
        print(f"  한파 — 공식 비중 {pct(s['cold_off_n'], s['n'])} · "
              f"공식 양성률 {pct(s['cold_off_pos'], s['cold_off_n'])} · "
              f"근사 양성률 {pct(s['cold_fb_pos'], s['cold_fb_n'])} · "
              f"합산 양성률 {pct(cold_pos, s['n'])}")
        print(f"  황사 — 판정가능 {s['dust_n']:,}개 · "
              f"양성률 {pct(s['dust_pos'], s['dust_n'])}")


if __name__ == "__main__":
    main()
