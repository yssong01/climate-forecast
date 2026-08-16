"""
partition_heterogeneity.py — Expert-Grid(공간 9분할) 아이디어를 이 데이터에
적용할 근거가 있는지, 있다면 "어느 축으로" 분할해야 하는지를 실측한다.

배경(2026-08-16): 참고자료 '1-0. XGboost.pptx'의 Expert-Grid는 특징 공간을
3×3=9구역으로 나눠 구역마다 독립 전문가를 두는 방식이다(Wine 178건에서
MSE 87%↓). 자료 스스로 명시한 전제가 있다 — "9구역 분할은 구역당 표본 수를
1/9로 줄이므로 데이터가 적으면 구역별 과적합 위험이 커진다. Expert-Grid의
가치는 '대용량·고노이즈·공간 비균일' 조건에서 발현된다."

우리 데이터는 138만 표본이라 '대용량' 조건은 만족한다. 남은 질문은
'공간 비균일'이 실제로 있는가, 그리고 어느 좌표축을 따라 있는가다.
분할 후보는 셋이다.
  · 계절(월)   — 모델 입력에 아예 없다(record_to_vec: 시각 sin/cos·위경도만)
  · 시각(시간) — 이미 sin/cos 로 들어가 있다
  · 관측소     — 이미 위경도로 들어가 있고, Re축이 이웃 관측소를 보고 있다

측정 방법 — 각 후보 축으로 데이터를 쪼갠 뒤, 셀마다 "그 셀 안에서만 성립하는
관계"가 전역 평균과 얼마나 다른지 본다. 지표는 둘.
  1) 조건부 평균의 분산비(설명력) — 그 축이 타깃 변동을 얼마나 설명하는가.
     ΔT = T(t+6) − T(t) 와 강수 발생 여부를 대상으로 잰다.
  2) 셀당 표본 수 — Expert-Grid 의 전제(구역당 표본 충분)를 만족하는지.

실행: python partition_heterogeneity.py
"""
import json
from collections import defaultdict
from datetime import timedelta

import numpy as np

from train import DATA_CACHE, LEAD_HOURS, WET_THRESH, _parse_ts


def eta_squared(groups, values):
    """분산비(η²) — 그룹 구분이 값의 총변동 중 몇 %를 설명하는가.
    0이면 그 축으로 나눠도 아무 정보가 없다는 뜻이고, 크면 그 축을 따라
    관계 자체가 달라진다는 뜻이다(= 전문가를 나눌 근거)."""
    total_mean = values.mean()
    ss_total = ((values - total_mean) ** 2).sum()
    ss_between = 0.0
    for g in np.unique(groups):
        m = groups == g
        if m.sum() == 0:
            continue
        ss_between += m.sum() * (values[m].mean() - total_mean) ** 2
    return ss_between / ss_total if ss_total > 0 else 0.0


def main():
    with open(DATA_CACHE, encoding="utf-8") as f:
        records = json.load(f)

    by_key = {(r.get("stn"), str(r["timestamp"])[:12]): i
              for i, r in enumerate(records)}
    src, tgt = [], []
    for i, r in enumerate(records):
        t2 = (_parse_ts(r["timestamp"]) + timedelta(hours=LEAD_HOURS)).strftime("%Y%m%d%H%M")
        j = by_key.get((r.get("stn"), t2))
        if j is not None:
            src.append(i); tgt.append(j)
    n = len(src)
    print(f"유효 쌍 {n:,}개\n")

    t_now = np.array([records[i]["temperature"] for i in src], dtype=np.float32)
    t_tgt = np.array([records[j]["temperature"] for j in tgt], dtype=np.float32)
    p_tgt = np.array([records[j]["precipitation"] for j in tgt], dtype=np.float32)
    dT = t_tgt - t_now                      # 모델이 실제로 학습하는 양(퍼시스턴스 잔차)
    wet = (p_tgt >= WET_THRESH).astype(np.float32)

    ts = [str(records[i]["timestamp"]) for i in src]
    month = np.array([int(s[4:6]) for s in ts])
    hour = np.array([int(s[8:10]) for s in ts])
    stn = np.array([records[i]["stn"] for i in src])

    print("=" * 88)
    print("1) 후보 분할축이 타깃 변동을 얼마나 설명하는가 (분산비 η², 클수록 분할 근거 큼)")
    print("=" * 88)
    print(f"{'분할축':<28} | {'ΔT 설명력':>10} | {'강수발생 설명력':>14} | {'모델이 이미 아는가':>18}")
    print("-" * 88)
    rows = [
        ("월(계절, 12구간)",      month, "❌ 입력에 없음"),
        ("시각(24구간)",          hour,  "✅ sin/cos 로 있음"),
        ("관측소(12개)",          stn,   "✅ 위경도로 있음"),
    ]
    for name, g, known in rows:
        print(f"{name:<28} | {100*eta_squared(g, dT):9.2f}% | "
              f"{100*eta_squared(g, wet):13.2f}% | {known:>18}")

    # 계절 × 시각 교차 — Expert-Grid 의 3×3 격자에 가장 가까운 형태
    season = np.select(
        [np.isin(month, [12, 1, 2]), np.isin(month, [3, 4, 5]),
         np.isin(month, [6, 7, 8])], [0, 1, 2], default=3)
    tod = np.select(
        [hour < 6, hour < 12, hour < 18], [0, 1, 2], default=3)
    grid = season * 4 + tod
    print(f"{'계절4 × 시간대4 = 16구역':<28} | {100*eta_squared(grid, dT):9.2f}% | "
          f"{100*eta_squared(grid, wet):13.2f}% | {'부분적':>18}")

    print()
    print("=" * 88)
    print("2) 월별 상세 — 같은 모델이 하나의 규칙으로 감당해야 하는 범위")
    print("=" * 88)
    print(f"{'월':>4} | {'표본':>10} | {'평균 ΔT':>9} | {'ΔT 표준편차':>11} | "
          f"{'|ΔT| 평균':>10} | {'강수율':>7}")
    print("-" * 88)
    for mo in range(1, 13):
        m = month == mo
        print(f"{mo:>4} | {m.sum():>10,} | {dT[m].mean():>9.3f} | {dT[m].std():>11.3f} | "
              f"{np.abs(dT[m]).mean():>10.3f} | {100*wet[m].mean():>6.2f}%")

    print()
    print("=" * 88)
    print("3) Expert-Grid 전제 점검 — 구역당 표본 수가 충분한가")
    print("=" * 88)
    for name, g in (("월 12구간", month), ("계절4 × 시간대4 = 16구역", grid),
                    ("관측소 12개", stn), ("관측소12 × 계절4 = 48구역", None)):
        if g is None:
            pairs = defaultdict(int)
            for s, se in zip(stn, season):
                pairs[(s, se)] += 1
            counts = np.array(list(pairs.values()))
        else:
            counts = np.array([(g == u).sum() for u in np.unique(g)])
        print(f"  {name:<26} 구역 {len(counts):>3}개 · "
              f"구역당 최소 {counts.min():>8,} / 중앙 {int(np.median(counts)):>8,} / "
              f"최대 {counts.max():>8,}")
    print()
    print("  (참고) 참고자료 Wine 데이터: 총 178건 → 9구역 분할 시 구역당 약 20건")


if __name__ == "__main__":
    main()
