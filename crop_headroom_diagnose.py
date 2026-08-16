"""
crop_headroom_diagnose.py — "의미 있는 부분만 먼저 학습(crop)" 전략의 여지를
데이터에서 실측한다. 학습을 돌리지 않는 CPU 전용 집계다.

배경(2026-08-15 사용자 제안): 곰팡이 진단 프로젝트(PangPangPang)에서
이미지 전체 대신 곰팡이 영역만 crop 해 먼저 학습하고, 그 가중치를 초기값으로
전체 이미지를 학습해 EfficientNet-B0 단일 모델로 90% 정확도를 얻었다.
같은 발상을 기후 데이터에 옮기려면 먼저 "무엇이 배경이고 무엇이 곰팡이인가"를
정의하고, 그 비율이 실제로 crop 할 만큼 치우쳐 있는지 재야 한다.

이미지에서 곰팡이가 화면의 일부만 차지하듯, 기후 데이터에서도 "모델이 배울
것이 있는 표본"은 일부다. +6시간 예보에서 배경에 해당하는 것은 두 가지다.

  1. 정적 표본 — T(t+6)이 T(t)와 거의 같은 시각. 퍼시스턴스(아무것도 안
     하는 기준선)가 이미 정답이라 모델이 더할 것이 없다. 손실에는 그대로
     들어가므로 학습 신호를 희석시킨다.
  2. 무사건 표본 — 강수·폭염·한파·황사 어느 것도 일어나지 않은 시각.
     희소사건 헤드 입장에서는 전부 음성 배경이다.

측정 항목
  A. |ΔT| = |T(t+6) − T(t)| 분포와, 구간별로 퍼시스턴스 오차가 전체 오차에
     기여하는 몫 — "모델의 값어치가 어디에 몰려 있는가"
  B. 사건 밀도 — 후보 crop 정의별로 남는 표본 수와 양성률
  C. 계절 편중 — 폭염·한파 공식 라벨이 실제로 존재하는 월 범위

실행: python crop_headroom_diagnose.py
"""
import json
from collections import defaultdict
from datetime import datetime, timedelta

import numpy as np

DATA_CACHE = "./cache/historical_data_1y.json"
WEATHER_ISSUE_LABELS = "./cache/weather_issue_labels.json"
LEAD_HOURS = 6
WET_THRESH = 0.1


def _parse_ts(ts):
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


def main():
    with open(DATA_CACHE, encoding="utf-8") as f:
        records = json.load(f)
    with open(WEATHER_ISSUE_LABELS, encoding="utf-8") as f:
        issue_labels = json.load(f)

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
    print(f"유효 (t, t+{LEAD_HOURS}h) 쌍 {n:,}개\n")

    t_now = np.array([records[i]["temperature"] for i in src_idx], dtype=np.float32)
    t_tgt = np.array([records[j]["temperature"] for j in tgt_idx], dtype=np.float32)
    p_tgt = np.array([records[j]["precipitation"] for j in tgt_idx], dtype=np.float32)
    dT = np.abs(t_tgt - t_now)

    # ── A. |ΔT| 분포와 오차 기여 몫 ──────────────────────────────────
    # 퍼시스턴스 MAE = mean(|ΔT|) 이므로, 구간별 |ΔT| 합이 전체 합에서
    # 차지하는 비율이 곧 "그 구간이 전체 오차에 기여하는 몫"이다.
    print("=" * 92)
    print("A. |ΔT| = |T(t+6) − T(t)| 분포 — 퍼시스턴스가 이미 맞히는 표본이 얼마나 되는가")
    print("=" * 92)
    print(f"  퍼시스턴스 MAE(=평균 |ΔT|): {dT.mean():.4f} °C   "
          f"중앙값 {np.median(dT):.4f} °C")
    print()
    print(f"{'구간':>14} | {'표본':>11} {'비중':>7} | {'오차 기여':>10} | {'누적 기여':>9}")
    print("-" * 92)
    edges = [0.0, 0.5, 1.0, 2.0, 3.0, 5.0, 8.0, 12.0, np.inf]
    total_err = dT.sum()
    cum = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (dT >= lo) & (dT < hi)
        share = dT[m].sum() / total_err if total_err else 0.0
        cum += share
        label = f"{lo:.1f}~{hi:.1f}" if np.isfinite(hi) else f"{lo:.1f}+"
        print(f"{label:>14} | {m.sum():>11,} {100*m.mean():>6.2f}% | "
              f"{100*share:>9.2f}% | {100*cum:>8.2f}%")

    # ── B. 사건 밀도 — 후보 crop 정의별 ─────────────────────────────
    heat_off, cold_off, dust_off = [], [], []
    for j in tgt_idx:
        r = records[j]
        ts = str(r["timestamp"])
        day = issue_labels.get(str(r.get("stn")), {}).get(
            f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}", {})
        heat_off.append(day.get("heatwave_advisory", -1))
        cold_off.append(day.get("coldwave_advisory", -1))
        dust_off.append(day.get("dust_observed", -1))
    heat_off = np.array(heat_off); cold_off = np.array(cold_off)
    dust_off = np.array(dust_off)

    is_wet = p_tgt >= WET_THRESH
    any_event = (heat_off == 1) | (cold_off == 1) | (dust_off == 1) | is_wet
    any_judgeable = (heat_off >= 0) | (cold_off >= 0) | (dust_off >= 0)

    print()
    print("=" * 92)
    print("B. 후보 crop 정의별 남는 표본 — '곰팡이 영역'에 해당하는 것이 무엇인가")
    print("=" * 92)

    def show(name, mask, note=""):
        k = int(mask.sum())
        print(f"  {name:<34} {k:>10,}개 ({100*k/n:5.2f}%)"
              + (f"  {note}" if note else ""))

    show("전체(현행 학습)", np.ones(n, bool))
    show("① 사건 표본만 (강수∪폭염∪한파∪황사)", any_event)
    show("② 변화 표본만 |ΔT| ≥ 2°C", dT >= 2.0)
    show("③ 변화 표본만 |ΔT| ≥ 3°C", dT >= 3.0)
    show("④ 공식 라벨 판정 가능 표본만", any_judgeable)
    show("⑤ ①∪② (사건 또는 변화)", any_event | (dT >= 2.0))
    show("⑥ ①∩④ (판정 가능한 사건)", any_event & any_judgeable)

    print()
    print("  각 정의에서의 사건 양성률(희소사건 헤드가 보게 되는 밀도):")
    for name, m in (("전체", np.ones(n, bool)),
                    ("① 사건 표본만", any_event),
                    ("② |ΔT| ≥ 2°C", dT >= 2.0),
                    ("⑤ ①∪②", any_event | (dT >= 2.0))):
        sub_wet = is_wet[m].mean() if m.sum() else 0
        hm = m & (heat_off >= 0)
        cm = m & (cold_off >= 0)
        dm = m & (dust_off >= 0)
        print(f"    {name:<16} 강수 {100*sub_wet:5.2f}% · "
              f"폭염 {100*(heat_off[hm] == 1).mean() if hm.sum() else 0:5.2f}% · "
              f"한파 {100*(cold_off[cm] == 1).mean() if cm.sum() else 0:5.2f}% · "
              f"황사 {100*(dust_off[dm] == 1).mean() if dm.sum() else 0:5.2f}%")

    # ── C. 계절 편중 ────────────────────────────────────────────────
    months = np.array([int(str(records[j]["timestamp"])[4:6]) for j in tgt_idx])
    print()
    print("=" * 92)
    print("C. 월별 공식 라벨 존재 여부 — 라벨 자체가 이미 계절로 crop 되어 있다")
    print("=" * 92)
    print(f"{'월':>4} | {'폭염 판정가능':>13} {'양성률':>8} | "
          f"{'한파 판정가능':>13} {'양성률':>8} | {'강수 양성률':>10}")
    print("-" * 92)
    for mo in range(1, 13):
        mm = months == mo
        hj = mm & (heat_off >= 0)
        cj = mm & (cold_off >= 0)
        hp = 100 * (heat_off[hj] == 1).mean() if hj.sum() else 0.0
        cp = 100 * (cold_off[cj] == 1).mean() if cj.sum() else 0.0
        print(f"{mo:>4} | {hj.sum():>13,} {hp:>7.2f}% | "
              f"{cj.sum():>13,} {cp:>7.2f}% | {100*is_wet[mm].mean():>9.2f}%")

    # ── D. 시간 분할 가능성 ─────────────────────────────────────────
    # README 한계 8: 30일 수집으로는 시간 분할 검증이 불가능했다(검증 구간에
    # 비가 아예 안 옴). 13년이면 연 단위 홀드아웃이 가능한지 확인한다.
    years = np.array([int(str(records[j]["timestamp"])[:4]) for j in tgt_idx])
    print()
    print("=" * 92)
    print("D. 연 단위 시간 분할 가능성 — 홀드아웃 후보 연도에 사건이 충분한가")
    print("=" * 92)
    print(f"{'연도':>6} | {'표본':>10} | {'강수양성':>9} | {'폭염양성':>9} | "
          f"{'한파양성':>9} | {'황사양성':>9}")
    print("-" * 92)
    for y in sorted(set(years.tolist())):
        m = years == y
        print(f"{y:>6} | {m.sum():>10,} | {int(is_wet[m].sum()):>9,} | "
              f"{int((heat_off[m] == 1).sum()):>9,} | "
              f"{int((cold_off[m] == 1).sum()):>9,} | "
              f"{int((dust_off[m] == 1).sum()):>9,}")


if __name__ == "__main__":
    main()
