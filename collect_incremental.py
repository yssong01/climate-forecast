"""
collect_incremental.py — 백필 완료 후의 일일 증분 수집기.

collect_year.py는 1회성 백필(과거 전체 구간)을 예산 캡을 걸고 나눠 받는
도구다. 이 스크립트는 그 뒤 단계 — 백필이 끝난 뒤 "어제 이후 새로 생긴
시간"만 매일 채워 넣는 용도다.

왜 이게 안전한가:
  관측소당 최대 24개(하루치) × 12관측소 = 최대 288회/일. 실측된 장애
  지점(~9,800회/일)의 3% 수준이다. 이미 받은 시간은 (관측소, 시각) 쌍으로
  걸러내므로, 하루에 몇 번을 돌리든(자동 스케줄이든 수동이든) 그날 실제로
  새로 생긴 시간 수 이상은 절대 요청하지 않는다 — 즉 호출 빈도를 따로
  조절할 필요가 없다. 유일한 안전장치는 MAX_GAP_HOURS: 오래 안 돌려서
  간격이 비정상적으로 벌어졌을 때(버그·중단 등) 그 큰 격차를 이 스크립트가
  조용히 다 채워버려 예산을 잡아먹지 않도록 상한을 두는 것뿐이다.

대시보드(app.py)의 "현재 관측값" 조회(RobustWeatherCollector.fetch())는
이 스크립트와 무관하다 — 사용자가 보고 있는 관측소 1개만, 대시보드
슬라이더 주기(기본 5분)로 호출되며 원래도 작았다.

실행 (스케줄러에 하루 1~수회 등록):
    python collect_incremental.py                 # 전 관측소, 최신까지
    python collect_incremental.py --stations 108   # 서울만
"""
import argparse
import signal
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

from weather_collector import (RobustWeatherCollector, STATIONS, KST,
                               api_call_stats)
from collect_year import load_existing, atomic_save, OUT_FILE, LEGACY_CACHE

CONCURRENCY = 6
# 간격이 이보다 크게 벌어져 있으면(장기간 미실행 등) 자동으로 다 채우지
# 않고 경고만 하고 멈춘다 — collect_year.py --budget 으로 의도적으로 채워야
# 한다. 조용히 큰 격차를 메우다 예산 한도에 부딪히는 것을 막기 위함.
MAX_GAP_HOURS = 72

_stop = threading.Event()
signal.signal(signal.SIGTERM, lambda *_: _stop.set())
signal.signal(signal.SIGINT, lambda *_: _stop.set())


def _obs_hour_ceiling() -> datetime:
    """
    지금 시각 기준, 이미 발표됐을 것으로 기대되는 마지막 정시.

    KST를 명시해 컨테이너의 시스템 시간대(기본 UTC)에 의존하지 않되
    (weather_collector.KST 참조), ASOS 타임스탬프 문자열 자체는 tz 정보가
    없는 KST 벽시계 표기이므로 latest_per_station()의 naive datetime과
    비교 가능하도록 tzinfo는 계산 직후 제거한다.
    """
    now = datetime.now(KST)
    if now.minute < 10:
        now -= timedelta(hours=1)
    return now.replace(minute=0, second=0, microsecond=0, tzinfo=None)


def latest_per_station(records: dict) -> dict:
    """(stn, timestamp) → record 딕셔너리에서 관측소별 가장 최근 시각."""
    latest = {}
    for stn, ts in records.keys():
        dt = datetime.strptime(ts[:12], "%Y%m%d%H%M")
        if stn not in latest or dt > latest[stn]:
            latest[stn] = dt
    return latest


def main():
    ap = argparse.ArgumentParser(description="ASOS 일일 증분 수집 (백필 이후용)")
    ap.add_argument("--stations", nargs="*", default=list(STATIONS.values()))
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()
    station_names = {v: k for k, v in STATIONS.items()}

    existing = {}
    existing.update(load_existing(LEGACY_CACHE))
    existing.update(load_existing(args.out))
    latest = latest_per_station(existing)

    ceiling = _obs_hour_ceiling()
    print("=" * 66)
    print(f" 일일 증분 수집 — 기준 시각 {ceiling.strftime('%Y-%m-%d %H:%M')} KST")
    print("=" * 66)

    targets = []
    for stn in args.stations:
        if stn not in latest:
            print(f"  [{station_names.get(stn, stn)}] 기존 데이터 없음 — "
                  f"collect_year.py로 먼저 백필하세요. 건너뜀.")
            continue
        gap_start = latest[stn] + timedelta(hours=1)
        gap_hours = int((ceiling - gap_start).total_seconds() // 3600) + 1
        if gap_hours <= 0:
            print(f"  [{station_names.get(stn, stn)}] 이미 최신 — 신규 0개")
            continue
        if gap_hours > MAX_GAP_HOURS:
            print(f"  [{station_names.get(stn, stn)}] 격차 {gap_hours}시간 — "
                  f"{MAX_GAP_HOURS}시간 상한 초과. collect_year.py --budget 으로 "
                  f"의도적으로 채우세요. 건너뜀.")
            continue
        print(f"  [{station_names.get(stn, stn)}] 신규 {gap_hours}개 "
              f"({gap_start.strftime('%m-%d %H:%M')} ~ {ceiling.strftime('%m-%d %H:%M')})")
        for i in range(gap_hours):
            dt = gap_start + timedelta(hours=i)
            targets.append((stn, dt.strftime("%Y%m%d%H00")))

    if not targets:
        print("\n수집할 신규 시간 없음 — 이미 최신 상태입니다.")
        return

    # 분모는 실제 호출 예산(weather_collector.DAILY_CALL_BUDGET)을 쓴다 —
    # 종전에는 근거 없는 리터럴 10000 으로 나눠, 예산이 3000 인데도 여유가
    # 3배 넉넉한 것처럼 표시됐다(2026-08-29 총점검에서 수정).
    _budget = api_call_stats().get("budget", 0)
    _pct = f"{len(targets) / _budget * 100:.1f}% 수준" if _budget > 0 else "예산 무제한"
    print(f"\n총 {len(targets)}건 요청 예정 (일일 예산 {_budget:,}건의 {_pct})\n")

    collectors = {stn: RobustWeatherCollector(stn=stn) for stn in args.stations}
    records = dict(existing)
    ok = fail = 0

    def fetch_one(stn, tm):
        return stn, tm, collectors[stn].fetch_at(tm)

    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(fetch_one, stn, tm) for stn, tm in targets}
        for fut in as_completed(futures):
            if _stop.is_set():
                for f in futures:
                    f.cancel()
                break
            stn, tm, data = fut.result()
            if data.get("status") == "SUCCESS_LIVE":
                records[(stn, tm)] = data
                ok += 1
            else:
                fail += 1

    atomic_save(args.out, list(records.values()))
    print(f"완료 — 신규 성공 {ok}개, 실패 {fail}개. 저장: {args.out}")
    if _stop.is_set():
        print("(중단 신호로 조기 종료 — 지금까지 받은 분은 저장됨)")


if __name__ == "__main__":
    main()
