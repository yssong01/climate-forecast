"""
refresh_deploy_data.py — 배포 저장소에 커밋되는 데이터 파일을 갱신한다.

## 왜 필요한가

Streamlit Community Cloud 의 컨테이너 파일시스템은 휘발성이다. 앱이 실행
중에 쓴 파일은 재시작·재배포 때 사라지고, 저장소에 커밋된 내용으로
되돌아간다. 그래서 배포판에서는 다음 두 가지가 시간이 갈수록 망가진다.

  ① cache/recent_window.json 이 배포 시점에 얼어붙는다 — 사이드바
     스파크라인과 '예보 추이' 탭의 "최근 72시간"이 서서히 비어간다.
  ② cache/accuracy_log.json 의 대기 항목(actual=None)이 영원히 해소되지
     않는다. accuracy.resolve_pending() 은 로컬 캐시만 조회하는데(쿼터
     보호 — accuracy.py 모듈 docstring 참고), 클라우드에는 그 캐시를
     채워주는 collect_incremental.py 가 돌지 않기 때문이다.

이 스크립트는 그 둘을 저장소 바깥에서 고친다. GitHub Actions
(.github/workflows/refresh-data.yml)가 주기적으로 실행해 결과를 main 에
커밋하면, Streamlit Cloud 가 푸시를 감지해 자동 재배포하면서 갱신된
데이터가 반영된다. 즉 "앱이 자기 파일을 유지한다"가 아니라 "저장소가
진실의 원본이고 앱은 그걸 읽기만 한다"로 구조를 바꾸는 것이다.

## collect_incremental.py 와 무엇이 다른가

collect_incremental.py 는 학습용 원본(cache/historical_data_1y.json,
69MB)을 기준으로 격차를 계산한다. 그 파일은 용량 때문에 저장소에 없으므로
CI 에서는 "기존 데이터 없음"으로 전부 건너뛴다. 이 스크립트는 저장소에
실제로 있는 recent_window.json(0.6MB)만을 기준으로 삼는다.

## API 호출량

관측소 12곳 × (마지막 수집 이후 경과 시간). 6시간 주기로 돌리면 회당 약
72건, 하루 약 288건이다 — 이 키가 차단됐던 지점(누적 약 9,800건,
CLAUDE.md 4절)의 3% 수준이고, collect_incremental.py 가 안전하다고 적어둔
일일 상한과 같은 규모다. MAX_GAP_HOURS 로 장기 미실행 시의 폭주를 막는다.

실행:
    python refresh_deploy_data.py                 # 전 관측소
    python refresh_deploy_data.py --stations 108  # 서울만
    python refresh_deploy_data.py --dry-run       # 요청 건수만 계산하고 종료
"""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import accuracy
from collect_year import atomic_save, load_existing
from weather_collector import (
    KST, STATIONS, RobustWeatherCollector, api_call_stats,
)

WINDOW_FILE = "./cache/recent_window.json"
WINDOW_DAYS = 10          # build_recent_window.py 와 동일 — 72시간 + 여유
MAX_GAP_HOURS = 72        # 장기 미실행 시 한 번에 채울 상한 (관측소당)
SEED_HOURS = 24           # 창에 아예 없는 관측소를 새로 채울 때의 초기 분량
CONCURRENCY = 6


def _obs_hour_ceiling() -> datetime:
    """
    지금 기준 이미 발표됐을 마지막 정시(KST, naive).

    ASOS 타임스탬프는 tz 정보 없는 KST 벽시계 표기다. GitHub Actions 러너는
    UTC 라 datetime.now() 를 그대로 쓰면 9시간 뒤처진 시각을 "현재"로 잡아
    조용히 묵은 구간만 요청하게 된다(weather_collector.py 가 같은 이유로
    KST 를 명시하는 것과 동일한 함정).
    """
    now = datetime.now(KST)
    if now.minute < 10:
        now -= timedelta(hours=1)
    return now.replace(minute=0, second=0, microsecond=0, tzinfo=None)


def _latest_per_station(records: dict) -> dict:
    latest = {}
    for stn, ts in records.keys():
        dt = datetime.strptime(ts[:12], "%Y%m%d%H%M")
        if stn not in latest or dt > latest[stn]:
            latest[stn] = dt
    return latest


def _plan_targets(existing: dict, stations: list, ceiling: datetime,
                  station_names: dict) -> list:
    """채워야 할 (관측소, 시각) 목록. 관측소별 상한을 적용한다."""
    latest = _latest_per_station(existing)
    targets = []
    for stn in stations:
        name = station_names.get(stn, stn)
        if stn in latest:
            gap_start = latest[stn] + timedelta(hours=1)
            gap_hours = int((ceiling - gap_start).total_seconds() // 3600) + 1
            if gap_hours <= 0:
                print(f"  [{name}] 이미 최신 — 신규 0개")
                continue
            if gap_hours > MAX_GAP_HOURS:
                # 조용히 다 채우면 예산을 크게 먹는다. 상한만큼만 최신 쪽부터.
                print(f"  [{name}] 격차 {gap_hours}시간 — 상한 {MAX_GAP_HOURS}시간 "
                      f"만큼만 최신 구간부터 채움")
                gap_hours = MAX_GAP_HOURS
                gap_start = ceiling - timedelta(hours=gap_hours - 1)
        else:
            # 창에서 완전히 빠진 관측소(신규 추가 등) — 최근 SEED_HOURS 만
            # 심어두면 다음 실행부터는 정상 격차 계산 경로를 탄다.
            gap_hours = SEED_HOURS
            gap_start = ceiling - timedelta(hours=gap_hours - 1)
            print(f"  [{name}] 창에 데이터 없음 — 최근 {SEED_HOURS}시간 초기 수집")

        print(f"  [{name}] 신규 {gap_hours}개 "
              f"({gap_start.strftime('%m-%d %H:%M')} ~ {ceiling.strftime('%m-%d %H:%M')})")
        for i in range(gap_hours):
            dt = gap_start + timedelta(hours=i)
            targets.append((stn, dt.strftime("%Y%m%d%H00")))
    return targets


def _trim(records: dict, cutoff: datetime) -> dict:
    kept = {}
    for key, rec in records.items():
        try:
            dt = datetime.strptime(str(rec["timestamp"])[:12], "%Y%m%d%H%M")
        except (ValueError, KeyError):
            continue
        if dt >= cutoff:
            kept[key] = rec
    return kept


def main():
    ap = argparse.ArgumentParser(description="배포용 데이터 파일 갱신")
    ap.add_argument("--stations", nargs="*", default=list(STATIONS.values()))
    ap.add_argument("--window", default=WINDOW_FILE)
    ap.add_argument("--dry-run", action="store_true",
                    help="요청 건수만 계산하고 API 호출 없이 종료")
    args = ap.parse_args()

    station_names = {v: k for k, v in STATIONS.items()}
    ceiling = _obs_hour_ceiling()

    print("=" * 66)
    print(f" 배포 데이터 갱신 — 기준 시각 {ceiling.strftime('%Y-%m-%d %H:%M')} KST")
    print("=" * 66)

    existing = load_existing(args.window)   # SUCCESS_LIVE 만 남긴다
    print(f"기존 창: {len(existing):,}개 레코드 ({args.window})\n")

    targets = _plan_targets(existing, args.stations, ceiling, station_names)
    print(f"\n총 {len(targets)}건 요청 예정")

    if args.dry_run:
        print("(--dry-run — API 호출 없이 종료)")
        return

    ok = fail = 0
    records = dict(existing)
    if targets:
        collectors = {stn: RobustWeatherCollector(stn=stn)
                      for stn in {s for s, _ in targets}}

        def fetch_one(stn, tm):
            return stn, tm, collectors[stn].fetch_at(tm)

        with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
            futures = [pool.submit(fetch_one, stn, tm) for stn, tm in targets]
            for fut in as_completed(futures):
                stn, tm, data = fut.result()
                # SUCCESS_LIVE 가 아닌 응답은 버린다 — 폴백값(과거값 복제 또는
                # 합성 기본값)을 이력에 섞으면 결측이 실측으로 위장된다
                # (CLAUDE.md 4절).
                if data.get("status") == "SUCCESS_LIVE":
                    records[(stn, tm)] = data
                    ok += 1
                else:
                    fail += 1

    cutoff = datetime.now(KST).replace(tzinfo=None) - timedelta(days=WINDOW_DAYS)
    before = len(records)
    records = _trim(records, cutoff)
    atomic_save(args.window, list(records.values()))

    size_mb = os.path.getsize(args.window) / 1e6
    print(f"\n수집: 성공 {ok}개 / 실패·미발표 {fail}개")
    print(f"창 정리: {before:,}개 → 최근 {WINDOW_DAYS}일 {len(records):,}개 "
          f"({size_mb:.2f}MB) → {args.window}")

    # ── 적중률 로그의 대기 항목 해소 ──────────────────────────────────
    # 방금 갱신한 창이 곧 실측 원본이다. 새 API 호출은 하지 않는다.
    resolved = accuracy.resolve_pending(records)
    stats = accuracy.stats()
    print(f"적중률 로그: 신규 해소 {resolved}건 / 대조 완료 누적 {stats['cum_n']}건")
    if stats["cum_n"]:
        print(f"  누적 적중률 — 기온 {stats['cum_temp']:.1%} · "
              f"강수 {stats['cum_precip']:.1%}")

    calls = api_call_stats()
    print(f"API 호출: {calls['count']}건 (예산 {calls['budget']}건, "
          f"차단 {calls['blocked']}건)")


if __name__ == "__main__":
    main()
