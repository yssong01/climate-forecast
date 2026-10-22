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
(.github/workflows/refresh-data.yml)가 15분마다(Cloudflare 트리거) 실행해
결과를 **`data` 브랜치**에 커밋하고, 앱은 그 브랜치를 raw.githubusercontent.com
에서 직접 읽는다(2026-08-20 — 종전에는 main 에 커밋해 매번 재배포가 일어났다).
즉 "앱이 자기 파일을 유지한다"가 아니라 "저장소가
진실의 원본이고 앱은 그걸 읽기만 한다"로 구조를 바꾸는 것이다.

## collect_incremental.py 와 무엇이 다른가

collect_incremental.py 는 학습용 원본(cache/historical_data_1y.json,
69MB)을 기준으로 격차를 계산한다. 그 파일은 용량 때문에 저장소에 없으므로
CI 에서는 "기존 데이터 없음"으로 전부 건너뛴다. 이 스크립트는 저장소에
실제로 있는 recent_window.json(0.6MB)만을 기준으로 삼는다.

## API 호출량

관측소 12곳 × (마지막 수집 이후 경과 시간)이므로, **모든 관측이 제때
발표되면** 호출량은 실행 주기와 무관하게 하루 약 288건이다 — 자주 돌리면
회당 건수가 그만큼 줄어들기 때문이다. 실측 평시는 약 270~330건이다.

**결측이 생기면 이 계산이 성립하지 않았다(2026-10-21 정정).** 발표되지 않은
시각은 다음 실행이 다시 요청하므로, 관측소 하나가 결측이면 그 공백 전체를
매 실행 재요청했다 — 실측 2026-08-28 하루 2,466건. 지금은 탐침·재시도 1회·
진짜 일일 상한으로 막는다(아래 FETCH_RETRIES 주석).

(2026-08-29 정정: 종전에는 "6시간 주기로 돌리면 회당 약 72건"이라고 적혀
있었으나, 실제 주기는 2026-08-20부터 Cloudflare Cron 15분 / GitHub Actions
안전망 55분이다. refresh-data.yml 헤더 주석은 이미 갱신돼 있었고 이 파일만
남아 있었다.)

실행:
    python refresh_deploy_data.py                 # 전 관측소
    python refresh_deploy_data.py --stations 108  # 서울만
    python refresh_deploy_data.py --dry-run       # 요청 건수만 계산하고 종료
"""
import argparse
import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timedelta

import accuracy
from collect_year import atomic_save, load_existing
from weather_collector import (
    KST, STATIONS, RobustWeatherCollector, api_call_stats, migrate_api_wind_dir,
    preload_call_count,
)

WINDOW_FILE = "./cache/recent_window.json"
WINDOW_DAYS = 10          # build_recent_window.py 와 동일 — 72시간 + 여유
MAX_GAP_HOURS = 72        # 장기 미실행 시 한 번에 채울 상한 (관측소당)
SEED_HOURS = 24           # 창에 아예 없는 관측소를 새로 채울 때의 초기 분량
CONCURRENCY = 6

# 호출 폭주 방지(2026-10-21) — 실측 2026-08-28 하루 2,466건(평시 약 300건).
#
# 종전에는 관측소 하나가 결측(빈 200 응답·-9 값)이면 매 실행이 마지막 관측
# 이후의 공백 **전체**(최대 72시간)를 재시도 2회로 다시 요청했다. 결측이
# 사흘이면 72 × 2 × 96회 ≈ 하루 13,800건으로, 이 키가 차단된 규모(약
# 9,800건)를 넘는다. 게다가 `KMA_DAILY_CALL_BUDGET` 은 프로세스 메모리의
# 계수라 실행마다 0으로 돌아가 하루 상한 노릇을 못 했다.
#
# 그래서 ① 저장소의 일일 사용량으로 계수를 시작해 상한을 하루 단위로 걸고
# (`preload_call_count`), ② 재시도를 1회로 줄이고(다음 실행이 15분 뒤 어차피
# 다시 시도한다), ③ 공백이 `PROBE_AFTER_HOURS` 를 넘은 관측소는 **최신
# 1시간만 탐침**하고 그게 성공한 실행에서만 나머지 공백을 채운다. 결측
# 관측소 하나의 비용이 실행당 144건에서 1건이 된다.
FETCH_RETRIES = 1
PROBE_AFTER_HOURS = 3     # 정상 실행의 공백은 1~2시간이다

# 창 **내부** 공백 재요청(2026-10-22). 위 격차 계산은 관측소별 최신 관측
# **이후**만 보므로, 그 앞에 생긴 구멍(그 실행에서 미발표였다가 나중에
# 발표된 시각)은 창에서 밀려날 때까지 영구히 남았다 — 수원(119) 10-20
# 12~15시 4시간이 실재했고, 그 시각을 목표로 한 적중률 대기 4건도 함께
# 영원히 대조되지 않았다. 결측이 KMA 쪽에서 영구적이면 매 실행 재요청이
# 곧 호출 폭주이므로(위 2,466건 사고와 같은 유형) 세 겹으로 묶는다:
# ① 최근 `INTERIOR_REFILL_HOURS` 안의 구멍만, ② **한 시간에 한 실행**
# (분이 15 미만인 실행 — Cloudflare 트리거의 :01 실행)에서만, ③ 실행당
# `INTERIOR_MAX_PER_RUN` 건까지(최신 쪽부터). 구멍 하나의 최대 비용은
# 48건이고, 최악(구멍이 상한 이상 계속 있는 경우)에도 하루 144건이다.
INTERIOR_REFILL_HOURS = 48
INTERIOR_MAX_PER_RUN = 6

# 최신성 감시 임계값 — 이 갱신이 끝난 뒤에도 가장 뒤처진 관측소가 기준
# 시각보다 이만큼 이상 묵었으면 실행을 실패로 표시한다(2026-08-16 실측:
# 워크플로가 6~7시간 주기로 "성공"을 계속 찍었는데도 실제 데이터는
# 03:00 KST 에서 멈춰 화면에 12시간 전 데이터가 떴다 — API 실패를 개수만
# 세고 넘어갈 뿐 아무도 알아채지 못하게 설계돼 있었다). 정상 실행이면
# ASOS 발표 지연을 감안해도 이 안에 들어와야 한다.
MAX_STALE_HOURS = int(os.getenv("MAX_STALE_HOURS", "3"))

# 인증키 유출 대응 모니터링(2026-08-19 추가) — KMA 포털의 "API 호출 현황"이
# 실제 총 호출량의 원본이지만, 그건 사람이 직접 봐야 한다. 이 저장소가 스스로
# 아는 "우리가 쓴 것으로 확실한 만큼"을 저장소에 남겨두면, 포털 수치와
# 비교했을 때 크게 벌어지는 경우(=우리가 모르는 호출)를 사람이 알아챌 수
# 있다. 이 워크플로(GitHub Actions)의 호출량만 기록한다 — Streamlit 서빙
# 프로세스는 컨테이너가 재시작될 때마다(코드 변경에 따른 재배포 등) 카운터가
# 0으로 돌아가므로, 그 프로세스 자신이 이 파일에 안전하게 누적 기록하기
# 어렵다. app.py 는 "이 프로세스 시작 이후" 값을 별도로 표시하고, 두 값의
# 합을 포털 수치와 비교하라고 안내한다.
USAGE_LOG_FILE = "./cache/api_usage_daily.json"


def _usage_today() -> dict:
    """저장소에 기록된 오늘(KST) 사용량. 없거나 날짜가 다르면 오늘 자 빈 기록."""
    today = datetime.now(KST).strftime("%Y%m%d")
    data = {"date": today, "sources": {}, "total": 0}
    if os.path.exists(USAGE_LOG_FILE):
        try:
            with open(USAGE_LOG_FILE, "r", encoding="utf-8") as f:
                existing = json.load(f)
            if existing.get("date") == today:
                data = existing
        except Exception:
            pass   # 손상된 파일이면 오늘 자로 새로 시작한다
    return data


def _record_usage(source: str, count_this_run: int) -> None:
    """오늘(KST) 이 소스가 사용한 것으로 확인된 호출 수를 저장소에 누적한다."""
    import tempfile

    data = _usage_today()
    today = data["date"]
    data["sources"][source] = data["sources"].get(source, 0) + count_this_run
    data["total"] = sum(data["sources"].values())

    directory = os.path.dirname(USAGE_LOG_FILE) or "."
    fd, tmp = tempfile.mkstemp(dir=directory, prefix="api_usage_daily.json.")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False)
        os.replace(tmp, USAGE_LOG_FILE)
    except Exception:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise
    print(f"사용량 기록: 오늘({today}) {source} {data['sources'][source]}건 "
          f"— 저장소 총합 {data['total']}건")


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
                  station_names: dict) -> tuple[list, dict]:
    """채워야 할 (관측소, 시각) 목록과, 탐침이 성공하면 이어서 채울 시각들.

    반환: (targets, backfill). 공백이 `PROBE_AFTER_HOURS` 를 넘거나 창에 아예
    없는 관측소는 targets 에 **최신 1시간만** 넣고 나머지를 backfill[stn] 에
    둔다(위 FETCH_RETRIES 주석 참고).
    """
    latest = _latest_per_station(existing)
    targets, backfill = [], {}
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

        hours = [(gap_start + timedelta(hours=i)).strftime("%Y%m%d%H00")
                 for i in range(gap_hours)]
        if stn not in latest or gap_hours > PROBE_AFTER_HOURS:
            targets.append((stn, hours[-1]))
            backfill[stn] = hours[:-1]
            print(f"  [{name}] 공백 {gap_hours}시간 — 최신 1시간만 탐침, 성공하면 "
                  f"나머지 {len(hours) - 1}시간을 채운다")
            continue
        print(f"  [{name}] 신규 {gap_hours}개 "
              f"({gap_start.strftime('%m-%d %H:%M')} ~ {ceiling.strftime('%m-%d %H:%M')})")
        targets.extend((stn, h) for h in hours)
    return targets, backfill


def _interior_holes(existing: dict, stations: list, skip: set) -> list:
    """관측소별 최신 관측 앞 `INTERIOR_REFILL_HOURS` 안의 빈 시각, 최신 쪽부터.

    `skip` 은 탐침 중인 관측소다 — 최신 관측조차 못 받는 관측소의 과거 구멍을
    먼저 메울 이유가 없다. 상한(`INTERIOR_MAX_PER_RUN`)은 전체 기준이다.
    """
    latest = _latest_per_station(existing)
    earliest = {}
    for stn, ts in existing.keys():
        dt = datetime.strptime(ts[:12], "%Y%m%d%H%M")
        if stn not in earliest or dt < earliest[stn]:
            earliest[stn] = dt
    holes = []
    for stn in stations:
        if stn in skip or stn not in latest:
            continue
        for i in range(1, INTERIOR_REFILL_HOURS + 1):
            dt = latest[stn] - timedelta(hours=i)
            # **양쪽에 관측이 있는 구멍만** 본다 — 첫 관측 앞은 구멍이 아니라
            # 창의 시작이다(갓 심은 관측소는 24시간뿐이라, 이 조건이 없으면
            # 그 앞 24시간을 매시 재요청한다).
            if dt <= earliest[stn]:
                break
            tm = dt.strftime("%Y%m%d%H00")
            if (stn, tm) not in existing:
                holes.append((i, stn, tm))
    holes.sort()
    return [(stn, tm) for _, stn, tm in holes[:INTERIOR_MAX_PER_RUN]]


def _fetch(collectors: dict, targets: list) -> tuple[dict, int]:
    """(관측소, 시각) 목록을 병렬 조회한다. 반환: ({(stn, tm): 레코드}, 실패 수).

    SUCCESS_LIVE 가 아닌 응답은 버린다 — 폴백값(과거값 복제 또는 합성
    기본값)을 이력에 섞으면 결측이 실측으로 위장된다(CLAUDE.md 4절).
    """
    got, fail = {}, 0
    with ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        futures = {pool.submit(collectors[stn].fetch_at, tm, FETCH_RETRIES): (stn, tm)
                   for stn, tm in targets}
        for fut in as_completed(futures):
            data = fut.result()
            if data.get("status") == "SUCCESS_LIVE":
                got[futures[fut]] = data
            else:
                fail += 1
    return got, fail


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
    # 이 창은 API 경로 레코드만 담는다 — 풍향이 10도 단위로 저장돼 있던
    # 이행 전 레코드를 도 단위로 바꿔 다시 저장한다(멱등, 2026-10-21).
    n_migrated = sum(1 for r in existing.values() if r.get("wind_dir_unit") is None)
    existing = {k: migrate_api_wind_dir(r) for k, r in existing.items()}
    print(f"기존 창: {len(existing):,}개 레코드 ({args.window})"
          + (f" · 풍향 단위 이행 {n_migrated:,}건" if n_migrated else "") + "\n")

    targets, backfill = _plan_targets(existing, args.stations, ceiling, station_names)
    # 창 내부 공백 — 한 시간에 한 실행만(위 INTERIOR_REFILL_HOURS 주석).
    if datetime.now(KST).minute < 15:
        holes = _interior_holes(existing, args.stations, set(backfill))
        if holes:
            print(f"  창 내부 공백 {len(holes)}건 재요청(최근 {INTERIOR_REFILL_HOURS}시간, "
                  f"실행당 상한 {INTERIOR_MAX_PER_RUN}): "
                  + ", ".join(f"{station_names.get(s, s)} {t[4:10]}" for s, t in holes))
            targets += holes
    print(f"\n총 {len(targets)}건 요청 예정"
          + (f"(탐침 {len(backfill)}곳 — 성공 시 최대 "
             f"{sum(map(len, backfill.values()))}건 추가)" if backfill else ""))

    # 일일 상한 — 저장소에 누적된 오늘 사용량으로 계수를 시작한다.
    usage = _usage_today()
    preloaded = int(usage.get("total", 0))
    preload_call_count(usage["date"], preloaded)
    print(f"오늘 사용량: {preloaded}건 (일일 상한 {api_call_stats()['budget']}건)")

    if args.dry_run:
        print("(--dry-run — API 호출 없이 종료)")
        return

    ok = fail = 0
    records = dict(existing)
    try:
        if targets:
            collectors = {stn: RobustWeatherCollector(stn=stn)
                          for stn in {s for s, _ in targets}}
            got, fail = _fetch(collectors, targets)
            # 탐침이 성공한 관측소만 나머지 공백을 채운다(최신 쪽부터). 탐침
            # 관측소는 1단계 요청이 최신 1시간 하나뿐이라, 그 관측소의 성공이
            # 하나라도 있으면 탐침이 성공한 것이다.
            rest = [(stn, tm) for stn, hrs in backfill.items()
                    if any(s == stn for s, _ in got) for tm in reversed(hrs)]
            if rest:
                print(f"탐침 성공 {len({s for s, _ in rest})}곳 — 공백 {len(rest)}건을 채운다")
                got2, fail2 = _fetch(collectors, rest)
                got.update(got2)
                fail += fail2
            records.update(got)
            ok = len(got)
    finally:
        # 실행이 도중에 죽어도 쓴 만큼은 남긴다 — 빠지면 다음 실행의 일일
        # 상한이 그만큼 느슨해진다.
        calls = api_call_stats()
        run_calls = calls["count"] - (preloaded if calls["date"] == usage["date"] else 0)
        print(f"API 호출: 이번 실행 {run_calls}건 · 오늘 누계 {calls['count']}건 "
              f"(일일 상한 {calls['budget']}건, 상한 초과로 건너뜀 {calls['blocked']}건)")
        _record_usage("refresh_workflow", max(0, run_calls))

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
    # 창보다 오래된 대기 항목은 영영 대조될 수 없다 — 대조 원본이 이 창뿐이다.
    if records:
        window_start = min(str(r["timestamp"])[:12] for r in records.values())
        expired = accuracy.expire_unresolvable(window_start)
        if expired:
            print(f"::warning::창({window_start[:10]}~)보다 오래돼 대조할 수 없는 대기 "
                  f"항목 {expired}건을 지웠다 — 그 시각 관측이 끝내 창에 들어오지 않았다")
    stats = accuracy.stats()
    print(f"적중률 로그: 신규 해소 {resolved}건 / 대조 완료 누적 {stats['cum_n']}건")
    if stats["cum_n"]:
        print(f"  누적 적중률 — 기온 {stats['cum_temp']:.1%} · "
              f"강수 {stats['cum_precip']:.1%}")

    # ── 최신성 게이트 — 조용한 실패를 CI 빨간불로 바꾼다 ─────────────────
    # 지금까지는 성공/실패 개수를 출력만 하고 종료 코드는 항상 0이었다.
    # "그 시각 관측이 아직 발표 안 됨"과 "API·네트워크가 진짜 막힘"이
    # 화면상 똑같이 "실패"로 찍혀 구분이 안 됐고, 후자가 반복돼도 워크플로는
    # 계속 초록불이었다(2026-08-16, 배포 화면에 12시간 전 데이터 노출로 발견).
    # 최신 관측이 기준 시각보다 MAX_STALE_HOURS 이상 뒤처진 관측소가 있으면
    # 실행 자체를 실패로 표시해 GitHub 알림·빨간불로 드러낸다.
    latest_after = _latest_per_station(records)
    stale = {station_names.get(s, s): (ceiling - dt).total_seconds() / 3600
            for s, dt in latest_after.items()}
    worst_name, worst_gap = (max(stale.items(), key=lambda kv: kv[1])
                             if stale else (None, 0.0))
    if worst_gap > MAX_STALE_HOURS:
        print(f"::error::최신성 게이트 실패 — {worst_name} 관측이 기준 시각보다 "
              f"{worst_gap:.1f}시간 뒤처졌습니다(허용 {MAX_STALE_HOURS}시간). "
              f"API 실패가 조용히 누적됐을 수 있습니다 — 위 '실패·미발표' 건수를 "
              f"확인하세요.")
        sys.exit(1)


if __name__ == "__main__":
    main()
