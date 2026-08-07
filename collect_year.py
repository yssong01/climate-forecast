"""
collect_year.py — ASOS 1년치 병렬 수집.

train.collect_historical()은 관측소×시간을 완전 순차로 돈다(관측소 하나를
끝내야 다음 관측소로 넘어감). 1년(8,760시간) × 12관측소 = 105,120회 호출을
이 방식으로 하면 호출당 0.3초 슬립만으로도 약 9시간, 실제 왕복 지연까지
더하면 하루를 넘긴다.

사전 실측(2026-08-07) 결과 kma_sfctm2 엔드포인트는:
  - tm1/tm2 범위 조회를 지원하지 않는다 — 응답이 단일 tm 조회와 완전히
    동일했다. 시간당 1회 조회가 이 API의 실제 설계다.
  - 짧은 버스트(12~36개 동시 호출)는 100% 성공한다. 문서화된 레이트리밋은
    응답 헤더에 없다.

그런데 동시성 16으로 실제 수집을 돌려보니, **누적 성공 약 9,800~10,000건
지점에서 이 키에 대한 모든 요청이 클린 에러 없이 커넥션 타임아웃으로
멈췄다**(단일 요청·다른 관측소로도 재현, 정지 후에도 즉시 회복 안 됨).
버스트 테스트에서는 안 보였던 현상 — 짧은 동시성 한도가 아니라 누적
요청수 기준 상한(추정: 일일 쿼터 성격)으로 보인다. 즉 병렬화로 줄일 수
있는 병목이 아니라, **하루 예산 안에서 나눠 받아야 하는 문제**다.

그래서 이 스크립트는 두 가지 역할을 한다:
  1. (관측소, 시각) 조합을 스레드풀로 병렬 처리해, 그날의 예산 안에서는
     최대한 빨리 받는다 — 여전히 유효한 최적화다.
  2. --budget 으로 1회 실행당 요청 수를 그 한도보다 낮게 자체 제한하고,
     한도에 닿기 *전에* 스스로 멈춘다. 담장에 부딪혀서 알아내는 대신
     미리 세어서 피한다.

weather_collector.fetch_at()은 self.stn/self.api_key 외의 공유 상태를
건드리지 않는다(디스크 캐시 쓰기는 fetch() 쪽에만 있다) — 그래서 이
병렬화는 추가 락 없이 안전하다.

안전장치:
  - --budget (기본 8000) — 실측된 장애 지점(~9,800건)보다 여유를 둔 상한.
    이 실행에서 성공 요청이 이 수에 닿으면 큐에 남은 작업을 취소하고
    깨끗하게 저장 후 종료한다.
  - 동시성 상한 --concurrency (기본 12)
  - 최근 200건 슬라이딩 윈도우 성공률 < 70% 이면 자동 반감(최소 2)
    + 30초 대기 — 예산 계산이 틀렸을 경우의 이중 안전망
  - 60초 또는 신규 2,000건마다 디스크에 증분 저장(원자적 교체) —
    중단돼도 처음부터 다시 하지 않는다
  - 기존 30일 캐시(cache/historical_data.json)와 이미 받은 진행 파일을
    미리 적재해 겹치는 (관측소, 시각)은 재요청하지 않는다
  - SIGTERM/SIGINT 수신 시 현재까지 수집분을 저장하고 종료 — docker stop
    이나 Ctrl-C로 중단해도 유실 없음

실행 (하루 1회, 여러 날에 걸쳐 나눠 받는다):
    python collect_year.py                    # 최근 365일, 예산 8000
    python collect_year.py --budget 5000       # 더 보수적으로
    python collect_year.py --concurrency 4     # 동시성도 낮추고 싶을 때
"""
import argparse
import json
import os
import signal
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from collections import deque
from datetime import datetime, timedelta

from weather_collector import RobustWeatherCollector, STATIONS, KST

OUT_FILE      = "./cache/historical_data_1y.json"
LEGACY_CACHE  = "./cache/historical_data.json"   # 기존 30일 수집 — 겹치면 재사용
CHECKPOINT_EVERY_N   = 2000
CHECKPOINT_EVERY_SEC = 60
BACKOFF_WINDOW   = 200
BACKOFF_MIN_RATE = 0.70
BACKOFF_SLEEP    = 30
MIN_CONCURRENCY  = 2

# 2026-08-07 실측: 동시성 16으로 돌리자 누적 성공 약 9,800건 지점에서
# 이 키에 대한 모든 요청이 클린 에러 없이 타임아웃되기 시작했고, 정지 후
# 즉시 단일 요청으로도 재현됐다(회복까지 걸리는 시간은 미확인). 실측치보다
# 여유를 둔 기본값 — 정확한 한도가 이보다 낮을 가능성에 대비한다.
DEFAULT_BUDGET = 8000

_stop = threading.Event()


def _handle_sigterm(signum, frame):
    """docker stop 등으로 종료 신호를 받으면 진행분을 먼저 저장하고 끝낸다."""
    print(f"\n[신호 {signum}] 종료 요청 — 현재까지 수집분 저장 후 정리합니다...")
    _stop.set()


signal.signal(signal.SIGTERM, _handle_sigterm)
signal.signal(signal.SIGINT, _handle_sigterm)


def load_existing(path: str) -> dict:
    """(stn, timestamp) → 레코드. SUCCESS_LIVE만 유지한다."""
    if not os.path.exists(path):
        return {}
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    return {
        (r["stn"], r["timestamp"]): r
        for r in records if r.get("status") == "SUCCESS_LIVE"
    }


def atomic_save(path: str, records: list) -> None:
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(records, f, ensure_ascii=False)
    os.replace(tmp, path)


def build_targets(days: int, stations: list) -> list:
    """전체 (stn, tm) 대상 목록. 관측소 블록이 아니라 station-major로 섞어
    스레드풀에 넣는다 — 어떤 순서로 완료되든 관측소별 균등 진행이 된다."""
    now = datetime.now(KST)
    hours = days * 24
    tms = [(now - timedelta(hours=i)).strftime("%Y%m%d%H00")
           for i in range(hours, 0, -1)]
    return [(stn, tm) for stn in stations for tm in tms]


def main():
    ap = argparse.ArgumentParser(description="ASOS 1년치 병렬 수집")
    ap.add_argument("--days", type=int, default=365)
    ap.add_argument("--concurrency", type=int, default=12)
    ap.add_argument("--budget", type=int, default=DEFAULT_BUDGET,
                    help="이번 실행에서 시도할 최대 요청 수(성공+실패 합산). "
                         f"실측된 장애 지점(~9,800건)보다 낮게 (기본 {DEFAULT_BUDGET})")
    ap.add_argument("--stations", nargs="*", default=list(STATIONS.values()))
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    station_names = {v: k for k, v in STATIONS.items()}
    concurrency = args.concurrency

    print("=" * 70)
    print(f" ASOS {args.days}일 병렬 수집 — {len(args.stations)}개 관측소")
    print("=" * 70)

    existing = {}
    existing.update(load_existing(LEGACY_CACHE))
    existing.update(load_existing(args.out))   # 진행 파일이 최신 소스
    print(f"재사용 가능한 기존 레코드: {len(existing)}개 "
          f"(기존 30일 캐시 + 이전 진행분)")

    targets = build_targets(args.days, args.stations)
    total = len(targets)
    pending = [(stn, tm) for stn, tm in targets if (stn, tm) not in existing]
    # 이번 실행은 예산만큼만 시도한다 — 나머지는 다음 실행(다음 날)이 이어서
    # 처리한다. 전량을 한꺼번에 큐에 넣지 않는 것 자체가 안전장치다.
    run_targets = pending[:args.budget]
    print(f"전체 목표: {total}개 | 이미 확보: {total - len(pending)}개 | "
          f"남은 수집: {len(pending)}개 | 이번 실행 예산: {len(run_targets)}개"
          + (f" (다음 실행에 {len(pending) - len(run_targets)}개 남음)"
             if len(pending) > len(run_targets) else "") + "\n")

    collectors = {stn: RobustWeatherCollector(stn=stn) for stn in args.stations}
    records = dict(existing)
    window = deque(maxlen=BACKOFF_WINDOW)
    lock = threading.Lock()

    done = 0
    last_checkpoint_t = time.time()
    last_checkpoint_n = 0
    t_start = time.time()
    per_station_ok = {s: 0 for s in args.stations}
    per_station_fail = {s: 0 for s in args.stations}

    def fetch_one(stn: str, tm: str):
        data = collectors[stn].fetch_at(tm)
        return stn, tm, data

    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {pool.submit(fetch_one, stn, tm) for stn, tm in run_targets}

        for fut in as_completed(futures):
            if _stop.is_set():
                for f in futures:
                    f.cancel()
                break

            stn, tm, data = fut.result()
            ok = data.get("status") == "SUCCESS_LIVE"
            with lock:
                window.append(ok)
                if ok:
                    records[(stn, tm)] = data
                    per_station_ok[stn] += 1
                else:
                    per_station_fail[stn] += 1
                done += 1

                # ── 적응형 백오프 — 최근 성공률이 떨어지면 자동으로 늦춘다 ──
                if len(window) == BACKOFF_WINDOW:
                    rate = sum(window) / BACKOFF_WINDOW
                    if rate < BACKOFF_MIN_RATE and concurrency > MIN_CONCURRENCY:
                        old = concurrency
                        concurrency = max(MIN_CONCURRENCY, concurrency // 2)
                        print(f"\n[경고] 최근 {BACKOFF_WINDOW}건 성공률 "
                              f"{rate:.0%} — 동시성 {old}→{concurrency}, "
                              f"{BACKOFF_SLEEP}초 대기 후 재개")
                        time.sleep(BACKOFF_SLEEP)
                        window.clear()

                # ── 진행 로그 ────────────────────────────────────────
                if done % 500 == 0 or done == len(run_targets):
                    elapsed = time.time() - t_start
                    rate_per_s = done / max(elapsed, 1e-6)
                    eta_s = (len(run_targets) - done) / max(rate_per_s, 1e-6)
                    print(f"  {done:6d}/{len(run_targets)} "
                          f"({done/len(run_targets)*100:5.1f}%) | "
                          f"{rate_per_s:5.1f}건/초 | "
                          f"경과 {elapsed/60:5.1f}분 | "
                          f"ETA {eta_s/60:5.1f}분 | 동시성 {concurrency}")

                # ── 증분 저장 ────────────────────────────────────────
                now_t = time.time()
                if (done - last_checkpoint_n >= CHECKPOINT_EVERY_N
                        or now_t - last_checkpoint_t >= CHECKPOINT_EVERY_SEC):
                    atomic_save(args.out, list(records.values()))
                    last_checkpoint_n, last_checkpoint_t = done, now_t

    atomic_save(args.out, list(records.values()))

    print(f"\n{'='*70}")
    print(f" 수집 종료 — 총 {len(records)}개 레코드 저장: {args.out}")
    print(f"{'='*70}")
    print(f"{'관측소':>6} {'성공':>6} {'실패':>6}")
    for stn in args.stations:
        print(f"{station_names.get(stn, stn):>6} "
              f"{per_station_ok[stn]:6d} {per_station_fail[stn]:6d}")
    if _stop.is_set():
        print("\n중단 요청으로 조기 종료됨 — 저장된 파일로 재실행하면 이어서 진행합니다.")

    remaining = len(pending) - done
    if remaining > 0:
        print(f"\n이번 예산 소진 — 아직 {remaining}개 남음. "
              f"내일(또는 한도가 풀린 뒤) 같은 명령으로 재실행하면 이어서 진행합니다.")


if __name__ == "__main__":
    main()
