"""
collect_island_backfill.py — 서해 도서 AWS 결손 구간(2026-03-15~현재) 보충 수집.

**배경.** Risk_Prediction 백필(`cache/island_aws_raw.parquet`, 497,107행)은
2022-01-01~2026-03-15까지만 있다(island-aws-ablation-gate-passed 메모 참고).
climate-forecast 배포 데이터(historical_data_1y.json)는 그 이후로도 계속
갱신되고 있어, 도서 특징을 P1 재학습에 실제로 넣으려면 이 격차를 메워야 한다.

**왜 관측소당 조회가 아니라 시각당 조회인가.** `awsh.php`는 ASOS 12관측소가
쓰는 `kma_sfctm2`(관측소당 1시간 단위만 조회 가능, collect_year.py 참고)와
달리, **단일 시각에 전 지점(736개)을 한 번에** 반환한다(Risk_Prediction
`src/sources/kma.py fetch_awsh()` 실측 확인, 2026-08-31 재확인: 시각 1개
호출로 도서 13개 전부 포함해 736개 관측소 회신, 0.21초, `#7777END` 완결성
마커 있음 — kma_sfctm2와 달리 문서화된 완결성 마커가 존재한다). 그래서
결손 구간(약 168일 = 4,032시간)을 채우는 데 필요한 호출은 4,032회이지
4,032×13회가 아니다.

**결측 규칙**(Risk_Prediction `_tag_missing()` 그대로 이식): 강수 필드
(rn_hr1)는 음수를 전부 결측으로 본다(-99.0/-9.0 둘 다 섞여 나옴, 고정
sentinel 하나로는 못 잡음). 그 외 필드는 -50 이하를 결측으로 본다(KMA
고정폭 API 전반의 관례, awsh.php 자체 문서화는 없음 — 방어적으로 적용).

**안전장치**(collect_year.py와 동일 원칙): --budget 으로 1회 실행 요청 수를
자체 제한하고, 도달 전에 스스로 멈춘다. 60초 또는 신규 500건마다 원자적
저장(고유 tmp + os.replace, CLAUDE.md 6절). SIGTERM/SIGINT 시 현재까지
저장 후 종료. 기존 parquet의 최대 tm 이후부터만 요청해 중복 수집을 피한다.

실행 (여러 날에 나눠 받는 것을 권장 — 이 키는 climate-forecast 자체 15분
주기 자동 갱신과 공유된다):
  python collect_island_backfill.py                  # 기본 예산 2500
  python collect_island_backfill.py --budget 1500     # 더 보수적으로
"""
import argparse
import os
import signal
import sys
import tempfile
import time
from datetime import datetime, timedelta

import pandas as pd
import requests
from dotenv import load_dotenv

load_dotenv()

BASE = "https://apihub.kma.go.kr/api"
AWSH_PATH = "typ01/url/awsh.php"
OUT_FILE = "./cache/island_aws_raw.parquet"

AWSH_COLUMNS = ["tm", "stn", "ta", "wd", "ws", "rn_day", "rn_hr1", "hm", "pa", "ps"]
# 기존 parquet 스키마와 맞춘다 — rn_day/ps는 애초에 저장하지 않았다.
KEEP_FIELDS = ["ta", "wd", "ws", "rn_hr1", "hm", "pa"]
RAIN_FIELDS = {"rn_hr1", "rn_day"}

# island-aws-ablation-gate-passed 메모의 14개 코드(가대암 956 포함 —
# 모델링 제외는 data_expansion_probe_island.py의 ISLAND_COORDS에서 하고,
# 원본 수집은 기존 parquet과 동일 지점 집합을 유지한다).
ISLAND_STNS = {"102", "169", "229", "303", "426", "501", "655",
               "663", "697", "700", "743", "797", "798", "956"}

BUDGET_DEFAULT = 2500
SAVE_EVERY_N = 500
SAVE_EVERY_SEC = 60

_stop = False


def _handle_signal(signum, frame):
    global _stop
    _stop = True
    print(f"\n[신호 {signum}] 현재까지 수집분 저장 후 종료합니다...")


signal.signal(signal.SIGINT, _handle_signal)
signal.signal(signal.SIGTERM, _handle_signal)


def _f(raw: str):
    try:
        return float(raw)
    except ValueError:
        return None


def _tag_missing(value, *, is_rain: bool, sentinel: float = -50.0):
    if value is None:
        return None, True
    if is_rain:
        return value, value < 0.0
    return value, value <= sentinel


def redact(text: str, key: str) -> str:
    return text.replace(key, "***") if key else text


def fetch_awsh(tm: datetime, key: str, timeout: int = 20) -> list[dict]:
    params = {"tm": tm.strftime("%Y%m%d%H%M"), "stn": 0, "disp": 0, "help": 0, "authKey": key}
    resp = requests.get(f"{BASE}/{AWSH_PATH}", params=params, timeout=timeout)
    resp.raise_for_status()
    text = resp.text
    rows = []
    for line in text.strip().split("\n"):
        if line.startswith("#") or not line.strip():
            continue
        fields = line.split()
        if len(fields) != len(AWSH_COLUMNS):
            continue
        stn = fields[1]
        if stn not in ISLAND_STNS:
            continue
        row = {"tm": fields[0], "stn": stn}
        for name, raw in zip(AWSH_COLUMNS[2:], fields[2:]):
            if name not in KEEP_FIELDS:
                continue
            val, missing = _tag_missing(_f(raw), is_rain=name in RAIN_FIELDS)
            row[name] = val
            row[f"{name}_missing"] = missing
        rows.append(row)
    return rows


def atomic_save(df: pd.DataFrame, path: str) -> None:
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".parquet")
    os.close(fd)
    df.to_parquet(tmp, index=False)
    os.replace(tmp, path)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--budget", type=int, default=BUDGET_DEFAULT)
    args = ap.parse_args()

    key = os.getenv("KMA_API_KEY", "")
    if not key:
        print("KMA_API_KEY 미설정 — .env 확인")
        return 1

    existing = pd.read_parquet(OUT_FILE)
    existing["tm"] = existing["tm"].astype(str)
    last_tm = datetime.strptime(existing["tm"].max(), "%Y%m%d%H%M")
    now_kst = datetime.utcnow() + timedelta(hours=9)
    now_kst = now_kst.replace(minute=0, second=0, microsecond=0)

    cursor = last_tm + timedelta(hours=1)
    total_hours = int((now_kst - cursor).total_seconds() // 3600) + 1
    print(f"기존 데이터 최대 시각: {last_tm} · 목표: {now_kst} · "
          f"필요 호출 {max(total_hours, 0):,}회 · 이번 실행 예산 {args.budget:,}회")

    if cursor > now_kst:
        print("결손 구간 없음 — 이미 최신입니다.")
        return 0

    new_rows = []
    requested = 0
    ok = 0
    fail = 0
    t_last_save = time.time()
    recent = []   # 최근 시도 성공/실패 슬라이딩 윈도우 — collect_year.py와 동일 안전장치

    while cursor <= now_kst and requested < args.budget and not _stop:
        try:
            rows = fetch_awsh(cursor, key)
            new_rows.extend(rows)
            ok += 1
            recent.append(True)
        except Exception as e:
            fail += 1
            recent.append(False)
            print(f"  [{cursor:%Y-%m-%d %H:%M}] 실패: {type(e).__name__}: "
                  f"{redact(str(e), key)[:150]}")
        requested += 1
        cursor += timedelta(hours=1)
        recent = recent[-50:]

        # 최근 50건 성공률이 70% 미만이면 서버 쪽 일시 장애로 보고 60초 쉰다 —
        # 실제로 2026-08-31 첫 실행에서 503이 연쇄됐다가 대기 후 회복된 것을
        # 통제 대조로 확인했다(같은 시각 재요청이 성공).
        if len(recent) >= 20 and (sum(recent) / len(recent)) < 0.7:
            print(f"  최근 {len(recent)}건 성공률 {sum(recent)/len(recent):.0%} — "
                  f"서버 일시 장애로 보고 60초 대기 후 재개")
            time.sleep(60)
            recent = []

        if new_rows and (len(new_rows) >= SAVE_EVERY_N or time.time() - t_last_save >= SAVE_EVERY_SEC):
            merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
            merged = merged.drop_duplicates(subset=["tm", "stn"], keep="last")
            atomic_save(merged, OUT_FILE)
            existing = merged
            print(f"  중간 저장: 누적 {len(existing):,}행 (요청 {requested}/{args.budget}, "
                  f"성공 {ok}, 실패 {fail})")
            new_rows = []
            t_last_save = time.time()

        time.sleep(0.05)

    if new_rows:
        merged = pd.concat([existing, pd.DataFrame(new_rows)], ignore_index=True)
        merged = merged.drop_duplicates(subset=["tm", "stn"], keep="last")
        atomic_save(merged, OUT_FILE)
        existing = merged

    remaining = max(0, int((now_kst - cursor).total_seconds() // 3600) + 1) if cursor <= now_kst else 0
    print(f"\n종료 — 요청 {requested}회(성공 {ok}, 실패 {fail}), 저장 {len(existing):,}행, "
          f"남은 결손 약 {remaining:,}시간(다음 실행에서 자동 재개)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
