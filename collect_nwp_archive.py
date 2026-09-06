"""
collect_nwp_archive.py — 수치예보(NWP) 예보값 수집기.

왜 필요한가(2026-09-06 실측, `nwp_feature_probe.py`):
  지상 관측만 쓰는 현재 입력으로는 강수 발생 판정 AUC 가 +6h 0.914 · +12h
  0.849 에서 멈춘다. 이웃 관측소의 시간 변화를 더 넣어도 증분이 +0.0003 에
  그쳐, 저장소는 이를 "기존 관측망의 정보는 소진됐다"고 결론지었다
  (README '관측 확장의 정보이론적 근거'). 같은 절제 실험에 NWP 예보값을
  한 단계 더 넣자 AUC 가 +6h +0.0285 · +12h +0.0795 올랐다 — 지금까지
  측정된 어떤 특징군보다 큰 증분이다. 물리적 이유는 이미 문서에 있다:
  +6h 뒤 도달할 강수계는 지금 약 404km 상류(서해상)에 있고, NWP 는 전 지구
  자료동화로 그 상류 상태를 이미 반영한다.

자료원 — Open-Meteo Previous Runs API (무료·인증키 불필요·CC BY 4.0)
  `<변수>_previous_day1` 은 **유효시각 24시간 전에 발표된** 예보값이다.
  T 시점에서 T+6h·T+12h 를 예측할 때 이 값은 각각 18h·12h 전에 이미
  발표돼 있으므로 시간 누수가 없다. 미래 유효시각에 대해서도 실시간으로
  채워지므로 **학습과 서빙이 문자 그대로 같은 필드를 쓴다** — 리드타임이
  다른 값을 학습·서빙에 나눠 쓸 때 생기는 분포 이동이 없다.

모델 선택 — `jma_gsm`
  실측으로 아카이브 소급 범위를 확인한 결과(2026-09-06):
    best_match / jma_msm            2022-01-01 ~
    gfs / ecmwf_ifs025 / icon       previous_day1 아카이브 없음
    **jma_gsm                       2016-01-01 ~**  ← 채택
  학습 캐시가 2013년부터이므로 2022년 기준을 쓰면 표본의 31.6% 만 남지만,
  2016년 기준이면 77.2% 가 남는다. 전 구간을 **한 모델**로 덮는 것이
  중간에 자료원이 바뀌어 모델이 두 체제를 학습하는 것보다 낫다고 보아
  단일 모델을 쓴다.

라이선스·한도 — CC BY 4.0 이므로 **화면에 출처를 표기해야 한다.** 무료
  티어는 비상업 용도 한정이고 10,000회/일·5,000회/시 제한이 있다. 공유 IP
  에서 429 가 보고되므로(Streamlit Cloud 는 발신 IP 를 공유한다) **배포
  앱이 직접 호출하지 않는다** — GitHub Actions 가 받아 저장소에 적재하고
  앱은 그 파일을 읽는다. ASOS 관측에 이미 쓰고 있는 구조와 같다.

실행:
    python collect_nwp_archive.py --backfill              # 전 구간 최초 수집
    python collect_nwp_archive.py --incremental --days 5  # 최근분 갱신(CI용)
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request
from datetime import datetime, timedelta

from weather_collector import STATION_COORDS

API = "https://previous-runs-api.open-meteo.com/v1/forecast"
MODEL = "jma_gsm"
ARCHIVE_START = "2016-01-01"        # 위 실측으로 확인된 jma_gsm 소급 한계
ARCHIVE_PATH = "./cache/nwp_archive.json"
RECENT_PATH = "./cache/nwp_recent.json"
RECENT_DAYS = 5                     # 배포용 창 — recent_window.json 과 같은 성격
SUFFIX = "_previous_day1"

# 변수 순서는 nwp_collector.NWP_VARS 와 반드시 같아야 한다 — 저장 파일이
# 이 순서에 의존하지 않도록 dict 로 저장하지만, 재구성 순서를 한 곳에
# 고정해 두는 편이 실수를 줄인다.
NWP_VARS = ["precipitation", "temperature_2m", "relative_humidity_2m",
            "cloud_cover", "wind_speed_10m", "wind_direction_10m",
            "surface_pressure"]


def _request(lat, lon, start, end, past_days=None, forecast_days=None, retries=4):
    hourly = ",".join(v + SUFFIX for v in NWP_VARS)
    params = dict(latitude=lat, longitude=lon, timezone="Asia/Seoul",
                  models=MODEL, hourly=hourly)
    if start is not None:
        params.update(start_date=start, end_date=end)
    else:
        params.update(past_days=past_days, forecast_days=forecast_days)
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                return json.loads(resp.read().decode())["hourly"]
        except Exception as exc:            # noqa: BLE001 — 재시도 대상 전부
            # 예외 문자열에 URL 이 통째로 들어가는 경우가 있다(CLAUDE.md
            # 4절). 이 엔드포인트는 인증키를 쓰지 않아 비밀값이 없지만,
            # 습관을 깨지 않도록 URL 을 지운 메시지만 남긴다.
            last = type(exc).__name__
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo 요청 실패({MODEL}): {last}")


def _to_records(hourly):
    """API 응답을 `{시각12자리: [변수값...]}` 로 바꾼다.

    Open-Meteo 시각은 `YYYY-MM-DDTHH:MM`(Asia/Seoul)이고 학습 캐시
    타임스탬프는 `YYYYMMDDHHmm` 이다. 둘 다 KST 벽시계 표기이므로 문자열
    정규화만으로 맞는다.
    """
    out = {}
    times = hourly["time"]
    cols = [hourly[v + SUFFIX] for v in NWP_VARS]
    for i, t in enumerate(times):
        row = [c[i] for c in cols]
        if any(x is None for x in row):
            # 결측은 채우지 않고 그 시각을 통째로 뺀다(CLAUDE.md 4절).
            continue
        key = t.replace("-", "").replace("T", "").replace(":", "")[:12]
        out[key] = row
    return out


def backfill(end_date=None):
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    archive = {}
    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            archive = json.load(f)
    for stn, (lat, lon) in STATION_COORDS.items():
        t0 = time.time()
        hourly = _request(lat, lon, ARCHIVE_START, end)
        rows = _to_records(hourly)
        archive.setdefault(stn, {}).update(rows)
        print(f"  {stn} {time.time() - t0:5.1f}s  유효 {len(rows):,}시각 "
              f"(누적 {len(archive[stn]):,})", flush=True)
        time.sleep(2)
    _save(ARCHIVE_PATH, archive)
    return archive


def incremental(days=RECENT_DAYS):
    """최근 며칠 + 앞으로 이틀을 받아 아카이브와 배포용 창을 함께 갱신한다.

    미래 구간이 필요한 이유: 서빙 시점 T 에서 쓰는 특징이 유효시각 T+6h·
    T+12h 의 예보값이라, 현재 시각 이후의 예보가 파일에 들어 있어야 한다.
    """
    archive = {}
    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            archive = json.load(f)
    recent = {}
    for stn, (lat, lon) in STATION_COORDS.items():
        hourly = _request(lat, lon, None, None, past_days=days, forecast_days=2)
        rows = _to_records(hourly)
        archive.setdefault(stn, {}).update(rows)
        recent[stn] = rows
        time.sleep(1)
    if os.path.exists(ARCHIVE_PATH):
        _save(ARCHIVE_PATH, archive)
    _save(RECENT_PATH, recent)
    n = min(len(v) for v in recent.values())
    print(f"  갱신 완료 — 관측소당 최소 {n:,}시각, 배포 창 {RECENT_PATH}")
    return recent


def _save(path, obj):
    """고유 tmp + os.replace 원자적 저장(CLAUDE.md 1절 6항)."""
    import tempfile
    d = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(dir=d, suffix=".tmp")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(obj, f, separators=(",", ":"))
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise
    print(f"  저장 {path} ({os.path.getsize(path) / 1e6:.1f}MB)")


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--backfill", action="store_true", help="2016년부터 전 구간 수집")
    ap.add_argument("--incremental", action="store_true", help="최근분만 갱신(CI용)")
    ap.add_argument("--days", type=int, default=RECENT_DAYS)
    args = ap.parse_args()
    if args.backfill:
        print(f"NWP 아카이브 백필 — {MODEL} {ARCHIVE_START}~현재, "
              f"{len(STATION_COORDS)}개 관측소")
        backfill()
    elif args.incremental:
        print(f"NWP 증분 갱신 — 최근 {args.days}일 + 예보 2일")
        incremental(args.days)
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
