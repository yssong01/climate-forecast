"""
collect_aerosol_archive.py — 황사 헤드용 에어로졸 자료 수집기.

왜 필요한가 — `improvement-levers-measured` 측정에서 네 확률 헤드 중 **황사만
변별력이 병목**이었다(AUC 0.824, 나머지는 0.878~0.962). 이 저장소는 그
이유를 이미 진단해 뒀다: PM10 은 국외 장거리 수송이 지배하는데 입력은 국내
12개 지점의 지상 관측뿐이라, 구조적으로 판별 단서가 부족하다. 결론은
"외부 에어로졸 자료가 사실상 유일한 레버"였고, 자료원이 없어 손대지 못했다.

자료원 — Open-Meteo Air Quality API(CAMS 전 지구 대기조성 모델).
  `dust`(먼지 농도)·`pm10`·`pm2_5`·`aerosol_optical_depth`·`carbon_monoxide`.
  소급 범위는 실측으로 **2022-11 부터**(2022-08 이전은 값이 없다).

**상류 지점을 함께 받는 것이 이 수집기의 핵심이다.** 관측소 자신의 값만
받으면 이미 입력에 있는 국내 PM10 과 크게 다르지 않다. 황사는 고비·내몽골
에서 발원해 편서풍으로 실려 오므로, 예측 시점에 필요한 정보는 **지금 상류에
얼마나 떠 있는가**다. 실측으로도 상류일수록 농도가 뚜렷이 크다(2025-03-15
기준 dust 최댓값: 서해중부 9 · 베이징 83 · 내몽골남부 619 · 고비남단 424).
이는 강수에서 "+6h 뒤 도달할 강수계는 지금 404km 상류에 있다"고 밝힌 것과
같은 구조의 문제다.

**시간 누수에 대한 경고 — 반드시 읽을 것.**
이 API 에는 수치예보(`collect_nwp_archive.py`)와 달리 **리드타임을 보존하는
아카이브가 없다**(`dust_previous_day1` 은 전부 null 로 확인). 따라서
  ① 특징은 **예측 시점 T 이하의 값만** 쓴다. 목표 시각 T+L 의 에어로졸 값을
     쓰면 그 시점 관측을 반영한 분석장이라 누수다.
  ② 그럼에도 아카이브의 T 시점 값은 **분석장**이고 실서빙의 T 시점 값은
     그보다 앞서 발표된 **예보**라, 학습이 서빙보다 유리하다.
따라서 이 자료로 잰 이득은 **상한**이다 — 수치예보 프로브가 하한이었던 것과
정반대다. 상한이 작으면 그대로 기각할 수 있고, 크면 리드타임을 보존하는
자료원을 찾아 다시 재야 한다.

실행:
    python collect_aerosol_archive.py --backfill
"""
import argparse
import json
import os
import time
import urllib.parse
import urllib.request

from weather_collector import STATION_COORDS

API = "https://air-quality-api.open-meteo.com/v1/air-quality"
ARCHIVE_START = "2022-11-01"          # 실측으로 확인한 소급 한계
ARCHIVE_PATH = "./cache/aerosol_archive.json"

VARS = ["dust", "pm10", "pm2_5", "aerosol_optical_depth", "carbon_monoxide"]

# 상류 지점 — 황사 이동 경로(고비/내몽골 → 화북 → 보하이/서해 → 한반도)를
# 따라 다섯 곳. 좌표는 경로를 대표하기 위한 것이지 특정 관측소가 아니다.
UPSTREAM = {
    "up_gobi":    (44.0, 108.0),   # 고비 남단 — 발원지
    "up_inner":   (42.0, 112.0),   # 내몽골 남부
    "up_beijing": (40.0, 116.0),   # 화북
    "up_bohai":   (38.5, 121.0),   # 보하이/랴오둥
    "up_yellow":  (37.5, 124.0),   # 서해 중부 — 도달 직전
}


def _request(lat, lon, start, end, retries=4):
    params = dict(latitude=lat, longitude=lon, timezone="Asia/Seoul",
                  hourly=",".join(VARS), start_date=start, end_date=end)
    url = API + "?" + urllib.parse.urlencode(params)
    last = None
    for attempt in range(retries):
        try:
            with urllib.request.urlopen(url, timeout=300) as resp:
                return json.loads(resp.read().decode())["hourly"]
        except Exception as exc:              # noqa: BLE001
            last = type(exc).__name__
            time.sleep(5 * (attempt + 1))
    raise RuntimeError(f"Open-Meteo 대기질 요청 실패: {last}")


def _to_records(hourly):
    """`{시각12: [값...]}`. 결측 시각은 통째로 뺀다(CLAUDE.md 4절)."""
    out = {}
    cols = [hourly[v] for v in VARS]
    for i, t in enumerate(hourly["time"]):
        row = [c[i] for c in cols]
        if any(x is None for x in row):
            continue
        out[t.replace("-", "").replace("T", "").replace(":", "")[:12]] = row
    return out


def backfill(end_date=None):
    from datetime import datetime
    end = end_date or datetime.now().strftime("%Y-%m-%d")
    points = {**{s: c for s, c in STATION_COORDS.items()}, **UPSTREAM}
    archive = {}
    if os.path.exists(ARCHIVE_PATH):
        with open(ARCHIVE_PATH, "r", encoding="utf-8") as f:
            archive = json.load(f)
    for name, (lat, lon) in points.items():
        t0 = time.time()
        rows = _to_records(_request(lat, lon, ARCHIVE_START, end))
        archive.setdefault(name, {}).update(rows)
        print(f"  {name:12s} {time.time() - t0:5.1f}s  유효 {len(rows):,}시각 "
              f"(누적 {len(archive[name]):,})", flush=True)
        time.sleep(2)
    _save(ARCHIVE_PATH, archive)
    return archive


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
    ap.add_argument("--backfill", action="store_true")
    args = ap.parse_args()
    if args.backfill:
        print(f"에어로졸 아카이브 백필 — CAMS {ARCHIVE_START}~현재, "
              f"관측소 {len(STATION_COORDS)}곳 + 상류 {len(UPSTREAM)}곳")
        backfill()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
