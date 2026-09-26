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


SNAPSHOT_PATH = "./cache/aerosol_snapshots.json"


def snapshot():
    """**서빙 시점의 값을 그대로 떠 둔다 — 시간 정합성을 재기 위해서다.**

    왜 필요한가(2026-09-27). 에어로졸 특징이 황사 F1 을 0.179→0.224 로
    올린다는 측정이 나왔는데(`aerosol_gbm_check.py`), 그 값이 **상한**인지
    실제 이득인지 가릴 수단이 없었다. 학습은 아카이브의 T 시점 값을 쓰고
    서빙은 그 시각에 조회 가능한 값을 쓰는데, 둘이 같은지 **아무도 재지
    않았다.** 수치예보에서는 `previous_day1` 이 리드타임을 보존해 이 문제가
    없었지만(CLAUDE.md 4절), 대기질 API 에는 그 필드가 없다.

    그래서 추측 대신 기록한다 — 지금 조회한 값을 **조회 시각과 함께** 남기고,
    며칠 뒤 같은 시각을 아카이브에서 다시 받아 대조한다. 값이 같으면 상한이
    아니라 실측 이득이고, 다르면 그 차이가 곧 학습·서빙 분포 격차다.

    파일은 조회 시각(`fetched_at`)으로 키를 잡아 **덮어쓰지 않는다** — 같은
    유효시각을 여러 번 조회한 기록이 남아야 "언제 갱신됐는가"를 볼 수 있다.
    """
    from datetime import datetime, timedelta, timezone
    now = datetime.now(timezone.utc)
    # 최근 2일만 본다 — 갱신이 일어난다면 그 구간에서 일어난다.
    start = (now - timedelta(days=2)).strftime("%Y-%m-%d")
    end = now.strftime("%Y-%m-%d")
    fetched_at = now.strftime("%Y%m%d%H%M")
    points = {**STATION_COORDS, **UPSTREAM}

    snaps = {}
    if os.path.exists(SNAPSHOT_PATH):
        with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
            snaps = json.load(f)

    # **6시간 버킷당 한 번만 뜬다.** CI 는 15분마다 도는 잡에 얹혀 있어,
    # 시(hour)만 보고 거르면 같은 시간대의 네 번이 전부 통과한다 — 하루
    # 16회가 되고 보관 20회가 **30시간치**밖에 안 남아 며칠에 걸친 갱신을
    # 볼 수 없다. 이 측정의 목적이 정확히 그 '며칠'이므로 여기서 막는다
    # (호출부가 어떻게 부르든 스스로 보장한다).
    def _bucket(ts12):
        """'YYYYMMDDHHmm' → 6시간 버킷 키. **키 앞자리를 그대로 비교하면
        안 된다** — 키의 앞 10자리는 실제 조회 시(hour)라 버킷 시와 다르다
        (16시 조회의 버킷은 12시다). 첫 구현이 이 때문에 같은 버킷을 두 번
        떴고, 실제로 파일에 2회가 쌓여 드러났다."""
        return f"{str(ts12)[:8]}{int(str(ts12)[8:10]) // 6 * 6:02d}"

    bucket = _bucket(fetched_at)
    if any(_bucket(k) == bucket for k in snaps):
        print(f"에어로졸 스냅숏: {bucket}xx 버킷은 이미 있다 — 건너뛴다 "
              f"(보관 {len(snaps)}회)")
        return snaps

    for name, (lat, lon) in points.items():
        rows = _to_records(_request(lat, lon, start, end))
        snaps.setdefault(fetched_at, {})[name] = rows
        time.sleep(1)
    # 6시간마다 1회 × 20회 = 약 5일치. 드리프트는 그 안에서 드러난다.
    for k in sorted(snaps)[:-20]:
        snaps.pop(k)
    _save(SNAPSHOT_PATH, snaps)
    n = sum(len(v) for v in snaps[fetched_at].values())
    print(f"에어로졸 스냅숏: {fetched_at} · 지점 {len(points)}곳 · {n:,}시각 "
          f"(보관 {len(snaps)}회)")
    return snaps


def drift_report():
    """스냅숏끼리 대조해 **같은 유효시각의 값이 바뀌었는지** 본다."""
    if not os.path.exists(SNAPSHOT_PATH):
        raise SystemExit(f"스냅숏이 없다: {SNAPSHOT_PATH} — --snapshot 을 먼저 돌릴 것")
    with open(SNAPSHOT_PATH, "r", encoding="utf-8") as f:
        snaps = json.load(f)
    keys = sorted(snaps)
    if len(keys) < 2:
        raise SystemExit(f"스냅숏이 {len(keys)}회뿐이라 대조할 수 없다 — "
                         f"시간 간격을 두고 다시 돌릴 것")
    first, last = snaps[keys[0]], snaps[keys[-1]]
    n_cmp = n_diff = 0
    worst = (0.0, None)
    for name, rows in last.items():
        old_rows = first.get(name, {})
        for ts, vals in rows.items():
            if ts not in old_rows:
                continue
            n_cmp += 1
            a, b = old_rows[ts], vals
            if a != b:
                n_diff += 1
                gap = max(abs(float(x) - float(y))
                          for x, y in zip(a, b) if x is not None and y is not None)
                if gap > worst[0]:
                    worst = (gap, f"{name} {ts}")
    from datetime import datetime
    t0 = datetime.strptime(keys[0], "%Y%m%d%H%M")
    t1 = datetime.strptime(keys[-1], "%Y%m%d%H%M")
    gap_h = (t1 - t0).total_seconds() / 3600

    print(f"\n스냅숏 대조 — {keys[0]} vs {keys[-1]} (간격 {gap_h:.1f}시간)")
    print(f"  같은 유효시각 {n_cmp:,}개 중 값이 바뀐 것 {n_diff:,}개 "
          f"({n_diff / max(n_cmp, 1):.1%})")
    if n_diff:
        print(f"  최대 변화 {worst[0]:.2f} ({worst[1]})")
        print("  → 조회 시점에 따라 값이 달라진다. 학습(아카이브)과 서빙의 "
              "분포가 어긋나므로 `aerosol_gbm_check.py` 의 이득은 **상한**이다.")
        print("  (변화가 있다는 결론은 간격과 무관하게 성립한다 — 한 번이라도 "
              "바뀌었으면 바뀌는 것이다.)")
        return n_cmp, n_diff

    # **'안 바뀐다'는 결론에는 간격 조건이 붙는다.** CAMS 는 하루 두 번
    # 발표하므로, 그보다 짧은 간격에서 값이 같은 것은 당연하고 아무것도
    # 증명하지 않는다. 짧은 간격의 0% 를 "채택해도 된다"로 읽으면 정확히
    # 이 저장소가 반복해 경고한 오류(측정하지 않은 것을 주장한다)가 된다.
    MIN_GAP_H = 48
    if gap_h < MIN_GAP_H:
        print(f"  → **아직 결론이 아니다.** 간격이 {gap_h:.1f}시간뿐이라 "
              f"CAMS 발표 주기(하루 2회)보다 짧거나 비슷하다. "
              f"{MIN_GAP_H}시간 이상 벌어진 뒤 다시 볼 것.")
    else:
        print("  → 조회 시점과 무관하게 같은 값이다. 학습·서빙 분포가 일치하므로 "
              "측정된 이득을 그대로 기대할 수 있다.")
    return n_cmp, n_diff


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
    ap.add_argument("--snapshot", action="store_true",
                    help="지금 조회 가능한 값을 조회 시각과 함께 남긴다 "
                         "(학습·서빙 시간 정합성 측정용)")
    ap.add_argument("--drift", action="store_true",
                    help="스냅숏끼리 대조해 값이 갱신되는지 본다")
    args = ap.parse_args()
    if args.backfill:
        print(f"에어로졸 아카이브 백필 — CAMS {ARCHIVE_START}~현재, "
              f"관측소 {len(STATION_COORDS)}곳 + 상류 {len(UPSTREAM)}곳")
        backfill()
    elif args.snapshot:
        snapshot()
    elif args.drift:
        drift_report()
    else:
        ap.print_help()


if __name__ == "__main__":
    main()
