"""
import_kma_fileset.py — 기상자료개방포털 "파일셋" CSV → 프로젝트 내부 스키마.

apihub 쿼터 문제를 완전히 우회하는 경로: 기상자료개방포털(data.kma.go.kr)
"종관기상관측(ASOS) → 파일셋" 메뉴에서 받은 CSV를 그대로 학습 캐시로
합친다. 이 경로는 apihub.kma.go.kr 를 전혀 거치지 않으므로 오늘 겪은
쿼터/차단과 완전히 무관하다.

CSV 실측 형식 (2026-08-07, 관측소 101 2023년 샘플로 확인):
  인코딩   : EUC-KR
  헤더     : 지점,일시,기온(°C),강수량(mm),풍속(m/s),풍향(16방위),습도(%),...
  일시     : "YYYY-MM-DD HH:MM" (매시 정각)
  강수량   : 빈 문자열 = 무강수(0.0) — 결측이 아니라 관측값 자체가 0
  풍향     : 이미 도(degree) 단위 — apihub WD 필드와 동일해 추가 변환 불필요

apihub 필드와의 대응 (weather_collector._parse_asos 와 동일 의미로 맞춤):
  지점→stn, 기온→temperature, 강수량→precipitation, 풍속→wind_speed,
  풍향→wind_dir, 습도→humidity, 해면기압→pressure(PS 와 동일 개념 —
  현지기압이 아니라 해면기압을 쓴다. 두 컬럼을 헷갈리면 관측소 고도차만큼
  체계적 오차가 생긴다)

현지 12개 관측소 외 지점은 조용히 건너뛴다 — "전체" 선택으로 105개 관측소
파일을 통째로 받아도 안전하게 필요한 것만 걸러진다.

실행:
    python import_kma_fileset.py 다운로드폴더/*.csv
    python import_kma_fileset.py 단일파일.csv --out ./cache/historical_data_1y.json
"""
import argparse
import csv
import glob
import sys
from datetime import datetime

from weather_collector import STATIONS
from collect_year import load_existing, atomic_save, OUT_FILE, LEGACY_CACHE

VALID_STNS = set(STATIONS.values())


def _sf(s: str, default: float) -> float:
    s = (s or "").strip()
    if not s:
        return default
    try:
        return float(s)
    except ValueError:
        return default


def parse_file(path: str) -> list:
    """CSV 1개 → 내부 레코드 리스트. 인코딩은 EUC-KR로 고정 시도, 실패하면
    UTF-8도 시도한다(포털이 형식을 바꿀 가능성에 대비)."""
    for enc in ("euc-kr", "utf-8-sig", "utf-8"):
        try:
            with open(path, encoding=enc) as f:
                text = f.read()
            break
        except UnicodeDecodeError:
            continue
    else:
        print(f"  [건너뜀] 인코딩 판별 실패: {path}")
        return []

    reader = csv.DictReader(text.splitlines())
    records = []
    for row in reader:
        stn = (row.get("지점") or "").strip()
        if stn not in VALID_STNS:
            continue
        try:
            dt = datetime.strptime(row["일시"].strip(), "%Y-%m-%d %H:%M")
        except (ValueError, KeyError):
            continue

        records.append({
            "timestamp":     dt.strftime("%Y%m%d%H00"),
            "temperature":   _sf(row.get("기온(°C)"), 20.0),
            "precipitation": max(0.0, _sf(row.get("강수량(mm)"), 0.0)),
            "humidity":      _sf(row.get("습도(%)"), 50.0),
            "wind_speed":    _sf(row.get("풍속(m/s)"), 1.5),
            "wind_dir":      _sf(row.get("풍향(16방위)"), 0.0),
            "pressure":      _sf(row.get("해면기압(hPa)"), 1013.0),
            "precip_type":   0,   # apihub 경로도 항상 0 — WW 코드 매핑 전 상태와 동일
            "stn":           stn,
            "status":        "SUCCESS_LIVE",
        })
    return records


def main():
    ap = argparse.ArgumentParser(description="기상자료개방포털 파일셋 CSV 가져오기")
    ap.add_argument("files", nargs="+", help="CSV 경로(들) — 와일드카드 가능")
    ap.add_argument("--out", default=OUT_FILE)
    args = ap.parse_args()

    paths = []
    for pattern in args.files:
        matched = glob.glob(pattern)
        paths.extend(matched if matched else [pattern])
    if not paths:
        print("[ERROR] 대상 파일이 없습니다.", file=sys.stderr)
        sys.exit(1)

    existing = {}
    existing.update(load_existing(LEGACY_CACHE))
    existing.update(load_existing(args.out))
    before = len(existing)

    per_stn_new = {}
    for path in paths:
        recs = parse_file(path)
        for r in recs:
            key = (r["stn"], r["timestamp"])
            if key not in existing:
                per_stn_new[r["stn"]] = per_stn_new.get(r["stn"], 0) + 1
            existing[key] = r
        print(f"  {path}: {len(recs)}행 파싱")

    atomic_save(args.out, list(existing.values()))

    added = len(existing) - before
    print(f"\n완료 — 총 {len(existing)}개 레코드 (신규 {added}개). 저장: {args.out}")
    if per_stn_new:
        names = {v: k for k, v in STATIONS.items()}
        for stn, n in sorted(per_stn_new.items()):
            print(f"  {names.get(stn, stn)}({stn}): +{n}")


if __name__ == "__main__":
    main()
