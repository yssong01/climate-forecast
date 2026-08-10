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


def _sf_or_none(s: str):
    """빈 값이면 None — '결측'과 '진짜 0'을 구분해야 하는 필드용.

    실측 사고(2026-08-07): 관측 자체가 결측인 시각(ASOS 장비 고장 등으로
    모든 필드가 빈 CSV 행)을 _sf() 로 채우면 "기온 20.0°C, 습도 50.0%,
    풍속 1.5m/s"라는 그럴듯한 값이 나오고, status="SUCCESS_LIVE"로 저장돼
    실제 관측과 구분이 안 된 채 학습·검증에 섞여 들어갔다. 학습된 모델의
    검증셋 최대 오차 사례 상위권이 전부 이 패턴이었다(예측 20°C 근처 vs
    실측 -6~-7°C인 1월 사례들 — 모델이 이 오염된 20.0 을 일부 학습한
    부작용으로 보인다). 328,802개 중 206개(0.063%)가 이 패턴으로 확인됨.
    """
    s = (s or "").strip()
    if not s:
        return None
    try:
        return float(s)
    except ValueError:
        return None


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

        # 핵심 필드가 결측이면 레코드 전체를 버린다 — 기본값으로 채워
        # "그럴듯한 가짜 관측"을 만들지 않는다. 강수량만 예외: 빈 문자열이
        # ASOS 관례상 "무강수(0)"를 뜻하므로 결측이 아니다(_sf 그대로 사용).
        temp = _sf_or_none(row.get("기온(°C)"))
        humid = _sf_or_none(row.get("습도(%)"))
        wind = _sf_or_none(row.get("풍속(m/s)"))
        pres = _sf_or_none(row.get("해면기압(hPa)"))
        if None in (temp, humid, wind, pres):
            continue

        records.append({
            "timestamp":     dt.strftime("%Y%m%d%H00"),
            "temperature":   temp,
            "precipitation": max(0.0, _sf(row.get("강수량(mm)"), 0.0)),
            "humidity":      humid,
            "wind_speed":    wind,
            "wind_dir":      _sf(row.get("풍향(16방위)"), 0.0),
            "pressure":      pres,
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
