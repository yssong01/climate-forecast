"""
import_weather_issues.py — 날씨 이슈별 데이터(폭염·한파·황사)를 파싱해
(관측소, 날짜) → 공식 라벨 조회표로 변환한다.

왜 필요한가: 기존 방식은 폭염/한파를 "그 순간 기온이 임계값을 넘었는가"로
근사했다(2026-08-08 구조 점검에서 지적된 "라벨 정의 근사 오차"). 이 파일들은
기상청이 실제로 발표한 폭염특보/한파특보 여부(O/X)를 일 단위로 담고 있어
그 근사 자체를 없앨 수 있다. 황사는 날씨이슈별 데이터(YDST 이벤트 리포트)가
2019~2020년치뿐이라 우리 학습 기간(2023~2026)과 아예 안 겹쳐 실질적으로
못 쓴다(2026-08-09 실측, 훈련 표본 0/1개) — 대신 황사관측(PM10) 연속
시계열(2023~2025)에서 임계값으로 직접 라벨을 만든다(parse_dust_pm10).

파일 형식: EUC-KR 인코딩, 탭구분 텍스트(확장자만 .xls — 진짜 바이너리 아님).
  폭염: 일시, 지점(코드), 폭염여부, 최고체감온도, ..., 폭염특보(O/X), ...
  한파: 일시, 지점(코드), ..., 한파특보(O/X), ...
  황사(YDST 이벤트, 2019~2020뿐이라 사실상 미사용): 일자, 지점번호, 지점명,
    황사관측여부(O/X), 평균미세먼지농도(PM10), ...
  황사관측(PM10, 실제로 쓰는 것): 지점, 일시, 1시간평균 미세먼지농도(㎍/㎥)
    — CSV, 콤마구분, EUC-KR

관측소 코드 갭(2026-08-09 확인): 날씨이슈별 데이터(162개 지점)엔 우리 12개
중 춘천(101)·강릉(105)이 없고 대신 인근 북춘천(93)·북강릉(104)이 있어
리매핑한다. 제주(184)는 처음엔 없는 줄 알았으나 재확인 결과 있다(EUC-KR
grep 실수였음, 2240일 커버 확인). PM10 관측망은 완전히 다른 문제 — 28개
지점 중 우리 12개와 겹치는 건 서울·수원·전주·광주·대구·춘천(북춘천 대체)
6개뿐이고, 나머지 6개(인천·강릉·대전·청주·부산·제주)는 관측망 자체에 없다.
사용자 결정(2026-08-09): 불연속 데이터를 억지로 프록시로 메우지 않고
그 6개는 라벨 없이 제외한다.

실행: python import_weather_issues.py
출력: ./cache/weather_issue_labels.json
  구조: { "<우리 관측소 코드>": { "YYYY-MM-DD": {
            "heatwave_advisory": 0/1 또는 없음,
            "coldwave_advisory": 0/1 또는 없음,
            "dust_observed":     0/1 또는 없음(PM10 6개 관측소만) } } }
"""
import glob
import json
import os
import re

STN_REMAP = {"93": "101", "104": "105"}   # 북춘천→춘천, 북강릉→강릉 근사
OUR_STATIONS = {"108", "112", "119", "101", "105", "133", "131",
                "146", "156", "143", "159", "184"}

HW_DIR   = "./dawnload/날씨이슈별데이터_폭염"
CW_DIR   = "./dawnload/날씨이슈별데이터_한파"
YDST_DIR = "./dawnload/날씨이슈별데이터_황사"
PM10_DIR = "./dawnload/기상관측_지상_황사관측(PM10)"
OUT_FILE = "./cache/weather_issue_labels.json"

# PM10 임계값 — 공식 황사주의보(400㎍/㎥ 2시간 지속)는 우리 6개 관측소 3년치에서
# 표본이 24건(0.016%)뿐이라 학습 자체가 불가능하다(2026-08-09 실측). 대신
# 대기환경지수 "매우나쁨" 등급 기준인 150㎍/㎥을 쓴다 — 815건(0.56%)로
# 한파(0.17~2.77%)와 비슷한 규모. "공식 황사특보"가 아니라 "미세먼지 매우
# 나쁨 수준"으로 정의를 낮춘 것임을 명시한다(폭염/한파의 임계값 근사와 같은
# 종류의 단순화).
PM10_THRESH = 150.0
PM10_MATCHED_RAW = {"108", "119", "146", "156", "143", "93"}  # 우리 12개 중 PM10 관측망에 있는 원본 코드


def _read_tsv(path: str) -> list[dict]:
    # errors="replace" 는 유지하되(바이트 하나 때문에 전체가 죽지 않도록)
    # 치환이 실제로 일어났는지 **헤더에서** 반드시 확인한다. 이 파서는
    # 컬럼을 이름으로 찾으므로(r.get("폭염특보(O/X)") 등), 헤더가 깨지면
    # 모든 조회가 조용히 None 을 돌려주고 행이 통째로 스킵된다 — 예외도
    # 로그도 없이 표본만 줄어든다(2026-08-29 총점검에서 가시화).
    with open(path, encoding="euc-kr", errors="replace") as f:
        header = next(f).rstrip("\n").split("\t")
        if any("�" in h for h in header):
            raise ValueError(
                f"헤더 디코딩 실패(EUC-KR 아님?): {os.path.basename(path)} — "
                f"깨진 컬럼 {[h for h in header if chr(0xFFFD) in h]}")
        rows = []
        for line in f:
            parts = line.rstrip("\n").split("\t")
            if len(parts) != len(header):
                continue
            rows.append(dict(zip(header, parts)))
        return rows


def _stn_code(raw: str) -> str | None:
    """'북강릉(104)' → '105' (리매핑) 또는 원본 코드. 우리 목록 밖이면 None."""
    m = re.search(r"\((\d+)\)", raw)
    code = m.group(1) if m else raw.strip()
    code = STN_REMAP.get(code, code)
    return code if code in OUR_STATIONS else None


def parse_heatwave(out: dict) -> int:
    n = 0
    for fp in glob.glob(os.path.join(HW_DIR, "ISSUE_HW_DAY_*.xls")):
        for r in _read_tsv(fp):
            stn = _stn_code(r.get("지점", ""))
            date = r.get("일시", "").strip()
            flag = r.get("폭염특보(O/X)", "").strip()
            if not stn or not date or flag not in ("O", "X"):
                continue
            out.setdefault(stn, {}).setdefault(date, {})["heatwave_advisory"] = int(flag == "O")
            n += 1
    return n


def parse_coldwave(out: dict) -> int:
    """
    컬럼명은 "한파특보(O/X)"지만 실제 값은 이진(O/X)이 아니라 3단계
    (X/주의보/경보)다(2026-08-09 실측, 187,554개 레코드 전수 확인 — 'O'는
    단 한 번도 등장하지 않고 X 172,034 / 주의보 11,724 / 경보 3,796). 컬럼명만
    보고 O/X로 파싱하면 전부 음성으로 잘못 읽힌다. 주의보·경보 둘 다
    "특보 발효"로 취급해 1로 매핑한다.
    """
    n = 0
    for fp in glob.glob(os.path.join(CW_DIR, "ISSUE_CW_DAY_*.xls")):
        for r in _read_tsv(fp):
            stn = _stn_code(r.get("지점", ""))
            date = r.get("일시", "").strip()
            raw = r.get("한파특보(O/X)", "").strip()
            if not stn or not date or not raw:
                continue
            out.setdefault(stn, {}).setdefault(date, {})["coldwave_advisory"] = int(raw != "X")
            n += 1
    return n


def parse_dust(out: dict) -> int:
    """
    황사는 이벤트가 발생한 날짜만 폴더가 존재한다 — 즉 폴더 자체가 없는 날짜는
    "그 어떤 관측소에서도 황사가 관측되지 않은 날"이라는 확정적 음성이다
    (train.py 쪽에서 조회 실패 시 0으로 채워야 하는 이유 — 결측이 아니라
    확정 음성). 여기서는 이벤트가 있었던 날짜의 관측소별 O/X만 채운다.
    """
    n = 0
    all_xls = glob.glob(os.path.join(YDST_DIR, "**", "ISSUE_YDST_YY_*.xls"), recursive=True)
    # _SNDE/_WSPF 변형이나 CHRT/SATI 하위 폴더 파일은 제외 — 폴더명과 파일명(확장자 제외)이
    # 정확히 같은 것만 "메인" 관측표다.
    main_files = [f for f in all_xls
                 if os.path.basename(f)[:-4] == os.path.basename(os.path.dirname(f))]
    for fp in main_files:
        for r in _read_tsv(fp):
            stn = _stn_code(r.get("지점번호", ""))
            date = r.get("일자", "").strip()
            flag = r.get("황사관측여부(O/X)", "").strip()
            if not stn or not date or flag not in ("O", "X"):
                continue
            out.setdefault(stn, {}).setdefault(date, {})["dust_observed"] = int(flag == "O")
            n += 1
    return n


def parse_dust_pm10(out: dict) -> int:
    """
    황사관측(PM10) 시간자료 — 실제로 학습에 쓰이는 황사 라벨 소스(parse_dust
    의 YDST 이벤트 리포트는 2019~2020년치뿐이라 우리 학습 기간과 안 겹침).

    지점(원본 코드, 이름 없음) + 일시(시간별) + 1시간평균 미세먼지농도.
    관측소당 하루 여러 시간대 값이 있으므로 그날의 최댓값이 PM10_THRESH를
    넘으면 그날을 dust_observed=1 로 표시한다(폭염/한파와 같은 "그 날 하루"
    단위 라벨 스타일).

    PM10_MATCHED_RAW 에 없는 원본 코드(우리 12개 중 6개는 이 관측망 자체가
    없음)는 무시한다 — 조회 실패 시 값을 0으로 채우지 않는다(사용자 결정:
    불연속 데이터를 프록시로 메우지 않고 제외, 2026-08-09).
    """
    daily_max: dict[tuple[str, str], float] = {}
    for fp in glob.glob(os.path.join(PM10_DIR, "ENV_YDST_*_HR_*.csv")):
        with open(fp, encoding="euc-kr", errors="replace") as f:
            # 이 파서는 컬럼을 위치(parts[0..2])로 읽어 헤더 이름에 의존하지
            # 않지만, 헤더가 깨졌다는 것은 파일 전체의 인코딩 가정이 틀렸다는
            # 신호다 — 값(관측소 코드·시각)도 함께 깨졌을 수 있으므로 조용히
            # 넘기지 않고 알린다(2026-08-29 추가).
            _hdr = next(f, None)
            if _hdr and "�" in _hdr:
                print(f"  [경고] 헤더 디코딩 실패(EUC-KR 아님?) — 건너뜀: "
                      f"{os.path.basename(fp)}")
                continue
            for line in f:
                parts = line.rstrip("\n").split(",")
                if len(parts) != 3:
                    continue
                raw_stn, ts, val = parts
                if raw_stn not in PM10_MATCHED_RAW:
                    continue
                stn = STN_REMAP.get(raw_stn, raw_stn)
                if stn not in OUR_STATIONS:
                    continue
                v = val.strip()
                if not v or v == "-":
                    continue
                try:
                    pm10 = float(v)
                except ValueError:
                    continue
                date = ts[:10]   # "YYYY-MM-DD HH:MM" → "YYYY-MM-DD"
                key = (stn, date)
                if pm10 > daily_max.get(key, -1.0):
                    daily_max[key] = pm10

    n = 0
    for (stn, date), maxval in daily_max.items():
        out.setdefault(stn, {}).setdefault(date, {})["dust_observed"] = int(maxval >= PM10_THRESH)
        n += 1
    return n


def main():
    out: dict = {}
    n_hw = parse_heatwave(out)
    n_cw = parse_coldwave(out)
    n_yd_event = parse_dust(out)          # 2019~2020뿐 — 우리 기간과 안 겹쳐 사실상 무효
    n_yd_pm10  = parse_dust_pm10(out)     # 실제로 쓰이는 것 (2023~2025, 6개 관측소)

    os.makedirs("./cache", exist_ok=True)
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    print(f"폭염 레코드 {n_hw:,}개 / 한파 레코드 {n_cw:,}개 / "
          f"황사(이벤트,2019~2020) {n_yd_event:,}개 / 황사(PM10,2023~2025) {n_yd_pm10:,}개 파싱")
    print(f"관측소 {len(out)}개 커버")
    for stn in sorted(out):
        n_days = len(out[stn])
        n_hw_o = sum(1 for d in out[stn].values() if d.get("heatwave_advisory") == 1)
        n_cw_o = sum(1 for d in out[stn].values() if d.get("coldwave_advisory") == 1)
        n_yd_o = sum(1 for d in out[stn].values() if d.get("dust_observed") == 1)
        print(f"  {stn}: {n_days}일 커버 (폭염특보 {n_hw_o}일 / 한파특보 {n_cw_o}일 / 황사관측 {n_yd_o}일)")
    print(f"저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
