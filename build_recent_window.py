"""
build_recent_window.py — 배포용 경량 이력 캐시 생성.

app.py 의 load_merged_history() 는 사이드바 스파크라인(최근 72시간)에만
쓰이는데, 기존 구현은 historical_data_1y.json 전체(3.5년치, 32만 레코드,
64MB)를 매번 파싱해 메모리에 올렸다 — 실측(docker stats) RAM 1.55GiB 중
가장 큰 절감 여지였다. 이 스크립트는 최근 WINDOW_DAYS 일치만 추려
recent_window.json 으로 저장한다. 배포 이미지/저장소에는 이 파일만
포함시키고, historical_data_1y.json(학습용 원본)은 제외한다.

재실행 시점: collect_incremental.py 로 새 데이터를 채운 뒤, 배포 저장소를
갱신하기 전에 다시 돌린다. 완전 자동화(cron 등)는 범위 밖 — 포트폴리오
데모는 갱신 주기가 매일 단위여도 충분하다.

실행: python build_recent_window.py
"""
import json
import os
from datetime import datetime, timedelta

from collect_year import atomic_save

SRC_FILES = ["./cache/historical_data_1y.json", "./cache/historical_data.json"]
OUT_FILE  = "./cache/recent_window.json"
WINDOW_DAYS = 10   # recent_series() 가 쓰는 72시간 + 여유


def main():
    cutoff = datetime.now() - timedelta(days=WINDOW_DAYS)
    merged = {}
    total_read = 0
    for path in SRC_FILES:
        if not os.path.exists(path):
            continue
        with open(path, "r", encoding="utf-8") as f:
            records = json.load(f)
        total_read += len(records)
        for r in records:
            # 실측만 통과시킨다(collect_year.load_existing 과 같은 기준).
            # 이 파일은 배포판이 직접 읽는 폴백 원본이라, 실측이 아닌 값이
            # 섞이면 그대로 화면에 나간다 — "폴백 경로도 가짜 데이터 금지
            # 적용 대상"(CLAUDE.md 4절). 현재 쓰기 경로들이 모두 SUCCESS_LIVE
            # 만 저장하므로 지금 걸리는 레코드는 없지만, 이 방어선이 여기만
            # 없었다(2026-08-29 총점검에서 추가).
            if r.get("status") != "SUCCESS_LIVE":
                continue
            ts = str(r.get("timestamp", ""))[:12]
            try:
                dt = datetime.strptime(ts, "%Y%m%d%H%M")
            except ValueError:
                continue
            if dt >= cutoff:
                merged[(r.get("stn"), r["timestamp"])] = r

    out = list(merged.values())
    # 고유 tmp + os.replace 로 원자적으로 쓴다(CLAUDE.md 1절 6항) — 같은
    # 파일을 refresh_deploy_data.py 는 이미 atomic_save 로 쓰는데 이 경로만
    # 직접 덮어쓰고 있었다. 도중에 죽으면 잘린 JSON 이 남아 배포판의
    # 스파크라인과 최신성 게이트가 함께 깨진다.
    atomic_save(OUT_FILE, out)

    src_size = sum(os.path.getsize(p) for p in SRC_FILES if os.path.exists(p))
    out_size = os.path.getsize(OUT_FILE)
    print(f"원본 {total_read:,}개 ({src_size/1e6:.1f}MB) → "
          f"최근 {WINDOW_DAYS}일 {len(out):,}개 ({out_size/1e6:.2f}MB)")
    print(f"저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
