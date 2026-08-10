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
            ts = str(r.get("timestamp", ""))[:12]
            try:
                dt = datetime.strptime(ts, "%Y%m%d%H%M")
            except ValueError:
                continue
            if dt >= cutoff:
                merged[(r.get("stn"), r["timestamp"])] = r

    out = list(merged.values())
    with open(OUT_FILE, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False)

    src_size = sum(os.path.getsize(p) for p in SRC_FILES if os.path.exists(p))
    out_size = os.path.getsize(OUT_FILE)
    print(f"원본 {total_read:,}개 ({src_size/1e6:.1f}MB) → "
          f"최근 {WINDOW_DAYS}일 {len(out):,}개 ({out_size/1e6:.2f}MB)")
    print(f"저장: {OUT_FILE}")


if __name__ == "__main__":
    main()
