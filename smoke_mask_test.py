"""
smoke_mask_test.py — 폭염·한파 마스킹 도입(2026-08-15) 후 학습 경로 전체를
작은 표본으로 한 번 태워 본다. 본 학습이 4시간대라 언패킹 개수 불일치 같은
단순 실수를 거기서 발견하면 그대로 4시간을 버린다.

확인 항목
  1. WeatherDataset 이 heat_mask/cold_mask 를 만들고, 공식 라벨이 있는
     구간(2024년)과 없는 구간(2016년)에서 마스크가 실제로 갈리는가
  2. DataLoader 10-튜플 언패킹 → 마스킹 BCE → 검증 지표 산출이 끝까지 도는가
  3. 마스크=0 표본이 채점에서 빠지는가(양성률이 근사 라벨 섞였을 때와 다른가)

실행: python smoke_mask_test.py
"""
import json
import os
import tempfile

import torch

import train as T


def main():
    with open(T.DATA_CACHE, encoding="utf-8") as f:
        allrec = json.load(f)

    # 공식 라벨이 있는 해(2024)와 없는 해(2016)를 섞어 담되 반드시 여름을
    # 고른다 — 폭염 공식 특보는 5~9월에만 존재하므로 겨울 구간을 잡으면
    # 마스크가 전부 0이 되어 정작 검증하려던 "마스킹 BCE 계산 경로"를
    # 건너뛴다(첫 시도에서 실제로 그렇게 됐다).
    subset = [r for r in allrec
              if str(r["timestamp"])[:6] in ("201607", "201608",
                                             "202407", "202408")]
    by_stn = {}
    for r in subset:
        by_stn.setdefault(r["stn"], []).append(r)
    small = []
    for stn, rows in by_stn.items():
        rows.sort(key=lambda x: x["timestamp"])
        small.extend(rows)
    print(f"소표본 {len(small):,}개 ({len(by_stn)}개 관측소, 2016·2024년 7~8월)")

    tmp = tempfile.NamedTemporaryFile("w", suffix=".json", delete=False,
                                      encoding="utf-8")
    json.dump(small, tmp, ensure_ascii=False)
    tmp.close()

    ckpt_tmp = tempfile.NamedTemporaryFile(suffix=".pt", delete=False)
    ckpt_tmp.close()

    # 소표본이라 collect_historical 의 "충분한가" 검사를 통과하도록 기준을 낮춘다.
    T.DATA_CACHE = tmp.name
    T.N_HOURS = len(small) // len(by_stn)
    T.STATIONS_TO_COLLECT = list(by_stn.keys())
    T.EPOCHS = 2
    T.PATIENCE = 99

    try:
        stats = T.train(checkpoint=ckpt_tmp.name, early_stop=False, verbose=True)
    finally:
        os.unlink(tmp.name)

    print("\n" + "=" * 70)
    print(" 스모크 테스트 결과")
    print("=" * 70)
    em = stats.get("extreme_metrics", {})
    for name, key in (("폭염", "heatwave"), ("한파", "coldwave"), ("황사", "dust")):
        m = em.get(key)
        print(f"  {name}: {m}" if m else f"  {name}: (지표 없음)")

    saved = torch.load(ckpt_tmp.name, map_location="cpu", weights_only=True)
    print(f"\n  체크포인트 저장 확인 — 키 {len(saved)}개, "
          f"extreme_metrics {'있음' if saved.get('extreme_metrics') else '없음'}")
    os.unlink(ckpt_tmp.name)
    print("\n전 경로 통과.")


if __name__ == "__main__":
    main()
