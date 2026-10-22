"""
migrate_training_wind_dir.py — 학습 캐시에 섞여 든 API 경로 레코드의 풍향을 도 단위로 고친다.

**왜 필요한가(2026-10-21 사고의 학습 쪽 절반).** API허브 ASOS 의 풍향 `WD` 는 10도
단위(0~36)이고 포털 CSV 는 도 단위(0~360)다. 서빙 쪽은 2026-10-21 에 고쳤지만
(`weather_collector._parse_asos` ×10 + `wind_dir_unit` 표식), 학습 캐시
(`cache/historical_data_1y.json`)에는 `collect_incremental.py` 가 API 로 이어 붙인
레코드가 **10도 단위 그대로** 들어 있다. 출처 필드가 없어(전부 `SUCCESS_LIVE`)
값만으로는 가를 수 없다 — 20 은 도 단위(북북동)와 10도 단위(200°) 양쪽에 있다.

**경계는 시각으로 정한다(2026-10-22 실측).** 12개 관측소 모두 포털 자료가
2025-12-31 23시(KST)에서 끝나고, 2026-01-01 이후 17,172건은 **한 건도 36 을
넘지 않으며** 값 집합이 API 의 17개(0·2·5·7·…·36)와 정확히 같다 — 도 단위
16방위 자료가 1만 7천 건 동안 36 이하에만 머물 수는 없다. 반대로 경계 앞의
포털 구간에는 36 이하 값이 0·20(도 단위 정북·북북동)과 2014년의 19건뿐이다
(2,7,9,23 등 — 한 시각에 여러 관측소가 함께 나타나 포털 원자료의 특이값으로
보이며, API 수집 이전이라 이 도구는 건드리지 않는다).

**지금 적용하지 않는 이유.** 배포 모델은 이 캐시로 학습됐고, 검증셋에도 2026년
표본이 들어 있다. 캐시를 고치면 그 표본의 특징이 바뀌어 체크포인트에 저장된 검증
지표가 더는 재현되지 않는다(`precip_gbm --stamp-norm` 같은 재현 대조가 깨진다).
**다음 재학습 직전에 `--apply` 한다** — `train.train()` 이 이행되지 않은 레코드가
있으면 학습을 시작하지 않는다(`assert_migrated`).

실행:
    python migrate_training_wind_dir.py           # 대상 건수·값 분포만 출력
    python migrate_training_wind_dir.py --apply   # 백업 후 ×10 + 표식, 원자적 저장
"""
import argparse
import collections
import json
import os
import shutil
import sys
import time

from weather_collector import WIND_DIR_UNIT, _API_WD_MAX

CACHE_PATH = "./cache/historical_data_1y.json"
# 포털 자료의 마지막 시각 다음 — 이 시각 이후는 전부 API 경로로 들어왔다(모듈 docstring).
API_START = "202601010000"


def unmigrated(records) -> list:
    """API 구간에 있으면서 아직 표식이 없는 레코드."""
    return [r for r in records
            if str(r.get("timestamp", ""))[:12] >= API_START and r.get("wind_dir_unit") is None]


def assert_migrated(records) -> None:
    """학습 전에 부른다 — 이행되지 않은 레코드가 있으면 멈춘다."""
    n = len(unmigrated(records))
    if n:
        raise SystemExit(
            f"[중단] 학습 캐시에 10도 단위 풍향 레코드 {n:,}건이 남아 있다(API 경로, "
            f"{API_START[:8]} 이후). 서빙은 도 단위라 이대로 학습하면 학습·서빙이 어긋난다.\n"
            f"  python migrate_training_wind_dir.py --apply  후 다시 실행할 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--apply", action="store_true")
    ap.add_argument("--path", default=CACHE_PATH)
    args = ap.parse_args()

    with open(args.path, "r", encoding="utf-8") as f:
        records = json.load(f)
    todo = unmigrated(records)
    vals = collections.Counter(r["wind_dir"] for r in todo)
    print(f"전체 {len(records):,}건 · 이행 대상 {len(todo):,}건({API_START[:8]} 이후, 표식 없음)")
    print(f"  대상 풍향 값 {len(vals)}종: {sorted(vals)}")
    over = [r for r in todo if float(r["wind_dir"]) > _API_WD_MAX]
    if over:
        # 경계 뒤에 도 단위 값이 있으면 경계 가정이 깨진 것이다 — 고치지 않는다.
        raise SystemExit(f"[중단] 대상 중 {len(over):,}건이 {_API_WD_MAX:.0f} 을 넘는다 — "
                         f"도 단위 레코드가 섞여 있어 시각 경계를 믿을 수 없다.")
    if not args.apply:
        print("(--apply 를 주지 않아 파일은 그대로다)")
        return 0
    if not todo:
        print("이행할 레코드가 없다.")
        return 0

    backup = args.path.replace(".json", time.strftime("_before_wd_%Y%m%d.json"))
    shutil.copy2(args.path, backup)
    print(f"백업: {backup}")
    for r in todo:
        r["wind_dir"] = float(r["wind_dir"]) * 10.0
        r["wind_dir_unit"] = WIND_DIR_UNIT
    from collect_year import atomic_save
    atomic_save(args.path, records)
    left = len(unmigrated(records))
    print(f"이행 완료 {len(todo):,}건 · 남은 대상 {left}건 → {args.path}")
    return 0 if left == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
