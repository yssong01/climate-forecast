"""
record_online_forecasts.py — 적중률 로그에 **온라인 예측**을 기록한다.

**왜 필요한가(2026-09-23).** '성능 검증' 탭의 '출력값 적중률'은 배포 화면에서
사실상 영구히 비어 있었다. 원인은 기록을 만드는 주체가 배포 앱뿐이었다는
것이다 — Streamlit Community Cloud 의 파일시스템은 휘발성이라, 앱이
`accuracy.record_prediction()` 으로 남긴 항목은 컨테이너가 재시작하면
사라지고 저장소 상태로 되돌아간다. `refresh-data.yml` 은 **대기 항목을
실측과 대조(resolve)** 할 뿐 새 예측을 만들지 않으므로, 대조할 대기 항목
자체가 저장소에 쌓이지 않았다. 화면은 그 사실을 모른 채 "영구 누적은
GitHub Actions 가 담당한다"고 설명하고 있었다.

이 스크립트가 그 빠진 고리다. CI 가 관측 창을 갱신한 직후에 실행되어,
**그 시점에 조회 가능한 자료만으로** 12개 관측소의 +6시간 출력값을 만들고
로그에 남긴다. 다음 실행들이 목표 시각의 실측이 들어오는 대로 대조한다.

**API 를 한 건도 호출하지 않는다.** `KMA_API_KEY` 를 비워 두면
`RobustWeatherCollector` 가 곧장 폴백 경로로 내려가고, 폴백 재료로 등록한
저장소 창(`cache/recent_window.json`)의 실측을 쓴다. 이 창은 바로 앞
단계가 방금 갱신한 것이다. 관측소 12곳이 같은 시각으로 채워지므로
Re축(공간 보간)의 "같은 시각 스냅샷" 조건도 그대로 만족한다. 예측 1건이
API 15회를 쓰는 경로를 그대로 뒀다면 실행마다 180회, 하루 17,000회가 되어
이 인증키가 차단됐던 지점(누적 약 9,800건)을 훌쩍 넘는다.

**백테스트가 아니다.** CLAUDE.md 11항은 적중률 로그에 백테스트·시딩 결과를
합치지 말라고 정한다. 여기서 만드는 것은 미래 시각에 대한 예측이며(목표
시각의 실측은 아직 존재하지 않는다), 기록 시점에 실제로 조회 가능했던
자료만 입력으로 쓴다 — 규약이 말하는 "실시간 온라인 기록" 그 자체다.
구분이 필요할 때를 대비해 `source="ci"` 로 남기고(앱이 남기는 것은
`"live"`), 같은 규약이 경계한 파일 비대는 `accuracy.trim()` 으로 막는다.
관측이 매시 정각 1회뿐이라 같은 정시 안의 반복 실행은 목표 시각이 같아
`record_prediction()` 의 중복 방지에 자동으로 걸린다(하루 약 288건).

실행: python record_online_forecasts.py [--dry-run] [--retain-days 30]
"""
import argparse
import os
import sys

import accuracy
from weather_collector import (
    STATIONS, set_offline_fallback, is_real_observation, api_call_stats,
)


def force_offline() -> None:
    """이 프로세스의 모든 관측 조회를 저장소 창으로 내린다.

    `RobustWeatherCollector` 는 **인스턴스를 만들 때** `KMA_API_KEY` 를
    읽고, 키가 비어 있으면 `fetch`/`fetch_at` 이 첫 줄에서 폴백으로 빠져
    네트워크에 닿지 않는다. 그래서 컬렉터가 하나라도 만들어지기 전에만
    부르면 되고, 모듈 임포트 시점일 필요는 없다 — 임포트만으로 남의
    환경을 바꾸는 모듈은 다른 곳(연기 시험 등)에서 임포트될 때 조용한
    사고를 만든다.
    """
    os.environ["KMA_API_KEY"] = ""
    # 키를 일부러 비웠다는 표식 — 컬렉터가 "설정을 빠뜨렸다"는 경고를
    # 관측소 12곳 × 조회 12회마다 찍는 것을 막는다(동작은 바뀌지 않는다).
    os.environ["KMA_OFFLINE_ONLY"] = "1"

WINDOW_PATH = "./cache/recent_window.json"
# 배포 경로는 **이 파일이 있는 디렉터리** 기준으로 찾는다. 워크플로가 코드와
# 데이터를 서로 다른 디렉터리에 체크아웃하고 데이터 쪽을 작업 디렉터리로
# 쓰기 때문에(`working-directory: data` + `PYTHONPATH=code`), 상대경로
# "./checkpoints/..." 는 데이터 쪽에서 찾게 되어 존재하지 않는다.
DEFAULT_CHECKPOINT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "checkpoints",
    "numerical_trichef.pt")


def load_window(path: str) -> list:
    import json
    with open(path, "r", encoding="utf-8") as f:
        records = json.load(f)
    if not isinstance(records, list) or not records:
        raise ValueError(f"{path} 가 비었거나 목록이 아니다")
    return records


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--checkpoint", default=os.getenv("CHECKPOINT_PATH")
                    or DEFAULT_CHECKPOINT)
    ap.add_argument("--window", default=WINDOW_PATH)
    ap.add_argument("--retain-days", type=int, default=accuracy.RETAIN_DAYS)
    ap.add_argument("--dry-run", action="store_true",
                    help="예측만 해보고 로그에 쓰지 않는다")
    args = ap.parse_args()

    force_offline()

    if not os.path.exists(args.checkpoint):
        print(f"::error::체크포인트가 없다: {args.checkpoint}")
        return 1

    records = load_window(args.window)
    n_stn = set_offline_fallback(records)
    print(f"저장소 창 {len(records):,}건 · 폴백 등록 관측소 {n_stn}곳")
    if n_stn < len(STATIONS):
        # 일부만 채워져도 진행한다 — 채워진 관측소의 기록은 정상이고,
        # 빠진 곳은 아래에서 실측 여부 검사에 걸려 건너뛰어진다.
        print(f"::warning::창에 실측이 있는 관측소가 {n_stn}/{len(STATIONS)}곳뿐이다")

    # torch 임포트는 여기서 한다 — 인자 오류나 창 부재로 끝날 실행에서
    # 수백 MB 를 로드하지 않기 위해서다.
    from predict import load_model, predict, NWPUnavailable

    model, ckpt = load_model(args.checkpoint, "cpu")
    model_id = accuracy.model_fingerprint(args.checkpoint)
    print(f"체크포인트 {args.checkpoint} (+{ckpt['lead_hours']}h) · "
          f"model_id={model_id}")

    written, skipped, failed = 0, 0, 0
    for name, stn in STATIONS.items():
        try:
            res = predict(stn=stn, model=model, ckpt=ckpt)
        except NWPUnavailable as e:
            # 수치예보 창이 그 시각을 못 덮으면 이 모델은 출력을 만들 수
            # 없다 — 없는 값을 채워 넣지 않는다(CLAUDE.md 1절 5항).
            print(f"  {name}({stn}) 건너뜀 — 수치예보 없음: {e}")
            failed += 1
            continue
        except Exception as e:                       # noqa: BLE001
            print(f"  {name}({stn}) 실패 — {type(e).__name__}: {e}")
            failed += 1
            continue

        if not is_real_observation(res.get("data_status")):
            # 자리표시자(FALLBACK_DEFAULT)로 만든 값은 기록하지 않는다.
            print(f"  {name}({stn}) 건너뜀 — 실측 없음({res.get('data_status')})")
            skipped += 1
            continue
        if not res.get("target_time"):
            skipped += 1
            continue

        f = res["forecast"]
        print(f"  {name}({stn}) {res['observed_at']} → {res['target_time']} "
              f"기온 {f['temperature']:.1f}°C · 강수 {f['precipitation']:.1f}mm")
        if args.dry_run:
            continue
        before = accuracy.log_summary()["total"]
        accuracy.record_prediction(
            station=stn, made_at=res["observed_at"],
            target_time=res["target_time"],
            pred_temp=f["temperature"], pred_precip=f["precipitation"],
            source="ci", model_id=model_id,
        )
        if accuracy.log_summary()["total"] > before:
            written += 1

    if args.dry_run:
        print(f"\n(--dry-run) 기록하지 않았다 — 실패 {failed}건 · 건너뜀 {skipped}건")
        return 0

    # 목표 시각의 실측이 들어온 항목을 대조한다. 창에서만 찾으므로 추가
    # 조회가 없다(accuracy.resolve_pending 의 계약).
    lookup = {(str(r["stn"]), str(r["timestamp"])[:12]): r for r in records}
    resolved = accuracy.resolve_pending(lookup)
    removed = accuracy.trim(args.retain_days)
    summary = accuracy.log_summary()
    print(f"\n신규 {written}건 · 대조 완료 {resolved}건 · 보존기간 초과 정리 "
          f"{removed}건 · 로그 {summary['total']}건(대기 {summary['pending']}건)")
    calls = api_call_stats()
    print(f"API 호출 {calls['count']}건 — 이 경로는 0이어야 한다")
    if calls["count"]:
        # 0이 아니면 오프라인 전제가 깨진 것이다. 조용히 넘기면 15분마다
        # 호출이 쌓여 인증키가 차단될 수 있다.
        print("::error::오프라인 전제가 깨졌다 — API 를 호출했다")
        return 1
    # 한 건도 못 남겼으면 **빨간불로 알린다**(2026-09-25 추가).
    #
    # 종전에는 12개 관측소가 전부 실패해도 0 을 반환했다. 그 사이 '대조'는
    # 계속 돌아 로그 파일이 바뀌므로 커밋도 일어나고, 워크플로 단계는 전부
    # success 로 표시된다 — **기록이 멈춘 것을 아무도 모른다.** 이 경로의
    # '출력값 적중률'이 구조적으로 비어 있던 것을 2026-09-23 에 겨우 고쳤는데,
    # 같은 침묵이 다시 가능한 상태였다(실제로 관측 5시간치가 기록되지 않은
    # 동안 워크플로는 계속 초록불이었다).
    #
    # 기록할 것이 원래 없었던 경우(모든 관측소가 이미 기록됨)와 구분하려고
    # 실패·건너뜀이 있었을 때만 실패로 본다. 커밋 스텝은 `!cancelled()` 라
    # 수집분은 그대로 보존된다(4절 — 게이트와 저장을 분리한다).
    if written == 0 and (failed or skipped):
        print(f"::error::온라인 예측을 한 건도 남기지 못했다 "
              f"(실패 {failed}건 · 건너뜀 {skipped}건) — 적중률 기록이 멈춘다")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
