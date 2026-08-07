"""
accuracy.py — 예보 적중률 로그.

하나의 로그(cache/accuracy_log.json)를 두 경로가 함께 채운다.
  - backtest_accuracy.py: 이미 확보된 과거 데이터 전체에 모델을 소급
    추론해 한 번에 대량 시딩한다 — "누적 적중률"을 기다리지 않고 즉시
    확보하기 위함.
  - app.py: 대시보드가 예보를 만들 때마다 항목을 추가하고(source="live"),
    시간이 지나 목표 시각이 지난 항목을 실측과 대조해 채운다.

같은 스키마를 쓰므로 두 출처가 자연스럽게 하나의 통계로 합쳐진다.

적중(hit) 정의 — 방식은 이 프로젝트의 다른 곳(Phase 3-6 POD/FAR)과 결을
맞췄다. 정답이 하나로 정해지는 값이 아니라 "실무적으로 쓸모 있었나"를
재는 것이므로 완전 일치가 아니라 허용오차/이진판정을 쓴다.
  - 기온: |예측−실측| ≤ HIT_TEMP_TOL(°C) 이면 적중.
          이 프로젝트 모델의 검증 MAE(약 0.85°C)의 약 2배 — MAE 자체를
          허용오차로 쓰면 정의상 언제나 ~50%에 수렴해 버려 무의미해진다.
  - 강수: 예측·실측이 같은 쪽(비/무비)이면 적중. 강수량 자체의 절대
          오차보다 "비가 올지 안 올지 맞았는가"가 실사용 가치에 더
          가깝고, 강수 대부분이 0 근처에 몰려 있어(README 한계 4) 절대
          오차 기준은 상시 "적중"으로 착시를 일으키기 쉽다.

live 경로에서 resolve_pending() 은 로컬 캐시만 조회하고 새 API 호출을
만들지 않는다 — 대시보드가 새로고침될 때마다 apihub 호출을 만들면
collect_incremental.py 의 예산 관리와 별개로 쿼터를 소비하게 된다
(오늘 실측: 누적 약 9,800건에서 이 키가 막힘). 아직 캐시에 없는 시각은
collect_incremental.py 가 채울 때까지 "대기" 상태로 남는다.
"""
import json
import os
from datetime import datetime

HIT_TEMP_TOL   = 1.5    # °C
PRECIP_THRESH  = 0.1    # mm — 이 이상이면 "비"로 판정 (train.py 이벤트 정의와 동일)
LOG_PATH       = "./cache/accuracy_log.json"


def _load(path: str = LOG_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(entries: list, path: str = LOG_PATH) -> None:
    tmp = path + ".tmp"
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(entries, f, ensure_ascii=False)
    os.replace(tmp, path)


def _hit(pred_temp, pred_precip, actual_temp, actual_precip):
    hit_temp = abs(pred_temp - actual_temp) <= HIT_TEMP_TOL
    hit_precip = (pred_precip >= PRECIP_THRESH) == (actual_precip >= PRECIP_THRESH)
    return bool(hit_temp), bool(hit_precip)


def record_prediction(station: str, made_at: str, target_time: str,
                      pred_temp: float, pred_precip: float,
                      source: str = "live", path: str = LOG_PATH) -> None:
    """
    새 예보 1건을 로그에 추가. (station, target_time, source) 가 이미 있으면
    건너뛴다 — 대시보드가 자동 갱신될 때마다 같은 +6h 목표시각을 다시
    예측해도 로그가 중복으로 쌓이지 않도록.
    """
    entries = _load(path)
    key = (station, target_time, source)
    if any((e["station"], e["target_time"], e["source"]) == key for e in entries):
        return
    entries.append({
        "station": station,
        "made_at": made_at,
        "target_time": target_time,
        "pred_temp": round(float(pred_temp), 3),
        "pred_precip": round(float(pred_precip), 3),
        "actual_temp": None,
        "actual_precip": None,
        "hit_temp": None,
        "hit_precip": None,
        "source": source,
    })
    _save(entries, path)


def resolve_pending(lookup: dict, path: str = LOG_PATH) -> int:
    """
    아직 실측과 대조 못 한 항목(actual=None)을 lookup 으로 채운다.

    lookup: {(station, timestamp): record} — 로컬 캐시에서 구성. 새 API
    호출은 여기서 하지 않는다(모듈 docstring 참고).

    반환: 새로 해소된 항목 수.
    """
    entries = _load(path)
    resolved = 0
    for e in entries:
        if e["actual_temp"] is not None:
            continue
        rec = lookup.get((e["station"], e["target_time"]))
        if rec is None:
            continue
        e["actual_temp"] = round(float(rec["temperature"]), 3)
        e["actual_precip"] = round(float(rec["precipitation"]), 3)
        e["hit_temp"], e["hit_precip"] = _hit(
            e["pred_temp"], e["pred_precip"], e["actual_temp"], e["actual_precip"]
        )
        resolved += 1
    if resolved:
        _save(entries, path)
    return resolved


def stats(station: str = None, recent_n: int = 20, path: str = LOG_PATH) -> dict:
    """
    누적/최근 적중률. station=None 이면 전 관측소 합산.
    반환: {"cum_n","cum_temp","cum_precip","recent_n","recent_temp","recent_precip"}
    값이 없으면 해당 항목은 None.
    """
    entries = [e for e in _load(path) if e["actual_temp"] is not None]
    if station:
        entries = [e for e in entries if e["station"] == station]
    entries.sort(key=lambda e: e["target_time"])

    def _rate(es, key):
        return (sum(1 for e in es if e[key]) / len(es)) if es else None

    recent = entries[-recent_n:]
    return {
        "cum_n": len(entries),
        "cum_temp": _rate(entries, "hit_temp"),
        "cum_precip": _rate(entries, "hit_precip"),
        "recent_n": len(recent),
        "recent_temp": _rate(recent, "hit_temp"),
        "recent_precip": _rate(recent, "hit_precip"),
    }
