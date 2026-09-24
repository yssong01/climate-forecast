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
import hashlib
import json
import os
import tempfile
import threading
from datetime import datetime, timedelta

HIT_TEMP_TOL   = 1.5    # °C
PRECIP_THRESH  = 0.1    # mm — 이 이상이면 "비"로 판정 (train.py 이벤트 정의와 동일)
LOG_PATH       = "./cache/accuracy_log.json"

# 로그 보존 기간(일). CI 가 관측소 12곳을 매시 기록하면 하루 약 288건씩
# 늘어난다 — 항목당 230바이트 안팎이라 무제한으로 두면 1년에 20MB를 넘고,
# CLAUDE.md 11항이 경계한 "GitHub 상한에 근접"에 그대로 들어간다. 화면이
# 보여주는 것은 현행 체크포인트의 누적이고 승격 때마다 어차피 0부터 다시
# 세므로, 한 달치를 남기면 표시에 필요한 범위를 충분히 덮는다.
RETAIN_DAYS = 30

# _save 자체는 이제 원자적이지만, record_prediction/resolve_pending은
# 읽기→수정→쓰기 세 단계로 나뉘어 있어 둘 사이에는 여전히 경합이 남는다 —
# 두 세션이 거의 동시에 읽으면 나중에 쓰는 쪽이 앞선 쪽의 수정을 덮어써
# 로그 항목이 조용히 사라질 수 있다(크래시는 아니고 유실). 이 락으로 두
# 함수의 읽기→수정→쓰기 구간 전체를 하나의 원자적 단위로 묶는다.
_lock = threading.Lock()


_FP_CACHE = {}


def model_fingerprint(checkpoint_path: str) -> str:
    """이 체크포인트가 만드는 **예측**의 신원 — 적중률 로그의 `model_id`.

    **무엇을 해시하는가가 핵심이다(2026-09-23, 두 번 고쳤다).**

    ① 처음에는 파일의 수정시각·크기(`mtime_ns:size`)였다. 그러면 같은
       모델이라도 **재배포할 때마다 값이 바뀐다** — Streamlit Cloud 는 매
       배포에서 저장소를 새로 체크아웃하므로 mtime 이 갱신된다. 화면의
       누적 적중률이 모델을 바꾸지 않아도 0에서 다시 시작했다.

    ② 그래서 파일 **내용** 해시로 바꿨는데, 같은 날 같은 증상이 다른 경로로
       재발했다. `metrics_report.py --patch-checkpoint`,
       `probability_calibration_fit.py --patch-metrics`,
       `promote_checkpoint.py` 의 게이트 기록처럼 **예측을 바꾸지 않는
       메타데이터만 적어 넣어도** 파일 내용이 달라져 기존 기록이 통째로
       고아가 된다(실제로 96건이 그렇게 됐다. 두 파일의 가중치 해시가
       동일함을 확인해 원인을 특정했다).

    그래서 **예측을 결정하는 것만** 해시한다 — 가중치, 정규화 통계, 입력
    차원, 예보 시계. 로그가 담는 것이 기온·강수 예측값이므로, 그 값을 바꾸지
    않는 변경은 같은 신원이어야 "이 모델의 누적"이 성립한다.

    **캐시 무효화 키로는 쓰지 않는다.** 화면은 확률 보정 곡선·예측구간까지
    보여주므로 그 메타데이터가 바뀌면 캐시는 **무효화돼야 한다** — 두 용도의
    요구가 반대다(app.py `ckpt_fingerprint` 주석 참고).

    torch 는 호출 시점에 들여온다 — 이 모듈은 수집 스크립트도 임포트한다.
    결과는 (mtime, size) 로 캐시해 매 재실행마다 다시 읽지 않는다.
    """
    # 기온 전용 보조 체크포인트가 켜져 있으면 **그것도 예측을 결정한다** —
    # 신원에 빠뜨리면 기온 출처를 바꿔도 같은 model_id 로 기록돼 서로 다른
    # 모델의 적중률이 한 줄로 뭉친다(2026-09-24 추가). 환경변수를 직접 읽는
    # 이유는 `predict.TEMP_CHECKPOINT` 와 같은 출처를 쓰면서도 이 모듈이
    # predict 를 임포트하지 않기 위해서다(수집 스크립트도 이 모듈을 쓴다).
    paths = [checkpoint_path]
    _temp = os.getenv("TEMP_CHECKPOINT_PATH", "")
    if _temp:
        paths.append(_temp)
    try:
        key = tuple((p, os.stat(p).st_mtime_ns, os.stat(p).st_size) for p in paths)
    except OSError:
        return "missing"
    if key in _FP_CACHE:
        return _FP_CACHE[key]
    try:
        import torch
        h = hashlib.sha256()
        for p in paths:
            ck = torch.load(p, map_location="cpu", weights_only=True)
            for name, tensor in ck["model_state"].items():
                h.update(name.encode())
                h.update(tensor.numpy().tobytes())
            for field in ("mean", "std"):
                h.update(repr(ck.get(field)).encode())
            h.update(f"{ck.get('num_features')}:{ck.get('lead_hours')}".encode())
        fp = "sha256:" + h.hexdigest()[:16]
    except Exception:                                # noqa: BLE001
        # torch 가 없거나 읽기에 실패하면 구분을 포기한다 — 틀린 신원으로
        # 서로 다른 모델의 기록을 뭉치는 것보다 낫다.
        fp = "unknown"
    _FP_CACHE[key] = fp
    return fp


def _load(path: str = LOG_PATH) -> list:
    if not os.path.exists(path):
        return []
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def _save(entries: list, path: str = LOG_PATH) -> None:
    """
    Streamlit Cloud는 세션(탭)마다 스레드를 띄우고 자동 갱신 때마다 이 함수가
    거의 동시에 호출된다. tmp 이름이 고정이면 두 스레드가 같은 tmp를 쓰다가
    한쪽이 os.replace 직전에 다른 쪽이 이미 옮겨간 tmp를 찾지 못해
    FileNotFoundError가 난다(2026-08-13 실측) — 호출마다 고유한 tmp로 회피.
    """
    directory = os.path.dirname(path)
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(dir=directory or ".", prefix=os.path.basename(path) + ".")
    try:
        with os.fdopen(fd, "w", encoding="utf-8") as f:
            json.dump(entries, f, ensure_ascii=False)
        os.replace(tmp, path)
    except BaseException:
        os.remove(tmp)
        raise


def _hit(pred_temp, pred_precip, actual_temp, actual_precip):
    hit_temp = abs(pred_temp - actual_temp) <= HIT_TEMP_TOL
    hit_precip = (pred_precip >= PRECIP_THRESH) == (actual_precip >= PRECIP_THRESH)
    return bool(hit_temp), bool(hit_precip)


def record_prediction(station: str, made_at: str, target_time: str,
                      pred_temp: float, pred_precip: float,
                      source: str = "live", path: str = LOG_PATH,
                      model_id: str = None) -> None:
    """
    새 예보 1건을 로그에 추가. (station, target_time, source) 가 이미 있으면
    건너뛴다 — 대시보드가 자동 갱신될 때마다 같은 +6h 목표시각을 다시
    예측해도 로그가 중복으로 쌓이지 않도록.

    model_id — 이 예측을 만든 체크포인트의 식별자(model_fingerprint() 참고).
    체크포인트를 교체하면 같은 로그에 서로 다른 모델의 기록이 쌓이는데,
    구분이 없으면 화면의 "누적 적중률"이 어느 모델도 아닌 혼합값이 된다
    (2026-08-16 배포 전 점검에서 발견). 옛 항목에는 이 필드가 없으므로
    stats() 는 그런 항목을 "legacy" 로 취급한다.
    """
    with _lock:
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
            "model_id": model_id,
        })
        _save(entries, path)


def resolve_pending(lookup: dict, path: str = LOG_PATH) -> int:
    """
    아직 실측과 대조 못 한 항목(actual=None)을 lookup 으로 채운다.

    lookup: {(station, timestamp): record} — 로컬 캐시에서 구성. 새 API
    호출은 여기서 하지 않는다(모듈 docstring 참고).

    반환: 새로 해소된 항목 수.
    """
    with _lock:
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


def trim(retain_days: int = RETAIN_DAYS, path: str = LOG_PATH) -> int:
    """보존 기간이 지난 **대조 완료** 항목을 지운다. 반환: 지운 건수.

    대기(actual=None) 항목은 기간과 무관하게 남긴다 — 아직 실측을 기다리는
    중일 수도 있고, 영영 대조되지 않는다면 그건 갱신 경로가 끊겼다는 신호라
    조용히 지우면 그 신호까지 지우는 셈이 된다.

    기준 시각은 목표 시각(`target_time`)이다. 벽시계가 아니라 데이터의
    시각을 쓰므로, 실행 환경의 시간대에 좌우되지 않는다.
    """
    with _lock:
        entries = _load(path)
        if not entries:
            return 0
        latest = max((e["target_time"] for e in entries), default=None)
        if not latest:
            return 0
        try:
            cutoff = (datetime.strptime(str(latest)[:12], "%Y%m%d%H%M")
                      - timedelta(days=retain_days)).strftime("%Y%m%d%H%M")
        except ValueError:
            return 0
        kept = [e for e in entries
                if e["actual_temp"] is None or str(e["target_time"])[:12] >= cutoff]
        removed = len(entries) - len(kept)
        if removed:
            _save(kept, path)
        return removed


def log_summary(path: str = LOG_PATH) -> dict:
    """
    로그 파일 자체의 규모 — 적중률이 아니라 "표본이 얼마나 있고 얼마나
    대조됐는지"를 본다. 배포판에서 특히 중요하다: Streamlit Community Cloud
    의 파일시스템은 휘발성이라, 앱이 실행 중에 추가한 항목은 재시작하면
    사라지고 저장소에 커밋된 스냅샷만 남는다. 화면이 "지금 보고 있는 통계가
    어디까지 영구인지"를 말할 수 있어야 한다.

    반환: {"total","resolved","pending","latest_target"}
          latest_target 은 대조 완료된 항목 중 가장 최근 목표시각(없으면 None).
    """
    entries = _load(path)
    resolved = [e for e in entries if e["actual_temp"] is not None]
    return {
        "total":         len(entries),
        "resolved":      len(resolved),
        "pending":       len(entries) - len(resolved),
        "latest_target": max((e["target_time"] for e in resolved), default=None),
    }


def stats(station: str = None, recent_n: int = 20, path: str = LOG_PATH,
          model_id: str = None) -> dict:
    """
    누적/최근 적중률. station=None 이면 전 관측소 합산.
    반환: {"cum_n","cum_temp","cum_precip","recent_n","recent_temp","recent_precip"}
    값이 없으면 해당 항목은 None.

    model_id — 지정하면 그 모델(model_fingerprint 값)이 만든 예측만 집계한다.
    체크포인트를 바꾸면 같은 로그에 다른 모델의 기록이 섞이는데, 필터링 없이
    합산하면 "누적 적중률"이 어느 모델의 성능도 아닌 값이 된다(2026-08-16
    배포 전 점검에서 발견 — 지금까지는 이 구분 없이 전부 합산했다). 지정하지
    않으면 예전처럼 전부 합산한다(레거시 호출부 호환용 기본값).
    """
    entries = [e for e in _load(path) if e["actual_temp"] is not None]
    if model_id is not None:
        entries = [e for e in entries if e.get("model_id") == model_id]
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
