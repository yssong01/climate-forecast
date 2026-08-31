"""
island_collector.py — 서해 도서 AWS 13개 지점 강수를 Z축 부가 특징으로 제공.

**왜 필요한가.** `island-aws-ablation-gate-passed` 절제 실험(ΔAUC +0.0143,
게이트 3배 초과)에서 검증된 신호를 실제 파이프라인에 편입한다. 서해
상류 관측(도서)이 하류(내륙 12관측소)보다 먼저 강수를 감지해 +6h 리드
타임 안에 도착하는 정보라는 가설(이류 지연 역산 상류거리 ~404km와 도서
거리 112~424km가 겹침)이 근거다.

**입력 형태 — 값 13개 + 결측 플래그 13개(2026-08-31 설계 결정).** 도서
13개 지점이 매 시각 전부 보고되지는 않는다(`cache/island_aws_raw.parquet`
실측: 결측률 8.7%, 13개 전부 요구 시 표본 84% 손실). `InterpolatedFieldCollector`
가 이웃 결측을 중립값(0.5)으로 채우는 것과 같은 원칙으로, 값이 없으면
중립값(0.5, sqrt 압축 정규화 범위의 중앙)으로 채우고 별도 플래그로
"결측"임을 명시한다. **주의**: 원 절제 실험은 "13개 전부 보고 시각만
사용"으로 게이트를 통과했다(중립값 채움이 아니다) — 이 구성은 표본
손실을 피하려는 다른 선택이라, 게이트 수치가 그대로 재현된다는 보장은
없다. 재학습 결과로 다시 검증해야 한다.

**정규화** — `interp_field_collector._norm_precip` 과 동일한 sqrt 압축을
쓴다(강수 분포가 0 근처에 몰리는 오른쪽 치우침, `historical_data_1y.json`
실측 p99=4.0mm 기준과 동일 근거). Z축 벡터에 그대로 이어붙여 다른 Z축
특성과 함께 표준화(평균/표준편차)된다 — 위성 IDW 격자와 달리 별도
파이프라인을 두지 않는다(Re축과 달리 flat 벡터라 공유 인코더의 그래디언트
경쟁 문제가 구조적으로 다르다, `re_axis_channel_saliency.py` 참고).

**가대암(956) 제외** — rn_hr1 이 전 기간 -99.0 고정값(결측 sentinel)이라
강수계가 없는 지점으로 추정, 처음부터 뺀다(`data_expansion_probe_island.py`
와 동일 결정, 2026-08-30 실측 확인).

실행: 별도 실행 없음 — `train.py`/`predict.py`가 import 해서 쓴다. 학습용
parquet 갱신은 `collect_island_backfill.py`. 실시간 서빙용 조회는 이 모듈의
`fetch_live_snapshot()`.
"""
import numpy as np
import pandas as pd
import requests

ISLAND_PARQUET = "./cache/island_aws_raw.parquet"

# 956(가대암)은 강수계 없음으로 추정돼 제외 — data_expansion_probe_island.py와 동일.
ISLAND_STNS = ["102", "169", "229", "303", "426", "501", "655",
               "663", "697", "700", "743", "797", "798"]
ISLAND_DIM = len(ISLAND_STNS) * 2   # 값 13 + 결측 플래그 13 = 26

AWSH_URL = "https://apihub.kma.go.kr/api/typ01/url/awsh.php"


def _norm_precip(v: float) -> float:
    return float(np.clip(np.sqrt(max(v, 0.0)) / np.sqrt(50.0), 0, 1))


def _vector_from_map(snapshot: dict) -> np.ndarray:
    """snapshot: {stn: rn_hr1 or None(결측)} → 26차원 벡터."""
    out = np.empty(ISLAND_DIM, dtype=np.float32)
    n = len(ISLAND_STNS)
    for i, stn in enumerate(ISLAND_STNS):
        val = snapshot.get(stn)
        if val is None:
            out[i] = 0.5
            out[n + i] = 1.0
        else:
            out[i] = _norm_precip(val)
            out[n + i] = 0.0
    return out


class IslandPrecipCollector:
    """(시각) → 26차원 벡터. TendencyCollector와 같은 get_batch/encode_single 인터페이스."""

    def __init__(self, parquet_path: str = ISLAND_PARQUET):
        df = pd.read_parquet(parquet_path, columns=["tm", "stn", "rn_hr1", "rn_hr1_missing"])
        df = df[df["stn"].isin(ISLAND_STNS)]
        # (tm, stn) -> rn_hr1 (결측이면 None). 중복 시각·지점 조합은 마지막이 이긴다
        # (이 저장소의 다른 dedup 병합과 동일 규칙).
        self.by_key: dict[tuple[str, str], float] = {}
        for tm, stn, rn, missing in zip(df["tm"], df["stn"], df["rn_hr1"], df["rn_hr1_missing"]):
            self.by_key[(str(tm), stn)] = None if bool(missing) else float(rn)

    def _vector(self, timestamp) -> np.ndarray:
        tm = str(timestamp)[:12]
        snapshot = {stn: self.by_key.get((tm, stn)) for stn in ISLAND_STNS}
        return _vector_from_map(snapshot)

    def encode_single(self, record: dict) -> np.ndarray:
        return self._vector(record["timestamp"])

    def get_batch(self, records: list) -> np.ndarray:
        return np.stack([self._vector(r["timestamp"]) for r in records], axis=0)


def fetch_live_snapshot(tm=None, timeout: int = 15) -> dict:
    """실시간 서빙용 — 도서 13개 지점의 현재(또는 지정 시각) 강수를 1회 호출로 가져온다.

    awsh.php 는 단일 시각에 전 지점(736개)을 반환하므로(2026-08-31 실측
    확인, 0.21초·#7777END 완결성 마커 있음), 지점당이 아니라 호출 1회면
    된다. 실패하거나 응답에 없는 지점은 결측(None)으로 남긴다 — 날조하지
    않는다(CLAUDE.md 1절 5항).
    """
    import os
    from datetime import datetime, timedelta

    key = os.getenv("KMA_API_KEY", "")
    if tm is None:
        tm = (datetime.utcnow() + timedelta(hours=9)).replace(
            minute=0, second=0, microsecond=0)
    snapshot = {stn: None for stn in ISLAND_STNS}
    if not key:
        return snapshot
    params = {"tm": tm.strftime("%Y%m%d%H%M"), "stn": 0, "disp": 0, "help": 0,
              "authKey": key}
    try:
        resp = requests.get(AWSH_URL, params=params, timeout=timeout)
        resp.raise_for_status()
        for line in resp.text.strip().split("\n"):
            if line.startswith("#") or not line.strip():
                continue
            fields = line.split()
            if len(fields) != 10:
                continue
            stn = fields[1]
            if stn not in snapshot:
                continue
            try:
                rn = float(fields[6])   # rn_hr1
            except ValueError:
                continue
            snapshot[stn] = None if rn < 0.0 else rn
    except Exception:
        pass   # 실패해도 전부 결측(None)으로 반환 — 화면에는 정보없음으로 표시
    return snapshot


def encode_live() -> np.ndarray:
    return _vector_from_map(fetch_live_snapshot())
