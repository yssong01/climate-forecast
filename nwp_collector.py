"""
nwp_collector.py — 수치예보(NWP) 예보값을 Z축 부가 특징으로 인코딩한다.

`collect_nwp_archive.py` 가 저장한 파일을 읽어 (관측소, 유효시각) → 예보값
7종 표를 만들고, 학습·서빙이 **같은 함수**로 같은 특징 벡터를 만든다.

특징 설계는 `nwp_feature_probe.py` 가 실측한 14차원을 그대로 옮긴 것이다
(+6h ΔAUC +0.0285 · +12h ΔAUC +0.0795). 설계 의도는 셋이다.

  ① **목표 시각의 예보**(0~6) — 핵심 신호. 지상 관측만으로는 볼 수 없는
     상류(서해상) 상태가 여기 반영돼 있다.
  ② **강수 타이밍 완충**(7~8) — 목표 시각 ±1시간 창의 합·최댓값. NWP 의
     강수 오차는 크기보다 시각 어긋남이 지배적이라, 한 시각만 보면
     "한 시간 빗나간 정확한 예보"를 놓친다.
  ③ **동시각 편향**(9~13) — 같은 시각의 예보와 실측의 차이. 모델이 그
     지점·그 시점에서 NWP 가 얼마나 치우쳐 있는지 알 수 있게 해준다
     (통계적 후처리(MOS)에서 표준적으로 쓰는 신호).

**결측은 채우지 않는다**(CLAUDE.md 1절 5항·4절). 필요한 시각의 예보가
하나라도 없으면 `None` 을 돌려주고, 학습에서는 그 표본을 통째로 뺀다.
아카이브 소급 한계가 2016-01-01 이라 그 이전 표본은 전부 빠진다 — 극한기상
공식 라벨이 2019년(폭염·황사)·2020년(한파)부터라 그 헤드들에는 영향이 없고,
회귀 헤드의 표본만 줄어든다.
"""
import json
import os

import numpy as np

ARCHIVE_PATH = "./cache/nwp_archive.json"
RECENT_PATH = "./cache/nwp_recent.json"

# collect_nwp_archive.NWP_VARS 와 같은 순서여야 한다.
NWP_VARS = ["precipitation", "temperature_2m", "relative_humidity_2m",
            "cloud_cover", "wind_speed_10m", "wind_direction_10m",
            "surface_pressure"]
I_PR, I_TP, I_RH, I_CC, I_WS, I_WD, I_SP = range(7)

NWP_DIM = 14

# 특징 부분집합(2026-09-07). `nwp_feature_probe.py --NWP_ABLATE` 절제에서
# 14차원의 기여가 극소수에 몰려 있음이 두 리드타임 모두에서 확인됐다 —
# 기온 이득의 94~95%를 `예보기온` 하나가, 강수 이득의 89~93%를 `창합계`
# 하나가 나른다. 아래 6개면 14차원 이득의 98.5~98.7%(강수)·98.6~99.2%
# (기온)를 회수하고, 동시각 편향 3개(11~13)는 빼도 영향이 없다.
#
# 차원을 줄이는 이유는 성능이 아니라 **Z축 예산**이다. 수치예보 도입 후
# Re축 게이트가 0.380→0.021 로 붕괴하고 황사가 유의 악화했는데, 신호 없는
# 차원이 Z축을 부풀린 것이 원인인지 이 축소로 가릴 수 있다.
#
# 인코딩 자체는 늘 14차원을 만들고 **여기서 열을 고른다** — 그래야
# 구버전 체크포인트(full14)와 새 체크포인트가 같은 코드 경로를 쓴다.
FEATURE_SETS = {
    "full14":  list(range(14)),
    # 예보강수·예보기온·예보습도·예보운량·창합계·기압변화
    "compact6": [0, 1, 2, 3, 7, 10],
}


def feature_dim(feature_set: str = "full14") -> int:
    return len(FEATURE_SETS[feature_set])


def select(vec, feature_set: str = "full14"):
    """14차원 벡터에서 그 집합의 열만 고른다."""
    cols = FEATURE_SETS[feature_set]
    return vec if len(cols) == NWP_DIM else vec[..., cols]


def _ts_add_hours(ts12: str, hours: int) -> str:
    from datetime import datetime, timedelta
    t = datetime.strptime(str(ts12)[:12], "%Y%m%d%H%M") + timedelta(hours=hours)
    return t.strftime("%Y%m%d%H%M")


def _encode(f_now, f_tgt, win, obs) -> np.ndarray:
    """예보값 → 14차원. `nwp_feature_probe.py` 와 같은 순서·정의."""
    wu = -np.sin(np.deg2rad(f_tgt[I_WD])) * f_tgt[I_WS]
    wv = -np.cos(np.deg2rad(f_tgt[I_WD])) * f_tgt[I_WS]
    return np.array([
        f_tgt[I_PR], f_tgt[I_TP], f_tgt[I_RH], f_tgt[I_CC], wu, wv, f_tgt[I_SP],
        float(np.sum(win)), float(np.max(win)),
        f_tgt[I_PR] - f_now[I_PR],
        f_tgt[I_SP] - f_now[I_SP],
        f_now[I_TP] - obs[0], f_now[I_PR] - obs[1], f_now[I_RH] - obs[2],
    ], dtype=np.float32)


class NWPForecastCollector:
    """(관측소, 시각) → NWP 예보 특징.

    `paths` 를 여러 개 주면 뒤엣것이 앞엣것을 덮어쓴다 — 서빙에서
    아카이브와 최근 창을 함께 실을 때 최신 값이 이기게 하기 위해서다.
    """

    def __init__(self, paths=(ARCHIVE_PATH,), required: bool = True):
        self.table = {}
        loaded = []
        for p in paths:
            if not os.path.exists(p):
                continue
            with open(p, "r", encoding="utf-8") as f:
                self.merge_raw(json.load(f))
            loaded.append(p)
        if required and not self.table:
            raise FileNotFoundError(
                f"NWP 아카이브가 없다({', '.join(paths)}) — "
                f"`python collect_nwp_archive.py --backfill` 을 먼저 실행할 것.")
        self.sources = loaded

    def merge_raw(self, raw: dict) -> None:
        """`{관측소: {시각: [값...]}}` 를 표에 합친다(나중 것이 이긴다)."""
        for stn, rows in raw.items():
            dst = self.table.setdefault(str(stn), {})
            for ts, vals in rows.items():
                dst[ts] = np.asarray(vals, dtype=np.float32)

    def coverage(self) -> str:
        if not self.table:
            return "없음"
        n = sum(len(v) for v in self.table.values())
        allts = [t for v in self.table.values() for t in v]
        return f"{len(self.table)}개 관측소 · {n:,}시각 ({min(allts)}~{max(allts)})"

    def raw_forecast(self, record: dict, lead_hours: int):
        """예보값 원본 `(f_now, f_tgt, win)`. 하나라도 없으면 None.

        단조성 시험(`coldwave_pathway_check.py`)이 이걸 쓴다 — 기온·계절을
        흔들면서 **예보는 고정**해야 "한 번에 하나만 바꾼다"는 원칙이
        지켜지기 때문이다. 흔든 시각의 예보를 매번 다시 조회하면 두 가지가
        동시에 바뀌어 판정이 무의미해지고, 애초에 미래 날짜는 아카이브에
        없다.
        """
        stn = str(record.get("stn"))
        rows = self.table.get(stn)
        if rows is None:
            return None
        ts = str(record.get("timestamp", ""))[:12]
        if len(ts) < 12:
            return None
        f_now = rows.get(ts)
        if f_now is None:
            return None
        got = [rows.get(_ts_add_hours(ts, lead_hours + d)) for d in (-1, 0, 1)]
        if any(g is None for g in got):
            return None
        return f_now, got[1], [g[I_PR] for g in got]

    @staticmethod
    def encode_from(forecast, record: dict, feature_set: str = "full14"):
        """고정한 예보값 + (흔들린) 관측 레코드 → 특징 벡터.

        관측에서 유도되는 편향 항(9~13번 중 뒤 세 개)은 레코드를 따라
        움직인다 — 기온을 흔들면 "예보−실측" 도 함께 움직이는 것이 물리적
        으로 일관된다. 예보값 자체만 고정한다.
        """
        f_now, f_tgt, win = forecast
        obs = (record.get("temperature"), record.get("precipitation"),
               record.get("humidity"))
        if any(o is None for o in obs):
            return None
        return select(_encode(f_now, f_tgt, win, obs), feature_set)

    def encode(self, record: dict, lead_hours: int, feature_set: str = "full14"):
        """한 레코드의 특징 벡터. 필요한 예보가 하나라도 없으면 None.

        **결측 판정은 `feature_set` 과 무관하다** — 늘 14차원을 만들 수
        있는지로 판단하고 그 뒤에 열을 고른다. 그래야 어떤 집합을 쓰든
        표본 구성이 같아 대조 실험이 성립한다(기준선 불일치가 생기지 않는다).
        """
        stn = str(record.get("stn"))
        rows = self.table.get(stn)
        if rows is None:
            return None
        ts = str(record.get("timestamp", ""))[:12]
        if len(ts) < 12:
            return None
        f_now = rows.get(ts)
        if f_now is None:
            return None
        need = [_ts_add_hours(ts, lead_hours + d) for d in (-1, 0, 1)]
        got = [rows.get(k) for k in need]
        if any(g is None for g in got):
            return None
        f_tgt = got[1]
        win = [g[I_PR] for g in got]
        obs = (record.get("temperature"), record.get("precipitation"),
               record.get("humidity"))
        if any(o is None for o in obs):
            return None
        return select(_encode(f_now, f_tgt, win, obs), feature_set)

    def get_batch(self, records: list, lead_hours: int, feature_set: str = "full14"):
        """(N, NWP_DIM) 배열과 유효 마스크를 함께 돌려준다.

        학습은 마스크가 False 인 표본을 데이터셋에서 제외한다 — 결측을
        중립값으로 메우면 "예보가 없다"와 "예보가 0mm 다"가 구분되지 않는다.
        """
        vecs, mask = [], []
        dim = feature_dim(feature_set)
        for r in records:
            v = self.encode(r, lead_hours, feature_set)
            mask.append(v is not None)
            vecs.append(v if v is not None else np.zeros(dim, dtype=np.float32))
        return np.stack(vecs, axis=0), np.asarray(mask, dtype=bool)


_SHARED = {}


def shared(paths=(ARCHIVE_PATH,)) -> "NWPForecastCollector":
    """프로세스당 한 번만 읽어 재사용한다.

    아카이브가 52MB JSON 이라, 진단 스크립트가 탐침마다 새로 만들면
    그것만으로 수십 초를 버린다(단조성 시험은 기준 레코드 12개를 돈다).
    """
    key = tuple(paths)
    if key not in _SHARED:
        _SHARED[key] = NWPForecastCollector(paths=paths)
    return _SHARED[key]


# ── 메모리 주입 경로 ────────────────────────────────────────────
# 배포 앱은 파일시스템 쓰기에 의존하면 안 된다(2026-09-07). Streamlit Cloud
# 의 파일시스템은 휘발성이고, 쓰기가 실패하면 예보를 아예 못 만들어 화면이
# 통째로 멈춘다 — 관측 자료는 폴백이 있지만 이 축은 없으면 출력이 불가능한
# 필수 입력이라 실패 비용이 훨씬 크다. 그래서 앱은 내려받은 내용을 그대로
# 메모리에 주입하고, 디스크 저장은 최선 노력(best effort)으로만 한다.
_SERVING = {"id": None, "collector": None}


def set_serving_payload(raw: dict, payload_id: str) -> None:
    """서빙용 표를 메모리로 주입한다. `payload_id` 가 같으면 재구성하지 않는다."""
    if _SERVING["id"] == payload_id and _SERVING["collector"] is not None:
        return
    c = NWPForecastCollector(paths=(), required=False)
    c.merge_raw(raw)
    _SERVING.update(id=payload_id, collector=c)


def serving_override():
    """주입된 표가 있으면 돌려준다 — 없으면 None(디스크 경로를 쓴다)."""
    return _SERVING["collector"]


def load_for_serving() -> "NWPForecastCollector":
    """서빙용 — 배포 창을 우선 읽고, 없으면 아카이브로 폴백한다.

    배포판은 `cache/nwp_recent.json`(작은 창)만 싣는다. 로컬에는 아카이브가
    있으므로 둘 다 읽되 최근 창이 이기게 순서를 잡는다.
    """
    return NWPForecastCollector(paths=(ARCHIVE_PATH, RECENT_PATH), required=False)


if __name__ == "__main__":
    c = NWPForecastCollector()
    print("아카이브:", c.coverage())
    stn = next(iter(c.table))
    ts = sorted(c.table[stn])[len(c.table[stn]) // 2]
    rec = dict(stn=stn, timestamp=ts, temperature=20.0, precipitation=0.0,
               humidity=60.0)
    for lead in (6, 12):
        v = c.encode(rec, lead)
        print(f"  lead=+{lead}h → {None if v is None else np.round(v, 3)}")
