"""
gbm_extra.py — GBM 전용 추가 특징(2026-10-22). 학습·진단·서빙이 **같은 함수**로 만든다.

## 왜 따로 두는가

강수·극한기상 GBM 은 주 체크포인트와 같은 표준화 28차원 입력을 공유한다(입력 계약,
`precip_gbm.check_input_contract`). 그 벡터에 열을 더하면 주 모델·기온 모델까지 다시 만들어야
한다. 그런데 탐침(NEXT.md 4-1)에서 의미 있던 후보는 **GBM 에만** 필요한 것들이었다 —
한파·폭염의 `previous_day2` '목표일·다음날' 창(교차 적합 한파 +0.078/+0.049), 강수의 `jma_msm`
(MSM 시대 +0.021). 그래서 공유 입력 뒤에 붙이는 **GBM 전용 블록**으로 분리한다. GBM 은 척도에
무관하므로 표준화하지 않고 원래 단위로 붙인다.

## 규약

- **사양은 모델 파일에 적힌다**(`meta_extra_spec`, `meta_extra_names`). 사양이 없는 GBM 파일
  (현행 배포본)에는 `augment()` 가 아무것도 하지 않는다 — 이 모듈은 채택 전까지 휴면이다.
- **학습과 서빙이 같은 함수**(`extra_vector`)를 부른다. 다른 것은 자료원뿐이다 — 학습은
  아카이브(`*_archive.json`), 서빙은 CI 가 15분마다 모으는 창(`*_recent.json`, data 브랜치).
- **시간 정합 상한**: 미래 유효시각의 previous_dayN 값은 실시간으로 잠정값이다(CLAUDE.md 4절
  정정). 예측 시점에서 `MAX_AHEAD_H[원천]` 시간을 넘는 유효시각은 **쓰지 않는다** — 학습에서도
  같은 상한을 적용해야 서빙이 재현할 수 있다. day2 의 33h 는 day1 결과에서 유추한 잠정값이며
  `nwp_drift_check.py --file cache/nwp_d2_recent.json` 측정 뒤 확정한다.
- **결측은 채우지 않는다.** 학습에서는 NaN(GBM 의 결측 가지)으로 두고, 서빙에서 필요한 값이
  없으면 `ExtraUnavailable` 로 멈춘다(수치예보 결측과 같은 처리 — 1절 5항).
- 하강 폭(`c_drop`)은 과거 24시간 관측이 필요해 서빙 경로가 복잡해지므로 v1 사양에서 뺐다.
"""
import json
import os
from datetime import datetime, timedelta

import numpy as np

# 행 배열 순서 — collect_nwp_archive.MSM_VARS 와 같다(강수·기온·습도·운량).
IP, IT, IRH, ICC = 0, 1, 2, 3

SOURCES = {
    "d2":  {"archive": "./cache/nwp_d2_archive.json",  "recent": "./cache/nwp_d2_recent.json"},
    "msm": {"archive": "./cache/nwp_msm_archive.json", "recent": "./cache/nwp_msm_recent.json"},
}
# 예측 시점에서 이 시간을 넘는 유효시각의 값은 쓰지 않는다(모듈 docstring).
MAX_AHEAD_H = {"d2": 33, "msm": 9}

SPECS = {
    "d2":  ["d2_c_min_d0", "d2_c_min_d1", "d2_c_2day", "d2_h_at_d0", "d2_h_at_d1", "d2_h_2day"],
    "msm": ["msm_p_V", "msm_pwin_sum", "msm_pwin_max", "msm_t_V", "msm_rh_V", "msm_cc_V"],
}
# 단조 제약 부호 — 한파는 더 추울수록, 폭염은 더 더울수록 확률이 오른다.
MONO = {"coldwave": {"d2_c_min_d0": -1, "d2_c_min_d1": -1, "d2_c_2day": -1},
        "heatwave": {"d2_h_at_d0": +1, "d2_h_at_d1": +1, "d2_h_2day": +1}}


class ExtraUnavailable(RuntimeError):
    """서빙에 필요한 추가 특징 자료가 없다 — 지어내지 않고 멈춘다."""


def names(spec):
    return [n for k in spec for n in SPECS[k]]


def kma_apparent_temp(ta, rh):
    """기상청 여름철 체감온도(2022-06-02~): Stull 습구온도 기반."""
    tw = (ta * np.arctan(0.151977 * np.sqrt(rh + 8.313659)) + np.arctan(ta + rh)
          - np.arctan(rh - 1.67633) + 0.00391838 * rh ** 1.5 * np.arctan(0.023101 * rh) - 4.686035)
    return -0.2442 + 0.55399 * tw + 0.45535 * ta - 0.0022 * tw ** 2 + 0.00278 * tw * ta + 3.0


def _key(dt):
    return dt.strftime("%Y%m%d%H00")


def load_rows(path):
    """{관측소: {시각12: [강수, 기온, 습도, 운량]}} — 아카이브(평평한)·스냅숏({'rows': …}) 둘 다."""
    with open(path, encoding="utf-8") as f:
        d = json.load(f)
    return d.get("rows", d) if isinstance(d, dict) and "rows" in d else d


def extra_vector(spec, stn, T, lead, src, missing=None):
    """한 표본의 추가 특징(사양 순서). `src` 는 {원천: {관측소: {시각: 행}}}.

    값이 NaN 인 이유는 둘이다 — ① 창이 시간 정합 상한(`MAX_AHEAD_H`) 너머라 **원래 쓰지 않는**
    시각(학습에서도 같은 규칙으로 NaN 이라 서빙에서도 정상), ② 상한 안인데 자료가 **없는** 시각.
    `missing` 리스트를 넘기면 ②에 해당하는 (원천, 시각)을 담는다 — 서빙은 이것이 비어 있지
    않으면 멈춘다.
    """
    out = []
    V0 = T + timedelta(hours=lead)
    for k in spec:
        rows = (src.get(k) or {}).get(str(stn), {})
        lim = T + timedelta(hours=MAX_AHEAD_H[k])

        def at(t, k=k, rows=rows, lim=lim):
            if t > lim:
                return None
            w = rows.get(_key(t))
            if w is None and missing is not None:
                missing.append((k, _key(t)))
            return w
        if k == "d2":
            d0 = V0.replace(hour=0, minute=0)

            def window(day, h0, h1):
                return [w for w in (at(day + timedelta(hours=h)) for h in range(h0, h1 + 1)) if w is not None]
            m0 = [w[IT] for w in window(d0, 0, 9)]
            m1 = [w[IT] for w in window(d0 + timedelta(days=1), 0, 9)]
            a0 = [float(kma_apparent_temp(w[IT], w[IRH])) for w in window(d0, 12, 17)]
            a1 = [float(kma_apparent_temp(w[IT], w[IRH])) for w in window(d0 + timedelta(days=1), 12, 17)]
            out += [min(m0) if m0 else np.nan, min(m1) if m1 else np.nan,
                    max(min(m0), min(m1)) if (m0 and m1) else np.nan,
                    max(a0) if a0 else np.nan, max(a1) if a1 else np.nan,
                    min(max(a0), max(a1)) if (a0 and a1) else np.nan]
        elif k == "msm":
            win = [at(V0 + timedelta(hours=h)) for h in range(-2, 3)]
            ok = all(w is not None for w in win)
            v = at(V0)
            out += [v[IP] if v else np.nan,
                    sum(w[IP] for w in win) if ok else np.nan,
                    max(w[IP] for w in win) if ok else np.nan,
                    v[IT] if v else np.nan, v[IRH] if v else np.nan, v[ICC] if v else np.nan]
        else:
            raise ValueError(f"알 수 없는 추가 특징 원천: {k}")
    return np.asarray(out, dtype=np.float64)


def _ts(x):
    return datetime.strptime(str(int(x))[:12] if not isinstance(x, str) else x[:12], "%Y%m%d%H%M")


def build_matrix(spec, stns, src_ts, lead, src):
    """여러 표본의 추가 특징 행렬(학습·진단용)."""
    return np.vstack([extra_vector(spec, s, _ts(t), lead, src) for s, t in zip(stns, src_ts)]) \
        if len(stns) else np.zeros((0, len(names(spec))))


def archive_sources(spec):
    """학습·진단용 자료원 — 아카이브가 없으면 멈춘다(collect_nwp_archive --backfill-extra)."""
    src = {}
    for k in spec:
        p = SOURCES[k]["archive"]
        if not os.path.exists(p):
            raise FileNotFoundError(f"{k} 아카이브가 없다: {p} — "
                                    f"collect_nwp_archive.py --backfill-extra 로 만든다")
        src[k] = load_rows(p)
    return src


def spec_of(meta):
    """GBM 메타에 적힌 추가 특징 사양(없으면 빈 튜플 — 현행 배포본)."""
    v = (meta or {}).get("meta_extra_spec")
    if v is None:
        return ()
    s = str(np.asarray(v).item() if hasattr(v, "shape") else v)
    return tuple(x for x in s.split(",") if x)


def augment(meta, x, stns=None, src_ts=None, lead=None, src=None, fixed=None):
    """GBM 입력에 추가 특징을 붙인다. 사양이 없으면 `x` 를 그대로 돌려준다.

    `fixed` 를 주면(1차원 벡터) 모든 행에 그 값을 붙인다 — 단조성 게이트처럼 한 기준 레코드의
    기온만 흔들 때 추가 특징을 기준값으로 고정하기 위해서다(규약 7-1).
    """
    spec = spec_of(meta)
    if not spec:
        return x
    if fixed is not None:
        ex = np.broadcast_to(np.asarray(fixed, dtype=np.float64), (len(x), len(fixed)))
    else:
        ex = build_matrix(spec, stns, src_ts, lead, src if src is not None else archive_sources(spec))
    want = [str(n) for n in np.ravel(meta["meta_extra_names"])]
    if want != names(spec):
        raise RuntimeError(f"추가 특징 이름이 코드와 다르다 — 파일 {want} vs 코드 {names(spec)}")
    return np.hstack([np.asarray(x, dtype=np.float64), ex])


_RECENT = {}


def recent_sources(spec):
    """서빙용 자료원 — CI 가 모으는 창 파일. 파일 mtime 으로 다시 읽는다."""
    src = {}
    for k in spec:
        p = SOURCES[k]["recent"]
        if not os.path.exists(p):
            raise ExtraUnavailable(f"{k} 창 파일이 없다: {p}")
        mt = os.path.getmtime(p)
        if _RECENT.get(k, (None,))[0] != mt:
            _RECENT[k] = (mt, load_rows(p))
        src[k] = _RECENT[k][1]
    return src


def serving_vector(meta, stn, T, lead):
    """서빙 한 건의 추가 특징 — 사양이 없으면 None. 필요한 값이 하나라도 없으면 멈춘다."""
    spec = spec_of(meta)
    if not spec:
        return None
    missing = []
    v = extra_vector(spec, stn, T, lead, recent_sources(spec), missing=missing)
    if missing:
        # 상한 안인데 자료가 없다 — 학습과 다른 입력이 되므로 지어내지 않고 멈춘다.
        raise ExtraUnavailable(f"{stn} {T:%Y%m%d%H%M} +{lead}h 추가 특징 자료 결측 "
                               f"{len(missing)}시각(예: {missing[0][0]} {missing[0][1]})")
    return v
