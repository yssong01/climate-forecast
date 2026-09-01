"""
predict.py — 학습된 Tri-CHEF 체크포인트로 실시간 +N시간 예보 생성.

사용법:
  python predict.py                    # 서울(108) 예보, 사람이 읽는 형식
  python predict.py --stn 159          # 부산 예보
  python predict.py --stn 159 --json   # JSON 출력 (app.py·API 연동용)
  python predict.py --list-stations    # 관측소 코드 목록

학습(train.py)과 추론이 동일한 전처리 경로를 쓰도록, 수치 벡터 변환은
train.record_to_vec() 을 그대로 재사용한다 — 별도 구현하면 두 경로가
조용히 어긋나는 것이 흔한 버그 원인이기 때문이다.
"""
import argparse
import json
import os
import sys
from datetime import datetime, timedelta

import numpy as np
import torch

from weather_collector import (
    RobustWeatherCollector, STATIONS, STATION_COORDS, is_real_observation,
)
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector, LAGS_HOURS
from text_collector import SimulatedTextCollector
from island_collector import encode_live as island_encode_live
from pipeline_model import TriCHEFPipeline
from train import record_to_vec, STATION_NAMES

# 환경변수로 뺀다(2026-08-18 수정, train.py의 CHECKPOINT_PATH 패턴과 통일).
#
# 이전엔 하드코딩 리터럴이었다 — head_decouple_finetune.py·
# coldwave_nested_pretrain.py 등 다수 스크립트가 위치 인자로 경로를 안 받고
# `from predict import CHECKPOINT` 의 기본값에만 의존하는데, 그 상태에서
# CHECKPOINT_PATH 를 지정해도 조용히 무시되고 항상 프로덕션 체크포인트가
# 로드됐다. 실제로 이 버그 때문에 한파 헤드 중첩 극한사건 전이학습 실험
# (2026-08-18)의 "최종 미세조정" 단계 두 번 모두 사전학습 체크포인트가 아닌
# 프로덕션 체크포인트를 그대로 불러왔다 — "재현성 확인"이 실은 같은 일반
# 디커플링을 두 번 돌린 것이었다. README·CLAUDE.md 의 해당 실험 결론은
# 무효로 표시했다(재실행 필요).
CHECKPOINT = os.getenv("CHECKPOINT_PATH", "./checkpoints/numerical_trichef.pt")
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# 강수 후처리 게이팅 — precip_prob_gate_sweep.py 실측(2026-08-29, 검증셋
# 평가용 절반·보정용 절반 분리): 매그니튜드(mm) 임계값은 강수 발생 판정
# F1과 같은 knob을 공유해(둘 다 rain_prob×amount 곱을 자름) MAE를 올리려
# 임계값을 키우면 F1이 무너졌다(t=2.95에서 F1 0.430→0.103, README
# '후처리 임계값을 올리면 이기지만, 채택하지 않는다' 참고). rain_prob 자체에
# 별도 임계값을 걸면(매그니튜드와 분리) F1·MAE가 동시에 개선되는 구간을
# 찾았다. τ가 오를수록 정밀도↑재현율↓ 트레이드오프가 있고, F1이 다시
# 기준선 아래로 꺾이는 절벽이 있어 절벽에서 떨어진 값을 쓴다.
#
# +6h 체크포인트에서 고른 값(0.85)을 +12h에 그대로 쓰면 안 된다 — rain_prob
# 분포 자체가 리드타임마다 달라서(+12h는 90백분위가 0.75에 그침) 같은
# 상수가 그 체크포인트에서는 F1을 오히려 깎았다(0.3560→0.3497, 실측). 두
# 체크포인트 모두 독립적으로 보정용/평가용 분할에서 재선정했다(규약 9번 —
# 헤드를 다시 조정하면 그 시점 표본에서 임계값도 다시 고른다).
#   +6h  — 평가용 F1 0.4296→0.4637, MAE 기준선 대비 +5.9%→+3.8%
#   +12h — 평가용 F1 0.3591→0.3805, MAE 기준선 대비 +17.7%→+17.2%
PRECIP_PROB_GATE_BY_LEAD = {6: 0.85, 12: 0.75}
PRECIP_PROB_GATE = PRECIP_PROB_GATE_BY_LEAD[6]  # lead_hours 미상일 때 기본값

# 구버전 체크포인트(Phase 3-7 hurdle 헤드 도입 전, head_rain 없음) 호환용
# 폴백 — extreme_event_probs()["rain"] 이 None이면 rain_prob 게이팅을 쓸 수
# 없으므로 예전 매그니튜드 클리핑으로 되돌아간다(precip_breakdown.py 실측
# 근거는 그대로 유효).
PRECIP_CLIP_THRESH = 0.2  # mm — head_rain 없는 구버전 체크포인트 전용 폴백

# 극한기상 헤드 판정 임계값 — threshold_validation.py 재실측(2026-08-17,
# signed seed7 체크포인트 기준: 13년·그룹분할·14차원·극한기상 헤드에 부호
# 표현 추가). 아래 값은 **확률 보정 전(원본) 공간**의 값이다 —
# event_threshold() 가 체크포인트의 보정 곡선으로 옮겨서 돌려준다.
#
# 채택 기준: 보정용/평가용 분할에서 t=0.5 대비 순이득이 0.01을 넘는 것만
# 재보정값을 쓴다(threshold_validation.py 자체 기준).
#   강수    — 순이득 +0.101 → 재보정값(0.83) 채택, F1 0.498 (t=0.5는 0.397)
#   폭염    — 순이득 +0.004 → 기준 미달, t=0.5 유지 (F1 0.819)
#   한파    — 순이득 +0.005 → 기준 미달, t=0.5 유지 (F1 0.454)
#   황사    — 순이득 +0.002 → 기준 미달, t=0.5 유지. v9에서는 0.81을 채택
#            했었으나 이 모델에서는 이득이 사라졌다. 어차피 정밀도가 낮아
#            화면에서는 제외하지만, 헤드 자체는 유지되어 값은 계속 계산된다.
EXTREME_EVENT_THRESH = {
    "rain":     0.83,
    "heatwave": 0.50,
    "coldwave": 0.50,
    "dust":     0.50,
}

# 관측소별 임계값 예외(2026-08-19 재검증, 한파 헤드 승격 이후).
#
# 부산 폭염(0.40)은 다시 검증했으나 순이득 −0.0033 으로 여전히 기각(전역
# 0.50 유지) — 2026-08-17 결론과 동일.
#
# 전주 한파는 calibration_plot_diagnose.py 에서 정밀도 18.1%로 다른 관측소
# (35~58%, 신뢰구간이 서로 겹치지 않음)와 뚜렷이 구분되는 이상치였다.
# station_threshold_check.py 로 보정용/평가용 분리 검증한 결과 t=0.33 이
# 전역 t=0.50 대비 순이득 +0.0689(F1 0.1216→0.1905)로 채택 기준(0.01)을
# 크게 넘었다. 여기 담는 값은 원본(raw, 보정 전) 확률 공간의 임계값이다 —
# event_threshold() 가 이 값을 calibrate_prob() 에 통과시켜 보정 공간으로
# 옮긴 뒤 비교하므로, 검증에 쓴 것과 같은 공간(raw)에서 고른 값을 그대로
# 넣는다.
# 널 캘리브레이션(2026-08-31, null_calibration_check.py — Risk_Prediction
# calibration.py fit_null() 이식, cross-project-technique-survey 메모 참고)
# 로 찾은 후보 중 6건. 방법: 무사건(공식 라벨 음성) 구간 확률의
# μ_null+z·σ_null 을 임계값 후보로 두고, z 는 보정용 절반의 F1 로 고른 뒤
# 평가용 절반에서만 채점했다(threshold_validation.py 와 동일 CALIB_SEED=1234
# 분할, 순이득 0.01 이상만 채택). **z 를 F1 로 고르므로 이 방식도 결국 F1
# 최대화이며, 기존 그리드서치와 목적함수가 같고 탐색 공간(관측소별 μ,σ 로
# 재매개변수화된 성긴 격자)만 다르다** — 2026-08-31 점검에서 "F1 을 직접
# 보지 않는 방법"이라는 최초 서술이 코드와 어긋남을 확인해 정정했다.
#
# 후보 8건 중 5건만 채택했다. 탈락 3건의 사유가 서로 다르므로 함께 남긴다.
#
# **① 평탄 구간 배제(2건).** 여기 담는 값은 원본(raw) 공간이고
# event_threshold() 가 보정 곡선으로 옮겨 판정하는데, 등온 회귀의 평탄
# 구간에 임계값이 떨어지면 판정이 보존되지 않는다(CLAUDE.md 12절, 실제
# F1 0.4566→0.4484 사고). 곡선에 대조한 결과 인천 폭염(raw 0.6565)과
# 수원 한파(raw 0.6000)가 τ±0.01 에서 보정폭 0.0 인 완전 평탄 구간에
# 떨어져 제외했다 — raw 공간의 순이득(+0.0240·+0.0502)이 서빙 공간에서
# 재현된다는 근거가 없다.
#
# **② 합산 지표 악화로 배제(1건) — 광주 한파(raw 0.1749).**
# 이 건은 평탄 구간 검사를 통과했고 광주 자체 표본에서는 F1 이 0.0000→
# 0.0980 으로 크게 올랐다. 그런데 **합산 서빙 한파 F1 은 0.5001→0.4859 로
# 떨어진다**(실측). 임계값이 0.1749 로 매우 낮아 광주에서 오탐이 쏟아지고,
# 그 오탐이 합산 정밀도를 끌어내리기 때문이다 — 양성 96건을 얻으려다
# 3,864건 전체의 지표를 깎는 셈이다. **F1 은 부분집합에서 합산으로
# 분해되지 않는다** — 관측소별 순이득이 양수라고 합산도 양수가 되지
# 않는다는 것을 2026-08-31 에 실측으로 확인했다. 앞으로 관측소별 예외는
# 그 관측소 표본의 순이득뿐 아니라 **합산 지표 비회귀**도 함께 확인한다.
#
# 채택한 5건의 합산 효과(실측): 폭염 0.8139→0.8165(+0.0026),
# 한파 0.5001→0.5004(+0.0003), 황사 변화 없음.
# 전주 한파(146)는 이미 그리드서치로 검증된 값이 있어 건드리지 않는다.
STATION_EVENT_THRESH_OVERRIDES: dict = {
    ("146", "coldwave"): 0.33,
    ("131", "heatwave"): 0.7142,  # 청주, 관측소 F1 0.7842→0.7995 (보정폭 4.7e-02)
    ("119", "heatwave"): 0.6781,  # 수원, 관측소 F1 0.8016→0.8162 (보정폭 4.5e-03)
    ("105", "coldwave"): 0.4167,  # 강릉, 관측소 F1 0.3947→0.4000 (보정폭 2.7e-02)
    ("108", "coldwave"): 0.6129,  # 서울, 관측소 F1 0.5451→0.5542 (보정폭 7.0e-03)
    ("112", "coldwave"): 0.5269,  # 인천, 관측소 F1 0.4444→0.4494 (보정폭 5.9e-02)
}


def raw_event_threshold(event: str, stn: str) -> float:
    """관측소별 재보정값이 있으면 그걸, 없으면 전역 기본값을 반환한다(보정 전)."""
    return STATION_EVENT_THRESH_OVERRIDES.get((stn, event), EXTREME_EVENT_THRESH[event])


def calibrate_prob(prob: float, event: str, ckpt: dict) -> float:
    """
    확률 보정(isotonic) 곡선을 적용한다. 곡선이 없는 체크포인트면 그대로 둔다.

    왜 필요한가(2026-08-16 실측, probability_calibration_check.py) — 희소 사건이
    '항상 아니다'로 붕괴하는 것을 막으려고 BCE에 준 `pos_weight`가 양성 확률을
    위로 밀어올린다. 그 결과 헤드 넷 모두 체계적으로 과신했다(한파는 "60~70%"
    라 해도 실제 빈도가 27%). 화면은 이 값을 확률 막대로 그대로 내놓고 있었다.

    보정은 단조 증가 함수이나 **순증가는 아니다** — 등온 회귀 결과는 평탄
    구간을 가지므로 서로 다른 원본 확률이 같은 보정값으로 접힌다. 따라서
    임계값을 같은 곡선으로 옮겨도 판정이 항상 보존되지는 않는다(2026-08-17
    한파 헤드에서 F1 0.4566→0.4484로 실측). 판정선은 `event_threshold()`가
    별도로 처리한다.

    강수("rain")에는 적용하지 않는다 — 화면에 확률이 아니라 mm로 나가고,
    양(amount) 헤드가 보정 전 확률과 곱해지도록 함께 학습됐다. 여기서 확률만
    바꾸면 학습된 곱이 깨져 강수량 자체가 틀어진다. 곡선은 분석용으로
    체크포인트에 남겨두되 서빙 경로에서는 호출하지 않는다.
    """
    cal = (ckpt or {}).get("prob_calibration") or {}
    head = (cal.get("heads") or {}).get(event)
    if not head or prob is None:
        return prob
    return float(np.interp(prob, head["x"], head["y"]))


def event_threshold(event: str, stn: str, ckpt: dict = None) -> float:
    """
    판정 임계값. 확률을 보정했으면 판정선도 보정 공간의 값이어야 한다 — 확률만
    보정하고 임계값을 그대로 두면 판정 기준이 어긋난다.

    보정 공간의 판정선은 두 경로로 정해진다.
      · `threshold_decision` — `probability_calibration_fit.py`가 보정 공간에서
        직접 고른 값. 곡선의 평탄 구간 때문에 '곡선으로 옮기기'가 판정을
        보존하지 못할 때만 재선정하며, 고른 표본과 채점한 표본을 분리해
        순이득이 0.01 이상일 때만 채택한다.
      · 그 키가 없으면(재선정 도입 이전 체크포인트) 원본 임계값을 곡선으로
        옮긴 값을 그대로 쓴다.

    관측소별 예외가 걸린 조합은 전역 재선정값을 쓰지 않는다 — 예외는 그
    관측소 표본에서 따로 검증해 고른 값이므로 전역값으로 덮으면 근거가 사라진다.
    """
    raw = raw_event_threshold(event, stn)
    if (stn, event) not in STATION_EVENT_THRESH_OVERRIDES:
        head = (((ckpt or {}).get("prob_calibration") or {}).get("heads") or {}).get(event)
        if head and head.get("threshold_decision") is not None:
            return float(head["threshold_decision"])
    return calibrate_prob(raw, event, ckpt)


def load_model(checkpoint_path: str = CHECKPOINT, device: str = DEVICE):
    """체크포인트에서 모델 아키텍처와 가중치를 복원한다."""
    ckpt = torch.load(checkpoint_path, map_location=device, weights_only=True)

    model = TriCHEFPipeline(
        embed_dim=ckpt["embed_dim"],
        num_features=ckpt["num_features"],
        alpha_init=ckpt["alpha_init"], phi_init=ckpt["phi_init"],
        orthogonalize=ckpt["orthogonalize"],
        temp_mean=ckpt["temp_mean"], precip_mean=ckpt["precip_mean"],
        persistence_residual=ckpt["persistence_residual"],
        feat_mean=ckpt["mean"], feat_std=ckpt["std"],
        # 구버전 체크포인트(이 키들 추가 전) 호환 — 당시 기본값과 동일
        dynamic_gate=ckpt.get("dynamic_gate", True),
        compact_satellite=ckpt.get("compact_satellite", True),
        re_channels=ckpt.get("re_channels", 4),   # 4=구버전(강수 채널 없음)
        im_dim=ckpt.get("im_dim", 384),   # 384=MiniLM(구버전), 12=경향벡터
        # 없으면 False — 이 플래그 도입 이전 체크포인트는 헤드 입력이
        # embed_dim 이므로 그대로 복원해야 state_dict 이 맞는다.
        signed_head_input=ckpt.get("signed_head_input", False),
        signed_precip_input=ckpt.get("signed_precip_input", False),
        head_dropout=ckpt.get("head_dropout", 0.0),
        coldwave_dropout=ckpt.get("coldwave_dropout", 0.0),
        precip_gamma_nll=ckpt.get("precip_gamma_nll", False),
        precip_gamma_alpha_init=ckpt.get("precip_gamma_alpha_init", 1.0),
        precip_gamma_mu_init=ckpt.get("precip_gamma_mu_init"),
        precip_quantile=ckpt.get("precip_quantile", False),
    ).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    return model, ckpt


def predict(stn: str = "108",
            checkpoint_path: str = CHECKPOINT,
            device: str = DEVICE,
            model=None, ckpt: dict = None) -> dict:
    """
    지정 관측소의 현재 관측값으로 +lead_hours 시간 후 기온·강수를 예측.

    Re축은 다시 켠다 — interp_field_collector.py 실측(2026-08-09): 가짜
    위성(정보량 0)을 12관측소 실측 IDW 공간보간장으로 바꾸자 게이트가 Re축에
    51.6%를 배분했고 폭염·한파·황사 F1 이 전부 크게 올랐다(황사 0.257→0.671).
    이전엔 img_x=None 으로 꺼서 배포 RAM 을 아꼈지만, 이제 Re축이 가장 큰
    비중을 차지하므로 끄면 방금 얻은 개선을 그대로 잃는다.

    Im축은 체크포인트에 따라 자동 분기한다(model.im_dim) — 신버전(12차원
    시간 경향 벡터, tendency_collector.py, Phase 3-11)이면 sentence-
    transformers 로드가 아예 없고, 구버전(384차원 MiniLM 텍스트)이면 기존
    방식으로 폴백한다. 전환기 호환을 위한 분기이며, 재학습이 끝나 구버전
    체크포인트를 더 안 쓰게 되면 단순화할 수 있다.

    두 축 모두 "다른 시각·다른 관측소"의 실측이 필요해 API 조회가 늘어난다:
      · Re축(공간): 같은 시각 다른 11개 관측소 → 12회
      · Im축(시간): 대상 관측소의 1h/3h/6h 전 관측 → 3회
    (호출당 총 15회 — 이전 1회 대비 응답 지연 증가는 감수). 일부 조회가
    실패해도 InterpolatedFieldCollector/TendencyCollector 가 남은 것만으로
    동작하거나(전무하면 중립값 0.5 또는 0) 정상 동작한다.

    model/ckpt 를 미리 만들어 넘기면 재사용한다. CLI(1회 실행)에서는 None 으로
    두면 내부에서 새로 만든다. app.py 처럼 반복 호출되는 환경에서는 호출자가
    st.cache_resource 로 캐싱해 넘기는 게 좋다(체크포인트 재로드 방지).
    """
    if model is None or ckpt is None:
        model, ckpt = load_model(checkpoint_path, device)
    lead_hours = ckpt["lead_hours"]

    # 대상 관측소 + 보간용 이웃 11개 — 전부 **같은 시각** 스냅샷이 필요하다.
    #
    # 가짜 데이터를 입력으로 쓰지 않는다는 규칙(CLAUDE.md 1절 5항, Re축이
    # 정보량 0이었던 바로 그 원인)이 여기 걸린다. 다만 '실측이 아닌 것'과
    # '지금 것이 아닌 것'은 구분해야 한다 — 전자만 배제한다.
    #
    # 대상 관측소 자신의 레코드는 폴백이어도 그대로 쓴다 — Z축 입력이 아예
    # 없으면 예보를 만들 수 없고, 그 상태는 data_status 로 호출부에 전달돼
    # 화면 상단 배지에 경고로 표시된다.
    record = None
    fetched = []
    for other_stn in STATIONS.values():
        try:
            r = RobustWeatherCollector(stn=other_stn).fetch()
        except Exception:
            continue
        if other_stn == stn:
            record = r
        fetched.append(r)
    if record is None:
        record = RobustWeatherCollector(stn=stn).fetch()
        fetched.append(record)

    # 보간장에 넣을 이웃을 고른다. 두 조건을 함께 건다(2026-08-19 개편).
    #
    #  ① 실측일 것 — is_real_observation() 이 날조 기본값(FALLBACK_DEFAULT)만
    #     걸러낸다. 저장소 창에서 온 값(FALLBACK_WINDOW)은 관측 시각이 과거일
    #     뿐 실제 관측이므로 받는다. 종전에는 SUCCESS_LIVE 만 받아, API 연결이
    #     막히면 이웃이 0곳이 되어 Re축 전체가 중립값(0.5)으로 죽었다.
    #
    #  ② 대상 레코드와 같은 관측 시각일 것 — IDW 는 '같은 시각의 공간 분포'를
    #     전제한다. 실측이라도 시각이 다른 값을 섞으면 실제로 존재한 적 없는
    #     공간장을 만든다. 연결이 막혀 저장소 창으로 내려가면 12곳이 같은
    #     시각으로 채워지므로 이 조건에서 그대로 살아남는다.
    snapshot_ts = str(record.get("timestamp", ""))[:12]
    neighbor_records = [
        r for r in fetched
        if is_real_observation(r.get("status"))
        and str(r.get("timestamp", ""))[:12] == snapshot_ts
    ]
    # get_image() 는 대상 관측소를 스스로 제외하므로, 실제 보간에 쓰이는 건
    # 여기서 자신을 뺀 수다.
    live_neighbors = sum(1 for r in neighbor_records if str(r.get("stn")) != stn)

    field_collector = InterpolatedFieldCollector(
        neighbor_records, STATION_COORDS, n_bands=getattr(model, "re_channels", 4))

    # Im축 인코더 종류에 따라 분기 — im_dim=384 면 구버전(MiniLM 텍스트),
    # 12 면 신버전(시간 경향 벡터). 재학습 전환기 동안 두 체크포인트 모두
    # 이 함수 하나로 서빙하기 위해 필요하다.
    live_lags = None   # MiniLM 분기에는 시차 조회 자체가 없다
    if getattr(model, "im_dim", 384) >= 128:
        txt_vec = SimulatedTextCollector().encode_single(record)
    else:
        tendency_records = [record]
        target_collector = RobustWeatherCollector(stn=stn)
        now_ts = datetime.strptime(str(record["timestamp"])[:12], "%Y%m%d%H%M")
        for lag in LAGS_HOURS:
            try:
                past = target_collector.fetch_at(
                    (now_ts - timedelta(hours=lag)).strftime("%Y%m%d%H00"))
            except Exception:
                continue
            # 과거 시각도 실측만 받는다. 날조 기본값을 넣으면
            # TendencyCollector 가 "그 시각에 실제 관측이 있었다"고 믿고
            # 거대한 가짜 변화량을 만들어낸다 — 빼면 설계대로 0(변화 정보
            # 없음)이 된다.
            #
            # 저장소 창에서 온 값(FALLBACK_WINDOW)은 받는다. fetch_at 이
            # exact_time=True 로 폴백하므로 **요청한 바로 그 시각의 실측**만
            # 돌아온다 — 타임스탬프가 요청 시각과 일치함이 보장되므로 차분이
            # 실제 관측 간의 변화량이 된다.
            if is_real_observation(past.get("status")):
                tendency_records.append(past)
        live_lags = len(tendency_records) - 1
        txt_vec = TendencyCollector(tendency_records).encode_single(record)

    mean = np.array(ckpt["mean"], dtype=np.float32)
    std  = np.array(ckpt["std"],  dtype=np.float32)
    # record_to_vec() 은 14차원(연중 시각 sin/cos, 2026-08-16 추가)인데 구버전
    # 체크포인트는 12차원으로 학습됐다. 새 축은 끝에 덧붙었으므로 그 체크포인트의
    # 차원만큼만 앞에서 잘라 쓰면 구버전 입력을 그대로 복원한다.
    nf = ckpt.get("num_features", len(mean))
    num_vec = np.array(record_to_vec(record), dtype=np.float32)
    if ckpt.get("use_island", False):
        # 도서 AWS 13개 지점 강수(26차원, island_collector.py) — 학습 때와
        # 같은 순서(record_to_vec 뒤에 이어붙임)로 실시간 조회한다. 실패해도
        # 전부 결측(중립값 0.5+플래그 1)으로 채워 정상 응답한다(날조 금지).
        #
        # **레코드의 관측 시각을 넘긴다** — 벽시계 현재 시각을 쓰면 학습
        # (소스 레코드 시각으로 조회)과 다른 시각을 보게 되어 짝이 깨진다.
        # 위 이웃 관측소 선별이 `timestamp == snapshot_ts` 를 강제하는 것과
        # 같은 이유다(2026-08-31 수정).
        num_vec = np.concatenate(
            [num_vec, island_encode_live(record.get("timestamp"))])
    if ckpt.get("use_climatology_anomaly", False):
        # 학습 때와 같은 순서(record_to_vec → 도서 특징 → 이상편차)로 붙인다.
        # 평년값 테이블은 체크포인트에 저장돼 있어 서빙 시점에 히스토리를
        # 다시 로드하지 않는다(train.py 저장부 주석 참고).
        from train import climatology_anomaly
        table = ckpt.get("climatology_table") or {}
        num_vec = np.concatenate(
            [num_vec, [climatology_anomaly(record, table)]]).astype(np.float32)
    num_vec  = num_vec[:nf]
    num_norm = (num_vec - mean) / std

    x_num = torch.tensor(num_norm, dtype=torch.float32).unsqueeze(0).to(device)
    x_img = torch.tensor(field_collector.get_image(record),
                         dtype=torch.float32).unsqueeze(0).to(device)
    x_txt = torch.tensor(txt_vec, dtype=torch.float32).unsqueeze(0).to(device)

    with torch.no_grad():
        pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)

    temp_pred   = pred[0, 0].item()
    precip_pred = max(0.0, pred[0, 1].item())

    # Phase 3-8/3-9 극한기상 확률 — 구버전 체크포인트(헤드 추가 전)는
    # extreme_event_probs() 가 전부 None 을 반환하므로 결과에서 빠진다.
    extreme = model.extreme_event_probs()

    # 강수 후처리 게이팅 — rain_prob 이 있으면(head_rain 도입 이후 체크포인트)
    # 그 확률 공간에서 자르고, 없으면(구버전) 매그니튜드로 폴백한다. rain_prob
    # 은 보정하지 않은 원본 값을 그대로 쓴다 — 강수 헤드에는 확률 보정을
    # 적용하지 않는다(README '확률 보정' 절, 회귀 헤드라 곱셈 구조가 보정
    # 전 확률과 맞물려 학습됐기 때문).
    if extreme["rain"] is not None:
        rain_prob = extreme["rain"][0].item()
        prob_gate = PRECIP_PROB_GATE_BY_LEAD.get(ckpt.get("lead_hours"), PRECIP_PROB_GATE)
        if rain_prob < prob_gate:
            precip_pred = 0.0
    else:
        rain_prob = None
        if precip_pred < PRECIP_CLIP_THRESH:
            precip_pred = 0.0

    gate = model.axis_weights()   # dynamic_gate=False 면 {"alpha":..,"phi":..}

    # 확률 보정을 여기서 적용한다 — 호출자(app.py·CLI)가 받는 값이 곧 화면에
    # 나가므로, 보정되지 않은 값이 밖으로 새어 나가지 않게 한 곳에서 처리한다.
    heatwave_prob = (calibrate_prob(extreme["heatwave"][0].item(), "heatwave", ckpt)
                     if extreme["heatwave"] is not None else None)
    coldwave_prob = (calibrate_prob(extreme["coldwave"][0].item(), "coldwave", ckpt)
                     if extreme["coldwave"] is not None else None)
    dust_prob     = (calibrate_prob(extreme["dust"][0].item(), "dust", ckpt)
                     if extreme["dust"] is not None else None)

    observed_at = record.get("timestamp")
    # accuracy.py 가 (관측소, 목표시각)으로 로그를 남기려면 이 시각이 필요하다
    # — 시간대는 관측시각 문자열이 이미 KST 벽시계 표기라 그대로 산술한다.
    target_time = (
        datetime.strptime(str(observed_at)[:12], "%Y%m%d%H%M")
        + timedelta(hours=lead_hours)
    ).strftime("%Y%m%d%H00") if observed_at else None

    return {
        "station_code": stn,
        "station_name": STATION_NAMES.get(stn, stn),
        "observed_at":  observed_at,
        "target_time":  target_time,
        "data_status":  record.get("status"),   # SUCCESS_LIVE / FALLBACK_*
        # 실측으로 채워진 보조 입력의 개수. 대상 관측소의 status 만으로는
        # Re·Im축이 얼마나 채워졌는지 알 수 없다 — 자기 조회는 성공했는데
        # 이웃·과거 조회가 예산 상한이나 API 장애로 폴백된 상태가 가능하다.
        # 그 경우 보간장은 중립값(0.5), 경향벡터는 0으로 채워지므로 예보의
        # 근거가 실제로 얇아진다. 화면이 그 사실을 말할 수 있게 내보낸다.
        "input_coverage": {
            "neighbors_live":  live_neighbors,
            "neighbors_total": len(STATIONS) - 1,
            "lags_live":       live_lags,
            "lags_total":      len(LAGS_HOURS) if live_lags is not None else None,
        },
        "current": {
            "temperature":   record.get("temperature"),
            "precipitation": record.get("precipitation"),
            "humidity":      record.get("humidity"),
            "wind_speed":    record.get("wind_speed"),
            "pressure":      record.get("pressure"),
        },
        "forecast_lead_hours": lead_hours,
        "forecast": {
            "temperature":   round(temp_pred, 2),
            "precipitation": round(precip_pred, 2),
            # 강수 확률(게이팅 전 원본, 미보정) — 2026-09-01 추가. 게이팅으로
            # 0.0mm 가 된 경우와 "비가 안 온다"고 확신한 경우가 화면에서
            # 구분되지 않는 문제(예: rain_prob=0.84 인데 τ=0.85 라 0mm 로
            # 표시)를 app.py 가 드러낼 수 있게 노출한다. 임계값 자체를
            # 옮기는 건 아니다 — 그 여지는 이미 소진됐다(improvement-levers
            # -measured 메모리, 임계값 재선정 여유 실측 +0.0015). 여기서는
            # 이미 계산돼 있던 값을 반환값에서 빠뜨리지 않는 것뿐이다.
            "rain_prob": round(rain_prob, 4) if rain_prob is not None else None,
        },
        "extreme_event_probs": {
            "heatwave": round(heatwave_prob, 4) if heatwave_prob is not None else None,
            "coldwave": round(coldwave_prob, 4) if coldwave_prob is not None else None,
            "dust":     round(dust_prob, 4)     if dust_prob     is not None else None,
        },
        "gate_weights": (
            {k: round(v, 4) for k, v in gate.items()} if "w_re" in gate else None
        ),
    }


def _print_human(result: dict) -> None:
    print(f"{'='*52}")
    print(f" Tri-CHEF 기후 예보 — {result['station_name']}({result['station_code']})")
    print(f"{'='*52}")
    status_note = "" if result["data_status"] == "SUCCESS_LIVE" else \
        f"  ⚠ {result['data_status']} (실시간 데이터 아님)"
    print(f" 관측 시각 : {result['observed_at']}{status_note}")

    c = result["current"]
    print(f" 현재 기온 : {c['temperature']:.1f} °C")
    print(f" 현재 강수 : {c['precipitation']:.1f} mm")
    print(f" 현재 습도 : {c['humidity']:.0f} %")
    print(f"{'-'*52}")

    f = result["forecast"]
    print(f" +{result['forecast_lead_hours']}시간 후 예보")
    print(f"   기온 : {f['temperature']:.1f} °C")
    print(f"   강수 : {f['precipitation']:.1f} mm")

    ev = result.get("extreme_event_probs") or {}
    if any(ev.get(k) is not None for k in ("heatwave", "coldwave", "dust")):
        print(f"{'-'*52}")
        if ev.get("heatwave") is not None:
            print(f" 폭염 확률 : {ev['heatwave']:.1%}")
        if ev.get("coldwave") is not None:
            print(f" 한파 확률 : {ev['coldwave']:.1%}")
        if ev.get("dust") is not None:
            print(f" 황사 확률 : {ev['dust']:.1%}")

    if result["gate_weights"]:
        g = result["gate_weights"]
        print(f"{'-'*52}")
        print(f" 축 배분 — 위성 {g.get('w_re', 0):.1%} | "
              f"텍스트 {g.get('w_im', 0):.1%} | 수치 {g.get('w_z', 0):.1%}")
    print(f"{'='*52}")


def main():
    parser = argparse.ArgumentParser(description="Tri-CHEF 기후 예보 추론")
    parser.add_argument("--stn", default="108",
                        help="관측소 코드 (기본: 108=서울)")
    parser.add_argument("--json", action="store_true",
                        help="JSON 형식으로 출력 (app.py·API 연동용)")
    parser.add_argument("--checkpoint", default=CHECKPOINT,
                        help=f"체크포인트 경로 (기본: {CHECKPOINT})")
    parser.add_argument("--list-stations", action="store_true",
                        help="사용 가능한 관측소 코드 목록 출력 후 종료")
    args = parser.parse_args()

    if args.list_stations:
        for name, code in STATIONS.items():
            print(f"  {code}  {name}")
        return

    if args.stn not in STATIONS.values():
        print(f"[ERROR] 알 수 없는 관측소 코드: {args.stn}", file=sys.stderr)
        print(f"  사용 가능: {dict(STATIONS)}", file=sys.stderr)
        sys.exit(1)

    result = predict(stn=args.stn, checkpoint_path=args.checkpoint)

    if args.json:
        print(json.dumps(result, ensure_ascii=False, indent=2))
    else:
        _print_human(result)


if __name__ == "__main__":
    main()
