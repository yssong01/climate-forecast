"""
train.py — Tri-CHEF 3축 파이프라인 학습.
논문 Eq.1: s = √((w_Re·Re)² + (w_Im·Im)² + (w_Z·Z)²)

Re축(공간): 인접 11개 관측소 실측 IDW 공간보간장 → 소형 CNN
Im축(시간): 1·3·6시간 전 대비 변화량(경향 벡터) → MLP 프로젝션
Z축(현재): 수치 센서(기온·강수·습도 등) → MLP 인코더

Re·Im축은 원래 위성 이미지·MiniLM 텍스트 임베딩이었으나, 정보량이 정확히
0으로 실측되어(2026-08-09) 위와 같이 실측 데이터로 교체했다 — 상세 경위는
README.md '정직한 한계 1번' 참조.

Phase 3-4에서 확정된 사항
  1. Gram-Schmidt 직교화(ORTHOGONALIZE 플래그로 ablation 가능) — 본 도메인에서는
     성능이 저하되어 기본값 False.
  2. dying-clamp 버그 수정(pipeline_model: clamp → softplus).
  3. 손실 스케일 정규화 — 기온 MSE가 강수 MSE를 1000배 압도하던 문제.
  4. 축 직교성·강수 예측 분산 진단 로그.
"""
import os
import json
import time
import torch
import torch.nn as nn
from torch.utils.data import (Dataset, DataLoader, Subset, WeightedRandomSampler,
                              BatchSampler, RandomSampler, SequentialSampler)
import numpy as np
from datetime import datetime, timedelta
from dotenv import load_dotenv

from weather_collector import RobustWeatherCollector, STATIONS, STATION_COORDS, KST
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector, TENDENCY_DIM
from text_collector import SimulatedTextCollector
from pipeline_model import TriCHEFPipeline

load_dotenv()

# ── 설정 ──────────────────────────────────────────────────────────
DEVICE     = "cuda" if torch.cuda.is_available() else "cpu"
EMBED_DIM  = 64

# 배치 크기 32 → 1024 (2026-08-15 perf_profile.py 실측).
# 데이터가 13년치(표본 138만)로 늘어난 뒤 학습 1회가 4시간 20분이 걸렸는데,
# 그동안 GPU 사용률은 31%, VRAM 은 8GB 중 747MiB 였다. 병목은 GPU 연산이
# 아니라 에폭당 34,524회 반복되는 파이썬 스텝 오버헤드였다. 실측 처리량:
#     batch   32 + workers  0 :  3,822 표본/초  (에폭 4.82분)  ← 종전
#     batch  512 + workers  8 : 45,426 표본/초  (에폭 0.41분)
#     batch 1024 + workers  8 : 74,956 표본/초  (에폭 0.25분)  ← 채택, 19.6배
# num_workers 만 늘리는 것으로는 오히려 느려진다(batch 32 에서 0.84배) —
# 배치가 작아 워커 통신 비용이 이득을 넘기 때문이다. 배치를 키우는 것이
# 본질적인 해법이다.
#
# 배치 크기는 학습 동역학을 바꾸므로 그냥 못 바꾼다. 이 모델에서 안전한
# 근거: 정규화 계층이 전부 LayerNorm 이라(pipeline_model.py) 배치 통계에
# 의존하지 않는다. BatchNorm 이 있었다면 배치 크기가 정규화 자체를 바꿨을
# 것이다. 게이트 워밍업도 에폭이 아니라 스텝 수에서 역산하도록 되어 있어
# (_REFERENCE_WARMUP_STEPS) 자동으로 따라간다.
#
# 다만 속도를 얻는 대신 정확도를 잃을 수 있다(2026-08-15 batch 1024 실행에서
# 기온 1.2033→1.2591, 강수 기준선 대비 −7.1%→−22.0%). 배치 크기 자체를
# 실험 변수로 다뤄야 하므로 환경변수로 빼둔다 — 코드를 고치지 않고
# BATCH_SIZE=256 처럼 지정해 대조 실행을 돌리기 위한 것이다.
BATCH_SIZE = int(os.getenv("BATCH_SIZE", "1024"))
_REFERENCE_BATCH = 32     # LR 스케일링 기준점 — 종전 배치 크기

# DataLoader 워커 수. 배치가 커진 뒤에야 이득이 나온다(위 실측 참고).
# 도커에서는 --shm-size 를 키워야 워커가 죽지 않는다(기본 64MB 로는 부족).
NUM_WORKERS = 8

EPOCHS     = 400    # 배치가 32배 커져 에폭당 스텝 수가 그만큼 줄었다.
                    # 같은 학습량을 확보하려면 에폭 수를 늘려야 한다.
                    # 에폭당 0.25분이라 400에폭도 100분이면 끝난다.

# AdamW 에서 배치를 k배 키울 때는 학습률을 √k 배 하는 것이 통용되는 규칙이다
# (SGD 의 선형 스케일링과 달리 Adam 계열은 제곱근 쪽이 안정적이다).
# 32 → 1024 이면 √32 ≈ 5.66배.
#
# 이 규칙이 이 모델에 맞는지는 아직 미검증이다 — batch 1024 + √32 스케일링
# 실행에서 LR 이 80에폭 만에 5.7e-3 → 4.4e-5 로 급락했는데, 최적화기가
# 헤맸다는 신호로 읽힌다. 스케일링이 과했는지 배치 크기 자체가 문제인지
# 가르려면 LR 을 배치와 독립적으로 지정할 수 있어야 한다. TRAIN_LR 로
# 직접 주면 스케일링을 건너뛴다.
LR         = float(os.getenv("TRAIN_LR",
                             1e-3 * (BATCH_SIZE / _REFERENCE_BATCH) ** 0.5))
ALPHA_INIT = 0.4    # Im 축(텍스트) 가중치 초기값 — 학습 중 자동 조정
PHI_INIT   = 0.2    # Z  축(수치)  가중치 초기값 — 학습 중 자동 조정
N_HOURS    = 720    # 관측소당 수집 기간: 30일 × 24시간 (장마철 포함 확보)
PATIENCE   = 40     # Early stopping — 에폭당 스텝 수가 줄어든 만큼 늘린다
VAL_RATIO  = 0.2

# 데이터 확장 — 서울 단일 관측소(표본 572개)는 이번 세션 대부분의 이상 현상
# (위성 인코더 과적합, 게이트 왜곡)의 근본 원인이었다. 12개 관측소 전체로
# 확장해 표본을 약 15배(≈8,600개)로 늘린다.
STATIONS_TO_COLLECT = list(STATIONS.values())          # 12개 관측소 전체
STATION_NAMES = {v: k for k, v in STATIONS.items()}    # 코드 → 한글명

# 예보 시계 — t 시각 관측으로 t+LEAD_HOURS 를 예측한다.
# 1시간에서는 퍼시스턴스(T(t+1)≈T(t))가 사실상 무적이라 모델 기여를 측정할
# 수 없었다(실측: 기온 +0.7%). 6시간이면 일주기가 지배적이 되어 시각 특성과
# 기압 추세를 학습한 모델이 baseline 을 이길 여지가 생긴다.
LEAD_HOURS = 6

# Gram-Schmidt 직교화 — ablation.py 실측 결과 기본값 False.
#   ON  : 기온 0.8886 / 강수 0.3267
#   OFF : 기온 0.8452 / 강수 0.2889   ← 우세
# OFF 조건에서 네트워크가 학습한 Re-Im 상관도가 0.4809(ON은 0.0846)로,
# 축 간 공유 정보가 회귀 성능에 기여함을 뜻한다. 검색 태스크(논문 원본)와
# 달리 예보 태스크에서는 직교성 강제가 손해다.
ORTHOGONALIZE = False
PERSISTENCE_RESIDUAL = True   # 기온을 delta 예측으로 전환 (아래 설명 참조)

# Phase 3-5 동적 게이팅 — 입력마다 축 가중치를 다르게 산출한다.
DYNAMIC_GATE  = True
# 게이트 엔트로피 정규화 계수. 가중치가 소수 축에 집중되도록 압력을 준다.
GATE_ENTROPY_WEIGHT = 0.05

# 엔트로피 항 워밍업 — 이 구간에서는 λ=0 으로 두고 이후 선형 증가시킨다.
# 에폭 1부터 집중 압력을 걸면 모델이 어느 축이 유용한지 배우기 전에 초기
# 순서가 고착된다(Phase 3-5 1차 실측: 위성 축 98% 로 붕괴). 먼저 배우게 한
# 뒤 집중시킨다 — 이 의도 자체는 옳다.
#
# 문제는 원래 이 워밍업을 "에폭 수"로 고정(50)했다는 점이다. 2026-08-07
# 재학습(328,802개, 이전 대비 38배)에서 학습이 36에폭 만에 조기종료됐는데,
# 워밍업이 50에폭이라 엔트로피 정규화가 단 한 번도 걸리지 않은 채 끝났다 —
# 게이트 배분이 예전과 달라진 게 실제 학습 때문인지 이 미작동 때문인지
# 구분이 안 되는 오염된 결과였다.
#
# 근본 원인: "50에폭"이 실제로 의미하는 건 에폭 수 자체가 아니라
# "그 50에폭 동안 모델이 본 그래디언트 스텝 수"다. 원래 캘리브레이션
# 지점(Phase 3-5, 학습 표본 6,837개, 배치 32 → 에폭당 배치 약 213개)에서
# 50에폭 = 약 10,650스텝. 데이터가 늘어나면 에폭당 스텝 수도 그만큼
# 늘어나므로, 같은 "에폭 수"가 전혀 다른 학습량을 의미하게 된다. 그래서
# 고정 에폭 수 대신 저 스텝 수를 재현하는 에폭 수를 학습 표본 수로부터
# 역산한다 — train() 내부에서 n_train 이 확정된 뒤 계산.
_REFERENCE_WARMUP_STEPS = 50 * (6837 // BATCH_SIZE)   # ≈ 10,650 (Phase 3-5 기준)
_MIN_WARMUP_EPOCHS = 3   # 극단적으로 큰 데이터셋에서도 최소한의 학습 기회는 보장
# 위성 인코더 용량. ResNet18(1,100만)은 표본 572개를 암기해 게이트를 왜곡시킨다.
COMPACT_SATELLITE = True
PRECIP_WEIGHT = 1.0    # 강수 손실 가중치 (기온 손실은 σ² 로 정규화되어 O(1))
SEED          = 42     # 학습/검증 분할 고정 → ablation 비교 가능
# 모델 초기화·학습 stochasticity(가중치 초기값, DataLoader 셔플 순서)만
# 따로 흔들고 싶을 때 쓴다 — SEED(분할)는 그대로 두고 이것만 바꾸면 같은
# train/val 분할에서 "이번 결과가 이 재학습 런의 우연인지, 재현되는
# 구조적 현상인지"를 가릴 수 있다(2026-08-16, A안 롤백 원인 조사).
INIT_SEED     = int(os.getenv("INIT_SEED", str(SEED)))

# Phase 3-7 hurdle 헤드 — precip_breakdown.py 실측(2026-08-07): 검증셋의
# 93.3%가 무강수인데 softplus 단독 강수 헤드는 정확히 0을 못 내, 그 미세한
# 비영 오차가 누적돼 baseline(상시 0예측)을 이긴 강수 구간의 이득(+10.2%)을
# 삼켜버렸다. head_rain(이진 분류) 을 추가해 "오는가"를 명시적으로 학습시킨다.
WET_THRESH = 0.1              # mm — accuracy.PRECIP_THRESH 와 동일 기준 재사용
HURDLE_BCE_WEIGHT = 1.0       # BCE 는 그 자체로 O(1) 스케일이라 강수 MSE 항과 비슷하게 둠

# Phase 3-8 극한기상 경보 헤드 — head_rain 과 같은 이진분류 패턴을 폭염·한파에
# 적용한다. 임계값은 기상청 주의보 발표기준(공식 규정값)이라 데이터에서
# 추정하지 않고 고정한다 — wet_prior/rain_pos_weight 와 달리 "무엇을 사건으로
# 볼지"는 기상청이 이미 정의했고, 우리가 반응형으로 계산하는 건 "그 정의에
# 해당하는 표본이 학습셋에 몇 % 있는가"(prior)와 "그로부터 나오는 클래스
# 불균형 보정(pos_weight)"뿐이다.
#   폭염주의보: 일 최고기온 33°C 이상   (기상청 특보 발표기준)
#   한파주의보: 아침 최저기온 -12°C 이하 (기상청 특보 발표기준)
# 강풍(순간풍속 14m/s)은 12관측소·3.5년 환산 표본이 약 300개(0.1%)로
# 지나치게 희박해 이번 확장에서 제외했다(실측, 2026-08-07).
HEATWAVE_THRESH = 33.0        # °C
COLDWAVE_THRESH = -12.0       # °C

# 극한기상 헤드가 3개(폭염/한파/황사)로 늘어나면서 강수 hurdle 이 계속
# 나빠졌다(precip_breakdown.py 실측, 2026-08-09: 무강수 구간 평균 예측
# 0.0546→0.0677mm, baseline 대비 손해 -16.0%→-25.6%). 헤드 4개(강수·폭염·
# 한파·황사) 모두 embed_dim=64 짜리 하나의 공유 magnitude 표현에서 갈라져
# 나오는데, EXTREME_BCE_WEIGHT=1.0(HURDLE_BCE_WEIGHT와 동급)이면 극한기상
# 3개가 강수 1개보다 그래디언트 예산을 3배 더 가져간다 — 항 개수 불균형이
# 곧 표현 경쟁으로 번진 것으로 보인다. 0.3 으로 낮춰 극한기상 쪽 압력을
# 줄이고 강수가 회복되는지 실측한다(가설 검증용 실험치, 결과 보고 추가
# 튜닝 예정).
# 13년 재학습(그룹 분할) 이후에도 강수는 기준선 대비 -17.9% 미달로 남아
# 있어 추가 재조정을 실험 중이다. EXTREME_BCE_WT 환경변수로 코드 수정 없이
# 값을 바꿔 돌릴 수 있다(BATCH_SIZE/TRAIN_LR 과 같은 패턴, 2026-08-16).
EXTREME_BCE_WEIGHT = float(os.getenv("EXTREME_BCE_WT", "0.3"))

# 황사 손실 가중 — 0 이면 황사 헤드를 손실에서 제외해 학습시키지 않는다.
# 왜 이런 스위치가 필요한가(2026-08-17): 황사는 정밀도 11.8%로 실용 수준에
# 미달해 화면에서 이미 제외돼 있고, 12개 관측소 중 6곳은 라벨이 아예 없어
# 채점된 적도 없다. 그런데 헤드가 늘면 강수가 나빠진다는 것은 이 저장소가
# 이미 확립한 사실이다(EXTREME_BCE_WT 를 1.0→0.3 으로 낮춰 강수를
# -25.6%→-19.3% 로 개선한 실측). 헤드 하나를 손실에서 빼는 것은 같은 방향의
# 더 강한 조치이므로, 공유 표현 경쟁 가설을 직접 검증할 수 있다.
# 모듈 자체는 남겨 둔다 — state_dict 키가 유지돼야 구버전 체크포인트와
# 로더가 호환되고, 되돌릴 때 코드를 고칠 필요가 없다.
DUST_LOSS_WEIGHT = float(os.getenv("DUST_LOSS_WT", "1.0"))

# Phase 3-9 황사 헤드 + 폭염·한파 라벨 소스 교체 — import_weather_issues.py
# 가 만든 기상청 "날씨 이슈별 데이터"(공식 폭염특보/한파특보/황사관측여부)
# 조회표. 순간 임계값 근사 대신 실제 발표된 특보를 쓴다(WeatherDataset.
# __init__ 참고). 이 파일이 없으면 폭염·한파는 기존 임계값 근사로, 황사는
# 전부 마스크 제외로 조용히 폴백한다 — import_weather_issues.py 를 먼저
# 실행하지 않아도 학습 자체는 깨지지 않는다.
WEATHER_ISSUE_LABELS = "./cache/weather_issue_labels.json"

# 폭염·한파에서 공식 라벨이 없는 표본을 어떻게 다룰지.
#   True  (기본) — 마스크로 손실·채점 양쪽에서 제외한다. 황사가 원래 쓰던 방식.
#   False (대조군) — 종전처럼 임계값 근사로 채워 넣는다. 배치 크기 변경 같은
#                    다른 조건을 바꾼 상태에서 "마스킹만의 효과"를 분리해
#                    재려면 같은 조건의 대조군이 필요하다.
# 환경변수 EXTREME_LABEL_MASKING=0 으로 끌 수 있다 — 코드를 고치지 않고
# 대조군을 돌리기 위한 것이다.
EXTREME_LABEL_MASKING = os.getenv("EXTREME_LABEL_MASKING", "1") != "0"

# 학습/검증 분할 방식 — make_split() 참고. 무작위 분할이 일 단위 극한기상
# 라벨을 100% 누수시킨다는 것이 실측되어(2026-08-15) 모드를 추가했다.
#   random   — 시각 단위 무작위. 종전 방식. 비교용으로만 남긴다.
#   group    — 날짜 단위. 하루를 통째로 한쪽에 배정해 일 단위 라벨 누수를
#              없앤다. 사계절이 양쪽에 다 들어가 분산이 작아 개발용에 적합.
#   temporal — 과거로 배워 미래를 예측. 실제 운영과 같은 구조라 보고용.
#
# 기본값을 group 으로 바꿨다(2026-08-17). 종전 기본값은 random 이었는데,
# 규약(CLAUDE.md 2절)은 사건 단위 라벨을 그룹 분할로 검증하도록 요구하므로
# 매 실험마다 SPLIT_MODE=group 을 기억해서 붙여야 했다. 실제로 황사 헤드
# 제외 실험에서 이 환경변수를 빠뜨려, 배포본(group)과 검증셋이 다른 결과를
# 만들어 놓고 비교할 뻔했다 — 기준선 MAE 가 달라 눈치챘다. 규약이 요구하는
# 값을 기본값으로 두면 이 실수 자체가 불가능해진다.
SPLIT_MODE = os.getenv("SPLIT_MODE", "group")
# temporal 모드 경계. 사이 하루(2024-01-01)는 완충으로 비운다 — Im축이
# 6시간 전을 보고 타깃이 +6시간이라 경계 표본은 양쪽에 걸친다.
TEMPORAL_TRAIN_END = os.getenv("TEMPORAL_TRAIN_END", "20231231")
TEMPORAL_VAL_START = os.getenv("TEMPORAL_VAL_START", "20240102")

# Crop 표본 재가중 — 곰팡이 진단 프로젝트의 "곰팡이 영역만 먼저 집중
# 학습"(2단계 사전학습) 아이디어를 여기서는 먼저 더 단순한 형태로
# 시험한다: 아키텍처·학습 단계를 그대로 두고, 사건 표본(강수·폭염·한파·
# 황사 중 하나라도 양성)이 배치에 더 자주 뽑히도록 WeightedRandomSampler로
# 재가중만 한다. crop_headroom_diagnose.py 실측(2026-08-15): 사건 표본은
# 전체의 14.1%인데 그 안에서 강수·폭염·한파·황사 양성률이 3~9배로 뛴다.
# 이 재가중만으로 개선되면 2단계 사전학습(재앙적 망각 위험이 있는 더
# 복잡한 방법)은 필요 없다 — 더 단순한 대안을 먼저 배제하는 것이 원칙이다.
# 1.0 = 끔(기존 동작과 동일, 균등 샘플링).
CROP_OVERSAMPLE = float(os.getenv("CROP_OVERSAMPLE", "1.0"))

# 극한기상 헤드에 부호 있는 표현(v_z)을 함께 넣을지 여부(2026-08-16).
# Eq.1 융합이 제곱으로 부호를 없애 헤드가 |편차|만 보게 되고, 그 결과 실제
# 관측에서 한겨울에 폭염 확률이 올라갔다(v9: −15°C 이하 85.7%가 0.5 초과).
# 상세 진단은 seasonal_falsealarm_check.py 참고. 환경변수로 빼둔 이유는
# 대조군(끈 상태)과 나란히 돌려 이 변경만의 효과를 분리하기 위함이다.
SIGNED_HEAD_INPUT = os.getenv("SIGNED_HEAD_INPUT", "1") != "0"

# 강수 경로(확률·양)에도 같은 부호 표현을 줄지 여부(2026-08-17 실험).
# 극한기상 헤드에 적용했더니 강수 MAE가 −17.9%에서 −9.9%로 좋아졌다 —
# 헤드 간 공유 표현 경쟁이 완화된 것으로 해석했으므로, 강수 헤드 자신에게도
# 같은 여지를 주면 더 좋아질 수 있다는 가설이다. 기본값 0(끔) — 검증 전까지
# 배포 동작을 바꾸지 않는다.
SIGNED_PRECIP_INPUT = os.getenv("SIGNED_PRECIP_INPUT", "0") != "0"

# 헤드 드롭아웃 비율(2026-08-17). >0 이면 MC Dropout 으로 예측별 불확실성을
# 낼 수 있다 — 현재 화면의 ± 는 검증셋 평균오차일 뿐 이 예측 하나의
# 신뢰구간이 아니다. 다만 파라미터가 5.4만개뿐인 작은 모델이라 드롭아웃이
# 성능을 깎을 위험이 있어, 부호 표현 실험과 **섞지 않고** 따로 검증한다.
HEAD_DROPOUT = float(os.getenv("HEAD_DROPOUT", "0"))

# 한파 헤드 전용 드롭아웃(2026-08-17). HEAD_DROPOUT(전체 헤드 일괄 적용)은
# 겨울 폭염 오탐을 38.8%로 재발시켜 기각했다 — 드롭아웃이 폭염 헤드의 부호
# 표현(v_Z) 경로 유닛까지 무작위로 꺼버리기 때문이다. 그런데 그 실행에서
# 한파 F1은 0.449→0.472로 개선됐었다(사건 8~40건뿐인 희소 헤드라 과적합
# 억제 효과가 실제로 있다). 폭염·황사 헤드는 건드리지 않고 한파 헤드에만
# 정규화를 걸어 그 개선만 부작용 없이 취할 수 있는지 검증한다.
COLDWAVE_DROPOUT = float(os.getenv("COLDWAVE_DROPOUT", "0"))

# 저장 경로도 환경변수로 뺀다(2026-08-16) — 기본값은 배포가 읽는 경로다.
# seed를 바꿔가며 재현성을 확인하는 대조 실행은 서로 다른 경로에 저장해야
# 한다. 안 그러면 ① 동시에 돌린 두 실행이 같은 파일을 덮어쓰고 ② 검증도
# 끝나지 않은 실험 결과가 배포 경로를 곧바로 파괴한다.
CHECKPOINT  = os.getenv("CHECKPOINT_PATH", "./checkpoints/numerical_trichef.pt")
# historical_data.json(원본 30일)이 아니라 historical_data_1y.json 을 쓴다.
# 후자는 apihub 수집(collect_year.py)과 기상자료개방포털 파일셋 임포트
# (import_kma_fileset.py)로 병합해온 누적 데이터셋 — 2026-08-07 기준
# 12개 관측소 × 2023-01-01~현재, 328,802개 레코드(원래 목표였던 "최근
# 1년"보다 훨씬 긴 3.5년치). collect_incremental.py 가 이 파일에 계속
# 새 시간을 채워 넣으므로 재학습 때마다 최신 상태를 그대로 반영한다.
DATA_CACHE  = "./cache/historical_data_1y.json"
os.makedirs("./checkpoints", exist_ok=True)


# ── 1. 수치 벡터 변환 ─────────────────────────────────────────────

def record_to_vec(r: dict) -> list[float]:
    """
    관측 레코드 → 14차원 수치 벡터.

    인덱스 8·9(시각 sin/cos)는 Phase 3-4 에서 추가했다. 기온의 일주기
    (diurnal cycle)는 기온 예측의 지배적 요인인데 기존 8차원에는 시각
    정보가 전혀 없어 모델이 낮/밤을 구분할 수 없었다.
    인덱스 10·11(위도·경도)은 다중 관측소 확장(1단계)에서 추가했다.
    관측소마다 기후가 다르므로(강릉의 동해 영향, 제주의 해양성 기후 등)
    지역 좌표 없이는 하나의 모델이 12개 관측소를 구분할 수 없다.
    인덱스 12·13(연중 시각 sin/cos)은 2026-08-16 partition_heterogeneity.py
    실측에서 추가했다 — 월(계절)은 기존 입력 어디에도 없는데, 시각을
    이미 안 상태에서도 월이 ΔT 설명력을 추가로 +2.59%p(약 6.2배) 더
    끌어올렸다(조건부 분산 분석). 관측소 η²=0.00%로 공간 분할 근거는
    없었던 것과 대조적으로 계절축은 실제로 누락된 정보였다.
    인덱스 0(기온)은 퍼시스턴스 잔차 계산에 쓰이므로 위치를 바꾸지 말 것.
    """
    wd_rad = np.deg2rad(r.get("wind_dir", 0.0))

    ts = str(r.get("timestamp", ""))            # YYYYMMDDHHmm
    hour = int(ts[8:10]) if len(ts) >= 10 else 0
    h_rad = 2.0 * np.pi * hour / 24.0

    if len(ts) >= 8:
        doy = datetime(int(ts[:4]), int(ts[4:6]), int(ts[6:8])).timetuple().tm_yday
    else:
        doy = 1
    y_rad = 2.0 * np.pi * doy / 365.25

    lat, lon = STATION_COORDS.get(str(r.get("stn", "108")), (36.5, 127.5))

    return [
        r.get("temperature",   20.0),   # 0: 기온      ← 퍼시스턴스 기준
        r.get("precipitation",  0.0),   # 1: 강수량
        r.get("humidity",      50.0),   # 2: 습도
        r.get("wind_speed",     1.5),   # 3: 풍속
        float(np.sin(wd_rad)),          # 4: 풍향 sin
        float(np.cos(wd_rad)),          # 5: 풍향 cos
        r.get("pressure",    1013.0),   # 6: 기압
        float(r.get("precip_type", 0)), # 7: 강수형태
        float(np.sin(h_rad)),           # 8: 시각 sin  ← 일주기
        float(np.cos(h_rad)),           # 9: 시각 cos  ← 일주기
        lat,                            # 10: 위도     ← 관측소 위치
        lon,                            # 11: 경도     ← 관측소 위치
        float(np.sin(y_rad)),           # 12: 연중 시각 sin ← 계절 주기
        float(np.cos(y_rad)),           # 13: 연중 시각 cos ← 계절 주기
    ]


NUM_FEATURES = 14   # record_to_vec 출력 차원


# ── 2. 과거 데이터 수집 ───────────────────────────────────────────

def collect_historical(n_hours: int = N_HOURS,
                       stations: list = None) -> list[dict]:
    """
    ASOS API로 각 관측소별 최근 n_hours 시간의 관측 데이터 수집.
    관측소를 순회하며 관측소 블록 단위로 이어붙인다(블록 내부는 시간순).
    """
    stations = stations or STATIONS_TO_COLLECT
    target_total = n_hours * len(stations)

    # 캐시 재사용 (API 호출 절약) — 레코드 수뿐 아니라 관측소 구성도 검증한다.
    # 이전 세션의 단일 관측소 캐시(720개)는 여기서 자동으로 부족 판정되어
    # 재수집이 트리거된다.
    if os.path.exists(DATA_CACHE):
        with open(DATA_CACHE, "r", encoding="utf-8") as f:
            cached = json.load(f)
        cached_stations = {r.get("stn") for r in cached}
        if len(cached) >= int(target_total * 0.9) and cached_stations >= set(stations):
            print(f"[캐시] {len(cached)}개 로드 완료 "
                  f"({len(cached_stations)}개 관측소, API 절약)")
            return cached
        print(f"[캐시] {len(cached)}개 ({len(cached_stations)}개 관측소) "
              f"— 목표({target_total}개, {len(stations)}개 관측소) 부족, 재수집")

    all_records = []
    # ASOS 시각은 KST 기준 — 컨테이너 기본 시간대(UTC)에 의존하지 않도록
    # 명시한다 (weather_collector.KST 참조). 여기서 datetime.now()를 그대로
    # 쓰면 Docker에서 수집 창 전체가 9시간 밀려 어긋난다.
    now = datetime.now(KST)

    print(f"ASOS 과거 {n_hours}시간 × {len(stations)}개 관측소 수집 시작 "
          f"(예상 {target_total}개, 약 {target_total * 0.3 / 60:.0f}분 소요)...")

    for si, stn in enumerate(stations, 1):
        stn_name = STATION_NAMES.get(stn, stn)
        collector = RobustWeatherCollector(stn=stn)
        records = []

        for i in range(n_hours, 0, -1):
            dt = now - timedelta(hours=i)
            tm = dt.strftime("%Y%m%d%H00")
            data = collector.fetch_at(tm)

            if data.get("status") == "SUCCESS_LIVE":
                records.append(data)

            time.sleep(0.3)  # API 과부하 방지

        print(f"  [{si:2d}/{len(stations)}] {stn_name}({stn}): "
              f"{len(records)}/{n_hours}개 성공")
        all_records.extend(records)

    print(f"\n전체 수집 완료: {len(all_records)}/{target_total}개 "
          f"({len(stations)}개 관측소)\n")

    with open(DATA_CACHE, "w", encoding="utf-8") as f:
        json.dump(all_records, f, ensure_ascii=False, indent=2)

    return all_records


# ── 3. 데이터셋 ───────────────────────────────────────────────────

def _parse_ts(ts) -> datetime:
    """YYYYMMDDHHmm 문자열 → datetime."""
    return datetime.strptime(str(ts)[:12], "%Y%m%d%H%M")


def _prf_metrics(probs: torch.Tensor, labels: torch.Tensor,
                 thresh: float = 0.5) -> dict:
    """
    이진분류 정밀도/재현율/F1 — 폭염·한파처럼 양성이 1% 미만인 사건은
    정확도(accuracy)가 "항상 무사건" 예측만으로도 99%를 넘어 무의미하다.
    양성을 실제로 잡아내는지가 관심사이므로 precision/recall/F1 로 본다.
    """
    pred_pos = (probs >= thresh).float()
    tp = (pred_pos * labels).sum().item()
    fp = (pred_pos * (1 - labels)).sum().item()
    fn = ((1 - pred_pos) * labels).sum().item()
    precision = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    recall    = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f1 = (2 * precision * recall / (precision + recall)
          if (precision + recall) > 0 and not (np.isnan(precision) or np.isnan(recall))
          else 0.0)
    return {"precision": precision, "recall": recall, "f1": f1,
            "n_pos": int(labels.sum().item()), "n_pred_pos": int(pred_pos.sum().item())}


class WeatherDataset(Dataset):
    """
    입력 X_num: 시각 t 의 12차원 수치 벡터 (표준화)
    입력 X_img: 시각 t 의 합성 위성 이미지 (4, 32, 32)
    입력 X_txt: 시각 t 의 MiniLM 텍스트 임베딩 (384)
    타깃 y    : [기온_{t+L}, 강수량_{t+L}]      L = lead_hours

    수집 실패로 시각이 건너뛴 구간이 있으면 (t, t+L) 쌍이 아예 존재하지
    않을 수 있다 — (관측소, 시각) 인덱스로 직접 조회해 그런 레코드는
    제외한다.

    이전 구현은 리스트에서 인덱스가 L 칸 떨어진 두 레코드의 시각차만
    검사했다. 이는 records 가 '관측소 블록 단위로 이어붙여지고, 블록
    내부는 시간순'이라는 저장 순서를 전제하는데, 단일 수집 실행
    (collect_historical 한 번)에서는 항상 성립했지만 apihub 수집
    (collect_year.py)과 웹 포털 파일셋 임포트(import_kma_fileset.py)를
    여러 번에 걸쳐 병합한 뒤로는 그 순서가 우연에 의존하게 됐다 — 실측
    328,802개 병합본에서 96.9%는 우연히 정렬이 맞았지만, 다음 병합에서
    순서가 달라지면 조용히 페어를 잃는 구조였다. 인덱스 조회 방식은
    records 의 순서와 완전히 무관하다.
    """

    def __init__(self, records: list[dict],
                 sat_collector=None,   # InterpolatedFieldCollector — Re축, 실측 공간보간장
                 txt_collector: SimulatedTextCollector = None,
                 lead_hours: int = 1,
                 mean: np.ndarray = None, std: np.ndarray = None):
        L = lead_hours
        self.lead_hours = L

        # ── 유효한 (t, t+L) 쌍만 선별 — (관측소, 시각) 인덱스 직접 조회 ──
        # records 의 저장 순서에 의존하지 않는다(클래스 docstring 참고).
        # 같은 (stn, 시각) 이 중복되면 나중 항목이 이긴다 — 이 저장소의
        # 다른 dedup 병합(collect_year.load_existing 등)과 같은 규칙.
        by_key = {(r.get("stn"), str(r["timestamp"])[:12]): i
                  for i, r in enumerate(records)}

        src_idx, tgt_idx = [], []
        for i, r in enumerate(records):
            tgt_ts = (_parse_ts(r["timestamp"]) + timedelta(hours=L)) \
                .strftime("%Y%m%d%H%M")
            j = by_key.get((r.get("stn"), tgt_ts))
            if j is not None:
                src_idx.append(i)
                tgt_idx.append(j)

        if not src_idx:
            raise ValueError(f"lead={L}시간 유효 쌍이 없다 — 데이터 연속성을 확인해야 한다.")
        dropped = len(records) - len(src_idx)
        if dropped > 0:
            print(f"  [정보] {dropped}개 레코드는 +{L}h 뒤 관측이 없어 제외 "
                  f"(사용 {len(src_idx)}개)")

        src_records = [records[i] for i in src_idx]
        tgt_records = [records[j] for j in tgt_idx]

        # 관측소·시각 메타데이터 — 학습에는 쓰이지 않지만 backtest_accuracy.py
        # 가 샘플별 예측을 (관측소, 목표시각)으로 정확도 로그에 남기는 데 필요.
        self.stns = [r.get("stn") for r in src_records]
        self.src_timestamps = [r["timestamp"] for r in src_records]
        self.tgt_timestamps = [r["timestamp"] for r in tgt_records]

        # ── Z축: 수치 특성 ───────────────────────────────────────
        vecs = np.array([record_to_vec(r) for r in src_records], dtype=np.float32)
        if mean is None:
            self.mean = vecs.mean(axis=0)
            self.std  = np.where(vecs.std(axis=0) > 1e-6, vecs.std(axis=0), 1.0)
        else:
            # record_to_vec()이 나중에 차원을 늘려도(2026-08-16: 12→14, 연중
            # 시각 sin/cos 추가) 구버전 체크포인트의 mean/std(짧은 차원)를 그대로
            # 넘겨받는 호출자가 많다(threshold_validation.py 등). 새 축은 벡터
            # 끝에 붙는 구조이므로 mean 길이만큼만 앞에서 잘라 구버전 입력을
            # 그대로 복원한다 — 안 자르면 브로드캐스트가 깨진다.
            vecs = vecs[:, :len(mean)]
            self.mean, self.std = mean, std
        self.X_num = torch.tensor((vecs - self.mean) / self.std, dtype=torch.float32)

        # ── Re축 / Im축 ──────────────────────────────────────────
        # float16 저장 — 13년치(약 130만 표본)에서 float32 그대로 두면
        # (4,32,32) 이미지만 약 22GB에 달해 WSL2/Docker 메모리 한도(31GB)를
        # 넘겨 OOM-kill 당한다(2026-08-15 실측, exit 137). 모델에 넣기
        # 직전(train.py 학습 루프)에서 다시 float32로 캐스팅한다.
        self.X_img = (torch.tensor(sat_collector.get_batch(src_records),
                                   dtype=torch.float16)
                      if sat_collector is not None else None)
        self.X_txt = (torch.tensor(txt_collector.get_batch(src_records),
                                   dtype=torch.float32)
                      if txt_collector is not None else None)

        # ── 타깃 ─────────────────────────────────────────────────
        temp_tgt   = np.array([r["temperature"]   for r in tgt_records], dtype=np.float32)
        precip_tgt = np.array([r["precipitation"] for r in tgt_records], dtype=np.float32)
        self.y = torch.tensor(np.stack([temp_tgt, precip_tgt], axis=1),
                              dtype=torch.float32)

        # 퍼시스턴스 기준선을 샘플 단위로 보관 → 검증셋과 동일한 표본에서
        # baseline 을 계산해야 공정한 비교가 된다(전체 평균과 비교하면 안 됨).
        temp_now = np.array([r["temperature"] for r in src_records], dtype=np.float32)
        self.temp_persist_abs = torch.tensor(np.abs(temp_tgt - temp_now),
                                             dtype=torch.float32)
        self.temp_persistence_mae = float(self.temp_persist_abs.mean())

        # 타깃 통계 — 헤드 편향 초기화 및 손실 스케일 정규화에 사용
        self.temp_mean   = float(temp_tgt.mean())
        self.temp_std    = float(max(temp_tgt.std(), 1e-3))
        self.precip_mean = float(precip_tgt.mean())

        # 관측소 구성 진단 — 다중 관측소 확장 검증용
        self.n_stations = len({r.get("stn") for r in src_records})

        # ── Phase 3-9 공식 라벨(폭염·한파·황사 특보) ────────────────
        # import_weather_issues.py 가 만든 (관측소,날짜)→기상청 공식판정
        # 조회표. 세 사건 모두 공식 기록이 있는 표본만 학습에 쓰고, 없는
        # 표본은 마스크로 손실에서 뺀다.
        #
        # 폭염·한파도 마스킹으로 바꾼 이유(2026-08-15 실측) — 예전에는 공식
        # 기록이 없으면 임계값 근사(그 시각 기온 ≥33°C / ≤−12°C)로 폴백했다.
        # 그런데 이 둘은 정의가 아예 다르다: 공식 특보는 "그 날 하루" 단위라
        # 24개 표본이 전부 양성이 되고(양성률 15~36%), 근사는 오후 몇 시간만
        # 양성이 된다(양성률 0.25~2.02%). 같은 현상에 50배 다른 양성률의
        # 라벨이 붙는 셈이다. 3.6년(2023~) 데이터에서는 공식 기록이 45%를
        # 덮어 영향이 제한적이었으나, 13년(2013~)으로 늘리자 공식 기록이
        # 없는 2013~2018년 63만 표본이 전부 근사 쪽으로 떨어지면서 정밀도가
        # 무너졌다(폭염 F1 0.842→0.665, 근사 구간만 보면 0.195).
        # 모델 입력에 연도가 없어(record_to_vec) 두 정의를 구분할 방법이
        # 원리적으로 없으므로, 황사가 이미 쓰던 마스킹 방식으로 통일한다.
        # 근사 라벨을 버려도 공식 라벨 표본 수는 오히려 늘어난다
        # (폭염 15만→33만, 한파 17만→29만 — 13년치 기준).
        issue_labels = {}
        if os.path.exists(WEATHER_ISSUE_LABELS):
            with open(WEATHER_ISSUE_LABELS, "r", encoding="utf-8") as f:
                issue_labels = json.load(f)

        heat_list, cold_list, dust_list = [], [], []
        heat_mask_list, cold_mask_list, dust_mask_list = [], [], []
        n_heat_off = n_cold_off = n_dust_off = 0
        for i, r in enumerate(tgt_records):
            ts = str(r["timestamp"])
            date_str = f"{ts[0:4]}-{ts[4:6]}-{ts[6:8]}"
            day = issue_labels.get(str(r.get("stn")), {}).get(date_str, {})

            if "heatwave_advisory" in day:
                heat_list.append(day["heatwave_advisory"]); heat_mask_list.append(1); n_heat_off += 1
            elif EXTREME_LABEL_MASKING:
                heat_list.append(0); heat_mask_list.append(0)
            else:   # 대조군 — 종전처럼 임계값 근사로 채우고 손실에도 넣는다
                heat_list.append(int(temp_tgt[i] >= HEATWAVE_THRESH)); heat_mask_list.append(1)

            if "coldwave_advisory" in day:
                cold_list.append(day["coldwave_advisory"]); cold_mask_list.append(1); n_cold_off += 1
            elif EXTREME_LABEL_MASKING:
                cold_list.append(0); cold_mask_list.append(0)
            else:
                cold_list.append(int(temp_tgt[i] <= COLDWAVE_THRESH)); cold_mask_list.append(1)

            if "dust_observed" in day:
                dust_list.append(day["dust_observed"]); dust_mask_list.append(1); n_dust_off += 1
            else:
                dust_list.append(0); dust_mask_list.append(0)

        self.y_heatwave = torch.tensor(heat_list, dtype=torch.float32)
        self.y_coldwave = torch.tensor(cold_list, dtype=torch.float32)
        self.y_dust     = torch.tensor(dust_list, dtype=torch.float32)
        self.heat_mask  = torch.tensor(heat_mask_list, dtype=torch.float32)
        self.cold_mask  = torch.tensor(cold_mask_list, dtype=torch.float32)
        self.dust_mask  = torch.tensor(dust_mask_list, dtype=torch.float32)
        self.n_heat_official = n_heat_off
        self.n_cold_official = n_cold_off
        self.n_dust_official = n_dust_off

    def __len__(self):
        return len(self.X_num)

    def __getitem__(self, idx):
        """
        정수 하나뿐 아니라 인덱스 목록도 받는다 — 목록이면 배치 하나를
        통째로 gather해서 돌려준다.

        왜 배치 단위 인덱싱이 필요한가(2026-08-16 실측) — 표본별로 꺼내면
        배치 1024개당 텐서 10,240개를 만들고 collate가 그걸 다시 10개로
        쌓는다. 에폭당 1,100만 번의 텐서 할당이다. 이 데이터셋은 전부
        RAM에 올라와 있어 표본별로 꺼낼 이유가 없다 — 한 번의 벡터화
        gather면 배치 텐서가 바로 나온다. forward+backward를 포함한
        실측에서 스텝당 18.3ms → 9.6ms(1.90배)였다.

        torch의 Subset이 목록 인덱싱을 그대로 전달해주므로(`Subset.
        __getitem__`), 학습/검증 분할을 거쳐도 이 경로가 유지된다.
        """
        if isinstance(idx, (list, np.ndarray, torch.Tensor)):
            idx = torch.as_tensor(idx, dtype=torch.long)
        return (self.X_num[idx], self.X_img[idx], self.X_txt[idx], self.y[idx],
                self.y_heatwave[idx], self.y_coldwave[idx], self.y_dust[idx],
                self.heat_mask[idx], self.cold_mask[idx], self.dust_mask[idx])


# ── 3-b. 학습/검증 분할 ───────────────────────────────────────────

def make_split(full_ds, mode: str, verbose: bool = True):
    """
    학습/검증 분할. mode 는 SPLIT_MODE 참고.

    왜 무작위 말고 다른 모드가 필요한가(2026-08-15 split_leakage_check.py 실측)
    ─ 폭염·한파·황사 라벨은 (관측소, 날짜) 단위다. 특보가 발효된 날은 그날
    24개 시각 표본이 전부 같은 라벨을 받는다. 그런데 시각 단위로 무작위
    분할하면 같은 날의 표본이 학습셋과 검증셋에 쪼개진다 — 하루 표본이
    평균 23.9개이므로 한쪽으로 몰릴 확률은 0.8^24 ≈ 0.5% 에 불과하다.
    실측 결과 판정 가능한 검증 표본의 100.00% 가, 양성만 따져도 100.00% 가
    학습셋에 같은 날 짝을 갖고 있었다(폭염 15,916/15,916 · 한파 3,852/3,852 ·
    황사 3,498/3,498). 모델은 학습 중에 그 관측소·그 날짜의 정답을 이미 봤고
    검증에서는 같은 날 다른 시각을 묻는 셈이라, 일반화가 아니라 "같은 날
    다른 시각 맞히기"를 채점해 왔다. 회귀(기온)는 같은 날 안에서도 값이
    평균 8.41°C 벌어지므로 이 문제에서 상대적으로 자유롭다.

    그룹 단위를 (관측소, 날짜)가 아니라 날짜로만 잡는 이유 ─ Re축은 설계상
    같은 시각 다른 11개 관측소의 실측값을 입력으로 쓴다(interp_field_
    collector.py). 관측소별로 쪼개면 서울 8/1은 학습, 대구 8/1은 검증으로
    갈릴 수 있는데 대구 표본의 Re축 입력 안에 이미 서울 관측이 들어 있어
    입력을 통한 누수가 남는다. 날짜 단위로 12개 관측소를 통째로 묶어야 막힌다.
    """
    n = len(full_ds)
    # 라벨 기준 날짜 = 타깃 시각의 날짜 (라벨이 붙는 단위와 일치시킨다)
    dates = np.array([str(ts)[:8] for ts in full_ds.tgt_timestamps])

    if mode == "random":
        n_val = max(2, int(n * VAL_RATIO))
        g = torch.Generator().manual_seed(SEED)
        perm = torch.randperm(n, generator=g).tolist()
        # random_split 과 같은 순서 — 앞에서부터 학습, 뒤가 검증.
        # 종전 체크포인트와 분할이 일치해야 비교가 성립한다.
        train_idx, val_idx = perm[:n - n_val], perm[n - n_val:]

    elif mode == "group":
        uniq = np.unique(dates)
        g = torch.Generator().manual_seed(SEED)
        order = torch.randperm(len(uniq), generator=g).tolist()
        n_val_days = max(1, int(len(uniq) * VAL_RATIO))
        val_days = {uniq[k] for k in order[:n_val_days]}
        is_val = np.array([d in val_days for d in dates])
        train_idx = np.flatnonzero(~is_val).tolist()
        val_idx = np.flatnonzero(is_val).tolist()

    elif mode == "temporal":
        # 과거로 배워 미래를 예측하는 실제 운영 구조. 경계에 완충 하루를
        # 비운다 — Im축이 6시간 전을 보고 타깃이 +6시간이라 경계 표본은
        # 양쪽 구간에 걸친다.
        is_train = dates <= TEMPORAL_TRAIN_END
        is_val = dates >= TEMPORAL_VAL_START
        train_idx = np.flatnonzero(is_train).tolist()
        val_idx = np.flatnonzero(is_val).tolist()

    else:
        raise ValueError(f"알 수 없는 SPLIT_MODE: {mode}")

    if verbose:
        # numpy 문자열 배열은 min/max 유니버설 함수가 없다 — 파이썬 set 으로 센다
        s_tr = set(dates[train_idx].tolist())
        s_va = set(dates[val_idx].tolist())
        print(f"분할 방식: {mode}")
        print(f"  학습 {len(train_idx):,}개 ({min(s_tr)}~{max(s_tr)}) / "
              f"검증 {len(val_idx):,}개 ({min(s_va)}~{max(s_va)})")
        shared = len(s_tr & s_va)
        print(f"  학습·검증이 공유하는 날짜 {shared:,}개"
              + ("  ← 일 단위 라벨 누수 경로" if shared else "  (누수 없음)"))
        for nm, t in (("폭염", full_ds.heat_mask), ("한파", full_ds.cold_mask),
                      ("황사", full_ds.dust_mask)):
            n_j = int(t[val_idx].sum().item())
            print(f"    검증셋 {nm} 판정가능 {n_j:,}개")

    return Subset(full_ds, train_idx), Subset(full_ds, val_idx)


# ── 4. 모델: pipeline_model.py의 TriCHEFPipeline 사용 ─────────────
#   Phase 3-1: Z축(수치) 활성, Re/Im 축 stub
#   Phase 3-2~: 순차적으로 실제 인코더 교체


# ── 5. 학습 ───────────────────────────────────────────────────────

def train(orthogonalize: bool = ORTHOGONALIZE,
          persistence_residual: bool = PERSISTENCE_RESIDUAL,
          lead_hours: int = LEAD_HOURS,
          use_re: bool = True, use_im: bool = True,
          early_stop: bool = True,
          dynamic_gate: bool = DYNAMIC_GATE,
          gate_entropy_weight: float = GATE_ENTROPY_WEIGHT,
          lr_schedule: str = "plateau",
          checkpoint: str = CHECKPOINT,
          verbose: bool = True) -> dict:
    """
    3축 Tri-CHEF 학습. 플래그를 바꿔 호출하면 ablation 대조군이 된다.

    use_re / use_im : 해당 축 입력을 차단(영벡터)해 기여도를 측정한다.
                      Z축(수치)은 퍼시스턴스 기준선에 필요하므로 항상 활성.
    early_stop      : ablation 비교 시 False 로 두어 모든 조건을 동일 에폭수로
                      학습한다. 조기 종료는 조건마다 학습 길이를 다르게 만들어
                      '성능 차이'와 '학습량 차이'를 구분할 수 없게 한다.
    lr_schedule     : "plateau" (기본) — ReduceLROnPlateau, 검증 손실이
                        8에폭 정체되면 LR 반감. 검증 손실 자체에 반응하는
                        보수적 스케줄.
                      "cosine_restarts" — CosineAnnealingWarmRestarts.
                        재시작 주기(T_0)를 고정 에폭이 아니라 게이트 워밍업의
                        2배로 둔다 — 엔트로피 정규화가 λ=0→목표값으로 램프업을
                        마치는 시점(gate_warmup_epochs~2×gate_warmup_epochs)에
                        정확히 첫 재시작이 걸리도록. 이 구간은 손실 함수 자체의
                        형태가 바뀌는 지점이라(엔트로피 항이 새로 추가됨),
                        LR을 다시 올려 최적화기가 바뀐 지형을 재탐색할 구체적
                        근거가 있다. 고정 에폭(예: 20/40/80)을 쓰지 않는 이유는
                        GATE_WARMUP_EPOCHS 와 같은 함정 때문이다 — 이번 데이터
                        규모(328k)에서 학습이 16~36에폭에 조기종료되는 것이
                        실측됐는데, 고정 재시작 지점이 그보다 늦으면 단 한 번도
                        발동하지 않는다.
    반환: 최종 성능 지표 dict (ablation.py 에서 비교용으로 사용)
    """
    axes = "Z" + ("+Re" if use_re else "") + ("+Im" if use_im else "")
    if verbose:
        print(f"{'='*70}")
        print(f" Tri-CHEF Phase 3-4 — 예보 시계 +{lead_hours}시간 | 활성 축: {axes}")
        print(f" 디바이스: {DEVICE} | embed_dim: {EMBED_DIM}")
        print(f" 배치 {BATCH_SIZE} | LR {LR:.2e} | 워커 {NUM_WORKERS} | "
              f"극한기상 라벨 마스킹: {'ON' if EXTREME_LABEL_MASKING else 'OFF(대조군)'}")
        print(f" EXTREME_BCE_WEIGHT: {EXTREME_BCE_WEIGHT} | "
              f"CROP_OVERSAMPLE: {CROP_OVERSAMPLE} | "
              f"DUST_LOSS_WEIGHT: {DUST_LOSS_WEIGHT}"
              f"{'  ← 황사 헤드 손실 제외' if DUST_LOSS_WEIGHT == 0 else ''}")
        print(f" 분할 방식: {SPLIT_MODE}"
              + (f" (학습 ~{TEMPORAL_TRAIN_END} / 검증 {TEMPORAL_VAL_START}~)"
                 if SPLIT_MODE == "temporal" else ""))
        print(f" Gram-Schmidt 직교화: {'ON' if orthogonalize else 'OFF'}"
              f" | 기온 Δ 예측: {'ON' if persistence_residual else 'OFF'}")
        gdesc = (f"동적 게이트 (균등 초기화, λ={gate_entropy_weight}, "
                 f"워밍업은 학습 표본 수에서 역산 — 아래 참조)" if dynamic_gate else "정적 α/φ")
        print(f" 축 가중치: {gdesc}")
        print(f" 위성 인코더: {'소형 CNN' if COMPACT_SATELLITE else 'ResNet18'}")
        print(f"{'='*70}\n")

    torch.manual_seed(INIT_SEED)
    np.random.seed(INIT_SEED)

    records = collect_historical()
    if len(records) < 20:
        print("[ERROR] 데이터 부족 (최소 20개 필요). API 연결을 확인하세요.")
        return {}

    if verbose:
        print("Re축 공간보간장 생성 중 (12관측소 실측 IDW 보간, 4ch 32×32)...")
    # Phase 3-10 실험 — SimulatedSatelliteCollector(노이즈, 정보량 0 확인됨)
    # 대신 같은 시각 다른 11개 관측소 실측값을 IDW 보간한 필드를 Re축에 넣는다.
    sat_collector = InterpolatedFieldCollector(records, STATION_COORDS)
    if verbose:
        print("Im축 시간 경향 벡터 생성 중 (기온·기압·습도·풍속 × 1h/3h/6h 차분)...")
    # Phase 3-11 실험 — SimulatedTextCollector(Z축 재포장, 중복정보 확인됨)
    # 대신 단일 스냅샷에서 유도 불가능한 시간 미분(∂s/∂t)을 Im축에 넣는다.
    txt_collector = TendencyCollector(records)

    full_ds = WeatherDataset(records,
                             sat_collector=sat_collector,
                             txt_collector=txt_collector,
                             lead_hours=lead_hours)
    train_ds, val_ds = make_split(full_ds, SPLIT_MODE, verbose=verbose)
    n_train, n_val = len(train_ds), len(val_ds)

    # 게이트 워밍업 — Phase 3-5 캘리브레이션 스텝 수를 현재 데이터 규모에서
    # 재현하는 에폭 수로 역산 (위 _REFERENCE_WARMUP_STEPS 주석 참고).
    steps_per_epoch = max(1, n_train // BATCH_SIZE)
    gate_warmup_epochs = max(_MIN_WARMUP_EPOCHS,
                             round(_REFERENCE_WARMUP_STEPS / steps_per_epoch))
    if verbose:
        print(f"게이트 워밍업: {gate_warmup_epochs}에폭 "
              f"(에폭당 {steps_per_epoch}배치 × {gate_warmup_epochs} ≈ "
              f"{steps_per_epoch*gate_warmup_epochs:,}스텝, 기준 "
              f"{_REFERENCE_WARMUP_STEPS:,}스텝)")

    # 배치 단위 인덱싱 — BatchSampler가 인덱스 목록을 만들고 데이터셋이 그걸
    # 한 번에 gather한다(WeatherDataset.__getitem__ 참고). batch_size=None은
    # "자동 배치 끄기"라 collate 단계 자체가 사라진다.
    #
    # 워커를 쓰지 않는다(num_workers=0). 이 데이터셋은 전부 RAM에 있어 워커가
    # 할 무거운 전처리가 없고, 오히려 배치 텐서(배치당 약 8MB)를 프로세스
    # 사이로 옮기는 비용만 추가된다. 실측(2026-08-16, forward+backward 포함):
    #   표본별 + 워커 8개 : 18.3 ms/스텝
    #   배치 gather + 워커 0 :  9.6 ms/스텝   ← 1.90배
    #   배치 gather + 워커 8개: 5.1 ms/배치(로딩만) — 워커를 붙이면 오히려 느리다
    _dl = dict(num_workers=0, pin_memory=(DEVICE == "cuda"))

    def _batched(dataset, sampler=None, shuffle=False):
        base = sampler if sampler is not None else (
            RandomSampler(dataset) if shuffle else SequentialSampler(dataset))
        return DataLoader(
            dataset, batch_size=None,
            sampler=BatchSampler(base, batch_size=BATCH_SIZE, drop_last=False),
            **_dl)

    if CROP_OVERSAMPLE > 1.0:
        is_wet = full_ds.y[:, 1] >= WET_THRESH
        is_event = (full_ds.y_heatwave.bool() | full_ds.y_coldwave.bool()
                   | full_ds.y_dust.bool() | is_wet)
        train_indices = torch.as_tensor(train_ds.indices)
        train_event = is_event[train_indices]
        sample_weights = torch.where(
            train_event, torch.tensor(CROP_OVERSAMPLE), torch.tensor(1.0))
        sampler = WeightedRandomSampler(
            sample_weights.double(), num_samples=n_train, replacement=True)
        train_loader = _batched(train_ds, sampler=sampler)
        if verbose:
            n_event = int(train_event.sum())
            print(f"Crop 재가중 ON — 학습셋 사건 표본 {n_event:,}개"
                  f"({100*n_event/n_train:.1f}%)를 {CROP_OVERSAMPLE:.1f}배 오버샘플링")
    else:
        train_loader = _batched(train_ds, shuffle=True)
    val_loader = _batched(val_ds)

    # baseline 은 반드시 검증셋과 동일한 표본에서 계산한다.
    # 전체 평균과 비교하면 표본 차이만으로 '개선'처럼 보일 수 있다.
    val_idx      = val_ds.indices
    temp_naive   = full_ds.temp_persist_abs[val_idx].mean().item()
    precip_naive = full_ds.y[val_idx, 1].abs().mean().item()

    if verbose:
        print(f"\n학습: {n_train}개 | 검증: {n_val}개 | 배치 크기: {BATCH_SIZE} "
              f"| 관측소: {full_ds.n_stations}개")
        print(f"기준선 — 검증셋 {n_val}개 기준 (모델과 동일 표본)")
        print(f"  기온 퍼시스턴스 T(t+{lead_hours})≈T(t) : {temp_naive:.4f} °C   "
              f"[전체 {full_ds.temp_persistence_mae:.4f}]")
        print(f"  강수 상시 0 예측            : {precip_naive:.4f} mm   "
              f"[전체 {full_ds.precip_mean:.4f}]\n")

    # hurdle 헤드 사전확률·클래스 불균형 가중치 — 학습 표본에서 직접 계산
    # (고정 상수 아님). 검증셋을 보면 데이터 누설이라 반드시 train_ds 에서만.
    train_precip = full_ds.y[train_ds.indices, 1]
    n_wet = int((train_precip >= WET_THRESH).sum().item())
    n_dry = len(train_precip) - n_wet
    wet_prior = n_wet / len(train_precip)
    rain_pos_weight = torch.tensor([n_dry / max(n_wet, 1)], device=DEVICE)
    if verbose:
        print(f"Hurdle 헤드 — 학습셋 강수 비율 {wet_prior:.4f} "
              f"({n_wet}/{len(train_precip)}), BCE pos_weight={rain_pos_weight.item():.2f}")

    # Phase 3-8/3-9 극한기상 헤드 사전확률·클래스 불균형 가중치.
    # 세 사건 모두 기상청 공식 라벨이 있는 표본에서만 비율을 잰다 — 마스크=0인
    # 표본은 "사건 없음으로 확인됨"이 아니라 "판정 불가"라 분모에서도 뺀다
    # (WeatherDataset 라벨 구성 주석 참고).
    def _masked_prior(labels, mask):
        m = mask[train_ds.indices].bool()
        y = labels[train_ds.indices][m]
        n_pos = int(y.sum().item())
        n_tot = max(len(y), 1)
        prior = n_pos / n_tot
        pw = torch.tensor([(n_tot - n_pos) / max(n_pos, 1)], device=DEVICE)
        return prior, pw, n_pos, n_tot

    heatwave_prior, heatwave_pos_weight, n_heat, n_heat_total = _masked_prior(
        full_ds.y_heatwave, full_ds.heat_mask)
    coldwave_prior, coldwave_pos_weight, n_cold, n_cold_total = _masked_prior(
        full_ds.y_coldwave, full_ds.cold_mask)
    dust_prior, dust_pos_weight, n_dust, n_dust_total = _masked_prior(
        full_ds.y_dust, full_ds.dust_mask)

    if verbose:
        print(f"극한기상 헤드 — 전부 공식 라벨 표본만 사용(판정 불가 표본은 손실에서 제외)")
        print(f"              학습셋 폭염 비율 {heatwave_prior:.4f} "
              f"({n_heat}/{n_heat_total}), pos_weight={heatwave_pos_weight.item():.2f}")
        print(f"              학습셋 한파 비율 {coldwave_prior:.4f} "
              f"({n_cold}/{n_cold_total}), pos_weight={coldwave_pos_weight.item():.2f}")
        print(f"              학습셋 황사 비율 {dust_prior:.4f} "
              f"({n_dust}/{n_dust_total}), pos_weight={dust_pos_weight.item():.2f}")

    # 모델 — 타깃 평균을 헤드 편향에 주입해 편향 학습 낭비 제거
    model = TriCHEFPipeline(
        embed_dim=EMBED_DIM, num_features=NUM_FEATURES,
        alpha_init=ALPHA_INIT, phi_init=PHI_INIT,
        orthogonalize=orthogonalize,
        temp_mean=full_ds.temp_mean, precip_mean=full_ds.precip_mean,
        persistence_residual=persistence_residual,
        feat_mean=full_ds.mean.tolist(), feat_std=full_ds.std.tolist(),
        dynamic_gate=dynamic_gate, compact_satellite=COMPACT_SATELLITE,
        wet_prior=wet_prior,
        heatwave_prior=heatwave_prior, coldwave_prior=coldwave_prior,
        dust_prior=dust_prior,
        signed_head_input=SIGNED_HEAD_INPUT,
        signed_precip_input=SIGNED_PRECIP_INPUT,
        head_dropout=HEAD_DROPOUT,
        coldwave_dropout=COLDWAVE_DROPOUT,
        im_dim=TENDENCY_DIM,   # Phase 3-11 — 384(MiniLM) 대신 12(경향벡터)
    ).to(DEVICE)

    # 차단된 축의 인코더는 forward 에서 호출되지 않아 gradient 를 받지 않는다.
    # 옵티마이저에 넘기기 전에 동결해 '실제 학습되는 파라미터'만 집계한다.
    if not use_re:
        for p in model.enc_re.parameters():
            p.requires_grad_(False)
    if not use_im:
        for p in model.enc_im.parameters():
            p.requires_grad_(False)

    trainable = [p for p in model.parameters() if p.requires_grad]
    n_params  = sum(p.numel() for p in trainable)
    if verbose:
        print(f"학습 파라미터: {n_params:,}개 / 표본 {n_train}개  "
              f"(비율 {n_params / n_train:,.0f}:1)")

    optimizer = torch.optim.AdamW(trainable, lr=LR, weight_decay=1e-4)
    if lr_schedule == "cosine_restarts":
        restart_t0 = max(2, gate_warmup_epochs * 2)
        scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(
            optimizer, T_0=restart_t0, T_mult=2, eta_min=LR * 0.01
        )
        if verbose:
            print(f"LR 스케줄: CosineAnnealingWarmRestarts "
                  f"(첫 재시작 {restart_t0}에폭 — 게이트 워밍업 종료 시점, "
                  f"이후 {restart_t0*2}, {restart_t0*6}, ...)")
    else:
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", patience=8, factor=0.5
        )
    mse = nn.MSELoss()

    # 손실 스케일 정규화: 기온 MSE 는 σ²(≈수십) 규모, 강수 MSE 는 O(1) 규모라
    # 그대로 더하면 강수 항이 전체의 0.1% 미만이 되어 학습되지 않는다.
    temp_var = full_ds.temp_std ** 2

    best_score = float("inf")
    best_stats = {}
    no_improve = 0
    # 지표별 최저치를 독립 추적 — 합산 score 기반 조기 종료가 한쪽 지표를
    # 가려버리는 문제를 피하고, ablation 조건 간 공정한 비교를 가능하게 한다.
    best_temp_only   = float("inf")
    best_precip_only = float("inf")

    if verbose:
        whead = "w Re/Im/Z" if dynamic_gate else "  α     φ"
        print(f"{'Epoch':>6} | {'Loss':>8} | {'기온MAE':>8} | {'강수MAE':>8} | "
              f"{'강수σ':>6} | {'|Δ기온|':>6} | {whead} | LR")
        print("-" * 92)

    for epoch in range(1, EPOCHS + 1):

        # 엔트로피 워밍업 — warmup 구간은 λ=0, 이후 목표값까지 선형 증가.
        if epoch <= gate_warmup_epochs:
            lam_now = 0.0
        else:
            ramp = min(1.0, (epoch - gate_warmup_epochs) / gate_warmup_epochs)
            lam_now = gate_entropy_weight * ramp

        model.train()
        train_loss = 0.0
        for (x_num, x_img, x_txt, y_b, heat_b, cold_b, dust_b,
             hmask_b, cmask_b, dmask_b) in train_loader:
            # non_blocking — pin_memory 로 고정한 페이지에서만 실제로 비동기
            # 전송이 일어난다. 둘은 짝으로 써야 의미가 있다.
            _nb = dict(non_blocking=True)
            x_num, y_b = x_num.to(DEVICE, **_nb), y_b.to(DEVICE, **_nb)
            heat_b = heat_b.to(DEVICE, **_nb).unsqueeze(-1)
            cold_b = cold_b.to(DEVICE, **_nb).unsqueeze(-1)
            dust_b = dust_b.to(DEVICE, **_nb).unsqueeze(-1)
            hmask_b = hmask_b.to(DEVICE, **_nb).unsqueeze(-1)
            cmask_b = cmask_b.to(DEVICE, **_nb).unsqueeze(-1)
            dmask_b = dmask_b.to(DEVICE, **_nb).unsqueeze(-1)
            x_img = x_img.to(DEVICE, **_nb).float() if use_re else None
            x_txt = x_txt.to(DEVICE, **_nb) if use_im else None

            optimizer.zero_grad()
            pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            loss = (mse(pred[:, 0:1], y_b[:, 0:1]) / temp_var
                    + PRECIP_WEIGHT * mse(pred[:, 1:2], y_b[:, 1:2]))
            # Hurdle BCE — "비가 오는가"에 직접 그래디언트를 준다. softplus
            # 회귀 손실만으로는 무강수 쪽으로 정확히 0을 향할 유인이 약해서
            # (기울기가 실측치와의 차이에만 비례) 이 항이 없으면 dynamic_gate
            # 케이스가 아니어도 동일한 문제가 재현된다(precip_breakdown.py).
            if hasattr(model, "head_rain"):
                is_wet = (y_b[:, 1:2] >= WET_THRESH).float()
                rain_bce = nn.functional.binary_cross_entropy_with_logits(
                    model._last_rain_logit, is_wet, pos_weight=rain_pos_weight
                )
                loss = loss + HURDLE_BCE_WEIGHT * rain_bce
            # Phase 3-8/3-9 극한기상 BCE — 회귀 손실(기온 MSE)만으로는 폭염·한파
            # 같은 꼬리 사건에 직접 그래디언트가 가지 않는다(전체 평균오차에
            # 묻힘). head_rain 과 같은 이유로 이진분류를 별도 항으로 둔다.
            #
            # 세 사건 모두 마스크=0(기상청 공식 라벨 없음)인 표본을 손실에서
            # 뺀다. 그 표본의 라벨 0 은 "사건 없음 확정"이 아니라 "판정 불가"라
            # 그대로 넣으면 근거 없는 음성 라벨을 학습시키게 된다.
            def _masked_bce(logit, target, mask, pos_weight):
                raw = nn.functional.binary_cross_entropy_with_logits(
                    logit, target, pos_weight=pos_weight, reduction="none")
                return (raw * mask).sum() / mask.sum()

            if hasattr(model, "head_heatwave") and hmask_b.sum() > 0:
                loss = loss + EXTREME_BCE_WEIGHT * _masked_bce(
                    model._last_heatwave_logit, heat_b, hmask_b, heatwave_pos_weight)
            if hasattr(model, "head_coldwave") and cmask_b.sum() > 0:
                loss = loss + EXTREME_BCE_WEIGHT * _masked_bce(
                    model._last_coldwave_logit, cold_b, cmask_b, coldwave_pos_weight)
            if (DUST_LOSS_WEIGHT > 0 and hasattr(model, "head_dust")
                    and dmask_b.sum() > 0):
                loss = loss + EXTREME_BCE_WEIGHT * DUST_LOSS_WEIGHT * _masked_bce(
                    model._last_dust_logit, dust_b, dmask_b, dust_pos_weight)
            # 게이트 엔트로피 정규화 — 가중치를 소수 축에 집중시키는 압력.
            # 이 항이 없으면 게이트도 정적 α/φ 와 마찬가지로 무용 축의
            # 가중치를 키우는 방향으로 학습된다(Phase 3-4 실측).
            if dynamic_gate and lam_now > 0:
                loss = loss + lam_now * model.gate_entropy()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            train_loss += loss.item()
        train_loss /= len(train_loader)

        # ── 검증 ──────────────────────────────────────────────────
        model.eval()
        temp_abs, precip_abs, precip_preds, deltas = [], [], [], []
        heat_probs, cold_probs, dust_probs = [], [], []
        heat_true, cold_true, dust_true = [], [], []
        heat_mask_all, cold_mask_all, dust_mask_all = [], [], []
        with torch.no_grad():
            for i, (x_num, x_img, x_txt, y_b, heat_b, cold_b, dust_b,
                    hmask_b, cmask_b, dmask_b) in enumerate(val_loader):
                x_num, y_b = x_num.to(DEVICE), y_b.to(DEVICE)
                x_img = x_img.to(DEVICE).float() if use_re else None
                x_txt = x_txt.to(DEVICE) if use_im else None
                pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt,
                             collect_diagnostics=(i == 0))
                temp_abs.append((pred[:, 0] - y_b[:, 0]).abs())
                precip_abs.append((pred[:, 1] - y_b[:, 1]).abs())
                precip_preds.append(pred[:, 1])
                if persistence_residual:
                    # 모델이 퍼시스턴스에 실제로 얹은 보정량 |Δ|
                    t_now = x_num[:, 0] * model.feat_std[0] + model.feat_mean[0]
                    deltas.append((pred[:, 0] - t_now).abs())
                heat_true.append(heat_b); cold_true.append(cold_b)
                dust_true.append(dust_b)
                heat_mask_all.append(hmask_b); cold_mask_all.append(cmask_b)
                dust_mask_all.append(dmask_b)
                if hasattr(model, "head_heatwave"):
                    heat_probs.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu())
                if hasattr(model, "head_coldwave"):
                    cold_probs.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu())
                if hasattr(model, "head_dust"):
                    dust_probs.append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu())

        # 샘플 단위 평균 (배치 크기가 달라도 편향되지 않도록)
        val_temp_mae   = torch.cat(temp_abs).mean().item()
        val_precip_mae = torch.cat(precip_abs).mean().item()
        precip_std     = torch.cat(precip_preds).std().item()
        delta_mag      = torch.cat(deltas).mean().item() if deltas else 0.0

        # 세 사건 모두 공식 라벨이 있는 표본에서만 채점한다 — 학습에서 뺀
        # 표본을 채점에 넣으면 "판정 불가"가 음성으로 둔갑해 수치가 부풀려진다.
        extreme_metrics = {}
        for _key, _probs, _true, _masks in (
                ("heatwave", heat_probs, heat_true, heat_mask_all),
                ("coldwave", cold_probs, cold_true, cold_mask_all),
                ("dust",     dust_probs, dust_true, dust_mask_all)):
            if not _probs:
                continue
            _m = torch.cat(_masks).bool()
            if _m.sum() > 0:
                extreme_metrics[_key] = _prf_metrics(
                    torch.cat(_probs)[_m], torch.cat(_true)[_m])

        # 조기 종료 기준: 두 baseline 대비 상대 오차의 합 (skill score)
        # 1.0 미만이면 해당 지표가 baseline 을 이긴 것.
        val_score = val_temp_mae / temp_naive + val_precip_mae / precip_naive

        best_temp_only   = min(best_temp_only,   val_temp_mae)
        best_precip_only = min(best_precip_only, val_precip_mae)

        # CosineAnnealingWarmRestarts 는 검증 지표가 아니라 에폭 진행 자체로
        # 스텝한다 — ReduceLROnPlateau 처럼 val_score 를 넘기면 타입 에러.
        if lr_schedule == "cosine_restarts":
            scheduler.step()
        else:
            scheduler.step(val_score)
        current_lr = optimizer.param_groups[0]["lr"]

        if verbose and (epoch % 10 == 0 or epoch == 1):
            w = model.axis_weights()
            mark = " ★" if val_score < best_score else ""
            if dynamic_gate:
                # 게이트가 각 축에 실제로 배분한 비중 (합 = 1)
                wcol = (f"{w.get('w_re',0):.3f}/{w.get('w_im',0):.3f}/"
                        f"{w.get('w_z',0):.3f}")
            else:
                wcol = f"{w['alpha']:5.3f} {w['phi']:5.3f}"
            print(f"{epoch:6d} | {train_loss:8.4f} | {val_temp_mae:7.4f}° | "
                  f"{val_precip_mae:7.4f}m | {precip_std:6.3f} | {delta_mag:6.3f} | "
                  f"{wcol} | {current_lr:.1e}{mark}")

        if val_score < best_score:
            best_score = val_score
            no_improve = 0
            w = model.axis_weights()
            best_stats = {
                "epoch":          epoch,
                "val_temp_mae":   val_temp_mae,
                "val_precip_mae": val_precip_mae,
                "precip_pred_std": precip_std,
                "delta_magnitude": delta_mag,
                "val_score":      val_score,
                "alpha_learned":  w["alpha"],
                "phi_learned":    w["phi"],
                "diagnostics":    dict(model.last_diagnostics),
                "extreme_metrics": extreme_metrics,
                # 기준선을 체크포인트에 같이 넣는다 — MAE 는 기준선 없이는
                # 우열을 말할 수 없는 지표인데, 지금까지는 학습 로그에만 찍고
                # 저장하지 않아 app.py 가 "잘 맞힌 값"처럼 보이는 강수 MAE 를
                # 비교 대상 없이 띄우고 있었다(2026-08-11 점검에서 발견).
                # 두 값 모두 검증셋과 동일 표본에서 계산된 것이다.
                "val_temp_naive_mae":   temp_naive,    # T(t+L) ≈ T(t) 퍼시스턴스
                "val_precip_naive_mae": precip_naive,  # 상시 0mm 예측
            }
            torch.save({
                **best_stats,
                "model_state":  model.state_dict(),
                "embed_dim":    EMBED_DIM,
                "orthogonalize": orthogonalize,
                "persistence_residual": persistence_residual,
                # dynamic_gate/compact_satellite 는 모델 아키텍처 자체를 바꾼다
                # (게이트 네트워크 유무, ResNet18 vs 소형 CNN). 이 값 없이는
                # 체크포인트만으로 TriCHEFPipeline 을 재구성할 수 없어
                # load_state_dict 가 shape mismatch 로 실패한다.
                "dynamic_gate": dynamic_gate,
                "compact_satellite": COMPACT_SATELLITE,
                "im_dim":       TENDENCY_DIM,   # Im축 인코더 종류 복원용(384 vs 12)
                # 극한기상 헤드 입력 폭 복원용 — True면 헤드가 embed_dim*2를
                # 받는다. 없으면 False(구버전)로 읽혀 기존 체크포인트가 그대로
                # 로드된다.
                "signed_head_input": SIGNED_HEAD_INPUT,
                "dust_loss_weight": DUST_LOSS_WEIGHT,
                "signed_precip_input": SIGNED_PRECIP_INPUT,
                "head_dropout": HEAD_DROPOUT,
                "coldwave_dropout": COLDWAVE_DROPOUT,
                "lead_hours":   lead_hours,
                "num_features": NUM_FEATURES,
                "alpha_init":   ALPHA_INIT,
                "phi_init":     PHI_INIT,
                "mean":         full_ds.mean.tolist(),
                "std":          full_ds.std.tolist(),
                "temp_mean":    full_ds.temp_mean,
                "precip_mean":  full_ds.precip_mean,
                # 이 체크포인트가 어떤 분할로 학습됐는지 — 진단 스크립트가
                # 같은 검증셋에서 채점하려면 필요하다. 없으면(구버전)
                # "random" 으로 읽는다.
                "split_mode":   SPLIT_MODE,
                "batch_size":   BATCH_SIZE,
                "lr":           LR,
                "extreme_label_masking": EXTREME_LABEL_MASKING,
                "extreme_bce_weight": EXTREME_BCE_WEIGHT,
                "crop_oversample": CROP_OVERSAMPLE,
            }, checkpoint)
        else:
            no_improve += 1
            if early_stop and no_improve >= PATIENCE:
                if verbose:
                    print(f"\nEarly stopping — epoch {epoch}")
                break

    # ── 결과 요약 ────────────────────────────────────────────────
    d = best_stats.get("diagnostics", {})
    result = {**best_stats,
              "temp_naive": temp_naive, "precip_naive": precip_naive,
              "orthogonalize": orthogonalize,
              "persistence_residual": persistence_residual,
              "lead_hours": lead_hours, "axes": axes,
              "epochs_run": epoch, "n_params": n_params,
              # 지표별 독립 최저치 — ablation 비교의 기준
              "best_temp_mae":   best_temp_only,
              "best_precip_mae": best_precip_only,
              "temp_gain_pct":   (1 - best_temp_only   / temp_naive)   * 100,
              "precip_gain_pct": (1 - best_precip_only / precip_naive) * 100}

    if verbose:
        print(f"\n{'='*70}")
        print(f" Phase 3-4 학습 완료  (+{lead_hours}h 예보 | 직교화 "
              f"{'ON' if orthogonalize else 'OFF'}"
              f" | 기온Δ {'ON' if persistence_residual else 'OFF'})")
        print(f"{'-'*70}")
        # 체크포인트에 실제로 저장된 값(= best_stats, 결합 val_score 최저
        # 에폭 기준)을 보고한다. best_temp_only/best_precip_only 는 그와
        # 다르다 — 두 지표가 각각 독립적으로 가장 좋았던(서로 다른 수 있는)
        # 에폭의 값이라, 어느 체크포인트도 실제로 동시에 내지 못한 조합일
        # 수 있다(2026-08-16 실측: 13년 재학습에서 기온 1.3290°C로
        # 찍혔지만 저장된 체크포인트는 1.3551°C — 결합 최적 에폭이 달랐다).
        # 화면·문서에 옮겨 적는 숫자가 실제 배포되는 체크포인트와 다르면
        # 안 되므로 여기서 헷갈릴 여지를 없앤다.
        temp_mae_deployed   = best_stats["val_temp_mae"]
        precip_mae_deployed = best_stats["val_precip_mae"]
        temp_gain   = (1 - temp_mae_deployed   / temp_naive)   * 100
        precip_gain = (1 - precip_mae_deployed / precip_naive) * 100
        print(f" 기온 MAE : {temp_mae_deployed:.4f} °C  vs 퍼시스턴스 "
              f"{temp_naive:.4f} °C  → {temp_gain:+.1f}%")
        print(f" 강수 MAE : {precip_mae_deployed:.4f} mm  vs 상시0예측 "
              f"{precip_naive:.4f} mm  → {precip_gain:+.1f}%")
        if abs(best_temp_only - temp_mae_deployed) > 1e-4 or \
           abs(best_precip_only - precip_mae_deployed) > 1e-4:
            print(f" (참고 — 지표별 독립 최저치, 체크포인트 값과 다름: "
                  f"기온 {best_temp_only:.4f}°C / 강수 {best_precip_only:.4f}mm)")
        print(f" 학습 에폭: {epoch}  (조기종료 {'ON' if early_stop else 'OFF'})")
        print(f" 강수 예측 표준편차: {best_stats['precip_pred_std']:.4f} mm  "
              f"({'정상 — 입력에 반응함' if best_stats['precip_pred_std'] > 1e-3 else '⚠ 붕괴 — 상수 출력'})")
        if persistence_residual:
            dm = best_stats['delta_magnitude']
            print(f" 기온 보정량 평균 |Δ|: {dm:.4f} °C  "
                  f"({'퍼시스턴스에 실질 보정 추가' if dm > 0.05 else '⚠ ≈0 — 사실상 퍼시스턴스 복사'})")
        print(f"{'-'*70}")
        d_final = best_stats.get("diagnostics", {})
        if dynamic_gate and "w_re" in d_final:
            print(f" 게이트 축 배분 (합=1): Re(위성) {d_final['w_re']:.4f} | "
                  f"Im(텍스트) {d_final['w_im']:.4f} | Z(수치) {d_final['w_z']:.4f}")
            print(f" 게이트 엔트로피: {d_final.get('gate_entropy', 0):.4f}  "
                  f"(최대 1.0986=균등배분, 낮을수록 한 축에 집중)")
        else:
            print(f" 학습된 축 가중치: α(Im)={best_stats['alpha_learned']:.4f} "
                  f"(초기 {ALPHA_INIT}) | φ(Z)={best_stats['phi_learned']:.4f} "
                  f"(초기 {PHI_INIT})")
        if d:
            print(f" 축 직교성 |cos| (0=직교, 무작위 기준 {1/np.sqrt(EMBED_DIM):.4f})")
            print(f"   직교화 전: Re-Im {d.get('cos_re_im_pre',0):.4f} | "
                  f"Re-Z {d.get('cos_re_z_pre',0):.4f} | Im-Z {d.get('cos_im_z_pre',0):.4f}")
            print(f"   직교화 후: Re-Im {d.get('cos_re_im_post',0):.4f} | "
                  f"Re-Z {d.get('cos_re_z_post',0):.4f} | Im-Z {d.get('cos_im_z_post',0):.4f}")
        em = best_stats.get("extreme_metrics", {})
        if em:
            print(f"{'-'*70}")
            print(f" 극한기상 헤드 (판정 임계 0.5, precision/recall/F1 — 양성희박이라 accuracy 무의미)")
            print(f"   라벨 소스 — 폭염·한파·황사 모두 기상청 공식 라벨만 사용"
                  f"(판정 불가 표본은 학습·채점 양쪽에서 제외)")
            for name, m in (("폭염", em.get("heatwave")),
                            ("한파", em.get("coldwave")),
                            ("황사", em.get("dust"))):
                if m is None:
                    continue
                print(f"   {name}: P={m['precision']:.3f} R={m['recall']:.3f} "
                      f"F1={m['f1']:.3f}  (실측양성 {m['n_pos']}개, 예측양성 {m['n_pred_pos']}개)")
        print(f" 체크포인트: {checkpoint}")
        print(f"{'='*70}")

    return result


if __name__ == "__main__":
    train()
