"""
pipeline_model.py — Phase 3: 3축 Tri-CHEF 파이프라인 모델
논문 Eq.1: s = √(A² + (α·B)² + (φ·C)²)

Re 축 (A): 위성 이미지  → ResNet18 4채널          (Phase 3-2)
Im 축 (B): 기상 문서    → MiniLM 384 → 프로젝션   (Phase 3-3)
Z  축 (C): IoT 수치 센서 → MLP 인코더             (Phase 3-1)

α: Im축 가중치 (log_param 학습 / Phase 3-5 동적 게이팅 예정)
φ: Z 축 가중치 (log_param 학습 / Phase 3-5 동적 게이팅 예정)

Phase 3-4 변경 사항
  1. Gram-Schmidt 3축 직교화 (orthogonalize 플래그 — ablation 가능)
  2. 강수 헤드 clamp → softplus  (dying-clamp gradient 소멸 버그 수정)
  3. 헤드 출력 편향 초기화       (타깃 평균에서 학습 시작)
  4. 축 상관/직교성 진단 지표 노출
"""
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
# torchvision 은 compact=False(ResNet18 경로) 일 때만 필요하다 — 배포
# 서빙 경로는 compact_satellite=True 고정(NumericalEncoder.__init__ 참고)
# 이라 이 임포트를 지연시키면 서빙 이미지에서 torchvision 의존성 자체를
# 뺄 수 있다(deploy_ablation.py 실측: Re축 기여도 0, 2026-08-08).


# ── 수치 안정 유틸 ───────────────────────────────────────────────────

def soft_normalize(v: torch.Tensor, eps: float = 1e-3) -> torch.Tensor:
    """
    소프트 L2 정규화:  v / √(‖v‖² + ε²)

    F.normalize 의 v/max(‖v‖,ε) 방식과 달리 ‖v‖=ε 지점의 미분 불연속(kink)이
    없고 v=0 에서도 매끄럽다. gradient 노름은 1/ε 로 유계이므로 직교화 잔차가
    0에 가까워져도 NaN·발산이 발생하지 않는다.

    ‖v‖=1 일 때 출력 노름 = 1/√(1+ε²) ≈ 1 − ε²/2  (ε=1e-3 → 오차 5e-7)
    """
    return v / torch.sqrt((v * v).sum(dim=-1, keepdim=True) + eps * eps)


def gram_schmidt_3axis(v_re: torch.Tensor,
                       v_im: torch.Tensor,
                       v_z:  torch.Tensor,
                       eps:  float = 1e-3):
    """
    배치별 3축 Gram-Schmidt 직교화 (논문 §III-B 축 독립성).

    순서 Re → Im → Z 는 논문 Eq.1의 축 우선순위(A, αB, φC)를 따른다.
      Re: 정보 그대로 보존
      Im: Re 성분 제거
      Z : Re·Im 성분 제거
    순서 의존적이므로 축 우선순위 자체가 ablation 대상이다.

    주의: soft_normalize 로 ‖u_re‖ = 1−5e-7 이므로 투영 제거 후 잔차
          <w_im, u_re> = <v_im,u_re>·(1−‖u_re‖²) ≈ <v_im,u_re>·1e-6 이
          남지만 실질 영향은 없다.

    입력/출력: 각 (B, D)
    """
    u_re = soft_normalize(v_re, eps)

    w_im = v_im - (v_im * u_re).sum(dim=-1, keepdim=True) * u_re
    u_im = soft_normalize(w_im, eps)

    w_z = (v_z
           - (v_z * u_re).sum(dim=-1, keepdim=True) * u_re
           - (v_z * u_im).sum(dim=-1, keepdim=True) * u_im)
    u_z = soft_normalize(w_z, eps)

    return u_re, u_im, u_z


def _pairwise_abs_cos(a: torch.Tensor, b: torch.Tensor) -> float:
    """배치 평균 |코사인 유사도| — 0이면 직교, 1이면 중복."""
    return F.cosine_similarity(a, b, dim=-1).abs().mean().item()


def hurdle_gamma_median(p: torch.Tensor, alpha: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """허들 혼합분포(Y=0 확률 1-p, Y~Gamma(alpha,rate) 확률 p, x>0)의 중앙값.

    2026-08-18 Gamma NLL 실험이 −147.5%로 실패한 원인은 최종 점추정으로
    p×E[Y|wet](조건부 평균)을 썼기 때문이다 — 이는 0에 확률질량이 몰린
    혼합분포에서 절대오차(MAE)를 최소화하는 점추정이 아니다. MAE 최적
    점추정은 중앙값이며, 닫힌 형태는 다음과 같다(gamma_median_test.py 에서
    몬테카를로 대조로 검증, 최대 상대오차 0.2%):
      F(x) = (1-p) + p·Gamma_CDF(x) 이므로
      p ≤ 0.5 → 중앙값 0 (F(0)=1-p 가 이미 0.5 이상)
      p > 0.5 → Gamma_분위수함수(1 - 0.5/p; alpha, rate)

    torch.special 에는 Gamma 분위수함수(불완전감마함수의 역함수)가 없어
    scipy.special.gammaincinv 로 CPU에서 계산한다 — 미분 가능성이 필요
    없는 순수 점추정 후처리이므로(NLL 손실은 이 함수를 거치지 않고
    log_prob 로 직접 계산된다) 문제되지 않는다.

    scipy는 지연 임포트한다(torchvision과 같은 이유 — 위 SatelliteEncoder
    주석 참고) — precip_gamma_nll=False(기본값, 배포 체크포인트)면 이
    함수 자체가 호출되지 않는데, 모듈 상단에서 무조건 import하면 scipy가
    requirements.txt(Streamlit Cloud 배포판, 최소 의존성 원칙)에 없어
    predict.py→pipeline_model.py 임포트 체인 전체가 깨진다.
    """
    from scipy.special import gammaincinv
    device = p.device
    q = torch.clamp(1.0 - 0.5 / p.clamp(min=1e-6), min=1e-6, max=1.0 - 1e-6)
    x = gammaincinv(alpha.detach().cpu().numpy(), q.detach().cpu().numpy())
    cont_median = torch.as_tensor(x, dtype=p.dtype, device=device) / rate
    return torch.where(p <= 0.5, torch.zeros_like(p), cont_median)


def _head_layers(in_dim: int, dropout: float = 0.0, out_dim: int = 1) -> list:
    """
    헤드 공통 층 구성. dropout>0 일 때만 Dropout 을 끼운다.

    p=0 이어도 무조건 끼우면 Sequential 의 인덱스가 밀려 state_dict 키가
    바뀌고(`head.2.weight` → `head.3.weight`) 구버전 체크포인트를 못 읽는다.
    Dropout 자체는 파라미터가 없으므로, 넣지 않으면 키가 그대로 유지된다.

    out_dim — 기본 1(회귀·이진분류 공용). Gamma NLL 강수 헤드처럼 분포
    모수 두 개(μ, α)를 함께 내야 할 때만 2로 지정한다.
    """
    layers = [nn.Linear(in_dim, 32), nn.GELU()]
    if dropout > 0:
        layers.append(nn.Dropout(dropout))
    layers.append(nn.Linear(32, out_dim))
    return layers


def _binary_head(embed_dim: int, prior: float, dropout: float = 0.0) -> nn.Sequential:
    """
    이진분류 헤드(강수/폭염/한파 공용) — 편향을 prior의 로짓으로 초기화.
    head_rain 이 처음 도입한 패턴(logit(wet_prior))을 그대로 재사용한다.
    prior 는 호출부(train.py)가 학습 표본에서 실측한 값을 넘긴다.
    """
    head = nn.Sequential(*_head_layers(embed_dim, dropout))
    p = min(max(prior, 1e-4), 1 - 1e-4)
    with torch.no_grad():
        head[-1].bias.fill_(math.log(p / (1 - p)))
    return head


# ── Z축: 수치 센서 인코더 ────────────────────────────────────────────

class NumericalEncoder(nn.Module):
    """
    Z축 (C): ASOS 지상관측 수치 8차원 → embed_dim.
    기온·강수·습도·풍속·풍향sin/cos·기압·강수형태를 단일 표현 공간으로 압축.
    """
    def __init__(self, in_dim: int = 8, embed_dim: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, 32), nn.LayerNorm(32), nn.GELU(),
            nn.Linear(32, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── Re축: 위성 이미지 인코더 ─────────────────────────────────────────

class SatelliteEncoder(nn.Module):
    """
    Re축 (A): 4채널 (R,G,B,NIR) 위성 이미지 → ResNet18 → embed_dim.
    입력 (B, 4, H, W), H/W ≥ 32.
    4채널 입력이라 ImageNet 사전학습 가중치는 사용하지 않는다.
    Phase 3-실제: Sentinel-2 연결 시 3ch pretrained + 채널 확장 검토.
    """
    def __init__(self, in_channels: int = 4, embed_dim: int = 64,
                 compact: bool = True):
        """
        compact=True (기본): 소형 CNN 약 2.8만 파라미터.

        ResNet18(1,100만)은 학습 표본 572개 대비 비율이 약 20,000:1 이라
        훈련셋을 그대로 암기한다. Phase 3-5 실측에서 게이트가 정보가 없는
        위성 축에 98% 를 배분한 원인이 이것이다 — 훈련 손실만 보면 암기력이
        가장 큰 축이 항상 이긴다. 표본이 수만 개 규모가 되기 전에는 소형
        인코더가 맞다.
        """
        super().__init__()
        if compact:
            self.net = nn.Sequential(
                nn.Conv2d(in_channels, 16, 3, stride=2, padding=1),
                nn.GroupNorm(4, 16), nn.GELU(),          # 32 → 16
                nn.Conv2d(16, 32, 3, stride=2, padding=1),
                nn.GroupNorm(8, 32), nn.GELU(),          # 16 → 8
                nn.Conv2d(32, 64, 3, stride=2, padding=1),
                nn.GroupNorm(8, 64), nn.GELU(),          # 8 → 4
                nn.AdaptiveAvgPool2d(1), nn.Flatten(),
                nn.Linear(64, embed_dim),
            )
        else:
            import torchvision.models as tv_models   # 지연 임포트 — 위 주석 참고
            base = tv_models.resnet18(weights=None)
            base.conv1 = nn.Conv2d(
                in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False
            )
            base.fc = nn.Linear(512, embed_dim)
            self.net = base

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.net(x), dim=-1)


# ── Im축: MiniLM 텍스트 인코더 ───────────────────────────────────────

class TextEncoder(nn.Module):
    """
    Im축 (B): MiniLM 384차원 임베딩 → embed_dim 프로젝션.
    MiniLM 추론 자체는 text_collector.py 에서 처리(캐싱 포함)하고,
    이 클래스는 학습 가능한 384 → embed_dim 프로젝션만 담당한다.
    """
    def __init__(self, text_dim: int = 384, embed_dim: int = 64):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(text_dim, 128), nn.LayerNorm(128), nn.GELU(),
            nn.Linear(128, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


class TendencyEncoder(nn.Module):
    """
    Im축 대안 (Phase 3-11): 시간 경향 벡터 12차원 → embed_dim.

    TextEncoder 를 대체한다. MiniLM 임베딩(384차원)은 Z축 스칼라를 버킷팅해
    재인코딩한 것이라 정보이론적으로 Z를 못 넘지만(중복), 시간 미분 ∂s/∂t 는
    단일 스냅샷에서 유도 불가능해 Z와 진짜로 독립이다 — tendency_collector.py
    docstring 참고.

    은닉폭 실험(2026-08-09): 32로 줄였더니(입력이 384→12로 작아졌다는 이유로)
    강수·폭염·한파·황사 F1 이 전부 소폭 하락했다(예: 황사 0.671→0.611).
    TextEncoder 가 주던 이득 일부가 "정보량"이 아니라 "학습 가능한 별도
    경로의 비선형 용량"(암묵적 정규화, 2026-08-08 A/B 테스트에서 이미 확인된
    현상) 이었을 가능성이 있어 — 입력 차원과 무관하게 128로 되돌려 용량
    자체가 원인인지 검증한다.
    """
    def __init__(self, in_dim: int = 12, embed_dim: int = 64, hidden: int = 128):
        super().__init__()
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.LayerNorm(hidden), nn.GELU(),
            nn.Linear(hidden, embed_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return F.normalize(self.proj(x), dim=-1)


# ── Phase 3-5: 입력 조건부 축 게이팅 ─────────────────────────────────

class ModalityGate(nn.Module):
    """
    입력에 따라 시각마다 달라지는 축 가중치 (w_re, w_im, w_z), Σw = 1.

    왜 필요한가 — Phase 3-4 ablation 실측:
      정적 log_alpha/log_phi 는 정보가 없는 축의 가중치를 오히려 키웠다
      (Z+Re+Im 조건에서 α 0.4→0.665, φ 0.2→0.462). 훈련 손실만 최소화하면
      무용 축도 훈련셋 암기에 기여하므로 가중치를 낮출 유인이 없다.

    왜 softmax 인가 — 스케일 자유도 제거:
      가중치가 자유로우면 (w_re, w_im, w_z) 를 k 배 줄이고 헤드가 1/k 로
      상쇄해 어떤 크기 정규화도 무력화된다. Σw=1 로 고정하면 축들이 정해진
      예산을 나눠 갖게 되어 '어느 축에 얼마를 줄 것인가'가 실제 선택이 된다.

    게이트 입력에 시각(sin/cos)이 포함되는 점이 중요하다. 광학 위성 영상은
    야간에 정보가 없으므로, 실데이터에서는 게이트가 밤에 Re 축을 낮추는
    방향을 학습할 수 있어야 한다(현재 시뮬레이션에서는 검증만 가능).
    """

    def __init__(self, in_dim: int = 12, hidden: int = 16,
                 init_weights: tuple = None):
        """
        init_weights=None (기본): 균등 초기화 (1/3, 1/3, 1/3).

        논문 기본값 (1, 0.4, 0.2) 로 초기화하면 정규화 후 (0.625, 0.25, 0.125)
        가 되어 Re 축이 처음부터 1등이 된다. Phase 3-5 실측에서 엔트로피 항이
        이 초기 순서를 그대로 증폭시켜, 모델이 어느 축이 유용한지 배우기도
        전에 위성 축으로 고착됐다. 논문의 축 우선순위(A > αB > φC)는 검색
        도메인의 것이며 기후 예보에는 이전되지 않는다.
        """
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden), nn.GELU(),
            nn.Linear(hidden, 3),
        )
        with torch.no_grad():
            # 학습 초기에는 입력 의존성을 거의 0 으로 두고 편향에서 출발한다.
            self.net[-1].weight.mul_(0.01)
            if init_weights is None:
                self.net[-1].bias.zero_()                  # softmax → 균등
            else:
                w = torch.tensor(init_weights, dtype=torch.float32)
                self.net[-1].bias.copy_(torch.log(w / w.sum()))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """x: (B, in_dim) 표준화된 수치 특성 → (B, 3) 축 가중치."""
        return torch.softmax(self.net(x), dim=-1)


# ── 메인: 3축 Tri-CHEF 파이프라인 ────────────────────────────────────

class TriCHEFPipeline(nn.Module):
    """
    Phase 3 3축 Tri-CHEF 파이프라인.

        s = √(A² + (α·B)² + (φ·C)²)      [논문 Eq.1]

    구현 현황
      Phase 3-1  ✓  Z축 수치 인코더, log_alpha/log_phi 학습
      Phase 3-2  ✓  Re축 위성 이미지 (ResNet18 4채널)
      Phase 3-3  ✓  Im축 기상 문서 (MiniLM 384 프로젝션)
      Phase 3-4  ✓  Gram-Schmidt 직교화 + dying-clamp 수정
      Phase 3-5  ✓  ModalityGate 입력 조건부 축 가중치
      Phase 3-6  ✓  PPO 경보 게이트 (rl_phase_filter.py) — 이 모델은 동결해
                    상태 특성 공급원으로만 쓴다. 논문의 위상각 θ 는 여기서
                    axis_report() 로 노출하지만, 축이 모두 단위 정규화라
                    게이트 배분으로 퇴화한다(README 한계 6).
    """

    def __init__(self, embed_dim: int = 64,
                 num_features: int = 12,
                 alpha_init: float = 0.4,
                 phi_init:   float = 0.2,
                 orthogonalize: bool = False,
                 gs_eps: float = 1e-3,
                 temp_mean:   float = 20.0,
                 precip_mean: float = 0.5,
                 persistence_residual: bool = True,
                 feat_mean: list = None,
                 feat_std:  list = None,
                 dynamic_gate: bool = False,
                 compact_satellite: bool = True,
                 re_channels: int = 4,
                 im_dim: int = 384,
                 wet_prior: float = 0.067,
                 heatwave_prior: float = 0.01,
                 coldwave_prior: float = 0.003,
                 dust_prior: float = 0.05,
                 signed_head_input: bool = False,
                 signed_precip_input: bool = False,
                 head_dropout: float = 0.0,
                 coldwave_dropout: float = 0.0,
                 precip_gamma_nll: bool = False,
                 precip_gamma_alpha_init: float = 1.0,
                 precip_gamma_mu_init: float = None):
        """
        orthogonalize        : Gram-Schmidt 직교화 on/off (ablation 스위치)
        gs_eps               : 소프트 정규화 ε — gradient 상한 1/ε
        temp_mean            : 기온 타깃 평균 — head_temp 편향 초기화용
        precip_mean          : 강수 타깃 평균 — head_precip 편향 초기화용
        persistence_residual : 기온을 절대값이 아닌 변화량(Δ)으로 예측
        feat_mean / feat_std : 입력 표준화 통계 (Δ 예측 시 필수)
        dynamic_gate         : True 면 정적 α/φ 대신 ModalityGate 사용 (Phase 3-5)
        wet_prior            : 검증셋 실측 강수 비율(기본 0.067, 2026-08-07
                                12관측소 3.5년 실측) — head_rain 편향을 이
                                사전확률의 로짓으로 초기화해 학습 시작부터
                                "대부분 무강수"를 반영한다.
        heatwave_prior       : 학습셋 실측 폭염 비율(목표시각 기온≥33°C,
                                기상청 폭염주의보 기준) — head_heatwave 편향
                                초기화. 임계값 자체는 규정값이라 고정이지만
                                비율은 train.py 가 실측해 넘긴다.
        coldwave_prior       : 학습셋 실측 한파 비율(목표시각 기온≤-12°C,
                                기상청 한파주의보 기준) — head_coldwave 편향
                                초기화.
        dust_prior           : 학습셋 실측 황사 비율(기상청 "날씨 이슈별
                                데이터" 공식 황사관측여부 라벨 기준) —
                                head_dust 편향 초기화.
        """
        super().__init__()
        self.embed_dim     = embed_dim
        self.orthogonalize = orthogonalize
        self.gs_eps        = gs_eps
        self.persistence_residual = persistence_residual
        self.dynamic_gate  = dynamic_gate

        # 표준화 통계를 버퍼로 보관 → forward 에서 현재 기온을 역표준화해
        # 퍼시스턴스 기준선을 복원한다 (state_dict 에 함께 저장/로드됨).
        if persistence_residual:
            if feat_mean is None or feat_std is None:
                raise ValueError(
                    "persistence_residual=True 이면 feat_mean/feat_std 가 필요합니다."
                )
            self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
            self.register_buffer("feat_std",  torch.tensor(feat_std,  dtype=torch.float32))

        # 3축 인코더
        self.re_channels = re_channels
        self.enc_re = SatelliteEncoder(re_channels, embed_dim,
                                       compact=compact_satellite)   # A: 위성
        # Im축 인코더 — im_dim=384 면 MiniLM 텍스트(기존), 12 면 시간 경향
        # 벡터(Phase 3-11). 체크포인트에 im_dim 을 저장해야 복원 시 아키텍처가
        # 맞는다(dynamic_gate/compact_satellite 와 같은 이유).
        self.im_dim = im_dim
        self.enc_im = (TextEncoder(im_dim, embed_dim) if im_dim >= 128
                       else TendencyEncoder(im_dim, embed_dim))
        self.enc_z  = NumericalEncoder(num_features, embed_dim)   # C: 수치

        # 축 가중치 — 정적(log 공간 학습) 또는 동적(입력 조건부 게이트)
        if dynamic_gate:
            # 균등 초기화 — 논문의 Re 우선 순서를 물려받지 않는다 (위 주석 참조)
            self.gate = ModalityGate(in_dim=num_features, init_weights=None)
        else:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
            self.log_phi   = nn.Parameter(torch.tensor(math.log(phi_init)))

        # 예측 헤드
        # signed_precip_input (2026-08-17) — 강수 경로(양·확률)에도 부호 표현을
        # 함께 넣는 실험 스위치. 극한기상 헤드에 같은 조치를 했더니 강수 MAE가
        # 기준선 대비 −17.9%에서 −9.9%로 좋아졌는데, 헤드들이 공유 표현을 두고
        # 벌이던 경쟁이 완화된 것으로 해석했다. 그렇다면 강수 헤드 자신에게도
        # 같은 여지를 주면 더 좋아질 수 있다는 가설을 검증하기 위한 것이다.
        # 기본값 False — 검증 전까지 배포 동작을 바꾸지 않는다.
        self.signed_precip_input = signed_precip_input
        self.head_dropout = head_dropout
        _reg_dim = embed_dim * 2 if signed_precip_input else embed_dim
        self.head_temp = nn.Sequential(*_head_layers(embed_dim, head_dropout))
        # precip_gamma_nll (2026-08-18) — 강수 amount 헤드를 MSE 회귀 대신
        # Gamma 분포 모수(μ, α) 회귀로 바꾸는 ablation 스위치. distribution_
        # diagnostics.py 실측: 습윤 표본(강수≥WET_THRESH) 의 KS 통계량이
        # 정규 0.294·감마 0.111·로그정규 0.066 순으로, 강수량 분포가 뚜렷하게
        # 우측 치우침(로그정규 최적합)이라 대칭 손실인 MSE 로는 호우 분위수를
        # 구조적으로 과소예측한다(99백분위 실측 6.55mm vs 예측 21.49mm, 0.30배
        # — precip_breakdown.py 별개 진단인 "무강수 구간 비영 오차"와는 다른
        # 결함). Gamma NLL 은 CRPS 처럼 적정 점수규칙이고, 오른쪽 꼬리가 두꺼운
        # 표본에 그래디언트가 (진실 - 예측) 대신 log-비율로 걸려 대칭 손실보다
        # 큰 값을 과소평가하지 않는다. 기본값 False — 검증 전까지 배포 동작을
        # 바꾸지 않는다(head_dropout/signed_precip_input 과 같은 패턴).
        self.precip_gamma_nll = precip_gamma_nll
        _precip_out = 2 if precip_gamma_nll else 1
        self.head_precip = nn.Sequential(
            *_head_layers(_reg_dim, head_dropout, out_dim=_precip_out))
        # Phase 3-7 hurdle 헤드 — "비가 오는가"를 별도 이진분류기로 분리.
        #
        # 왜 필요한가(2026-08-07 실측, precip_breakdown.py): 검증셋의
        # 93.3%가 무강수인데 head_precip 은 softplus 출력이라 수학적으로
        # 정확히 0을 낼 수 없다(softplus(x)>0 for all finite x). 무강수
        # 구간에서 이 미세한 비영 오차가 표본 수(6만+)만큼 누적되면, 강수
        # 구간(6.7%)에서 baseline을 이기고 얻는 이득(모델이 baseline보다
        # +10.2% 나음, 실측 확인됨)을 통째로 삼켜버린다 — 전체 격차의
        # 139.7%가 무강수 구간에서만 발생했다. 후처리 클리핑만으로는 최선의
        # 경우(0.2mm 임계)에도 여전히 baseline에 8.2% 졌다: 약한 강수 예측을
        # 함께 깎아먹기 때문이다. 즉 회귀 헤드 하나가 "오는가"와 "얼마나"를
        # 동시에 표현하려는 구조 자체가 문제였다.
        #
        # head_rain 은 sigmoid 로 [0,1] 확률을 내고, 최종 예측은
        # P(비) × softplus(양) 로 부드럽게 결합한다 — 이산 게이트(하드 0/1)
        # 대신 이렇게 두는 이유는 P(비)→0 이면 예측이 연속적으로 0에
        # 수렴하면서도, 학습 내내 미분 가능해 그래디언트가 끊기지 않기
        # 때문이다(Phase 3-3 의 dying-clamp 교훈과 같은 이유로 하드 게이트를
        # 피한다).
        self.head_rain = _binary_head(_reg_dim, wet_prior, head_dropout)

        # Phase 3-8 극한기상 경보 헤드 — head_rain 과 동일한 구조·목적
        # (연속 회귀로는 못 내는 "일어나는가"를 명시적으로 학습). 임계값은
        # 기상청 주의보 발표기준(공식 규정값, 데이터에서 추정하지 않음):
        #   폭염주의보: 일 최고기온 33°C 이상
        #   한파주의보: 아침 최저기온 -12°C 이하
        # 강풍(≥14m/s)은 실측 빈도가 학습셋 환산 약 300표본(0.1%)으로
        # 지나치게 희박해 이번 확장에서 제외했다(정량 확인, 2026-08-07).
        #
        # signed_head_input (2026-08-16) — 극한기상 헤드에만 부호 있는 표현을
        # 함께 넣는다. Eq.1 융합 `s=√((w·v)²)`는 제곱을 거치며 부호를 없애므로
        # 헤드는 |편차|만 볼 뿐 "위로 벗어났는지 아래로 벗어났는지"를 모른다.
        # 그 결과 극단적 저온과 극단적 고온이 비슷한 크기로 들어와, 실제
        # 관측에서 한겨울에 폭염 확률이 올라갔다(v9 실측: −15°C 이하 809건 중
        # 85.7%가 0.5 초과, seasonal_falsealarm_check.py). 회귀 기온 헤드가
        # 퍼시스턴스 잔차 구조를 쓰는 이유와 정확히 같은 문제이며, 분류
        # 헤드에서도 같은 원인이 작동한다는 것을 그동안 놓쳤다.
        #
        # v_z(정규화된 Z축 임베딩)는 제곱을 거치지 않아 부호가 살아 있다.
        # 원본 num_x를 직접 넣지 않는 이유 — 거기엔 위도·경도가 들어 있어
        # 헤드가 관측소를 암기할 통로가 된다(Phase 3-6 PPO에서 실제로 발생).
        #
        # 회귀(기온·강수)·강수 헤드는 건드리지 않는다. 진단된 결함은 극한기상
        # 분류에 있고, 강수 경로는 이미 기준선 미달로 취약해 함께 바꾸면
        # 원인 분리가 불가능해진다.
        _ext_dim = embed_dim * 2 if signed_head_input else embed_dim
        self.signed_head_input = signed_head_input
        self.head_heatwave = _binary_head(_ext_dim, heatwave_prior, head_dropout)
        # coldwave_dropout (2026-08-17) — head_dropout과 별도로 한파 헤드에만
        # 거는 드롭아웃. head_dropout을 전체 헤드에 걸었더니(2026-08-17 기각)
        # 겨울 폭염 오탐이 38.8%로 재발했다 — 드롭아웃이 폭염 헤드의 부호
        # 표현(v_Z) 경로 유닛까지 무작위로 꺼버려, 그 결함을 고친 경로 자체를
        # 학습 중 반복 차단했기 때문이다. 한파는 사건이 8~40건뿐이라 과적합
        # 억제가 필요하지만(실제로 head_dropout 전체 적용 시 한파 F1이
        # 0.449→0.472로 개선됐었다), 폭염·황사 헤드는 건드리지 않아야 그 개선을
        # 부작용 없이 취할 수 있다.
        self.head_coldwave = _binary_head(_ext_dim, coldwave_prior,
                                          max(head_dropout, coldwave_dropout))
        # Phase 3-9 황사 헤드 — 폭염/한파와 동일 패턴. 라벨은 ASOS 현상번호
        # 추정치가 아니라 기상청 "날씨 이슈별 데이터"의 공식 황사관측여부를
        # 쓴다(import_weather_issues.py, 2026-08-09).
        self.head_dust = _binary_head(_ext_dim, dust_prior, head_dropout)

        # 출력 편향 초기화 — magnitude 원소값이 ~1/√D 로 작아서 편향만
        # 학습하는 데 수십 에폭이 낭비되는 문제를 제거한다.
        with torch.no_grad():
            # Δ 예측이면 목표 평균이 ≈0 이므로 편향도 0에서 시작한다.
            self.head_temp[-1].bias.fill_(0.0 if persistence_residual else temp_mean)
            # softplus(b) = precip_mean  →  b = log(exp(precip_mean) − 1)
            b = math.log(math.expm1(max(precip_mean, 1e-3)))
            if precip_gamma_nll:
                # μ 는 **습윤 조건부** 평균으로 초기화한다 — Gamma NLL 이
                # 습윤 표본에서만 계산되므로 μ 의 학습 목표가 E[y | 강수] 다.
                # 전체 평균(precip_mean≈0.16mm)을 쓰면 목표(≈2.4mm)보다 15배
                # 낮은 데서 출발해, 이 블록이 없애려던 "편향만 학습하느라
                # 에폭을 낭비하는" 문제가 그대로 남는다(2026-08-19 수정).
                mu0 = precip_gamma_mu_init if precip_gamma_mu_init else precip_mean
                self.head_precip[-1].bias[0].fill_(
                    math.log(math.expm1(max(mu0, 1e-3))))
                a0 = max(precip_gamma_alpha_init - 1e-3, 1e-3)
                self.head_precip[-1].bias[1].fill_(math.log(math.expm1(a0)))
            else:
                self.head_precip[-1].bias.fill_(b)

        # 마지막 forward 의 축 진단 지표 (detach 된 float, 학습에 무관)
        self.last_diagnostics: dict = {}
        self._gate_w = None   # 동적 게이트의 마지막 출력 (엔트로피 정규화용)
        self._last_rain_logit = None      # hurdle BCE 손실 계산용 (train.py 가 읽음)
        self._last_precip_mu = None       # Gamma NLL μ (precip_gamma_nll=True 일 때만)
        self._last_precip_alpha = None    # Gamma NLL α (precip_gamma_nll=True 일 때만)
        self._last_heatwave_logit = None  # 폭염 BCE 손실 계산용
        self._last_coldwave_logit = None  # 한파 BCE 손실 계산용
        self._last_dust_logit = None      # 황사 BCE 손실 계산용

    # ── 축 가중치 ────────────────────────────────────────────────

    @property
    def alpha(self) -> torch.Tensor:
        """Im축 가중치 (정적 모드 전용)."""
        return torch.exp(self.log_alpha)

    @property
    def phi(self) -> torch.Tensor:
        """Z축 가중치 (정적 모드 전용)."""
        return torch.exp(self.log_phi)

    def axis_weights(self) -> dict:
        """
        현재 축 가중치. 동적 모드에서는 마지막 forward 의 배치 평균을 쓴다.
        Eq.1 의 (1, α, φ) 형태와 맞추기 위해 Re 축 기준으로 정규화한다.
        """
        if self.dynamic_gate:
            if self._gate_w is None:
                return {"alpha": float("nan"), "phi": float("nan")}
            w = self._gate_w.detach().mean(dim=0)
            base = w[0].clamp_min(1e-6)
            return {"alpha": (w[1] / base).item(), "phi": (w[2] / base).item(),
                    "w_re": w[0].item(), "w_im": w[1].item(), "w_z": w[2].item()}
        return {"alpha": self.alpha.item(), "phi": self.phi.item()}

    def gate_entropy(self) -> torch.Tensor:
        """
        마지막 forward 의 게이트 엔트로피 (정규화 항으로 사용).
        낮을수록 가중치가 소수 축에 집중된다. 최대 ln3 ≈ 1.0986.
        어느 축을 살릴지는 지정하지 않고 '집중하라'는 압력만 준다
        — 정답을 코드에 심지 않기 위한 대칭적 설계.
        """
        if self._gate_w is None:
            return torch.zeros((), device=next(self.parameters()).device)
        w = self._gate_w
        return -(w * torch.log(w + 1e-9)).sum(dim=-1).mean()

    # ── 축 임베딩 ────────────────────────────────────────────────

    def encode(self,
               num_x: torch.Tensor,
               img_x: torch.Tensor = None,
               txt_x: torch.Tensor = None):
        """
        3축 임베딩 반환 (직교화 전 원본).
        Phase 3-6 RL 상태 구성 및 위상각 계산의 입력이 된다.
        img_x / txt_x 가 None 이면 해당 축은 영벡터 → Eq.1 기여 0.
        """
        B = num_x.size(0)
        v_z  = self.enc_z(num_x)
        v_re = (self.enc_re(img_x) if img_x is not None
                else torch.zeros(B, self.embed_dim, device=num_x.device))
        v_im = (self.enc_im(txt_x) if txt_x is not None
                else torch.zeros(B, self.embed_dim, device=num_x.device))
        return v_re, v_im, v_z

    # ── Phase 3-6: 샘플별 축 진단 (RL 상태 구성용) ───────────────

    @torch.no_grad()
    def axis_report(self,
                    num_x: torch.Tensor,
                    img_x: torch.Tensor = None,
                    txt_x: torch.Tensor = None) -> dict:
        """
        샘플별 축 진단 — Phase 3-6 경보 게이트의 상태 특성.

        last_diagnostics 는 배치 평균만 남겨서 RL 상태로 쓸 수 없다.
        여기서는 샘플마다 개별 값을 (B,) 텐서로 돌려준다.

            theta = atan2(‖w_im·v_im‖, ‖w_re·v_re‖)   [논문의 위상각]

        주의 — theta 는 이 구현에서 퇴화한다. 세 인코더가 모두 F.normalize
        출력이라 ‖v‖ = 1 이므로 theta = atan2(w_im, w_re) 가 되어, 축 자체의
        per-sample 정보가 아니라 게이트 배분의 재표현일 뿐이다. 실제 축별
        정보는 축 간 코사인에 남아 있으므로 함께 반환한다.
        이 퇴화는 test_phase36.py T1 에서 수치로 확인한다.
        """
        v_re, v_im, v_z = self.encode(num_x, img_x, txt_x)

        if self.dynamic_gate:
            w = self.gate(num_x)
        else:
            ones = torch.ones(num_x.size(0), 1, device=num_x.device)
            w = torch.cat([ones, self.alpha * ones, self.phi * ones], dim=-1)

        a_re = w[:, 0] * v_re.norm(dim=-1)
        a_im = w[:, 1] * v_im.norm(dim=-1)

        return {
            "gate":      w,                                        # (B, 3)
            "theta":     torch.atan2(a_im, a_re),                  # (B,)
            "cos_re_im": F.cosine_similarity(v_re, v_im, dim=-1),  # (B,)
            "cos_re_z":  F.cosine_similarity(v_re, v_z,  dim=-1),
            "cos_im_z":  F.cosine_similarity(v_im, v_z,  dim=-1),
        }

    # ── 순전파 ───────────────────────────────────────────────────

    def forward(self,
                num_x: torch.Tensor,
                img_x: torch.Tensor = None,
                txt_x: torch.Tensor = None,
                collect_diagnostics: bool = False) -> torch.Tensor:
        """
        num_x : (B, 8)        수치 센서 (필수)
        img_x : (B, 4, H, W)  위성 이미지
        txt_x : (B, 384)      MiniLM 텍스트 임베딩

        반환  : (B, 2) — [기온 °C, 강수량 mm]  (물리 단위)
        """
        v_re, v_im, v_z = self.encode(num_x, img_x, txt_x)

        if collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics = {
                    "cos_re_im_pre": _pairwise_abs_cos(v_re, v_im),
                    "cos_re_z_pre":  _pairwise_abs_cos(v_re, v_z),
                    "cos_im_z_pre":  _pairwise_abs_cos(v_im, v_z),
                }

        if self.orthogonalize:
            v_re, v_im, v_z = gram_schmidt_3axis(v_re, v_im, v_z, self.gs_eps)

        if collect_diagnostics:
            with torch.no_grad():
                self.last_diagnostics.update({
                    "cos_re_im_post": _pairwise_abs_cos(v_re, v_im),
                    "cos_re_z_post":  _pairwise_abs_cos(v_re, v_z),
                    "cos_im_z_post":  _pairwise_abs_cos(v_im, v_z),
                    "norm_im": v_im.norm(dim=-1).mean().item(),
                    "norm_z":  v_z.norm(dim=-1).mean().item(),
                })

        # ── 축 가중치 결정 ───────────────────────────────────────
        if self.dynamic_gate:
            # (B, 3) — 샘플마다 다른 가중치. 엔트로피 정규화를 위해
            # detach 하지 않고 보관하며, train() 이 직후에 읽어 간다.
            w = self.gate(num_x)
            self._gate_w = w
            w_re, w_im, w_z = w[:, 0:1], w[:, 1:2], w[:, 2:3]
        else:
            self._gate_w = None
            w_re, w_im, w_z = 1.0, self.alpha, self.phi

        # Hermitian-style modulus (논문 Eq.1, 원소별)
        magnitude = torch.sqrt(
            (w_re * v_re) ** 2 + (w_im * v_im) ** 2 + (w_z * v_z) ** 2 + 1e-7
        )

        if collect_diagnostics and self._gate_w is not None:
            with torch.no_grad():
                gw = self._gate_w
                self.last_diagnostics.update({
                    "w_re": gw[:, 0].mean().item(),
                    "w_im": gw[:, 1].mean().item(),
                    "w_z":  gw[:, 2].mean().item(),
                    # 가중치가 한 축에 몰릴수록 낮아진다 (최대 ln3 = 1.0986)
                    "gate_entropy": float(
                        -(gw * torch.log(gw + 1e-9)).sum(dim=-1).mean()
                    ),
                })

        # 기온: 퍼시스턴스 기준선 + Eq.1 이 학습한 보정량(Δ).
        #
        # 각 축은 F.normalize 로 단위 벡터가 되고 √(v²) 가 부호까지 제거하므로,
        # magnitude 만으로 절대 기온(≈28°C)을 복원하기는 구조적으로 불리하다.
        # 반면 T_{t+1} ≈ T_t 퍼시스턴스는 1시간 예보에서 매우 강한 기준선이다.
        # 따라서 기준선을 물리적으로 제공하고 Eq.1 은 그 위의 편차만 학습한다
        # — 논문의 오프셋 보정(offset correction) 구조와 동일하다.
        delta_temp = self.head_temp(magnitude)
        if self.persistence_residual:
            # num_x 는 표준화된 값이므로 역표준화해 현재 기온을 복원
            temp_now  = num_x[:, 0:1] * self.feat_std[0] + self.feat_mean[0]
            temp_pred = temp_now + delta_temp
        else:
            temp_pred = delta_temp

        # Hurdle: P(비) × 강수량. softplus 단독으로는 못 냈던 정확한 0을
        # P(비)→0 극한에서 연속적으로 표현한다(위 head_rain 주석 참고).
        # clamp(x,0,·) 대신 softplus 를 양(amount) 헤드에 쓰는 이유는 여전히
        # Phase 3-3 과 동일 — x<0 에서 gradient 0 이 되는 dying-clamp을
        # 피한다.
        _reg_in = (torch.cat([magnitude, v_z], dim=-1)
                   if self.signed_precip_input else magnitude)
        rain_logit = self.head_rain(_reg_in)
        self._last_rain_logit = rain_logit   # train.py 가 BCE 손실 계산에 사용
        rain_prob = torch.sigmoid(rain_logit)
        if self.precip_gamma_nll:
            _precip_out = self.head_precip(_reg_in)
            mu    = F.softplus(_precip_out[:, 0:1]) + 1e-6
            alpha = F.softplus(_precip_out[:, 1:2]) + 1e-3
            self._last_precip_mu, self._last_precip_alpha = mu, alpha
            if self.training:
                # 학습 중엔 이 열이 손실 계산에 쓰이지 않는다(train.py 가
                # _last_precip_mu/_last_precip_alpha 로 NLL을 직접 계산) —
                # 그런데도 매 스텝 scipy CPU 왕복을 시키면 학습 루프만
                # 느려진다. 저렴하지만 틀린 값(곱셈 점추정)으로 채워
                # forward() shape만 맞춘다.
                precip_pred = rain_prob * mu
            else:
                # 추론/검증 시점엔 hurdle_gamma_median() 이 계산하는 올바른
                # (MAE 최적) 점추정을 쓴다 — 위 함수 docstring 참고.
                rate = alpha / mu
                precip_pred = hurdle_gamma_median(rain_prob.squeeze(-1), alpha.squeeze(-1),
                                                   rate.squeeze(-1)).unsqueeze(-1)
        else:
            amount = F.softplus(self.head_precip(_reg_in))
            precip_pred = rain_prob * amount

        # 극한기상 경보 확률 — 회귀 출력(temp_pred/precip_pred)과 별개로
        # 학습·추론 시 model._last_*_logit / extreme_event_probs() 로 읽는다.
        # forward() 반환 shape(B,2)를 그대로 유지해 predict.py 등 기존
        # 호출부를 건드리지 않는다(head_rain 도입 때와 같은 이유).
        # 극한기상 헤드에는 부호가 살아 있는 v_z 를 함께 넣는다(위 __init__
        # signed_head_input 주석 참고). magnitude 는 제곱을 거쳐 |편차| 만
        # 남으므로, 이것만으로는 한겨울과 한여름이 구분되지 않는다.
        _ext_in = (torch.cat([magnitude, v_z], dim=-1)
                   if self.signed_head_input else magnitude)
        self._last_heatwave_logit = self.head_heatwave(_ext_in)
        self._last_coldwave_logit = self.head_coldwave(_ext_in)
        self._last_dust_logit = self.head_dust(_ext_in)

        return torch.cat([temp_pred, precip_pred], dim=-1)

    @torch.no_grad()
    def extreme_event_probs(self) -> dict:
        """마지막 forward() 호출의 극한기상 확률(독립 이진분류, 합=1 아님)."""
        return {
            "heatwave": torch.sigmoid(self._last_heatwave_logit)
                        if self._last_heatwave_logit is not None else None,
            "coldwave": torch.sigmoid(self._last_coldwave_logit)
                        if self._last_coldwave_logit is not None else None,
            "dust":     torch.sigmoid(self._last_dust_logit)
                        if self._last_dust_logit is not None else None,
            "rain":     torch.sigmoid(self._last_rain_logit)
                        if self._last_rain_logit is not None else None,
        }


# ── Tri-CHEF 없이 Z축만 — 대조군 ─────────────────────────────────────

class ZOnlyBaseline(nn.Module):
    """
    Hermitian modulus를 거치지 않는 대조군. NumericalEncoder 출력을 헤드에
    직접 넣는다 — TriCHEFPipeline과 embed_dim·persistence_residual·softplus
    헤드·편향 초기화가 전부 동일하고, 딱 하나만 다르다: magnitude 연산을
    거치는가.

    이 비교가 필요한 이유(2026-08-07 실측): 4번의 학습 모두 게이트가
    97~99.9% Z축으로 수렴했다. 그런데 magnitude = √((w_re·v_re)²+...+eps)
    는 원소별 연산이라, w_re≈w_im≈0 인 지금도 magnitude[i] ≈ w_z·|v_z[i]|
    로 축약될 뿐 v_z[i] 자체가 되지는 않는다 — 각 차원의 **부호가
    사라진다**. 즉 게이트가 이미 Z축만 쓰기로 정했더라도, Tri-CHEF 구조를
    유지하는 한 부호 정보 손실이라는 대가는 계속 치르고 있다. 이 클래스는
    그 대가가 실제로 성능에 영향을 주는지 직접 재기 위한 것이다.
    """

    def __init__(self, embed_dim: int = 64, num_features: int = 12,
                 temp_mean: float = 20.0, precip_mean: float = 0.5,
                 persistence_residual: bool = True,
                 feat_mean: list = None, feat_std: list = None):
        super().__init__()
        self.persistence_residual = persistence_residual

        if persistence_residual:
            if feat_mean is None or feat_std is None:
                raise ValueError(
                    "persistence_residual=True 이면 feat_mean/feat_std 가 필요합니다."
                )
            self.register_buffer("feat_mean", torch.tensor(feat_mean, dtype=torch.float32))
            self.register_buffer("feat_std",  torch.tensor(feat_std,  dtype=torch.float32))

        self.enc_z = NumericalEncoder(num_features, embed_dim)

        self.head_temp = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.head_precip = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        with torch.no_grad():
            self.head_temp[-1].bias.fill_(0.0 if persistence_residual else temp_mean)
            b = math.log(math.expm1(max(precip_mean, 1e-3)))
            self.head_precip[-1].bias.fill_(b)

    def forward(self, num_x: torch.Tensor,
               img_x: torch.Tensor = None, txt_x: torch.Tensor = None,
               collect_diagnostics: bool = False) -> torch.Tensor:
        """img_x/txt_x는 TriCHEFPipeline과 동일한 학습 루프를 재사용하기
        위한 자리만 차지하는 인자 — 이 모델은 Z축만 본다."""
        v_z = self.enc_z(num_x)   # 부호 보존 (F.normalize, abs 아님)

        delta_temp = self.head_temp(v_z)
        if self.persistence_residual:
            temp_now  = num_x[:, 0:1] * self.feat_std[0] + self.feat_mean[0]
            temp_pred = temp_now + delta_temp
        else:
            temp_pred = delta_temp

        precip_pred = F.softplus(self.head_precip(v_z))
        return torch.cat([temp_pred, precip_pred], dim=-1)
