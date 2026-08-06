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
import torchvision.models as tv_models


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

    def __init__(self, in_dim: int = 10, hidden: int = 16,
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
      Phase 3-6  —  PPO RL 위상 필터 θ<30°/80°
    """

    def __init__(self, embed_dim: int = 64,
                 num_features: int = 10,
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
                 compact_satellite: bool = True):
        """
        orthogonalize        : Gram-Schmidt 직교화 on/off (ablation 스위치)
        gs_eps               : 소프트 정규화 ε — gradient 상한 1/ε
        temp_mean            : 기온 타깃 평균 — head_temp 편향 초기화용
        precip_mean          : 강수 타깃 평균 — head_precip 편향 초기화용
        persistence_residual : 기온을 절대값이 아닌 변화량(Δ)으로 예측
        feat_mean / feat_std : 입력 표준화 통계 (Δ 예측 시 필수)
        dynamic_gate         : True 면 정적 α/φ 대신 ModalityGate 사용 (Phase 3-5)
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
        self.enc_re = SatelliteEncoder(4, embed_dim,
                                       compact=compact_satellite)   # A: 위성
        self.enc_im = TextEncoder(384,      embed_dim)   # B: 텍스트
        self.enc_z  = NumericalEncoder(num_features, embed_dim)   # C: 수치

        # 축 가중치 — 정적(log 공간 학습) 또는 동적(입력 조건부 게이트)
        if dynamic_gate:
            # 균등 초기화 — 논문의 Re 우선 순서를 물려받지 않는다 (위 주석 참조)
            self.gate = ModalityGate(in_dim=num_features, init_weights=None)
        else:
            self.log_alpha = nn.Parameter(torch.tensor(math.log(alpha_init)))
            self.log_phi   = nn.Parameter(torch.tensor(math.log(phi_init)))

        # 예측 헤드
        self.head_temp = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )
        self.head_precip = nn.Sequential(
            nn.Linear(embed_dim, 32), nn.GELU(), nn.Linear(32, 1),
        )

        # 출력 편향 초기화 — magnitude 원소값이 ~1/√D 로 작아서 편향만
        # 학습하는 데 수십 에폭이 낭비되는 문제를 제거한다.
        with torch.no_grad():
            # Δ 예측이면 목표 평균이 ≈0 이므로 편향도 0에서 시작한다.
            self.head_temp[-1].bias.fill_(0.0 if persistence_residual else temp_mean)
            # softplus(b) = precip_mean  →  b = log(exp(precip_mean) − 1)
            b = math.log(math.expm1(max(precip_mean, 1e-3)))
            self.head_precip[-1].bias.fill_(b)

        # 마지막 forward 의 축 진단 지표 (detach 된 float, 학습에 무관)
        self.last_diagnostics: dict = {}
        self._gate_w = None   # 동적 게이트의 마지막 출력 (엔트로피 정규화용)

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

        # 강수는 물리적으로 음수 불가 → softplus 로 강제.
        # clamp(x,0,·) 는 x<0 에서 gradient 가 정확히 0 이라 전 샘플이 음수로
        # 떨어지면 학습이 영구 정지한다(Phase 3-3 실측 확인). softplus 는
        # gradient = sigmoid(x) > 0 이므로 항상 회복 가능하다.
        precip_pred = F.softplus(self.head_precip(magnitude))

        return torch.cat([temp_pred, precip_pred], dim=-1)
