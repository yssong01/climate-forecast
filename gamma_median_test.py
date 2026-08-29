"""
gamma_median_test.py — Gamma NLL 재시도의 첫 단계: 허들(hurdle) 혼합분포의
중앙값(median) 점추정 공식이 맞는지 검증한다.

**배경.** 2026-08-18 Gamma NLL 실험이 −147.5%로 크게 실패한 원인은 Gamma
모델링 자체가 아니라 최종 점추정 방식이었다(README 참고) — 곱셈
`rain_prob × μ`(조건부 평균)는 절대오차(MAE)를 최소화하는 점추정이 아니다.
0에 확률질량이 몰린 혼합분포에서 MAE 최적 점추정은 **중앙값**이다.

**공식.** Y = 0 (확률 1-p) 또는 Y ~ Gamma(α, rate) (확률 p, x>0)인 허들
혼합분포에서, CDF는 F(x) = (1-p) + p·Gamma_CDF(x) (x≥0). 중앙값 m은
F(m)=0.5를 만족하는 가장 작은 m이다.
  - p ≤ 0.5 이면 F(0) = 1-p ≥ 0.5 이므로 m = 0.
  - p > 0.5 이면 Gamma_CDF(m) = (0.5-(1-p))/p = 1 - 0.5/p 를 풀어야 하고,
    m = Gamma_분위수함수(1 - 0.5/p; α, rate).

**검증 방법.** 닫힌 형태 공식과 몬테카를로(대량 샘플링 후 실측 중앙값)를
대조한다 — 독립적인 두 방법이 일치해야 공식을 신뢰할 수 있다. torch.special
에는 Gamma 분위수함수가 없어 정규화 불완전감마함수의 역함수
(gammaincinv)로 직접 구현한다: Gamma_CDF(x;α,rate) = P(α, rate·x) 이므로
Gamma_분위수함수(q;α,rate) = gammaincinv(α,q) / rate.

실행: python gamma_median_test.py
"""
import numpy as np
import torch
from scipy.special import gammaincinv

# torch.special 에는 gammaincinv(정규화 불완전감마함수의 역함수)가 없다
# (torch 2.6 실측 — gammainc/gammaincc는 있지만 역함수는 없음). 이 함수는
# 학습 손실(NLL)의 backward 경로에 들어가지 않는 순수 후처리(추론 시점 점
# 추정)라 미분 가능성이 필요 없다 — predict.py의 calibrate_prob()(등온
# 회귀, sklearn 기반)와 같은 성격이다. scipy.special.gammaincinv 로 CPU에서
# 계산한다(distribution_diagnostics.py 가 이미 이 프로젝트에서 scipy를
# 쓰는 선례).


def gamma_quantile(alpha: torch.Tensor, rate: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Gamma(alpha, rate) 분포의 q-분위수. q는 (0,1) 구간."""
    x = gammaincinv(alpha.detach().cpu().numpy(), q.detach().cpu().numpy())
    return torch.as_tensor(x, dtype=torch.float32) / rate.detach().cpu()


def hurdle_median(p: torch.Tensor, alpha: torch.Tensor, rate: torch.Tensor) -> torch.Tensor:
    """허들 혼합분포(Y=0 확률 1-p, Y~Gamma(alpha,rate) 확률 p)의 중앙값."""
    q = torch.clamp(1.0 - 0.5 / p.clamp(min=1e-6), min=1e-6, max=1.0 - 1e-6)
    cont_median = gamma_quantile(alpha, rate, q)
    return torch.where(p <= 0.5, torch.zeros_like(p), cont_median)


def monte_carlo_median(p: float, alpha: float, rate: float, n: int = 2_000_000, seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    is_wet = torch.rand(n, generator=g) < p
    gamma_dist = torch.distributions.Gamma(concentration=alpha, rate=rate)
    samples = torch.where(
        is_wet,
        gamma_dist.sample((n,)),
        torch.zeros(n),
    )
    return samples.median().item()


def main():
    # (p, alpha, rate) 조합 — WET_THRESH 근방 강수량 분포에서 실측된 값대(
    # train.py 로그의 precip_gamma_alpha_init≈0.30, 습윤 평균≈2.4mm 근방)와
    # p<0.5·p>0.5 경계 양쪽을 모두 포함하도록 고른다.
    cases = [
        (0.10, 0.30, 0.30 / 2.4),   # 전형적 무강수 우세 구간 — median 0 이어야 함
        (0.50, 0.30, 0.30 / 2.4),   # 경계
        (0.62, 1.50, 1.50 / 2.4),   # +6h rain_prob 게이트(0.85) 근방은 아니지만 p>0.5 구간
        (0.85, 2.00, 2.00 / 2.4),
        (0.95, 0.80, 0.80 / 2.4),
    ]
    print(f"{'p':>6}{'alpha':>8}{'rate':>8}{'closed-form':>14}{'monte-carlo':>14}{'|diff|':>10}")
    max_rel_err = 0.0
    for p, alpha, rate in cases:
        p_t = torch.tensor(p)
        alpha_t = torch.tensor(alpha)
        rate_t = torch.tensor(rate)
        m_closed = hurdle_median(p_t, alpha_t, rate_t).item()
        m_mc = monte_carlo_median(p, alpha, rate)
        diff = abs(m_closed - m_mc)
        denom = max(abs(m_mc), 1e-6)
        max_rel_err = max(max_rel_err, diff / denom if m_mc > 1e-6 else diff)
        print(f"{p:6.2f}{alpha:8.2f}{rate:8.3f}{m_closed:14.5f}{m_mc:14.5f}{diff:10.5f}")

    print(f"\n최대 상대오차(0에 가까운 경우 절대오차): {max_rel_err:.4f}")
    if max_rel_err < 0.02:
        print("판정: 닫힌 형태 공식이 몬테카를로와 일치 — 중앙값 공식 채택 가능.")
    else:
        print("판정: 불일치 — 공식 또는 gammaincinv 사용법을 재점검할 것.")

    # 경계 조건 벡터화 확인 — 배치 텐서로도 정상 동작하는지.
    p_batch = torch.tensor([0.1, 0.5, 0.62, 0.85, 0.95])
    alpha_batch = torch.tensor([0.30, 0.30, 1.50, 2.00, 0.80])
    rate_batch = alpha_batch / 2.4
    m_batch = hurdle_median(p_batch, alpha_batch, rate_batch)
    print(f"\n배치 계산: {m_batch.tolist()}")


if __name__ == "__main__":
    main()
