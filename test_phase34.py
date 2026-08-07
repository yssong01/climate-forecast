"""
test_phase34.py — Phase 3-4 수치 검증

Gram-Schmidt 직교화는 잔차 노름이 0에 가까워질 때 NaN·발산이 발생하기 쉽다.
긴 학습을 돌리기 전에 아래를 확인한다.

  T1. 직교성       — 직교화 후 모든 축 쌍이 |cos| ≈ 0
  T2. 단위 노름    — 직교화 후 각 축 노름 ≈ 1
  T3. 평행 입력    — 거의 평행한 두 축에서 NaN 없이 유한 gradient
  T4. 영벡터 입력  — 영벡터에서 NaN 없음
  T5. gradient 흐름 — 전체 파이프라인 역전파에서 NaN/Inf 없음
  T6. softplus 수정 — clamp 는 gradient 0, softplus 는 항상 > 0
"""
import sys

import torch
import torch.nn.functional as F

from pipeline_model import (soft_normalize, gram_schmidt_3axis,
                            TriCHEFPipeline)

PASS, FAIL = "  [PASS]", "  [FAIL]"
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append(ok)
    print(f"{PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


print("=" * 66)
print(" Phase 3-4 수치 검증")
print("=" * 66)

torch.manual_seed(0)
B, D = 16, 64


def abs_cos(a, b):
    return F.cosine_similarity(a, b, dim=-1).abs().max().item()


# ── T1/T2: 무작위 단위 벡터 직교화 ────────────────────────────────
print("\n[T1/T2] 무작위 단위 벡터 직교화")
v_re = F.normalize(torch.randn(B, D), dim=-1)
v_im = F.normalize(torch.randn(B, D), dim=-1)
v_z  = F.normalize(torch.randn(B, D), dim=-1)

u_re, u_im, u_z = gram_schmidt_3axis(v_re, v_im, v_z)

c1, c2, c3 = abs_cos(u_re, u_im), abs_cos(u_re, u_z), abs_cos(u_im, u_z)
check("Re⊥Im", c1 < 1e-5, f"max|cos| = {c1:.2e}")
check("Re⊥Z ", c2 < 1e-5, f"max|cos| = {c2:.2e}")
check("Im⊥Z ", c3 < 1e-5, f"max|cos| = {c3:.2e}")

for nm, u in [("Re", u_re), ("Im", u_im), ("Z ", u_z)]:
    n = u.norm(dim=-1)
    check(f"{nm} 노름 ≈ 1", torch.allclose(n, torch.ones_like(n), atol=1e-4),
          f"평균 {n.mean().item():.6f}")

# ── T3: 거의 평행한 입력 (최악 조건) ──────────────────────────────
print("\n[T3] 거의 평행한 축 (잔차 노름 → 0, 최악 조건)")
a = F.normalize(torch.randn(B, D), dim=-1)
p_re = a.clone().requires_grad_(True)
p_im = (a + 1e-9 * torch.randn(B, D)).requires_grad_(True)   # 사실상 동일
p_z  = (a + 1e-9 * torch.randn(B, D)).requires_grad_(True)

o_re, o_im, o_z = gram_schmidt_3axis(p_re, p_im, p_z)
out = (o_re.sum() + o_im.sum() + o_z.sum())
out.backward()

finite_fwd = all(torch.isfinite(t).all().item() for t in (o_re, o_im, o_z))
finite_bwd = all(torch.isfinite(g.grad).all().item() for g in (p_re, p_im, p_z))
check("순전파 유한 (NaN/Inf 없음)", finite_fwd)
check("역전파 유한 (NaN/Inf 없음)", finite_bwd,
      f"max|grad| = {max(g.grad.abs().max().item() for g in (p_re,p_im,p_z)):.2e}")
check("퇴화 축 노름 → 0 (기여 소멸, 폭주 아님)",
      o_im.norm(dim=-1).mean().item() < 0.1,
      f"‖u_im‖ = {o_im.norm(dim=-1).mean().item():.2e}")

# ── T4: 영벡터 입력 ───────────────────────────────────────────────
print("\n[T4] 영벡터 입력")
z_re = torch.zeros(B, D, requires_grad=True)
z_im = torch.zeros(B, D, requires_grad=True)
z_z  = F.normalize(torch.randn(B, D), dim=-1).requires_grad_(True)
q_re, q_im, q_z = gram_schmidt_3axis(z_re, z_im, z_z)
(q_re.sum() + q_im.sum() + q_z.sum()).backward()
check("영벡터 순전파 유한",
      all(torch.isfinite(t).all().item() for t in (q_re, q_im, q_z)))
check("영벡터 역전파 유한",
      all(torch.isfinite(g.grad).all().item() for g in (z_re, z_im, z_z)))
check("soft_normalize(0) = 0",
      soft_normalize(torch.zeros(1, D)).abs().max().item() == 0.0)

# ── T5: 전체 파이프라인 gradient ──────────────────────────────────
print("\n[T5] 전체 파이프라인 역전파")
model = TriCHEFPipeline(embed_dim=D, num_features=12,
                        temp_mean=28.0, precip_mean=0.5,
                        orthogonalize=True, persistence_residual=True,
                        feat_mean=[28.0] + [0.0] * 11,
                        feat_std=[1.0] * 12)
num = torch.randn(4, 12)
img = torch.rand(4, 4, 32, 32)
txt = torch.randn(4, 384)
pred = model(num, img, txt, collect_diagnostics=True)
pred.sum().backward()

bad = [n for n, p in model.named_parameters()
       if p.grad is not None and not torch.isfinite(p.grad).all()]
check("전 파라미터 gradient 유한", len(bad) == 0,
      f"이상 파라미터: {bad[:3]}" if bad else "")
no_grad = [n for n, p in model.named_parameters() if p.grad is None]
check("모든 파라미터가 gradient 수신", len(no_grad) == 0,
      f"미수신: {no_grad[:3]}" if no_grad else "")
check("출력 형태 (4, 2)", tuple(pred.shape) == (4, 2), str(tuple(pred.shape)))
check("강수 예측 ≥ 0 (softplus)", (pred[:, 1] >= 0).all().item(),
      f"최소 {pred[:,1].min().item():.4f}")
check("퍼시스턴스 잔차 경로 동작 (기온 ≈ 현재기온 + Δ)",
      abs(pred[:, 0].mean().item() - (28.0 + num[:, 0].mean().item())) < 5.0,
      f"예측 {pred[:,0].mean().item():.2f} °C")

# ── T6: clamp vs softplus gradient ────────────────────────────────
print("\n[T6] dying-clamp 버그 재현 및 수정 확인")
x1 = torch.tensor([-0.10], requires_grad=True)     # 진단에서 관측된 실제 값
torch.clamp(x1, 0.0, 200.0).backward()
x2 = torch.tensor([-0.10], requires_grad=True)
F.softplus(x2).backward()
check("clamp: gradient = 0 (학습 정지 — 버그 재현)",
      x1.grad.item() == 0.0, f"grad = {x1.grad.item():.6f}")
check("softplus: gradient > 0 (회복 가능 — 수정 확인)",
      x2.grad.item() > 0.0, f"grad = {x2.grad.item():.6f}")

# ── 요약 ──────────────────────────────────────────────────────────
print("\n" + "=" * 66)
n_pass, n_all = sum(results), len(results)
print(f" 결과: {n_pass}/{n_all} 통과"
      + ("  — 학습 진행 가능" if n_pass == n_all else "  — 실패 항목 수정 필요"))
print("=" * 66)

# 실패를 종료 코드로 알린다. 이게 없으면 항목이 깨져도 프로세스가 0을 반환해
# CI(.github/workflows/ci.yml)가 초록불을 띄운다 — 테스트가 있으나 마나가 된다.
sys.exit(0 if n_pass == n_all else 1)
