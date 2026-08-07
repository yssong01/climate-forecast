"""
test_phase36.py — Phase 3-6 검증 (PPO 경보 게이트)

RL 코드는 조용히 틀리기 쉽다. 보상 부호가 뒤집혀도, GAE 가 종료 상태를
잘못 부트스트랩해도 학습은 그럭저럭 돌아가는 것처럼 보이고 결과만 나빠진다.
아래는 그런 무증상 오류를 학습 전에 잡는 항목이다.

  T1. 위상각 퇴화   — theta 가 게이트 배분의 재표현일 뿐임을 수치로 확인
  T2. 축 진단 형태  — axis_report 출력 shape·유한성
  T3. 보상표        — (이벤트 × 행동) 6조합의 보상이 명세와 일치
  T4. 경보 피로     — 감쇠·누적 및 상시 경보 시 정상상태 수렴
  T5. MDP 성질      — 같은 관측·같은 행동인데 이력이 다르면 보상이 다르다
                      (이게 성립해야 PPO 를 쓸 근거가 생긴다)
  T6. GAE           — 손계산 대조 + 종료 상태 부트스트랩 0
  T7. 정책 출력     — shape·확률합·유한성
  T8. 에피소드 분할 — 관측소 경계를 넘지 않음
"""
import sys

import numpy as np
import torch

from pipeline_model import TriCHEFPipeline
import rl_phase_filter as rp

PASS, FAIL = "  [PASS]", "  [FAIL]"
results = []


def check(name: str, ok: bool, detail: str = ""):
    results.append(bool(ok))
    print(f"{PASS if ok else FAIL} {name}" + (f" — {detail}" if detail else ""))


print("=" * 70)
print(" Phase 3-6 검증 — PPO 경보 게이트")
print("=" * 70)

torch.manual_seed(0)
np.random.seed(0)


# ── T1/T2: 축 진단 ────────────────────────────────────────────────
print("\n[T1/T2] 축 진단 (axis_report)")

model = TriCHEFPipeline(
    embed_dim=64, num_features=12, temp_mean=28.0, precip_mean=0.5,
    persistence_residual=True, feat_mean=[28.0] + [0.0] * 11,
    feat_std=[1.0] * 12, dynamic_gate=True,
)
model.eval()

B = 32
num = torch.randn(B, 12)
img = torch.rand(B, 4, 32, 32)
txt = torch.randn(B, 384)
rep = model.axis_report(num, img, txt)

check("gate shape (B, 3)", tuple(rep["gate"].shape) == (B, 3),
      str(tuple(rep["gate"].shape)))
check("gate 합 = 1 (softmax)",
      torch.allclose(rep["gate"].sum(-1), torch.ones(B), atol=1e-5))
for k in ("theta", "cos_re_im", "cos_re_z", "cos_im_z"):
    check(f"{k} shape (B,) · 유한",
          tuple(rep[k].shape) == (B,) and torch.isfinite(rep[k]).all().item())

# theta = atan2(w_im·‖v_im‖, w_re·‖v_re‖) 인데 인코더가 모두 F.normalize
# 출력이라 ‖v‖ = 1 → theta 는 atan2(w_im, w_re) 로 퇴화한다.
# 즉 논문의 '위상각'은 이 구현에서 축 자체의 per-sample 정보를 담지 못하고
# 게이트 배분을 다르게 쓴 값일 뿐이다. RL 상태에 넣되 그 사실을 명시한다.
w = rep["gate"]
theta_from_gate = torch.atan2(w[:, 1], w[:, 0])
gap = (rep["theta"] - theta_from_gate).abs().max().item()
check("위상각 θ ≡ atan2(w_im, w_re) — 게이트의 재표현으로 퇴화",
      gap < 1e-5, f"최대 차이 {gap:.2e}")

# 반면 축 간 코사인은 게이트와 무관한 per-sample 값이라 분산이 살아 있다.
cos_std = rep["cos_re_im"].std().item()
check("cos(Re,Im) 은 표본마다 변동 — 실제 축 정보 보존",
      cos_std > 1e-4, f"std = {cos_std:.4f}")


# ── T3: 보상표 ────────────────────────────────────────────────────
print("\n[T3] 보상표 (이벤트 × 행동)")

feats = np.zeros((1, 4, rp.N_STATIC), dtype=np.float32)
events = np.array([[1, 0, 1, 0]], dtype=np.float32)
env = rp.AlarmEnv(feats, events)
env.reset()

_, r0, _ = env.step(np.array([0]))          # 이벤트 + 정상 → 미탐지
check("이벤트·정상 = 미탐지", r0[0] == rp.R_MISS, f"{r0[0]:+.1f}")

_, r1, _ = env.step(np.array([2]))          # 비이벤트 + 경보, 피로 0
check("비이벤트·경보 = 오경보×(1+0)", r1[0] == rp.R_ALARM_FA, f"{r1[0]:+.1f}")

_, r2, _ = env.step(np.array([2]))          # 이벤트 + 경보 → 정탐
check("이벤트·경보 = 정탐", r2[0] == rp.R_HIT, f"{r2[0]:+.1f}")

# 직전 두 스텝의 경보로 피로 = 0.9·1.0 + 1.0 = 1.9
_, r3, _ = env.step(np.array([2]))
check("비이벤트·경보 = 오경보×(1+피로) — 피로 반영",
      abs(r3[0] - rp.R_ALARM_FA * (1 + 1.9)) < 1e-5,
      f"{r3[0]:+.3f} (기대 {rp.R_ALARM_FA*2.9:+.3f})")

env2 = rp.AlarmEnv(feats, events)
env2.reset()
_, ra, _ = env2.step(np.array([1]))
check("이벤트·주의 = 부분 인정", ra[0] == rp.R_WATCH_HIT, f"{ra[0]:+.1f}")
_, rb, _ = env2.step(np.array([1]))
check("비이벤트·주의 = 값싼 오경보",
      abs(rb[0] - rp.R_WATCH_FA * (1 + rp.FATIGUE_WATCH)) < 1e-5,
      f"{rb[0]:+.3f}")


# ── T4: 경보 피로 ─────────────────────────────────────────────────
print("\n[T4] 경보 피로 감쇠·누적")

T = 200
env = rp.AlarmEnv(np.zeros((1, T, rp.N_STATIC), np.float32),
                  np.zeros((1, T), np.float32))
env.reset()
for _ in range(T - 1):
    env.step(np.array([2]))                 # 상시 경보
steady = rp.FATIGUE_ALARM / (1 - rp.FATIGUE_DECAY)
check(f"상시 경보 정상상태 ≈ {steady:.0f}",
      abs(env.fatigue[0] - steady) < 0.05, f"{env.fatigue[0]:.4f}")

env.reset()
env.step(np.array([2]))
f1 = env.fatigue[0]
for _ in range(30):
    env.step(np.array([0]))                 # 이후 무경보 → 감쇠
check("무경보 시 피로 감쇠 → 0",
      env.fatigue[0] < 0.05 * f1, f"{f1:.3f} → {env.fatigue[0]:.5f}")

# 상시 경보가 자명한 해가 되지 않는지 — 피로가 없으면 무조건 경보가 이긴다
alarm_cost = rp.R_ALARM_FA * (1 + steady)
check("상시 경보는 피로로 인해 비용이 폭증 (자명해 배제)",
      abs(alarm_cost) > abs(rp.R_MISS) * 0.5,
      f"오경보 비용 {alarm_cost:.1f} vs 미탐지 {rp.R_MISS:.1f}")


# ── T5: MDP 성질 ──────────────────────────────────────────────────
print("\n[T5] MDP 성질 — 보상이 과거 행동에 의존하는가")

# 두 에피소드는 관측·이벤트가 완전히 동일하다. 첫 스텝의 행동만 다르게 준 뒤,
# 두 번째 스텝에서 같은 행동을 취했을 때 보상이 갈리는지 본다.
# 갈리지 않으면 이 문제는 밴딧이고, PPO 를 쓸 근거 자체가 사라진다.
feats = np.zeros((2, 3, rp.N_STATIC), np.float32)
events = np.zeros((2, 3), np.float32)
env = rp.AlarmEnv(feats, events)
env.reset()
env.step(np.array([0, 2]))                  # 이력만 다르게
_, r, _ = env.step(np.array([2, 2]))        # 동일 행동
check("동일 관측·동일 행동인데 이력이 다르면 보상이 다르다 → MDP",
      abs(r[0] - r[1]) > 1e-6,
      f"이력 없음 {r[0]:+.2f} vs 직전 경보 {r[1]:+.2f}")
check("관측에 피로가 실제로 노출됨 (상태의 일부)",
      abs(env._obs()[0, -1] - env._obs()[1, -1]) > 1e-6,
      f"{env._obs()[0,-1]:.3f} vs {env._obs()[1,-1]:.3f}")


# ── T6: GAE ───────────────────────────────────────────────────────
print("\n[T6] GAE(λ) 이점 추정")

rew = np.array([[1.0], [2.0], [3.0]], dtype=np.float32)
val = np.array([[0.5], [0.5], [0.5]], dtype=np.float32)
adv = rp.compute_gae(rew, val, gamma=1.0, lam=1.0)
# 손계산: t=2 delta=3+0-0.5=2.5 / t=1 delta=2+0.5-0.5=2 → 4.5 / t=0 → 5.5
expect = np.array([[5.5], [4.5], [2.5]], dtype=np.float32)
check("γ=λ=1 손계산 일치", np.allclose(adv, expect, atol=1e-5),
      f"{adv.ravel()} vs {expect.ravel()}")
check("종료 상태 부트스트랩 = 0 (val[T-1] 재사용 아님)",
      abs(adv[-1, 0] - 2.5) < 1e-5,
      f"마지막 이점 {adv[-1,0]:.3f} (재사용 시 3.000)")

adv2 = rp.compute_gae(np.zeros((5, 3), np.float32), np.zeros((5, 3), np.float32))
check("보상·가치 0 → 이점 0", np.abs(adv2).max() < 1e-6)


# ── T7: 정책 출력 ─────────────────────────────────────────────────
print("\n[T7] ActorCritic 출력 · 상태 집합")

# minimal 은 관측소 지문(게이트·위상각·기압·습도)을 반드시 배제해야 한다 —
# 게이트 입력에 위도·경도가 있어 w_* 와 theta 가 사실상 관측소 식별자다.
FINGERPRINT = {"theta", "w_re", "w_im", "w_z",
               "cos_re_im", "cos_re_z", "cos_im_z", "humidity", "pressure"}
check("minimal 상태에 관측소 지문 특성이 없음",
      not (set(rp.STATE_SETS["minimal"]) & FINGERPRINT),
      f"{sorted(set(rp.STATE_SETS['minimal']) & FINGERPRINT)}")
check("minimal 에 피로가 포함됨 (MDP 항 유지)",
      "fatigue" in rp.STATE_SETS["minimal"])
check("full 은 전체 상태", len(rp.state_cols("full")) == rp.N_STATE)

cols = rp.state_cols("minimal")
mu = np.zeros(len(cols), np.float32)
sd = np.ones(len(cols), np.float32)
net = rp.ActorCritic(mu, sd, cols)
obs = torch.randn(16, rp.N_STATE)
logits, value = net(obs)
check("logits shape (16, 3)", tuple(logits.shape) == (16, rp.N_ACTION))
check("value shape (16,)", tuple(value.shape) == (16,))
probs = torch.softmax(logits, dim=-1)
check("행동 확률 합 = 1", torch.allclose(probs.sum(-1), torch.ones(16), atol=1e-5))
check("출력 유한",
      torch.isfinite(logits).all().item() and torch.isfinite(value).all().item())
# 환경은 항상 15차원을 내보내고 정책만 자기 열로 좁혀 본다 — 지문 열을 바꿔도
# minimal 정책의 출력이 흔들리지 않아야 배제가 실제로 걸린 것이다.
obs2 = obs.clone()
obs2[:, rp.FEAT_NAMES.index("w_re")] += 5.0
obs2[:, rp.FEAT_NAMES.index("pressure")] -= 3.0
check("minimal 정책은 배제된 열 변화에 반응하지 않음",
      torch.allclose(net(obs)[0], net(obs2)[0], atol=1e-6))

logits.sum().backward()
bad = [n for n, p in net.named_parameters()
       if p.grad is not None and not torch.isfinite(p.grad).all()]
check("gradient 유한", len(bad) == 0, str(bad[:3]) if bad else "")


# ── T8: 에피소드 분할 ─────────────────────────────────────────────
print("\n[T8] 에피소드 분할 무결성")

L = rp.EPISODE_LEN
# 관측소 식별자를 특성값에 심어 에피소드가 섞이는지 본다
series = {
    "A": {"feat": np.ones((L * 2 + 40, rp.N_STATIC), np.float32),
          "event": np.zeros(L * 2 + 40, np.float32)},
    "B": {"feat": np.full((L * 2 + 40, rp.N_STATIC), 2.0, np.float32),
          "event": np.zeros(L * 2 + 40, np.float32)},
}
f, e = rp.to_episodes(series, ["A", "B"])
check("에피소드 수 = 관측소당 floor(N/L)", f.shape[0] == 4, f"{f.shape[0]}개")
check(f"에피소드 길이 = {L}", f.shape[1] == L, str(f.shape))
homogeneous = all(len(np.unique(f[i])) == 1 for i in range(f.shape[0]))
check("각 에피소드가 단일 관측소에서만 구성됨 (경계 오염 없음)", homogeneous)
check("나머지 구간은 버려짐 (길이 불균일 방지)",
      f.shape[0] * L <= sum(len(v["feat"]) for v in series.values()))

sub = rp.to_episodes(series, ["A"])[0]
check("검증 관측소만 선택 시 학습 관측소가 섞이지 않음",
      np.allclose(sub, 1.0), f"평균 {sub.mean():.2f} (A=1.0)")


# ── 요약 ──────────────────────────────────────────────────────────
print("\n" + "=" * 70)
n_pass, n_all = sum(results), len(results)
print(f" 결과: {n_pass}/{n_all} 통과"
      + ("  — Phase 3-6 학습 진행 가능" if n_pass == n_all else "  — 실패 항목 수정 필요"))
print("=" * 70)

sys.exit(0 if n_pass == n_all else 1)
