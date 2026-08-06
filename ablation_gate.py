"""
ablation_gate.py — Phase 3-5 동적 게이팅 검증

배경 (Phase 3-4 실측, +6시간 예보, 150에폭 고정):
    Z 단독      기온 MAE 0.7263 °C   ← 최고
    Z+Re+Im     기온 MAE 1.0513 °C   (-44.7%)
  정보가 없는 시뮬레이션 축이 성능을 떨어뜨렸고, 정적 α/φ 는 이를 억제하기는
  커녕 가중치를 키웠다 (α 0.4→0.665, φ 0.2→0.462).

검증 가설:
  동적 게이트 + 엔트로피 정규화가 무용 축(Re/Im)의 배분을 낮춰,
  3축 조건의 성능을 Z 단독(0.7263) 수준으로 회복시킨다.

  게이트가 목표를 달성했다면 w_z 가 1에 가깝고 w_re/w_im 이 0에 가까워야
  한다. 어느 축을 살릴지는 코드에 심지 않았다 — 엔트로피 항은 '집중하라'는
  대칭적 압력만 주고, 어느 축이 선택되는지는 데이터가 결정한다.

λ(엔트로피 계수)를 훑어 억제 압력과 성능의 관계를 본다.
"""
from train import train, LEAD_HOURS

# (라벨, 3축여부, 동적게이트, λ, 체크포인트)
CONDITIONS = [
    ("Z 단독 (기준선) ", False, False, 0.00, "./checkpoints/gate_zonly.pt"),
    ("3축 정적 α/φ    ", True,  False, 0.00, "./checkpoints/gate_static.pt"),
    ("3축 게이트 λ=0  ", True,  True,  0.00, "./checkpoints/gate_l000.pt"),
    ("3축 게이트 λ=.05", True,  True,  0.05, "./checkpoints/gate_l005.pt"),
]

print("\n" + "#" * 74)
print(f"#  Phase 3-5 동적 게이팅 검증  (3축 전부 활성, +{LEAD_HOURS}시간 예보)")
print("#" * 74)

runs = {}
for label, tri, gate, lam, ckpt in CONDITIONS:
    print(f"\n\n{'#'*74}\n#  {label.strip()} 학습 시작\n{'#'*74}")
    runs[label] = train(use_re=tri, use_im=tri,
                        dynamic_gate=gate, gate_entropy_weight=lam,
                        early_stop=False,               # 전 조건 동일 에폭수
                        checkpoint=ckpt, verbose=True)

Z_ONLY_REFERENCE = runs["Z 단독 (기준선) "]["best_temp_mae"]

# ── 비교표 ────────────────────────────────────────────────────────
print("\n\n" + "=" * 74)
print(f" Phase 3-5 결과  (+{LEAD_HOURS}시간 예보, 3축 전부 활성)")
print("=" * 74)
print(f" Z 단독 기준선(동일 조건 재측정): 기온 MAE {Z_ONLY_REFERENCE:.4f} °C")
print("-" * 74)
print(f"{'조건':<18} | {'기온MAE':>8} | {'Z단독대비':>9} | {'w_re':>6} {'w_im':>6} {'w_z':>6} | {'엔트로피':>7}")
print("-" * 74)

for label, *_ in CONDITIONS:
    r = runs[label]
    d = r.get("diagnostics", {})
    gap = r["best_temp_mae"] - Z_ONLY_REFERENCE     # 음수면 Z단독보다 우수
    if "w_re" in d:
        wcols = f"{d['w_re']:6.3f} {d['w_im']:6.3f} {d['w_z']:6.3f}"
        ent   = f"{d.get('gate_entropy', 0):7.4f}"
    else:
        wcols = f"{'—':>6} {'—':>6} {'—':>6}"
        ent   = f"{'—':>7}"
    print(f"{label:<18} | {r['best_temp_mae']:8.4f} | {gap:+9.4f} | {wcols} | {ent}")

print("-" * 74)

# ── 판정 ──────────────────────────────────────────────────────────
tri_conditions = [c for c in CONDITIONS if c[1]]     # 3축 조건만 비교
best_label = min(tri_conditions, key=lambda c: runs[c[0]]["best_temp_mae"])[0]
best = runs[best_label]
gap  = best["best_temp_mae"] - Z_ONLY_REFERENCE
d    = best.get("diagnostics", {})

print(f"\n 최우수 조건: {best_label.strip()}  (기온 MAE {best['best_temp_mae']:.4f} °C)")
if gap <= 0.03:
    print(" → 게이팅이 무용 축의 피해를 제거해 Z 단독 수준을 회복했다.")
    print("   Eq.1 구조가 '정보 없는 모달리티에 강건'해졌음을 뜻한다.")
    if d.get("w_z", 0) > 0.6:
        print(f"   게이트가 Z축에 {d['w_z']:.1%} 를 배분 — 정답을 코드에 심지")
        print("   않았음에도 데이터가 유용한 축을 스스로 찾아냈다.")
else:
    print(f" → Z 단독 대비 {gap:+.4f} °C. 게이팅만으로는 완전 회복에 미달.")
    print("   무용 축이 magnitude 에 섞이는 구조적 손실이 남아 있으며,")
    print("   축을 완전히 끄는(hard gating) 설계나 실데이터 연결이 필요하다.")
print("=" * 74)
