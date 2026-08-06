"""
ablation.py — 축 기여도 실측

질문: +6시간 기온 예보의 +48.8% 개선은 정말 멀티모달(3축) 덕분인가,
      아니면 Z축 수치 + 시각 특성만으로 이미 달성되는가?

현재 Re축(위성)과 Im축(텍스트)은 Z축의 ASOS 수치에서 결정론적으로 생성된
시뮬레이션이다. 즉 정보 이론적으로 새 정보가 없으므로 기여도가 0에 가까울
것으로 예상되지만, 추측 대신 측정한다.

Z축은 퍼시스턴스 기준선 계산에 필요하므로 항상 활성이며,
Re/Im 축 입력을 영벡터로 차단해 대조군을 만든다.

동일 시드·동일 분할·동일 하이퍼파라미터·**동일 에폭수**로 4개 조건을 학습한다.

조기 종료는 끈다. 1차 시도에서 조건별 종료 에폭이 38~148로 4배까지 벌어져
'성능 차이'와 '학습량 차이'가 분리되지 않았다(합산 score 기준 조기 종료가
기온이 아직 개선 중인 조건을 먼저 중단시킴). 비교 지표도 합산 score 시점의
값이 아니라 **지표별 독립 최저치**를 쓴다.

── 앞선 ablation 결과 (Gram-Schmidt 직교화, +1시간 시계) ──────────
    ON  : 기온 0.8886 / 강수 0.3267
    OFF : 기온 0.8452 / 강수 0.2889   ← 우세
  OFF 조건에서 학습된 Re-Im 상관도가 0.4809(ON은 0.0846)로, 축 간 공유
  정보가 회귀 성능에 기여했다. 검색 태스크(논문 원본)와 달리 예보 태스크
  에서는 직교성 강제가 손해 → ORTHOGONALIZE 기본값 False 로 확정.
"""
from train import train, LEAD_HOURS

CONDITIONS = [
    ("Z 단독        ", False, False, "./checkpoints/abl_z.pt"),
    ("Z + Re(위성)  ", True,  False, "./checkpoints/abl_z_re.pt"),
    ("Z + Im(텍스트)", False, True,  "./checkpoints/abl_z_im.pt"),
    ("Z + Re + Im   ", True,  True,  "./checkpoints/abl_all.pt"),
]

print("\n" + "#" * 72)
print(f"#  Ablation: 축 기여도 측정  (예보 시계 +{LEAD_HOURS}시간)")
print("#" * 72)

runs = {}
for label, use_re, use_im, ckpt in CONDITIONS:
    print(f"\n\n{'#'*72}\n#  {label.strip()} 학습 시작\n{'#'*72}")
    runs[label] = train(use_re=use_re, use_im=use_im,
                        early_stop=False,          # 전 조건 동일 에폭수
                        checkpoint=ckpt, verbose=True)

# ── 비교표 ────────────────────────────────────────────────────────
base = runs["Z 단독        "]

print("\n\n" + "=" * 72)
print(f" 축 기여도 Ablation 결과  (+{LEAD_HOURS}시간 예보)")
print("=" * 72)
print(f" 기준선: 기온 퍼시스턴스 {base['temp_naive']:.4f} °C | "
      f"강수 상시0예측 {base['precip_naive']:.4f} mm")
print("-" * 72)
print(f"{'조건':<16} | {'기온MAE':>9} {'개선':>8} | {'강수MAE':>9} {'개선':>8} | "
      f"{'Z대비':>8} | {'파라미터':>11}")
print("-" * 72)

for label, *_ in CONDITIONS:
    r = runs[label]
    d_temp = base["best_temp_mae"] - r["best_temp_mae"]   # 양수면 Z 단독보다 우수
    print(f"{label:<16} | {r['best_temp_mae']:9.4f} {r['temp_gain_pct']:+7.1f}% | "
          f"{r['best_precip_mae']:9.4f} {r['precip_gain_pct']:+7.1f}% | "
          f"{d_temp:+8.4f} | {r['n_params']:>11,}")

print("-" * 72)
print(f" 전 조건 동일 {base['epochs_run']} 에폭 학습 | 학습 표본 572개")

# ── 판정 ──────────────────────────────────────────────────────────
full = runs["Z + Re + Im   "]
gain = base["best_temp_mae"] - full["best_temp_mae"]
rel  = gain / base["best_temp_mae"] * 100

# 강수 기여도도 함께 본다 — 타깃에 따라 결론이 갈리기 때문이다.
p_gain = base["best_precip_mae"] - full["best_precip_mae"]
p_rel  = p_gain / base["best_precip_mae"] * 100

print(f"\n 멀티모달 순기여 (3축 − Z단독)")
print(f"   기온: {gain:+.4f} °C ({rel:+.1f}%)   강수: {p_gain:+.4f} mm ({p_rel:+.1f}%)")
print()
if rel < -1.0 and p_rel > 3.0:
    print(" → 타깃에 따라 결론이 갈린다. 기온은 Z축 단독이 낫고, 강수는")
    print("   멀티모달이 기여한다. 시뮬레이션 위성(강수 시 밝기 상승)과")
    print("   텍스트('강수 상황' 서술)가 강수 신호를 중복 인코딩하기 때문으로,")
    print("   정보 추가가 아니라 앙상블 강화 효과로 보는 것이 타당하다.")
elif abs(rel) < 3.0 and abs(p_rel) < 3.0:
    print(" → 양 타깃 모두 오차 범위. Re/Im 은 Z축의 재인코딩일 뿐이며 새로운")
    print("   정보가 없다는 가설과 일치한다. 실제 가치 입증에는 Sentinel-2")
    print("   실위성 또는 기상청 예보문 실데이터 연결이 필요하다.")
elif rel > 0 and p_rel > 0:
    print(" → 양 타깃 모두 멀티모달이 기여. 시뮬레이션 축이라도 앙상블 효과로")
    print("   성능이 개선되는 것으로 해석된다.")
else:
    print(" → 멀티모달이 순손해. 정보 이득 없이 파라미터만 늘어 과적합으로")
    print("   작용한 것으로 보인다. 인코더 용량 축소를 검토할 것.")
print("=" * 72)
