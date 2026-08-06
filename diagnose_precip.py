"""
diagnose_precip.py — Phase 3-3 강수 MAE 고정 현상 진단

가설: torch.clamp(x, 0, 200)이 x<0 구간에서 gradient=0 이므로,
      head_precip 출력이 전 샘플 음수가 되면 학습이 영구 정지한다.

검증 항목
  1. head_precip pre-clamp 출력 분포 (음수 비율)
  2. 3축 pairwise 코사인 유사도 (Gram-Schmidt 필요성 측정)
  3. 예측이 항상 0일 때의 MAE (관측된 0.7293과 대조)
"""
import torch
import numpy as np

from train import (WeatherDataset, collect_historical,
                   DEVICE, ALPHA_INIT, PHI_INIT)
from satellite_collector import SimulatedSatelliteCollector
from text_collector import SimulatedTextCollector
from pipeline_model import TriCHEFPipeline

CHECKPOINT = "./checkpoints/numerical_trichef.pt"

print("=" * 62)
print(" Phase 3-3 강수 예측 고정 현상 진단")
print("=" * 62)

# 데이터 재구성 (캐시 사용 → API 호출 없음)
records = collect_historical()
ds = WeatherDataset(records,
                    sat_collector=SimulatedSatelliteCollector(),
                    txt_collector=SimulatedTextCollector())

# 체크포인트 로드
ckpt = torch.load(CHECKPOINT, map_location=DEVICE, weights_only=True)
model = TriCHEFPipeline(embed_dim=ckpt["embed_dim"],
                        alpha_init=ALPHA_INIT, phi_init=PHI_INIT).to(DEVICE)
model.load_state_dict(ckpt["model_state"])
model.eval()

x_num = ds.X_num.to(DEVICE)
x_img = ds.X_img.to(DEVICE)
x_txt = ds.X_txt.to(DEVICE)
y     = ds.y.to(DEVICE)

with torch.no_grad():
    # forward()와 동일한 경로로 magnitude 재구성
    v_re, v_im, v_z = model.encode(x_num, x_img, x_txt)
    magnitude = torch.sqrt(
        v_re ** 2 + (model.alpha * v_im) ** 2 + (model.phi * v_z) ** 2 + 1e-7
    )
    pre_clamp = model.head_precip(magnitude)          # clamp 적용 전 원본 출력
    post      = torch.clamp(pre_clamp, 0.0, 200.0)    # 현재 코드가 하는 일

# ── 1. clamp 가설 검증 ────────────────────────────────────────────
neg_ratio = (pre_clamp < 0).float().mean().item() * 100
print(f"\n[1] head_precip pre-clamp 출력 분포 (전체 {len(ds)}개)")
print(f"    최소: {pre_clamp.min().item():+.4f}")
print(f"    최대: {pre_clamp.max().item():+.4f}")
print(f"    평균: {pre_clamp.mean().item():+.4f}")
print(f"    음수 비율: {neg_ratio:.1f}%")
if neg_ratio > 99.0:
    print(f"    → 확정: 전 샘플 음수 → clamp gradient=0 → 학습 영구 정지")
    print(f"       (dying-clamp: 한 번 빠지면 절대 회복 불가)")
else:
    print(f"    → 음수 비율이 100%가 아님. 다른 원인 조사 필요.")

# ── 2. 3축 상관관계 (Gram-Schmidt 필요성) ──────────────────────────
def cos_sim(a, b):
    return torch.nn.functional.cosine_similarity(a, b, dim=-1).abs().mean().item()

print(f"\n[2] 3축 pairwise 코사인 유사도 (|cos|, 0=직교 1=중복)")
print(f"    Re-Im : {cos_sim(v_re, v_im):.4f}")
print(f"    Re-Z  : {cos_sim(v_re, v_z ):.4f}")
print(f"    Im-Z  : {cos_sim(v_im, v_z ):.4f}")
print(f"    (64차원 무작위 벡터 기준값 ≈ {1/np.sqrt(64):.4f})")

# ── 3. 항상 0 예측 시 MAE ─────────────────────────────────────────
precip_true = y[:, 1]
mae_zero = precip_true.abs().mean().item()
mae_actual = (post[:, 0] - precip_true).abs().mean().item()
print(f"\n[3] MAE 대조")
print(f"    전체 데이터 강수 평균 (=0 예측 시 MAE): {mae_zero:.4f} mm")
print(f"    현재 모델 실제 MAE (전체)            : {mae_actual:.4f} mm")
print(f"    학습 로그의 검증셋 MAE               : {ckpt['val_precip_mae']:.4f} mm")
print(f"    → 세 값이 일치하면 '모델이 항상 0을 출력'하는 상태 확정")
print("=" * 62)
