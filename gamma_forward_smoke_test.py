"""
gamma_forward_smoke_test.py — pipeline_model.py 의 precip_gamma_nll 분기가
train/eval 모드에서 모두 정상 동작하는지, forward() shape이 그대로인지
확인하는 일회성 스모크 테스트. 실제 데이터·체크포인트는 쓰지 않는다.

실행: python gamma_forward_smoke_test.py
"""
import torch

from pipeline_model import TriCHEFPipeline

torch.manual_seed(0)

NUM_DIM = 14   # 배포 Z축 차원(연중 시각 포함)
IMG_SHAPE = (4, 16, 16)
TXT_DIM = 12   # 배포 Im축(TendencyEncoder) 차원
B = 64

model = TriCHEFPipeline(
    num_features=NUM_DIM, im_dim=TXT_DIM,
    embed_dim=64, temp_mean=15.0, precip_mean=0.5,
    persistence_residual=True,
    feat_mean=[0.0] * NUM_DIM, feat_std=[1.0] * NUM_DIM,
    precip_gamma_nll=True, precip_gamma_alpha_init=0.3, precip_gamma_mu_init=2.4,
)

x_num = torch.randn(B, NUM_DIM)
x_img = torch.randn(B, *IMG_SHAPE)
x_txt = torch.randn(B, TXT_DIM)

print("=== train() 모드 ===")
model.train()
out = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
print("shape:", tuple(out.shape), "(기대: (64, 2))")
assert out.shape == (B, 2), "forward() 반환 shape이 깨짐"
print("mu 범위:", model._last_precip_mu.min().item(), model._last_precip_mu.max().item())
print("alpha 범위:", model._last_precip_alpha.min().item(), model._last_precip_alpha.max().item())
print("precip_pred(=rain_prob*mu, 학습 중 저렴한 값) 통계:",
      out[:, 1].min().item(), out[:, 1].max().item())

print("\n=== eval() 모드 ===")
model.eval()
with torch.no_grad():
    out_eval = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
print("shape:", tuple(out_eval.shape), "(기대: (64, 2))")
assert out_eval.shape == (B, 2), "forward() 반환 shape이 깨짐(eval)"
precip_eval = out_eval[:, 1]
n_zero = int((precip_eval == 0.0).sum())
n_pos = int((precip_eval > 0.0).sum())
print(f"중앙값 점추정 — 0인 표본 {n_zero}/{B}, 양수인 표본 {n_pos}/{B}")
print("양수 표본 범위:", precip_eval[precip_eval > 0].min().item() if n_pos else None,
      precip_eval[precip_eval > 0].max().item() if n_pos else None)
assert not torch.isnan(precip_eval).any(), "NaN 발생"
assert (precip_eval >= 0).all(), "음수 발생 — 강수량은 항상 0 이상이어야 함"

# rain_prob 과 점추정의 관계 확인 — rain_prob<=0.5인 표본은 전부 0이어야 한다.
rain_prob_eval = torch.sigmoid(model._last_rain_logit).squeeze(-1)
low_p = rain_prob_eval <= 0.5
print(f"\nrain_prob<=0.5 표본 {int(low_p.sum())}개 중 "
      f"precip_pred==0 인 것 {int((precip_eval[low_p] == 0.0).sum())}개 (전부 일치해야 함)")
assert (precip_eval[low_p] == 0.0).all(), "p<=0.5인데 0이 아닌 점추정이 있음 — 공식 오류"

print("\n판정: 모든 검증 통과 — forward() shape 유지, train/eval 분기 정상, "
      "중앙값 공식 경계조건(p<=0.5 → 0) 충족.")
