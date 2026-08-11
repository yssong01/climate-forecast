"""
train_z_only.py — Tri-CHEF 적용 효과 측정용 Z축 단독 대조군 학습.

왜 필요한가: 포트폴리오의 핵심 근거인 "3축 Tri-CHEF vs Z축 단독" 비교는
두 조건이 **같은 데이터셋·같은 분할**에서 나와야 성립한다. 기존
checkpoints/z_only_baseline.pt 는 현행 데이터셋보다 이전 세대에서 학습된
것이라(2026-08-11 확인: temp_mean 15.3251 vs 현행 15.2646) 그대로 쓰면
표본이 다른 값을 비교하게 된다 — 이 프로젝트가 규약으로 금지한 오류다.

공정성 조건:
  · 같은 데이터 캐시(cache/historical_data_1y.json) — API 재수집 없음
  · 같은 SEED(42) → random_split 이 동일한 학습/검증 분할을 만든다
  · 같은 조기 종료 기준(early_stop=True, PATIENCE=20)
    → 두 조건 모두 '자기 기준으로 수렴한 최고 검증 체크포인트'를 저장한다.
      train() 주석은 ablation 에 early_stop=False 를 권하지만, 여기서는
      "실제 배포되는 것이 최고 검증 체크포인트"라는 점을 우선했다.
      학습 길이가 조건마다 다를 수 있다는 한계는 결과 보고에 명시한다.

Z축은 퍼시스턴스 기준선 계산에 필요해 항상 활성이다(train() 주석 참고).
use_re/use_im 를 False 로 두면 해당 축 입력이 영벡터로 차단된다.

실행: python train_z_only.py
"""
from train import train

OUT = "./checkpoints/z_only_current_dataset.pt"

if __name__ == "__main__":
    print("=" * 70)
    print(" Z축 단독 대조군 학습 (Tri-CHEF 미적용) — 현행 데이터셋 기준")
    print(" 3축 조건과 동일: 같은 캐시 / 같은 SEED / 같은 조기 종료 기준")
    print("=" * 70)
    result = train(use_re=False, use_im=False, checkpoint=OUT, verbose=True)
    print(f"\n저장: {OUT}")
