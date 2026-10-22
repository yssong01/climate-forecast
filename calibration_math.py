"""
calibration_math.py — 확률 보정 곡선의 **적용** 함수(numpy 만 쓴다, 2026-10-22).

서빙(`extreme_gbm.calibrated`)이 보정 곡선을 적용하려고 진단 모듈
`probability_calibration_fit` 을 임포트하고 있었다. 그 모듈은 맨 위에서
`eval_cache`·`train`·`predict` 를 불러오므로, 화면의 첫 극한기상 확률 계산이
학습 파이프라인 전체를 메모리에 올렸다. 적용 수식 둘만 여기로 옮기고 두
쪽이 같은 함수를 쓴다 — 수식을 두 곳에 적으면 한쪽만 고치게 된다.
적합(fit_*)은 학습 쪽 일이라 `probability_calibration_fit` 에 남긴다.
"""
import numpy as np


def apply_calibration(probs, xs, ys):
    """등온 회귀 조회표를 선형보간으로 적용. 양 끝은 끝값으로 고정된다(np.interp 기본)."""
    return np.interp(probs, xs, ys)


def apply_beta(probs, a, b, c, eps=1e-6):
    """베타 보정(Kull et al. 2017) m(p) = sigmoid(a·ln p − b·ln(1−p) + c)."""
    p = np.clip(probs, eps, 1 - eps)
    z = a * np.log(p) - b * np.log(1 - p) + c
    return 1.0 / (1.0 + np.exp(-z))
