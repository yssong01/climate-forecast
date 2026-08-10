"""
threshold_validation.py — 임계값 재보정이 진짜 개선인지, 검증셋 과적합인지 검증.

배경(2026-08-08): hurdle_diagnose.py 가 검증셋 전체에서 F1 을 최대화하는
임계값을 찾아 "재학습 없이 F1 +0.05~+0.14 개선 가능"이라고 보고했다
(폭염 t=0.93, 한파 t=0.99, 강수 t=0.75). 그런데 deploy_ablation.py 에서
게이트 가중치 0.003 짜리 Im 축을 끄자 폭염 F1 이 0.65→0.11 로 붕괴했다.

가중치 0.003 인 축이 성능을 좌우할 리 없다. 더 그럴듯한 해석은 t=0.93 이
과도하게 예민한 동작점이라는 것 — 즉 그 임계값이 검증셋의 특정 표본 배치에
맞춰진 값이고, 입력이 조금만 흔들려도 무너진다는 뜻이다. 그렇다면 보고했던
"개선"은 일반화되지 않는 허수다.

검증 방법: 검증셋을 보정용(calibration)/평가용(test) 두 개로 무작위 분할해
  · 보정용에서 F1 최적 임계값을 고르고
  · 그 임계값을 평가용에 적용했을 때 F1 이 유지되는지 본다.
보정용 F1 과 평가용 F1 의 격차가 크면 과적합이다. 비교 기준으로 t=0.5 도
같은 평가용에서 잰다 — 정직한 비교는 동일 표본에서만 성립한다.

라벨 소스(2026-08-10 수정): 폭염·한파·황사는 순간 온도 임계값으로 재계산하지
않는다 — WeatherDataset 이 학습 때 실제로 쓴 공식 라벨(ds.y_heatwave 등)을
그대로 쓴다. 이전 버전은 여기서 temp_a>=HEATWAVE_THRESH 식으로 순간 임계값을
다시 계산했는데, 이건 학습 목표(공식 특보 기준, 하루 단위)와 다른 정의라 양성
표본 수가 10~17배 적게 나오고 F1도 크게 낮게 측정됐다(hurdle_diagnose.py 와
불일치 발견, 2026-08-10). 두 스크립트가 같은 라벨 정의를 쓰도록 통일한다.

실행: python threshold_validation.py
"""
import numpy as np
import torch
from torch.utils.data import random_split

from train import collect_historical, WeatherDataset, VAL_RATIO, SEED, WET_THRESH
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from text_collector import SimulatedTextCollector
from predict import load_model, CHECKPOINT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
CALIB_SEED = 1234   # SEED(42)와 다른 값 — 학습/검증 분할과 독립적인 재분할


def prf(probs, labels, t):
    pred = probs >= t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else float("nan")
    r = tp / (tp + fn) if (tp + fn) else float("nan")
    f = (2 * p * r / (p + r)) if (p + r) and not (np.isnan(p) or np.isnan(r)) else 0.0
    return p, r, f


def best_thresh(probs, labels, grid=None):
    if grid is None:
        grid = np.linspace(0.01, 0.99, 99)
    scored = [(t, prf(probs, labels, t)[2]) for t in grid]
    return max(scored, key=lambda x: x[1])


def sensitivity(probs, labels, t, delta=0.02):
    """임계값을 ±delta 흔들었을 때 F1 변동폭 — 동작점의 안정성 지표."""
    fs = [prf(probs, labels, tt)[2]
          for tt in (max(0.001, t - delta), t, min(0.999, t + delta))]
    return min(fs), max(fs)


def main():
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    model.eval()
    records = collect_historical()
    # 2026-08-09: 체크포인트가 Re·Im축 모두 실제 데이터에 의존하므로 학습
    # 때와 같은 컬렉터를 써야 한다. im_dim 으로 구/신버전을 자동 판별한다.
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    n_val = max(2, int(len(ds) * VAL_RATIO))
    _, val_ds = random_split(ds, [len(ds) - n_val, n_val],
                             generator=torch.Generator().manual_seed(SEED))
    val_idx = np.array(val_ds.indices)

    rain_p, heat_p, cold_p, dust_p = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(val_idx), BATCH):
            idx = val_idx[b:b + BATCH].tolist()
            model(num_x=ds.X_num[idx].to(DEVICE),
                  img_x=ds.X_img[idx].to(DEVICE),
                  txt_x=ds.X_txt[idx].to(DEVICE))
            rain_p.append(torch.sigmoid(model._last_rain_logit).squeeze(-1).cpu().numpy())
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            if hasattr(model, "head_dust"):
                dust_p.append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu().numpy())

    precip_a = ds.y[val_idx, 1].numpy()
    heads = [
        ("강수", np.concatenate(rain_p), (precip_a >= WET_THRESH).astype(int)),
        ("폭염", np.concatenate(heat_p), ds.y_heatwave[val_idx].numpy().astype(int)),
        ("한파", np.concatenate(cold_p), ds.y_coldwave[val_idx].numpy().astype(int)),
    ]
    if dust_p:
        dmask = ds.dust_mask[val_idx].numpy().astype(bool)
        heads.append(("황사", np.concatenate(dust_p)[dmask],
                     ds.y_dust[val_idx].numpy().astype(int)[dmask]))

    rng = np.random.RandomState(CALIB_SEED)

    print(f"검증 {len(val_idx)}개 (황사는 PM10 실측 가능 관측소만 별도 표본)\n")
    print("="*86)
    print(" 임계값 일반화 검증 — 보정용에서 고른 t 가 평가용에서도 통하는가?")
    print("="*86)

    for name, probs, labels in heads:
        # 헤드마다 표본 길이가 다르다(황사는 PM10 커버 관측소로 미리 필터링된
        # 부분집합) — 그래서 마스크도 헤드별로 그 길이에 맞춰 새로 뽑는다.
        mask = rng.rand(len(probs)) < 0.5      # 보정용 절반 / 평가용 절반
        pc, lc = probs[mask], labels[mask]      # calibration
        pt, lt = probs[~mask], labels[~mask]    # test
        if lc.sum() == 0 or lt.sum() == 0:
            print(f"\n[{name}] 표본 부족으로 분할 검증 불가")
            continue

        t_opt, f1_calib = best_thresh(pc, lc)
        _, _, f1_test = prf(pt, lt, t_opt)          # 고른 t 를 평가용에 적용
        _, _, f1_test05 = prf(pt, lt, 0.5)          # 같은 평가용에서 t=0.5
        t_oracle, f1_oracle = best_thresh(pt, lt)   # 평가용을 훔쳐본 상한(도달 불가)
        lo, hi = sensitivity(pt, lt, t_opt)

        gap = f1_calib - f1_test
        real_gain = f1_test - f1_test05
        print(f"\n[{name}] 양성 보정용 {int(lc.sum())}개 / 평가용 {int(lt.sum())}개")
        print(f"  보정용에서 고른 t = {t_opt:.2f}  (보정용 F1 {f1_calib:.4f})")
        print(f"  → 평가용 F1        = {f1_test:.4f}   (일반화 격차 {gap:+.4f})")
        print(f"  평가용 t=0.50 F1   = {f1_test05:.4f}")
        print(f"  실제 순이득        = {real_gain:+.4f}  "
              f"{'✅ 진짜 개선' if real_gain > 0.01 else '⚠️ 허수 — 과적합 의심'}")
        print(f"  t±0.02 흔들 때 F1 범위 = [{lo:.4f}, {hi:.4f}] (폭 {hi-lo:.4f})"
              f"{'  ⚠️ 동작점 불안정' if hi - lo > 0.05 else ''}")
        print(f"  (참고) 평가용을 훔쳐본 상한 t={t_oracle:.2f} → F1 {f1_oracle:.4f}")


if __name__ == "__main__":
    main()
