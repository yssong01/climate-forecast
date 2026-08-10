"""
hurdle_diagnose.py — 이진분류 헤드(강수 hurdle·폭염·한파·황사) 진단 + 임계값 재보정.

재학습 없이 현재 체크포인트로 검증셋을 다시 통과시켜:
  1) 검증셋을 보정용/평가용으로 반씩 나눠, 보정용에서 F1 최적 임계값을 찾고
     평가용에서 그 임계값이 실제로 통하는지 확인한다(threshold_validation.py
     에서 확립한 방법 — 같은 데이터에서 임계값을 고르고 채점하면 과적합
     허수가 섞인다는 걸 2026-08-08에 실측으로 확인했다).
  2) 강수 hurdle 은 추가로 rain_prob(분류)과 amount(회귀)를 분리해,
     무강수 구간에서 어느 쪽이 잔여 오차를 만드는지 진단한다.

라벨 소스(2026-08-09부터): 폭염·한파·황사는 순간 임계값 근사가 아니라
train.py/WeatherDataset 이 이미 계산해둔 공식 라벨(ds.y_heatwave 등)을 그대로
쓴다 — 학습 때 실제로 맞춘 목표와 동일한 기준으로 검증해야 재보정이 의미가
있다. 황사는 dust_mask=1(PM10 실측 가능 관측소 6곳)인 표본만 채점한다.

실행: python hurdle_diagnose.py
  → 콘솔 표 + ./checkpoints/threshold_calibration.png
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from torch.utils.data import random_split

from train import collect_historical, WeatherDataset, VAL_RATIO, SEED, WET_THRESH
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from text_collector import SimulatedTextCollector
from predict import load_model, CHECKPOINT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
OUT_PNG = "./checkpoints/threshold_calibration.png"
CALIB_SEED = 1234   # 학습/검증 분할(SEED)과 별개 — 검증셋 내부를 보정/평가로 재분할


def _prf(probs: np.ndarray, labels: np.ndarray, t: float) -> tuple:
    pred = probs >= t
    tp = int((pred & (labels == 1)).sum())
    fp = int((pred & (labels == 0)).sum())
    fn = int((~pred & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) > 0 else float("nan")
    r = tp / (tp + fn) if (tp + fn) > 0 else float("nan")
    f = (2 * p * r / (p + r)) if (p + r) > 0 and not (np.isnan(p) or np.isnan(r)) else 0.0
    return p, r, f, tp, fp, fn


def _pr_curve(probs: np.ndarray, labels: np.ndarray) -> list[dict]:
    rows = []
    for t in np.linspace(0.01, 0.99, 99):
        p, r, f, tp, fp, fn = _prf(probs, labels, t)
        rows.append({"thresh": t, "precision": p, "recall": r, "f1": f,
                     "tp": tp, "fp": fp, "fn": fn})
    return rows


def _best_f1(rows: list[dict]) -> dict:
    valid = [r for r in rows if not np.isnan(r["precision"])]
    return max(valid, key=lambda r: r["f1"]) if valid else rows[0]


def main():
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    records = collect_historical()
    # 2026-08-09: 체크포인트가 Re축(게이트 41~52%)·Im축 모두 실제 데이터에
    # 의존하므로 학습 때와 같은 컬렉터를 써야 한다 — 안 맞추면 모든 지표가
    # 무너진다(실측: 황사 F1 0.661→0.089, Re축 컬렉터 불일치로 확인됨).
    # im_dim 으로 구버전(MiniLM 384)/신버전(경향벡터 12) 체크포인트를 자동
    # 판별한다 — 같은 실수를 축마다 반복하지 않기 위함.
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    n_val = max(2, int(len(ds) * VAL_RATIO))
    n_train = len(ds) - n_val
    _, val_ds = random_split(ds, [n_train, n_val],
                             generator=torch.Generator().manual_seed(SEED))
    val_idx = np.array(val_ds.indices)

    rain_probs, heat_probs, cold_probs, dust_probs = [], [], [], []
    amounts = []
    model.eval()
    with torch.no_grad():
        for b in range(0, len(val_idx), BATCH):
            idx = val_idx[b:b + BATCH].tolist()
            x_num = ds.X_num[idx].to(DEVICE)
            x_img = ds.X_img[idx].to(DEVICE)
            x_txt = ds.X_txt[idx].to(DEVICE)
            pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)

            rain_logit = model._last_rain_logit
            rp = torch.sigmoid(rain_logit).squeeze(-1)
            rain_probs.append(rp.cpu().numpy())
            amounts.append((pred[:, 1] / rp.clamp_min(1e-6)).cpu().numpy())

            if hasattr(model, "head_heatwave"):
                heat_probs.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
            if hasattr(model, "head_coldwave"):
                cold_probs.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            if hasattr(model, "head_dust"):
                dust_probs.append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu().numpy())

    rain_prob = np.concatenate(rain_probs)
    amount = np.concatenate(amounts)
    precip_actual = ds.y[val_idx, 1].numpy()
    is_wet = (precip_actual >= WET_THRESH).astype(int)

    heads = [("강수(hurdle)", rain_prob, is_wet, None)]
    if heat_probs:
        heads.append(("폭염", np.concatenate(heat_probs),
                      ds.y_heatwave[val_idx].numpy().astype(int), None))
    if cold_probs:
        heads.append(("한파", np.concatenate(cold_probs),
                      ds.y_coldwave[val_idx].numpy().astype(int), None))
    if dust_probs:
        dmask = ds.dust_mask[val_idx].numpy().astype(bool)
        heads.append(("황사", np.concatenate(dust_probs)[dmask],
                      ds.y_dust[val_idx].numpy().astype(int)[dmask], None))

    print(f"검증 {len(val_idx)}개\n")
    print("="*86)
    print(" 임계값 재보정 — 보정용/평가용 분할, 라벨은 학습 때와 동일한 공식 라벨(폭염·한파·황사)")
    print("="*86)

    rng = np.random.RandomState(CALIB_SEED)
    curves = {}
    for name, probs, labels, _ in heads:
        mask = rng.rand(len(probs)) < 0.5
        pc, lc = probs[mask], labels[mask]
        pt, lt = probs[~mask], labels[~mask]
        n_pos_c, n_pos_t = int(lc.sum()), int(lt.sum())

        rows_full = _pr_curve(probs, labels)   # 시각화용 — 전체 검증셋 기준
        curves[name] = rows_full

        if n_pos_c == 0 or n_pos_t == 0:
            print(f"\n[{name}] 표본 부족(보정용 양성 {n_pos_c}개/평가용 양성 {n_pos_t}개) — 재보정 생략")
            continue

        calib_rows = _pr_curve(pc, lc)
        t_opt = _best_f1(calib_rows)["thresh"]
        p05, r05, f05, tp05, fp05, fn05 = _prf(pt, lt, 0.5)
        pOpt, rOpt, fOpt, tpOpt, fpOpt, fnOpt = _prf(pt, lt, t_opt)

        print(f"\n[{name}] 보정용 양성 {n_pos_c}개 / 평가용 양성 {n_pos_t}개")
        print(f"  평가용 t=0.50    : P={p05:.3f} R={r05:.3f} F1={f05:.3f}  (TP={tp05} FP={fp05} FN={fn05})")
        print(f"  평가용 t={t_opt:.2f}(보정용에서 선택): P={pOpt:.3f} R={rOpt:.3f} F1={fOpt:.3f}  "
              f"(TP={tpOpt} FP={fpOpt} FN={fnOpt})")
        gain = fOpt - f05
        print(f"  실제 순이득: {gain:+.3f}  "
              f"({'✅ 진짜 개선' if gain > 0.01 else '⚠️ 0.5가 이미 근접 최적이거나 과적합 의심'})")

    print("\n" + "="*86)
    print(" 강수 hurdle 분해 — rain_prob(분류) vs amount(회귀), 무강수 구간")
    print("="*86)
    dry = precip_actual < WET_THRESH
    p = rain_prob[dry]
    a = amount[dry]
    print(f"무강수 {dry.sum()}개({dry.mean()*100:.2f}%)")
    print("[무강수 구간] rain_prob 분포")
    for q in (0, 10, 25, 50, 75, 90, 99, 100):
        print(f"  p{q:3d} = {np.percentile(p, q):.4f}")
    print(f"  평균 = {p.mean():.4f}")
    print("\n[무강수 구간] amount(softplus 헤드) 분포 — mm")
    for q in (0, 10, 25, 50, 75, 90, 99, 100):
        print(f"  p{q:3d} = {np.percentile(a, q):.4f}")
    print(f"  평균 = {a.mean():.4f}")
    print(f"\n[무강수 구간] 최종 예측(prob×amount) 평균 = {(p*a).mean():.4f}mm")

    # ── PR 곡선 시각화(전체 검증셋 기준, 참고용) ────────────────────
    fig, axes = plt.subplots(1, len(curves), figsize=(6*len(curves), 5))
    if len(curves) == 1:
        axes = [axes]
    for ax, (name, rows) in zip(axes, curves.items()):
        t   = [r["thresh"] for r in rows]
        pr  = [r["precision"] for r in rows]
        rc  = [r["recall"] for r in rows]
        f1s = [r["f1"] for r in rows]
        ax.plot(t, pr, label="precision", color="#3b82f6")
        ax.plot(t, rc, label="recall", color="#f97316")
        ax.plot(t, f1s, label="F1", color="#10b981", linewidth=2)
        best = _best_f1(rows)
        ax.axvline(best["thresh"], color="#10b981", linestyle="--", alpha=0.5)
        ax.axvline(0.5, color="gray", linestyle=":", alpha=0.5)
        ax.set_title(f"{name} (전체검증셋 best t={best['thresh']:.2f}, F1={best['f1']:.3f})")
        ax.set_xlabel("threshold")
        ax.set_ylabel("score")
        ax.legend()
        ax.set_ylim(0, 1.02)
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=120)
    print(f"\n저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
