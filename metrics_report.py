"""
metrics_report.py — 정확도·정밀도·신뢰도 세 축을 같은 표본에서 한 번에 산출한다.

**왜 필요한가.** 세 지표가 지금까지 서로 다른 스크립트에서, 서로 다른 분할로
나왔다. 회귀 MAE 는 `train.py`, 분류 F1 은 `patch_extreme_metrics.py`,
보정 오차는 `probability_calibration_check.py`가 각각 냈고, 임계값 검증은 또
검증셋을 절반으로 쪼개 쓴다. 그래서 "F1 은 올랐는데 신뢰도는 어떻게 됐나"를
물으면 표본이 다른 숫자를 비교하게 된다. 여기서는 `eval_cache.py`가 만든
동일 표본 하나에서 셋을 모두 계산한다.

**세 지표는 서로 다른 질문에 답한다.**
  정확도(accuracy)    — 값이 실제와 얼마나 가까운가            → 회귀
  정밀도(precision)   — 사건이라 말한 것 중 몇 개가 맞았나      → 분류
  신뢰도(calibration) — "70%"라 한 상황에서 실제로 70%가 났나  → 확률 자체
한 축이 정상이어도 다른 축이 깨질 수 있다는 것이 이 프로젝트의 반복된 경험이라
(계절 오탐·확률 과신·판정선 붕괴) 셋을 분리해 함께 본다.

**배포 기준으로 잰다.** 분류 지표는 원본 임계값 0.5 뿐 아니라 서빙이 실제로
쓰는 판정선(`predict.event_threshold`, 보정 후 공간)에서도 계산한다. 화면에
나가는 판정과 문서의 F1 이 다른 값을 가리키는 일을 막기 위해서다.

실행: python metrics_report.py [체크포인트]
출력: docs/images/metrics_report.png + cache/metrics_report.json + 표준출력
"""
import argparse
import json
import os

import numpy as np

import eval_cache
from predict import (CHECKPOINT, EXTREME_EVENT_THRESH, calibrate_prob,
                     event_threshold, PRECIP_PROB_GATE, PRECIP_PROB_GATE_BY_LEAD)

OUT_PNG = "./docs/images/metrics_report.png"
OUT_JSON = "./cache/metrics_report.json"

HIT_TEMP_TOL = 1.5    # °C — accuracy.py 와 동일 기준
HIT_PRECIP_THRESH = 0.1

EVENTS = [("heatwave", "폭염"), ("coldwave", "한파"), ("dust", "황사")]


# ── 지표 계산 ────────────────────────────────────────────────
def prf(prob, label, t):
    pred = (prob >= t).astype(int)
    tp = int(((pred == 1) & (label == 1)).sum())
    fp = int(((pred == 1) & (label == 0)).sum())
    fn = int(((pred == 0) & (label == 1)).sum())
    tn = int(((pred == 0) & (label == 0)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return dict(precision=p, recall=r, f1=f, tp=tp, fp=fp, fn=fn, tn=tn)


def reliability(prob, label, n_bins=10):
    """구간별 (평균 예측확률, 실제 빈도, 표본 수)와 ECE·MCE·Brier."""
    edges = np.linspace(0, 1, n_bins + 1)
    rows, ece, mce = [], 0.0, 0.0
    n = len(prob)
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (prob >= lo) & (prob < hi if hi < 1 else prob <= 1)
        k = int(m.sum())
        if k == 0:
            rows.append((float(lo), float(hi), None, None, 0))
            continue
        pm, fr = float(prob[m].mean()), float(label[m].mean())
        rows.append((float(lo), float(hi), pm, fr, k))
        ece += k / n * abs(pm - fr)
        mce = max(mce, abs(pm - fr))
    brier = float(((prob - label) ** 2).mean())
    return rows, float(ece), float(mce), brier


def accuracy_block(d, ckpt=None):
    """① 정확도 — 회귀. 기준선은 반드시 같은 표본에서 계산한다."""
    tp_, tt = d["temp_pred"], d["temp_true"]
    pp, pt = d["precip_pred"], d["precip_true"]
    temp_mae = float(np.abs(tp_ - tt).mean())
    temp_naive = float(d["temp_persist_abs"].mean())          # T(t+6) ≈ T(t)
    precip_mae = float(np.abs(pp - pt).mean())
    precip_naive = float(np.abs(pt).mean())                    # 상시 0mm

    # 서빙 후처리(rain_prob < PRECIP_PROB_GATE 면 0)를 반영한 값도 함께
    # 잰다 — 문서의 MAE 는 후처리 전 값이라, 화면에 실제로 나가는 값과
    # 다르다는 점을 드러낸다. 2026-08-29부터 매그니튜드(mm) 임계값 대신
    # 확률 공간에서 게이팅한다(predict.py PRECIP_PROB_GATE 주석 참고,
    # precip_prob_gate_sweep.py 실측 근거). 게이트 값은 리드타임마다 다르게
    # 선정했으므로(+6h/+12h 분포가 달라 공유 상수를 쓰면 안 됨, predict.py
    # 주석 참고) 체크포인트의 lead_hours 로 고른다.
    prob_gate = (PRECIP_PROB_GATE_BY_LEAD.get(ckpt.get("lead_hours"), PRECIP_PROB_GATE)
                 if ckpt is not None else PRECIP_PROB_GATE)
    pp_served = np.where(d["rain_prob"] < prob_gate, 0.0, pp)
    precip_mae_served = float(np.abs(pp_served - pt).mean())

    hit_temp = float((np.abs(tp_ - tt) <= HIT_TEMP_TOL).mean())
    wet_pred = pp_served >= HIT_PRECIP_THRESH
    wet_true = pt >= HIT_PRECIP_THRESH
    hit_precip = float((wet_pred == wet_true).mean())

    # 강수의 주 지표(2026-08-17 변경). MAE 단독은 표본의 93.8%가 무강수인
    # 분포에서 "거의 항상 0"이라는 퇴화 해로 끌린다 — 후처리 임계값을 2.95mm
    # 로 올리면 MAE 로는 기준선을 넘지만 발생 판정 F1 이 0.430→0.103 으로
    # 무너진다(error_breakdown.py 실측). 따라서 발생 판정 F1 과 강수 구간
    # 조건부 MAE 를 주 지표로 두고, 전체 MAE 는 정직성 차원에서 병기한다.
    tp_w = int((wet_pred & wet_true).sum())
    fp_w = int((wet_pred & ~wet_true).sum())
    fn_w = int((~wet_pred & wet_true).sum())
    p_w = tp_w / (tp_w + fp_w) if tp_w + fp_w else 0.0
    r_w = tp_w / (tp_w + fn_w) if tp_w + fn_w else 0.0
    f1_w = 2 * p_w * r_w / (p_w + r_w) if p_w + r_w else 0.0
    mae_wet = float(np.abs(pp_served[wet_true] - pt[wet_true]).mean())
    base_wet = float(np.abs(pt[wet_true]).mean())
    mae_dry = float(np.abs(pp_served[~wet_true] - pt[~wet_true]).mean())

    return {
        "precip_wet_precision": p_w, "precip_wet_recall": r_w,
        "precip_wet_f1": f1_w,
        "precip_mae_wet": mae_wet, "precip_baseline_mae_wet": base_wet,
        "precip_mae_dry": mae_dry,
        "n_wet": int(wet_true.sum()),
        "n": int(len(tt)),
        "temp_mae": temp_mae,
        "temp_baseline_mae": temp_naive,
        "temp_delta_pct": (temp_mae - temp_naive) / temp_naive * 100,
        "precip_mae": precip_mae,
        "precip_mae_served": precip_mae_served,
        "precip_baseline_mae": precip_naive,
        "precip_delta_pct": (precip_mae - precip_naive) / precip_naive * 100,
        "precip_delta_pct_served": (precip_mae_served - precip_naive) / precip_naive * 100,
        "hit_rate_temp": hit_temp,
        "hit_rate_precip": hit_precip,
    }


def precision_block(d, ckpt):
    """② 정밀도 — 분류. 원본 0.5 기준과 서빙 판정선 기준을 함께 낸다."""
    out = {}
    for key, ko in EVENTS:
        mask = d[f"{key.replace('heatwave', 'heat').replace('coldwave', 'cold')}_mask"].astype(bool)
        prob = d[f"{key.replace('heatwave', 'heat').replace('coldwave', 'cold')}_prob"][mask]
        label = d[f"y_{key}"][mask].astype(int)
        if len(prob) == 0:
            continue
        t_raw = EXTREME_EVENT_THRESH.get(key, 0.5)
        prob_cal = np.array([calibrate_prob(float(p), key, ckpt) for p in prob])
        t_served = event_threshold(key, "108", ckpt)
        out[key] = {
            "ko": ko,
            "n": int(mask.sum()),
            "n_pos": int(label.sum()),
            "raw": {"threshold": float(t_raw), **prf(prob, label, t_raw)},
            "served": {"threshold": float(t_served), **prf(prob_cal, label, t_served)},
        }
    return out


def calibration_block(d, ckpt):
    """③ 신뢰도 — 확률 자체. 보정 전후를 같은 표본에서 비교한다."""
    out = {}
    for key, ko in EVENTS + [("rain", "강수")]:
        if key == "rain":
            prob = d["rain_prob"]
            label = (d["precip_true"] >= float(d["wet_thresh"])).astype(int)
        else:
            short = key.replace("heatwave", "heat").replace("coldwave", "cold")
            mask = d[f"{short}_mask"].astype(bool)
            prob = d[f"{short}_prob"][mask]
            label = d[f"y_{key}"][mask].astype(int)
        if len(prob) == 0:
            continue
        prob_cal = np.array([calibrate_prob(float(p), key, ckpt) for p in prob])
        rows_raw, ece0, mce0, br0 = reliability(prob, label)
        rows_cal, ece1, mce1, br1 = reliability(prob_cal, label)
        out[key] = {
            "ko": ko, "n": int(len(prob)), "base_rate": float(label.mean()),
            "raw": {"ece": ece0, "mce": mce0, "brier": br0, "bins": rows_raw},
            "calibrated": {"ece": ece1, "mce": mce1, "brier": br1, "bins": rows_cal},
        }
    return out


# ── 출력 ─────────────────────────────────────────────────────
def print_report(acc, pre, cal):
    print(f"\n{'=' * 78}\n ① 정확도(accuracy) — 회귀 · 검증 표본 {acc['n']:,}개\n{'=' * 78}")
    print(f"  {'지표':<14}{'모델':>12}{'기준선':>12}{'증감':>12}")
    print(f"  {'기온 MAE':<14}{acc['temp_mae']:>12.4f}{acc['temp_baseline_mae']:>12.4f}"
          f"{acc['temp_delta_pct']:>11.1f}%")
    print(f"  {'강수 MAE':<14}{acc['precip_mae']:>12.4f}{acc['precip_baseline_mae']:>12.4f}"
          f"{acc['precip_delta_pct']:>11.1f}%")
    print(f"  {'강수(후처리)':<13}{acc['precip_mae_served']:>12.4f}"
          f"{acc['precip_baseline_mae']:>12.4f}{acc['precip_delta_pct_served']:>11.1f}%")
    print(f"\n  적중률 — 기온 ±{HIT_TEMP_TOL}°C 이내 {acc['hit_rate_temp']:.1%} · "
          f"강수 발생 여부 일치 {acc['hit_rate_precip']:.1%}")
    print(f"\n  강수 주 지표(후처리 적용) — MAE 단독은 무강수 93.8% 분포에서"
          f" 퇴화 해로 끌리므로 아래를 먼저 본다")
    print(f"    발생 판정  정밀도 {acc['precip_wet_precision']:.1%} · "
          f"재현율 {acc['precip_wet_recall']:.1%} · F1 {acc['precip_wet_f1']:.3f}"
          f"  (실제 강수 {acc['n_wet']:,}건)")
    print(f"    강수 구간 조건부 MAE  모델 {acc['precip_mae_wet']:.4f}mm  vs  "
          f"기준선(상시 0) {acc['precip_baseline_mae_wet']:.4f}mm  → "
          f"{(acc['precip_mae_wet'] / acc['precip_baseline_mae_wet'] - 1) * 100:+.1f}%")
    print(f"    무강수 구간 MAE       모델 {acc['precip_mae_dry']:.4f}mm  vs  "
          f"기준선 0.0000mm  ← 전체 격차의 대부분이 여기서 나온다")

    print(f"\n{'=' * 78}\n ② 정밀도(precision) — 분류\n{'=' * 78}")
    print(f"  {'사건':<6}{'기준':<8}{'임계값':>8}{'정밀도':>9}{'재현율':>9}{'F1':>9}"
          f"{'TP':>8}{'FP':>8}{'FN':>8}")
    for key, m in pre.items():
        for tag, label in (("raw", "원본"), ("served", "서빙")):
            b = m[tag]
            print(f"  {m['ko'] if tag == 'raw' else '':<6}{label:<8}{b['threshold']:>8.3f}"
                  f"{b['precision']:>9.1%}{b['recall']:>9.1%}{b['f1']:>9.3f}"
                  f"{b['tp']:>8,}{b['fp']:>8,}{b['fn']:>8,}")

    print(f"\n{'=' * 78}\n ③ 신뢰도(calibration) — 확률 자체\n{'=' * 78}")
    print(f"  {'사건':<6}{'표본':>10}{'양성률':>9}{'ECE 전':>9}{'ECE 후':>9}"
          f"{'MCE 전':>9}{'MCE 후':>9}{'Brier 전':>10}{'Brier 후':>10}")
    for key, m in cal.items():
        print(f"  {m['ko']:<6}{m['n']:>10,}{m['base_rate']:>9.1%}"
              f"{m['raw']['ece']:>9.4f}{m['calibrated']['ece']:>9.4f}"
              f"{m['raw']['mce']:>9.4f}{m['calibrated']['mce']:>9.4f}"
              f"{m['raw']['brier']:>10.4f}{m['calibrated']['brier']:>10.4f}")
    print("\n  ※ '보정 후' 열은 검증셋 전체에서 잰 값이다. 보정 곡선은 이 검증셋의"
          "\n     절반으로 적합했으므로 그 절반이 표본에 섞여 있어 낙관적이다."
          "\n     한 번도 보지 않은 표본에서의 값은 probability_calibration_fit.py 가"
          "\n     평가용 절반에서 따로 보고한다 — 배포 판정의 근거는 그쪽을 쓴다.")


def plot(acc, pre, cal, out_png):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from probability_calibration_check import _ensure_korean_font
    from matplotlib import font_manager

    fp = _ensure_korean_font()
    if fp:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 3, figsize=(16.5, 5.4))

    # ① 정확도 — 기준선 대비 증감 막대
    ax = axes[0]
    names = ["기온 MAE", "강수 MAE", "강수(후처리)"]
    vals = [acc["temp_delta_pct"], acc["precip_delta_pct"], acc["precip_delta_pct_served"]]
    colors = ["#2C6849" if v < 0 else "#95372A" for v in vals]
    ax.barh(names[::-1], vals[::-1], color=colors[::-1], alpha=0.85)
    ax.axvline(0, color="#444", lw=1.2)
    for i, v in enumerate(vals[::-1]):
        ax.text(v + (2 if v > 0 else -2), i, f"{v:+.1f}%", va="center",
                ha="left" if v > 0 else "right", fontsize=11, fontweight="bold")
    ax.set_xlim(min(vals) - 18, max(vals) + 18)
    ax.set_xlabel("기준선 대비 증감 (음수가 개선)", fontsize=12, fontweight="bold")
    ax.set_title(f"① 정확도 — 회귀\n적중률 기온 {acc['hit_rate_temp']:.1%} · "
                 f"강수 {acc['hit_rate_precip']:.1%}", fontsize=13, fontweight="bold")
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", alpha=0.25)

    # ② 정밀도 — 사건별 정밀도·재현율·F1
    ax = axes[1]
    keys = list(pre.keys())
    x = np.arange(len(keys))
    width = 0.26
    for i, (field, lab, c) in enumerate((("precision", "정밀도", "#4C78A8"),
                                         ("recall", "재현율", "#E2954F"),
                                         ("f1", "F1", "#54A24B"))):
        vals = [pre[k]["served"][field] for k in keys]
        ax.bar(x + (i - 1) * width, vals, width, label=lab, color=c, alpha=0.9)
        for xi, v in zip(x + (i - 1) * width, vals):
            ax.text(xi, v + 0.015, f"{v:.2f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([f"{pre[k]['ko']}\n(양성 {pre[k]['n_pos']:,})" for k in keys],
                       fontsize=11)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("값", fontsize=12, fontweight="bold")
    ax.set_title("② 정밀도 — 분류 (서빙 판정선 기준)", fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.25)

    # ③ 신뢰도 — 보정 전후 ECE
    ax = axes[2]
    keys = list(cal.keys())
    x = np.arange(len(keys))
    ax.bar(x - 0.19, [cal[k]["raw"]["ece"] for k in keys], 0.36,
           label="보정 전", color="#B0413E", alpha=0.85)
    ax.bar(x + 0.19, [cal[k]["calibrated"]["ece"] for k in keys], 0.36,
           label="보정 후", color="#2C6849", alpha=0.85)
    for xi, k in zip(x, keys):
        ax.text(xi - 0.19, cal[k]["raw"]["ece"] + 0.004,
                f"{cal[k]['raw']['ece']:.3f}", ha="center", fontsize=9)
        ax.text(xi + 0.19, cal[k]["calibrated"]["ece"] + 0.004,
                f"{cal[k]['calibrated']['ece']:.3f}", ha="center", fontsize=9)
    ax.set_xticks(x)
    ax.set_xticklabels([cal[k]["ko"] for k in keys], fontsize=11)
    ax.set_ylabel("ECE (0에 가까울수록 좋다)", fontsize=12, fontweight="bold")
    ax.set_title("③ 신뢰도 — 표시 확률이 실제 빈도와 맞는가",
                 fontsize=13, fontweight="bold")
    ax.legend(fontsize=10)
    ax.tick_params(labelsize=11)
    ax.grid(axis="y", alpha=0.25)

    fig.suptitle("세 축 지표 — 같은 검증 표본에서 한 번에 산출",
                 fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n저장: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--batch", type=int, default=eval_cache.DEFAULT_BATCH)
    args = ap.parse_args()

    import torch
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=True)
    d = eval_cache.load(args.ckpt, args.batch)

    acc = accuracy_block(d, ckpt)
    pre = precision_block(d, ckpt)
    cal = calibration_block(d, ckpt)
    print_report(acc, pre, cal)
    plot(acc, pre, cal, OUT_PNG)

    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump({"accuracy": acc, "precision": pre, "calibration": cal},
                  f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
