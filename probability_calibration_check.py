"""
probability_calibration_check.py — 확률 보정(probability calibration) 상태 측정.

**이 프로젝트에서 '보정'이라는 말이 두 가지로 쓰인다.** 혼동을 막기 위해
용어를 먼저 구분한다.

- `threshold_validation.py`의 '보정' = **판정 임계값 고르기.** 확률 자체는
  그대로 두고 "몇 이상을 사건으로 칠 것인가"만 정한다.
- `calibration_plot_diagnose.py`의 '캘리브레이션 플롯' = **관측소별
  정밀도-재현율 산점도.** 신뢰도 곡선(reliability diagram)이 아니다.
- 이 스크립트의 '보정' = **확률 자체가 빈도와 맞는가.** 모델이 "70%"라고
  한 표본들에서 실제로 70%가 사건이었는지를 잰다.

세 번째는 지금까지 한 번도 측정한 적이 없다. 대시보드는 이 확률을 막대로
그대로 노출하므로, 어긋나 있다면 화면 문구로 밝혀야 한다.

지표
  ECE(Expected Calibration Error) — 구간별 |예측확률 − 실제빈도|를 표본 수로
    가중평균. 0에 가까울수록 잘 맞는다.
  MCE(Maximum Calibration Error) — 그 최댓값. 최악의 구간이 얼마나 어긋나는가.
  Brier score — (예측확률 − 실제)²의 평균. 보정과 변별력을 함께 반영한다.

폭염·한파·황사는 기상청 공식 라벨이 있는 표본에서만 채점한다 — 마스크=0인
표본의 라벨 0은 '사건 없음'이 아니라 '판정 불가'다(train.py와 동일 규약).

실행: python probability_calibration_check.py [체크포인트]
출력: docs/images/probability_calibration.png + 표준출력 지표
"""
import os
import subprocess
import sys

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import (WeatherDataset, collect_historical, make_split, WET_THRESH)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from text_collector import SimulatedTextCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096
N_BINS = 10
OUT_PNG = "./docs/images/probability_calibration.png"


def _ensure_korean_font():
    """컨테이너 기본 이미지에는 한글 폰트가 없다 — 없으면 설치한다."""
    path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if not os.path.exists(path):
        subprocess.run("apt-get update -qq && apt-get install -y -qq fonts-nanum",
                       shell=True, check=False,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    return path if os.path.exists(path) else None


def reliability(probs, labels, n_bins=N_BINS):
    """
    구간별 (평균 예측확률, 실제 빈도, 표본 수).

    등간격 구간을 쓴다. 등빈도(quantile) 구간이 표본 균형에는 낫지만, 확률
    0.0~0.1에 표본이 몰리는 희소 사건에서는 구간 경계가 전부 0 근처에 몰려
    "0.7이라고 했을 때 실제로 어땠나"를 볼 수 없게 된다.
    """
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    rows = []
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            rows.append((lo, hi, np.nan, np.nan, 0))
            continue
        rows.append((lo, hi, float(probs[m].mean()), float(labels[m].mean()), int(m.sum())))
    return rows


def ece_mce(rows, n_total):
    """구간별 오차를 표본 수로 가중해 평균(ECE)과 최댓값(MCE)을 낸다."""
    gaps, weights = [], []
    for _, _, p_mean, freq, n in rows:
        if n == 0:
            continue
        gaps.append(abs(p_mean - freq))
        weights.append(n)
    if not gaps:
        return float("nan"), float("nan")
    gaps, weights = np.array(gaps), np.array(weights, dtype=float)
    return float((gaps * weights).sum() / n_total), float(gaps.max())


def main():
    ckpt_path = sys.argv[1] if len(sys.argv) > 1 else CHECKPOINT
    model, ckpt = load_model(ckpt_path, DEVICE)
    model.eval()
    print(f"체크포인트: {ckpt_path} (num_features={ckpt.get('num_features')}, "
          f"분할={ckpt.get('split_mode', 'random')})")

    records = collect_historical()
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False, ckpt=ckpt)
    val_idx = np.array(val_ds.indices)

    rain_p, heat_p, cold_p, dust_p = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(val_idx), BATCH):
            idx = val_idx[b:b + BATCH].tolist()
            model(num_x=ds.X_num[idx].to(DEVICE),
                  img_x=ds.X_img[idx].to(DEVICE).float(),
                  txt_x=ds.X_txt[idx].to(DEVICE))
            rain_p.append(torch.sigmoid(model._last_rain_logit).squeeze(-1).cpu().numpy())
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).squeeze(-1).cpu().numpy())
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).squeeze(-1).cpu().numpy())
            if hasattr(model, "head_dust"):
                dust_p.append(torch.sigmoid(model._last_dust_logit).squeeze(-1).cpu().numpy())

    precip_a = ds.y[val_idx, 1].numpy()
    hmask = ds.heat_mask[val_idx].numpy().astype(bool)
    cmask = ds.cold_mask[val_idx].numpy().astype(bool)
    heads = [
        ("강수", np.concatenate(rain_p), (precip_a >= WET_THRESH).astype(int)),
        ("폭염", np.concatenate(heat_p)[hmask],
         ds.y_heatwave[val_idx].numpy().astype(int)[hmask]),
        ("한파", np.concatenate(cold_p)[cmask],
         ds.y_coldwave[val_idx].numpy().astype(int)[cmask]),
    ]
    if dust_p:
        dmask = ds.dust_mask[val_idx].numpy().astype(bool)
        heads.append(("황사", np.concatenate(dust_p)[dmask],
                     ds.y_dust[val_idx].numpy().astype(int)[dmask]))

    results = []
    for name, probs, labels in heads:
        rows = reliability(probs, labels)
        ece, mce = ece_mce(rows, len(probs))
        brier = float(((probs - labels) ** 2).mean())
        base = float(labels.mean())
        results.append((name, probs, labels, rows, ece, mce, brier, base))

        print(f"\n{'='*72}\n[{name}]  표본 {len(probs):,}개 · 실제 양성률 {base:.3%}")
        print(f"  ECE {ece:.4f} · MCE {mce:.4f} · Brier {brier:.4f}")
        print(f"  {'예측확률 구간':<16}{'평균예측':>10}{'실제빈도':>10}{'차이':>10}{'표본':>12}")
        for lo, hi, p_mean, freq, n in rows:
            if n == 0:
                print(f"  {f'{lo:.1f}~{hi:.1f}':<16}{'—':>10}{'—':>10}{'—':>10}{0:>12,}")
                continue
            d = p_mean - freq
            flag = "  ←과신" if d > 0.10 else ("  ←과소" if d < -0.10 else "")
            print(f"  {f'{lo:.1f}~{hi:.1f}':<16}{p_mean:>10.3f}{freq:>10.3f}"
                  f"{d:>+10.3f}{n:>12,}{flag}")

    # ── 신뢰도 곡선 ─────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager

    fp = _ensure_korean_font()
    if fp:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    # 배치(2026-08-17 조정) — 1행(신뢰도 곡선)은 x·y가 모두 확률이라 축척이
    # 같아야 대각선을 눈으로 판정할 수 있다. `set_box_aspect(1)`로 정사각형을
    # 강제하고, 그만큼 남는 세로 공간을 2행(표본 수 히스토그램)에 넘긴다.
    n = len(results)
    fig, axes = plt.subplots(2, n, figsize=(4.8 * n, 8.9),
                             gridspec_kw={"height_ratios": [4.2, 2.6]})
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, (name, probs, labels, rows, ece, mce, brier, base) in enumerate(results):
        ax, axh = axes[0, j], axes[1, j]
        xs = [r[2] for r in rows if r[4] > 0]
        ys = [r[3] for r in rows if r[4] > 0]
        ns = [r[4] for r in rows if r[4] > 0]

        ax.plot([0, 1], [0, 1], "--", color="gray", lw=1.4, label="완전 보정선(y = x)")
        ax.plot(xs, ys, "o-", color="#4C78A8", lw=2.2, ms=7,
                label="실측값(점 크기 = 표본 수)")
        # 점 크기로 표본 수를 함께 보여준다 — 오른쪽 끝 구간은 표본이 극히
        # 적어 실제 빈도가 요동치므로, 크기 없이 보면 과대해석하기 쉽다.
        ax.scatter(xs, ys, s=[max(12, 320 * k / max(ns)) for k in ns],
                   color="#4C78A8", alpha=0.30, zorder=3)
        ax.axhline(base, color="#B0413E", ls=":", lw=1.4,
                   label=f"실제 양성률 {base:.1%}")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        ax.set_box_aspect(1)
        ax.set_xlabel("모델이 낸 확률", fontsize=13, fontweight="bold")
        ax.set_ylabel("실제 사건 빈도", fontsize=13, fontweight="bold")
        ax.set_title(f"{name} — ECE {ece:.3f} · Brier {brier:.3f}",
                     fontsize=15, fontweight="bold")
        ax.tick_params(labelsize=11)
        # 범례와 설명 상자는 첫 열에만 둔다 — 네 패널의 기호가 모두 같으므로
        # 열마다 반복하면 곡선만 가린다(2026-08-17 조정).
        if j == 0:
            ax.legend(fontsize=10.5, loc="upper left", labelspacing=0.35,
                      borderpad=0.5, handlelength=1.6, framealpha=0.92)
            # 범례 바로 아래에 '실제 양성률'이 무엇이고 무엇을 뜻하는지 밝힌다 —
            # 그림만 떼어 문서·발표자료에 실릴 때 본문 설명이 따라가지 않는다.
            ax.text(0.035, 0.755,
                    f"실제 양성률({base:.1%})은 전체 표본 중\n"
                    "사건이 실제로 일어난 비율이다.\n"
                    "아무 정보 없이 늘 이 값만 답할 때\n"
                    "그려지는 수평선이며, 곡선이 이 선에서\n"
                    "위아래로 크게 벌어질수록 위험한 시각과\n"
                    "그렇지 않은 시각을 실제로 가려낸다는 뜻이다.\n"
                    "네 패널의 기호는 모두 같다.",
                    transform=ax.transAxes, fontsize=9.5, va="top", ha="left",
                    linespacing=1.55,
                    bbox=dict(boxstyle="round,pad=0.45", facecolor="#F7F7F7",
                              edgecolor="gray", alpha=0.92))
        ax.grid(alpha=0.25)

        axh.bar([r[2] for r in rows if r[4] > 0], ns,
                width=0.08, color="#54A24B", alpha=0.75)
        axh.set_yscale("log")
        axh.set_xlim(0, 1)
        axh.set_xlabel("모델이 낸 확률", fontsize=13, fontweight="bold")
        axh.set_ylabel("표본 수(로그 눈금)", fontsize=13, fontweight="bold")
        axh.set_title("구간별 표본 수", fontsize=13, fontweight="bold")
        axh.tick_params(labelsize=11)
        axh.grid(alpha=0.25)

    fig.suptitle("확률 보정 상태 — 대각선에 근접할수록 '확률'의 신뢰도가 높아진다",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.965])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n저장: {OUT_PNG}")


if __name__ == "__main__":
    main()
