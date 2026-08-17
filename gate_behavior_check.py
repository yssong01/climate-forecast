"""
gate_behavior_check.py — 동적 게이트의 판단이 무엇에 반응하는지 잰다.

**배경.** 문서와 화면은 게이트 배분을 "학습된 판단"이라고 서술한다. 헤드는
기온을 흔들어 물리적 방향성을 확인했지만(`coldwave_pathway_check.py`),
**게이트는 그런 식으로 흔들어 본 적이 없다.**

먼저 구조를 분명히 해야 한다. 게이트의 입력은 Z축뿐이다.

    w = self.gate(num_x)        # pipeline_model.py — num_x 는 Z축 14차원

즉 게이트는 Re축 보간장도, Im축 경향도 보지 않는다. "인접 관측소가 결측이라
Re 를 덜 믿는다" 같은 판단은 **구조적으로 불가능**하다 — 그런 정보가 입력에
없기 때문이다. 게이트가 할 수 있는 것은 "지금이 어떤 상황인가(Z)"에 따라
축 배분을 바꾸는 것뿐이다.

**무엇을 재는가.**
  1. 배분의 분포 — 평균·표준편차·최소·최대. 표준편차가 0에 가까우면 사실상
     정적 가중치이며, '동적'이라는 서술이 과장이 된다.
  2. 특성별 민감도 — Z축 14개 특성을 하나씩 관측 범위(5~95 백분위)로 흔들
     었을 때 각 축 가중치가 움직이는 폭. 무엇이 배분을 지배하는지 보인다.
  3. 물리적 타당성 — 배분이 실제로 의미 있는 축에 몰리는지, 아니면 특정
     특성 하나에 기계적으로 끌려다니는지.

실행: python gate_behavior_check.py [체크포인트] [--n 20000]
출력: docs/images/gate_behavior.png + 표준출력 + VERDICT 줄
"""
import argparse
import os

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import WeatherDataset, collect_historical, make_split
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PNG = "./docs/images/gate_behavior.png"

# record_to_vec() 의 특성 순서(train.py). 14차원 구성이다.
FEATURES = ["기온", "강수", "습도", "풍속", "풍향 sin", "풍향 cos", "기압",
            "강수형태", "시각 sin", "시각 cos", "위도", "경도",
            "연중시각 sin", "연중시각 cos"]

# 판정 기준 — 배분 표준편차가 이보다 작으면 사실상 정적이다.
MIN_DYNAMIC_STD = 0.01


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--n", type=int, default=20000)
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, DEVICE)
    model.eval()
    records = collect_historical()
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=TendencyCollector(records), lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False)
    val_idx = np.array(val_ds.indices)
    rng = np.random.RandomState(1234)
    idx = np.sort(rng.choice(val_idx, size=min(args.n, len(val_idx)), replace=False))
    X = ds.X_num[idx.tolist()].to(DEVICE)
    n_feat = X.shape[1]
    names = FEATURES[:n_feat] + [f"특성{i}" for i in range(len(FEATURES), n_feat)]
    print(f"체크포인트: {args.ckpt} · 표본 {len(idx):,}개 · Z축 {n_feat}차원")

    if not getattr(model, "dynamic_gate", True):
        print("이 체크포인트는 동적 게이트를 쓰지 않는다 — 점검 대상이 아니다.")
        return

    # ── 1. 배분 분포 ─────────────────────────────────────────
    with torch.no_grad():
        W = model.gate(X).cpu().numpy()          # (N, 3)
    print(f"\n{'=' * 80}\n 1. 축 배분 분포 (표본 {len(W):,})\n{'=' * 80}")
    print(f"  {'축':<6}{'평균':>9}{'표준편차':>10}{'최소':>9}{'5%':>9}"
          f"{'95%':>9}{'최대':>9}")
    for j, ax in enumerate(["Re", "Im", "Z"]):
        c = W[:, j]
        print(f"  {ax:<6}{c.mean():>9.3f}{c.std():>10.4f}{c.min():>9.3f}"
              f"{np.percentile(c, 5):>9.3f}{np.percentile(c, 95):>9.3f}{c.max():>9.3f}")
    dyn_std = float(W.std(axis=0).max())
    print(f"\n  배분이 표본마다 얼마나 달라지는가 — 최대 표준편차 {dyn_std:.4f}")
    print("  (0에 가까우면 사실상 정적 가중치이고, '입력 조건부'라는 서술이 과장이 된다)")

    # ── 2. 특성별 민감도 ─────────────────────────────────────
    # 한 특성만 5→95 백분위로 옮기고 나머지는 그대로 둔 채 배분 변화를 본다.
    print(f"\n{'=' * 80}\n 2. Z축 특성별 민감도 — 5%→95% 로 옮길 때 배분 변화폭"
          f"\n{'=' * 80}")
    print(f"  {'특성':<12}{'Δw_Re':>10}{'Δw_Im':>10}{'Δw_Z':>10}{'합계 변화폭':>13}")
    sens, undetermined = [], []
    Xc = X.clone()
    with torch.no_grad():
        for f in range(n_feat):
            lo = torch.quantile(X[:, f], 0.05)
            hi = torch.quantile(X[:, f], 0.95)
            # 5%와 95%가 같으면 흔들 폭이 없다 — "게이트가 무시한다"가 아니라
            # "측정이 성립하지 않는다". 둘을 구분하지 않으면 0.0000 을
            # 무시의 증거로 오독하게 된다(2026-08-17 강수형태에서 실제 발생).
            if torch.isclose(lo, hi):
                undetermined.append(names[f])
                sens.append(np.zeros(3))
                print(f"  {names[f]:<12}{'—':>10}{'—':>10}{'—':>10}"
                      f"{'측정 불가':>13}")
                continue
            Xc.copy_(X); Xc[:, f] = lo
            w_lo = model.gate(Xc).mean(dim=0).cpu().numpy()
            Xc.copy_(X); Xc[:, f] = hi
            w_hi = model.gate(Xc).mean(dim=0).cpu().numpy()
            d = w_hi - w_lo
            sens.append(d)
            print(f"  {names[f]:<12}{d[0]:>+10.4f}{d[1]:>+10.4f}{d[2]:>+10.4f}"
                  f"{np.abs(d).sum():>13.4f}")
    sens = np.array(sens)
    if undetermined:
        print(f"\n  측정 불가 {len(undetermined)}개: {', '.join(undetermined)}"
              f" — 표본의 5%와 95% 백분위가 같아 흔들 구간이 없다(대부분 한 값에 몰림).")
    order = np.argsort(-np.abs(sens).sum(axis=1))
    top = ", ".join(f"{names[i]}({np.abs(sens[i]).sum():.3f})" for i in order[:3])
    print(f"\n  배분을 가장 크게 움직이는 특성: {top}")

    # ── 3. 구조적 한계 ───────────────────────────────────────
    print(f"\n{'=' * 80}\n 3. 게이트가 볼 수 없는 것\n{'=' * 80}")
    print("  게이트의 입력은 Z축뿐이다(pipeline_model.py: w = self.gate(num_x)).")
    print("  따라서 Re축 보간장의 품질(이웃 관측이 몇 개나 있었는지, 결측이었는지)과")
    print("  Im축 경향의 품질은 배분에 반영될 수 없다 — 그런 정보가 입력에 없다.")
    print("  '축의 신뢰도에 따라 가중치를 조절한다'는 해석은 이 구조에서 성립하지")
    print("  않으며, 게이트가 하는 일은 '지금이 어떤 상황인가'에 따른 배분 전환이다.")

    # ── 그림 ─────────────────────────────────────────────────
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import font_manager
    from probability_calibration_check import _ensure_korean_font

    fp = _ensure_korean_font()
    if fp:
        font_manager.fontManager.addfont(fp)
        plt.rcParams["font.family"] = font_manager.FontProperties(fname=fp).get_name()
    plt.rcParams["axes.unicode_minus"] = False

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 6.2))
    ax = axes[0]
    for j, (ax_name, c) in enumerate(zip(["Re", "Im", "Z"],
                                         ["#1C6779", "#684A82", "#A96406"])):
        ax.hist(W[:, j], bins=60, alpha=0.6, label=f"{ax_name} (평균 {W[:, j].mean():.3f})",
                color=c)
    ax.set_xlabel("축 가중치", fontsize=13, fontweight="bold")
    ax.set_ylabel("표본 수", fontsize=13, fontweight="bold")
    ax.set_title("축 배분의 분포 — 표본마다 얼마나 달라지는가",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(alpha=0.25)

    ax = axes[1]
    y = np.arange(n_feat)
    for j, (ax_name, c) in enumerate(zip(["Re", "Im", "Z"],
                                         ["#1C6779", "#684A82", "#A96406"])):
        ax.barh(y + (j - 1) * 0.26, sens[:, j], 0.26, label=ax_name, color=c, alpha=0.88)
    ax.set_yticks(y)
    ax.set_yticklabels(names, fontsize=11)
    ax.invert_yaxis()
    ax.axvline(0, color="#444", lw=1.2)
    ax.set_xlabel("특성을 5%→95%로 옮길 때 배분 변화", fontsize=13, fontweight="bold")
    ax.set_title("무엇이 배분을 움직이는가 — Z축 특성별 민감도",
                 fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.tick_params(labelsize=11)
    ax.grid(axis="x", alpha=0.25)

    fig.suptitle("동적 게이트의 거동 — 게이트는 Z축만 보고 배분을 정한다",
                 fontsize=18, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    os.makedirs(os.path.dirname(OUT_PNG), exist_ok=True)
    fig.savefig(OUT_PNG, dpi=130, bbox_inches="tight")
    print(f"\n저장: {OUT_PNG}")

    verdict = "PASS" if dyn_std >= MIN_DYNAMIC_STD else "WARN"
    print(f"\nVERDICT gate_dynamics {verdict} max_std={dyn_std:.4f} "
          f"top_feature={names[order[0]]}")


if __name__ == "__main__":
    main()
