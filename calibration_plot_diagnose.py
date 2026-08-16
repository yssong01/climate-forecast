"""
calibration_plot_diagnose.py — 관측소별 재현율×정밀도 신뢰도 플롯을 만든다.

배경(2026-08-16, 사용자 제안): 극한기상 헤드 성능을 관측소별로 쪼개
가로축 정확도·세로축 정밀도, 신뢰구간을 십자형 에러바로 표시하자는
아이디어. 가로축을 정확도(accuracy) 그대로 쓰면 이 프로젝트가 이미
정한 규약(CLAUDE.md 검증 규약 "양성이 희박한 이벤트는 accuracy를 쓰지
않는다")과 충돌한다 — 한파(6.8%)·황사(2.7%)처럼 양성이 드물면 accuracy가
거의 항상 90%대로 나와 변별력이 없다. 그래서 가로축을 재현율(recall)로
바꿨다 — 정밀도·재현율은 이 프로젝트가 이미 쓰는 지표쌍이다.

에러바는 이항분포 신뢰구간(Wilson score, 95%)이다 — 지어낸 값이 아니라
표본 수에서 바로 계산되는 실측 불확실성이다. 표본이 적은 관측소일수록
에러바가 커진다(=신뢰도가 낮다는 뜻을 그대로 반영).

한글 라벨(2026-08-16 추가): 컨테이너 기본 이미지엔 한글 글리프가 있는
폰트가 전혀 없다(fc-list 0건 실측) — 라벨을 한글로 쓰면 네모(tofu)로
깨진다. 이 스크립트가 실행 시점에 `fonts-nanum`을 직접 설치하고
matplotlib 폰트로 등록한다(매번 재설치 — 이미지에 굽지 않고 컨테이너는
`--rm`으로 매번 새로 뜨므로).

실행: python calibration_plot_diagnose.py [--out calibration.png]
"""
import argparse
import math
import subprocess
import sys

import numpy as np
import torch

from train import collect_historical, WeatherDataset, make_split, STATION_NAMES
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from predict import load_model

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024

_NANUM_PATH = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"


def _ensure_korean_font():
    """matplotlib 라벨용 한글 폰트를 확보한다. 없으면 설치 시도, 실패하면 None."""
    import os
    if os.path.exists(_NANUM_PATH):
        return _NANUM_PATH
    try:
        subprocess.run(["apt-get", "update", "-qq"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.run(["apt-get", "install", "-y", "-qq", "fonts-nanum"], check=True,
                       stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
                       env={**os.environ, "DEBIAN_FRONTEND": "noninteractive"})
    except Exception as e:
        print(f"폰트 설치 실패({e}) — root 권한·네트워크 필요", file=sys.stderr)
        return None
    return _NANUM_PATH if os.path.exists(_NANUM_PATH) else None


def wilson_ci(k: int, n: int, z: float = 1.96):
    """이항비율 k/n 의 Wilson score 95% 신뢰구간 (정규근사보다 소표본에 안전)."""
    if n == 0:
        return 0.0, 0.0, 0.0
    p = k / n
    denom = 1 + z * z / n
    center = (p + z * z / (2 * n)) / denom
    half = (z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n))) / denom
    return p, max(0.0, center - half), min(1.0, center + half)


def prf(pred_pos, labels):
    tp = int((pred_pos & (labels == 1)).sum())
    fp = int((pred_pos & (labels == 0)).sum())
    fn = int((~pred_pos & (labels == 1)).sum())
    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="./docs/images/calibration_plot.png")
    args = ap.parse_args()

    model, ckpt = load_model()
    records = collect_historical()
    txt = TendencyCollector(records)
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=True)
    idx = np.array(val_ds.indices)
    stns = np.array([ds.stns[i] for i in idx])

    heat_p, cold_p = [], []
    model.eval()
    with torch.no_grad():
        for s in range(0, len(idx), BATCH):
            sl = idx[s:s + BATCH]
            model(num_x=ds.X_num[sl].to(DEVICE), img_x=ds.X_img[sl].to(DEVICE).float(),
                  txt_x=ds.X_txt[sl].to(DEVICE))
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).cpu().squeeze(-1))
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).cpu().squeeze(-1))
    heat_p = torch.cat(heat_p).numpy()
    cold_p = torch.cat(cold_p).numpy()
    heat_y = ds.y_heatwave[idx].numpy()
    cold_y = ds.y_coldwave[idx].numpy()
    heat_m = ds.heat_mask[idx].numpy().astype(bool)
    cold_m = ds.cold_mask[idx].numpy().astype(bool)

    results = {"heatwave": [], "coldwave": []}
    for name, p, y, m in (("heatwave", heat_p, heat_y, heat_m),
                          ("coldwave", cold_p, cold_y, cold_m)):
        for stn_code in sorted(set(stns)):
            sel = m & (stns == stn_code)
            n = int(sel.sum())
            if n < 30:
                continue
            pred_pos = p[sel] >= 0.5
            labels = y[sel]
            precision, recall, tp, fp, fn = prf(pred_pos, labels)
            n_pos = int((labels == 1).sum())
            # Wilson CI — precision 은 "양성 예측" 모집단 기준, recall 은 "실제 양성" 모집단 기준
            _, p_lo, p_hi = wilson_ci(tp, tp + fp)
            _, r_lo, r_hi = wilson_ci(tp, tp + fn)
            results[name].append({
                "station": STATION_NAMES.get(stn_code, stn_code),
                "code": stn_code,
                "n": n, "n_pos": n_pos,
                "precision": precision, "p_lo": p_lo, "p_hi": p_hi,
                "recall": recall, "r_lo": r_lo, "r_hi": r_hi,
            })

    print("=" * 90)
    print(" 관측소별 재현율×정밀도 (t=0.5, Wilson 95% CI)")
    print("=" * 90)
    for name, rows in results.items():
        print(f"\n[{name}]")
        print(f"{'관측소':<6} {'표본':>7} {'양성':>6} | "
              f"{'재현율':>7} {'CI':>16} | {'정밀도':>7} {'CI':>16}")
        for r in sorted(rows, key=lambda x: -x["n_pos"]):
            print(f"{r['station']:<6} {r['n']:>7,} {r['n_pos']:>6,} | "
                  f"{r['recall']:>6.1%} [{r['r_lo']:.1%},{r['r_hi']:.1%}] | "
                  f"{r['precision']:>6.1%} [{r['p_lo']:.1%},{r['p_hi']:.1%}]")

    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.font_manager as fm
    except ImportError:
        print("\nmatplotlib 없음 — 표만 출력하고 종료(플롯 생략)")
        return

    korean_font = _ensure_korean_font()
    if korean_font:
        fm.fontManager.addfont(korean_font)
        plt.rcParams["font.family"] = fm.FontProperties(fname=korean_font).get_name()
        plt.rcParams["axes.unicode_minus"] = False  # 한글 폰트는 유니코드 마이너스 글리프가 없다
    else:
        print("\n한글 폰트 설치 실패 — 관측소 코드만 라벨로 사용")

    fig, axes = plt.subplots(1, 2, figsize=(13, 6))
    colors = {"heatwave": "#E2954F", "coldwave": "#4C78A8"}
    labels_ko = {"heatwave": "폭염", "coldwave": "한파"}
    for ax, name in zip(axes, ["heatwave", "coldwave"]):
        rows = results[name]
        for r in rows:
            x, y = r["recall"], r["precision"]
            xerr = [[x - r["r_lo"]], [r["r_hi"] - x]]
            yerr = [[y - r["p_lo"]], [r["p_hi"] - y]]
            size = 20 + 4 * math.sqrt(r["n_pos"])
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="+",
                        color=colors[name], markersize=size / 5,
                        markeredgewidth=2, capsize=4, elinewidth=1.5, alpha=0.85)
            label = f"{r['station']}({r['code']})" if korean_font else r["code"]
            ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=(6, 6), fontsize=8)
        ax.set_xlabel("재현율 (Recall)")
        ax.set_ylabel("정밀도 (Precision)")
        ax.set_title(f"{labels_ko[name]} — 관측소별 (십자 = Wilson 95% 신뢰구간)")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.grid(alpha=0.3)

    fig.tight_layout()
    fig.savefig(args.out, dpi=130)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
