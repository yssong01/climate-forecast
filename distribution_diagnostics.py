"""
distribution_diagnostics.py — 학습이 만들어낸 값들이 실제로 어떤 분포인지 검증.

무엇을 왜 보는가:

  게이트 배분 (w_re, w_im, w_z) — [0,1] 구간, 셋의 합이 항상 1이다. 이건
  구조적으로 디리클레(Dirichlet) 분포의 표본이고, 디리클레의 각 성분을
  따로 떼어보면 정확히 베타(Beta) 분포를 따른다 — 그래서 여기서 베타
  피팅은 임의 선택이 아니라 게이트의 실제 구조(softmax over 3, 합=1)에서
  나오는 정칙 형태다. 정규분포로 피팅하면 [0,1] 경계 밖에 확률질량이
  생겨 구조적으로 틀린다.

  위상각 θ — 한계 6(README)에서 θ = atan2(w_im, w_re) 로 수학적으로
  퇴화함을 증명했다. 그 증명이 맞다면 θ의 분포는 w_im/w_re 비율 분포를
  그대로 반영해야 하고, 폭넓게 퍼진 자유 위상이 아니라 게이트가 실제로
  쓰는 좁은 구간에 몰려 있어야 한다 — 수식 증명을 분포 형태로 다시 확인.

  축 간 코사인 (cos_re_im 등) — 0 근처에 대칭이면 축이 서로 독립적으로
  움직인다는 뜻. 정규분포 피팅 + 평균이 0에서 얼마나 벗어나는지가
  "이 두 축이 진짜 다른 정보를 담는가"의 직접적인 근거가 된다.

  예측 잔차 (기온·강수 오차) — MAE 하나로 요약하면 오차 분포의 모양은
  안 보인다. 정규분포면 MAE·표준편차만으로 오차 구조를 다 설명할 수
  있지만, 두꺼운 꼬리(heavy tail)면 극단 사례에서 훨씬 크게 틀리고
  있다는 뜻이고 이는 Tri-CHEF의 '변별력' 주장과 직결된다 — 융합이
  극단 사례를 더 잘 판별한다면 잔차 분포의 꼬리가 얇아야 한다.

각 분포마다 Kolmogorov-Smirnov 검정을 함께 낸다. p < 0.05 면 "이 분포가
맞다"는 가설을 기각한다 — 즉 그 이론적 분포와 실제로 다르다는 증거다.
표본이 수만 개 규모라 아주 작은 편차도 통계적으로 유의해지기 쉽다는
점은 감안해서 읽어야 한다(유의성 ≠ 실질적 크기).

실행:
    python distribution_diagnostics.py
    → ./checkpoints/distribution_diagnostics.png 저장 + 콘솔에 수치 요약
"""
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy import stats

from train import collect_historical, WeatherDataset, VAL_RATIO, SEED, make_split
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from weather_collector import STATION_COORDS
from text_collector import SimulatedTextCollector
from predict import load_model, CHECKPOINT

DEVICE  = "cuda" if torch.cuda.is_available() else "cpu"
OUT_PNG = "./checkpoints/distribution_diagnostics.png"
BATCH   = 1024


def fit_normal(data: np.ndarray) -> dict:
    mu, sigma = stats.norm.fit(data)
    ks_stat, ks_p = stats.kstest(data, "norm", args=(mu, sigma))
    return {"dist": "Normal", "params": f"μ={mu:.4f}, σ={sigma:.4f}",
            "ks_stat": ks_stat, "ks_p": ks_p,
            "curve": lambda x: stats.norm.pdf(x, mu, sigma)}


def fit_beta(data: np.ndarray) -> dict:
    eps = 1e-6
    d = np.clip(data, eps, 1 - eps)
    a, b, _, _ = stats.beta.fit(d, floc=0, fscale=1)
    ks_stat, ks_p = stats.kstest(d, "beta", args=(a, b, 0, 1))
    return {"dist": "Beta", "params": f"α={a:.2f}, β={b:.2f}",
            "ks_stat": ks_stat, "ks_p": ks_p,
            "curve": lambda x: stats.beta.pdf(x, a, b)}


def panel(ax, data: np.ndarray, title: str, fit: dict, bins: int = 60):
    ax.hist(data, bins=bins, density=True, color="#4C78A8", alpha=0.75,
           edgecolor="none")
    xs = np.linspace(data.min(), data.max(), 300)
    ax.plot(xs, fit["curve"](xs), color="#E2954F", linewidth=2)
    sig = "✓" if fit["ks_p"] >= 0.05 else "✗"
    ax.set_title(
        f"{title}\n{fit['dist']}({fit['params']}) — KS p={fit['ks_p']:.1e} {sig}",
        fontsize=9,
    )
    ax.tick_params(labelsize=8)


def main():
    model, ckpt = load_model(CHECKPOINT, DEVICE)
    print(f"체크포인트 — 기온MAE {ckpt['val_temp_mae']:.3f}°C  "
          f"강수MAE {ckpt['val_precip_mae']:.3f}mm  (+{ckpt['lead_hours']}h)")

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
    # 학습 때와 동일한 시드로 분할해 검증셋만 뽑는다 — 학습에 쓰인 표본을
    # 분포 진단에 섞으면 과적합된 부분까지 "정상"으로 보게 된다.
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False)
    val_idx = val_ds.indices
    print(f"검증 표본: {len(val_idx)}개")

    buf = {k: [] for k in ("w_re", "w_im", "w_z", "theta",
                           "cos_re_im", "cos_re_z", "cos_im_z",
                           "temp_err", "precip_err",
                           "temp_pred", "temp_actual",
                           "precip_pred", "precip_actual")}
    stns  = [ds.stns[i] for i in val_idx]
    stimes = [ds.src_timestamps[i] for i in val_idx]
    ttimes = [ds.tgt_timestamps[i] for i in val_idx]

    model.eval()
    with torch.no_grad():
        for b in range(0, len(val_idx), BATCH):
            idx = val_idx[b:b + BATCH]
            x_num = ds.X_num[idx].to(DEVICE)
            x_img = ds.X_img[idx].to(DEVICE).float()
            x_txt = ds.X_txt[idx].to(DEVICE)
            y = ds.y[idx]

            pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt).cpu()
            rep = model.axis_report(x_num, x_img, x_txt)

            buf["w_re"].append(rep["gate"][:, 0].cpu().numpy())
            buf["w_im"].append(rep["gate"][:, 1].cpu().numpy())
            buf["w_z"].append(rep["gate"][:, 2].cpu().numpy())
            buf["theta"].append(rep["theta"].cpu().numpy())
            buf["cos_re_im"].append(rep["cos_re_im"].cpu().numpy())
            buf["cos_re_z"].append(rep["cos_re_z"].cpu().numpy())
            buf["cos_im_z"].append(rep["cos_im_z"].cpu().numpy())
            buf["temp_err"].append((pred[:, 0] - y[:, 0]).numpy())
            buf["precip_err"].append((pred[:, 1] - y[:, 1]).numpy())
            buf["temp_pred"].append(pred[:, 0].numpy())
            buf["temp_actual"].append(y[:, 0].numpy())
            buf["precip_pred"].append(pred[:, 1].numpy())
            buf["precip_actual"].append(y[:, 1].numpy())

    d = {k: np.concatenate(v) for k, v in buf.items()}

    # matplotlib 기본 폰트(DejaVu Sans)에 한글 글리프가 없어 그림 안 라벨은
    # 전부 깨진다(콘솔 출력은 터미널이 처리하니 문제없어 한글 유지). 이미지에
    # 한글 폰트를 새로 설치하는 대신 라벨만 영문/기호로 바꾼다 — 재빌드 없이
    # 바로 해결되고, 이 스크립트의 실제 소비자(개발자 콘솔)에게는 어차피
    # 표 형태 콘솔 출력이 주 정보원이다.
    fig, axes = plt.subplots(3, 3, figsize=(15, 12))
    fig.suptitle("Tri-CHEF Distribution Diagnostics — Validation Set",
                fontsize=13, fontweight="bold")

    specs = [
        ("w_re", "gate w_re (satellite)", fit_beta),
        ("w_im", "gate w_im (text)", fit_beta),
        ("w_z",  "gate w_z (numeric)",  fit_beta),
        ("theta", "phase angle θ (rad) — expect degenerate (limit 6)", fit_normal),
        ("cos_re_im", "axis cosine Re-Im", fit_normal),
        ("cos_re_z",  "axis cosine Re-Z",  fit_normal),
        ("cos_im_z",  "axis cosine Im-Z",  fit_normal),
        ("temp_err",   "temp residual (°C)", fit_normal),
        ("precip_err", "precip residual (mm)", fit_normal),
    ]

    print(f"\n{'항목':<14} {'분포':<6} {'파라미터':<22} {'KS p-value':>12} {'적합':>6}")
    print("-" * 66)
    for ax, (key, title, fitter) in zip(axes.flat, specs):
        fit = fitter(d[key])
        panel(ax, d[key], title, fit)
        sig = "적합" if fit["ks_p"] >= 0.05 else "부적합"
        print(f"{key:<14} {fit['dist']:<6} {fit['params']:<22} "
              f"{fit['ks_p']:>12.2e} {sig:>6}")

    plt.tight_layout(rect=[0, 0, 1, 0.96])
    plt.savefig(OUT_PNG, dpi=130)
    print(f"\n저장: {OUT_PNG}")

    # 게이트 배분 합=1 검증 — 디리클레 표본이 맞다면 항상 성립해야 한다.
    total = d["w_re"] + d["w_im"] + d["w_z"]
    print(f"\n게이트 배분 합 검증: 평균 {total.mean():.6f} "
          f"(1.0에서 벗어남 {abs(total.mean()-1.0):.2e})")

    # ── 잔차 분포의 실질적 모양 — 정규분포 σ만으로는 안 보이는 것 ──
    #
    # KS 검정과 별개로, 정규분포였다면 MAE ≈ σ·√(2/π) ≈ 0.7979σ 여야 한다.
    # 이 배율에서 크게 벗어난다는 건 히스토그램이 뾰족한 중심(대부분 표본이
    # 아주 작은 오차) + 얇지만 긴 꼬리(소수 극단 표본)로 이뤄져 있다는
    # 뜻이다 — 그림 1의 x축이 비정상적으로 넓게 뻗은 이유이기도 하다.
    for key, unit in (("temp_err", "°C"), ("precip_err", "mm")):
        e = d[key]
        sigma = e.std()
        mae = np.abs(e).mean()
        implied_mae = sigma * np.sqrt(2 / np.pi)
        pct = np.percentile(np.abs(e), [50, 90, 99, 99.9])
        print(f"\n[{key}] MAE={mae:.4f}{unit} vs 정규분포라면 기대되는 "
              f"MAE={implied_mae:.4f}{unit} (σ={sigma:.4f} 기준, "
              f"배율 {mae/implied_mae:.2f}x)")
        print(f"  절대오차 백분위수 — p50={pct[0]:.3f} p90={pct[1]:.3f} "
              f"p99={pct[2]:.3f} p99.9={pct[3]:.3f} {unit}")

    # ── 극단 잔차 Top-10 — 실제로 어떤 관측소·시각에서 크게 틀렸는가 ──
    from weather_collector import STATIONS
    names = {v: k for k, v in STATIONS.items()}
    for key, pred_key, actual_key, unit in (
        ("temp_err", "temp_pred", "temp_actual", "°C"),
        ("precip_err", "precip_pred", "precip_actual", "mm"),
    ):
        order = np.argsort(-np.abs(d[key]))[:10]
        print(f"\n[{key}] 절대오차 상위 10건:")
        print(f"  {'관측소':<6} {'관측시각':<14} {'목표시각':<14} "
              f"{'예측':>8} {'실측':>8} {'오차':>8}")
        for i in order:
            print(f"  {names.get(stns[i], stns[i]):<6} {stimes[i]:<14} "
                  f"{ttimes[i]:<14} {d[pred_key][i]:>8.2f} "
                  f"{d[actual_key][i]:>8.2f} {d[key][i]:>8.2f}")


if __name__ == "__main__":
    main()
