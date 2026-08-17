"""
error_breakdown.py — 회귀 오차를 구간별로 쪼개 "어디서 지고 있는가"를 특정한다.

**왜 필요한가.** 강수 MAE 가 기준선(상시 0mm)에 −9.9% 미달인 것은 알지만,
그 격차가 어느 구간에서 생기는지는 재본 적이 없다. 처방이 정반대이기 때문에
이 구분이 중요하다.

  · 무강수 구간에서 진다면 → 모델이 0 을 못 내는 문제다. 후처리 임계값이나
    허들의 확률 항을 손봐야 한다.
  · 강수 구간에서 진다면 → 강수량 자체를 못 맞히는 문제다. 표현력·입력을
    손봐야 하고, 트렁크 분리 같은 구조 변경이 후보가 된다.

**분해 방식.** 전체 MAE 는 구간별 MAE 의 표본 수 가중평균이므로,
각 구간의 기여도를 `n_i / N × MAE_i` 로 정확히 분해할 수 있다. 기준선도 같은
방식으로 분해하면 두 기여도의 차이가 곧 그 구간이 만든 격차다 — 이 값들의
합은 전체 격차와 일치한다(검산으로 확인한다).

실행: python error_breakdown.py [체크포인트]
출력: docs/images/error_breakdown.png + cache/error_breakdown.json + 표준출력
"""
import argparse
import json
import os

import numpy as np

import eval_cache
from predict import CHECKPOINT, PRECIP_CLIP_THRESH

OUT_PNG = "./docs/images/error_breakdown.png"
OUT_JSON = "./cache/error_breakdown.json"

STATION_NAMES = {
    "101": "춘천", "105": "강릉", "108": "서울", "112": "인천", "119": "수원",
    "131": "청주", "133": "대전", "143": "대구", "146": "전주", "156": "광주",
    "159": "부산", "184": "제주",
}


def segments(d):
    """구간 정의 — (이름, 라벨 배열, 표시 순서)."""
    ts = d["tgt_ts"].astype(str)
    month = np.array([int(t[4:6]) for t in ts])
    hour = np.array([int(t[8:10]) for t in ts])
    pt = d["precip_true"]
    tn = d["temp_now"]

    season_map = {12: "겨울", 1: "겨울", 2: "겨울", 3: "봄", 4: "봄", 5: "봄",
                  6: "여름", 7: "여름", 8: "여름", 9: "가을", 10: "가을", 11: "가을"}
    season = np.array([season_map[m] for m in month])

    hour_band = np.where(hour < 6, "00-05",
                np.where(hour < 12, "06-11",
                np.where(hour < 18, "12-17", "18-23")))

    temp_band = np.where(tn < -5, "-5°C 미만",
                np.where(tn < 5, "-5~5",
                np.where(tn < 15, "5~15",
                np.where(tn < 25, "15~25", "25°C 이상"))))

    rain_band = np.where(pt < 0.1, "무강수",
                np.where(pt < 1.0, "0.1~1mm",
                np.where(pt < 5.0, "1~5mm",
                np.where(pt < 20.0, "5~20mm", "20mm 이상"))))

    station = np.array([STATION_NAMES.get(s, s) for s in d["stn"].astype(str)])

    return [
        ("강수 강도", rain_band, ["무강수", "0.1~1mm", "1~5mm", "5~20mm", "20mm 이상"]),
        ("계절", season, ["봄", "여름", "가을", "겨울"]),
        ("시각", hour_band, ["00-05", "06-11", "12-17", "18-23"]),
        ("현재 기온", temp_band, ["-5°C 미만", "-5~5", "5~15", "15~25", "25°C 이상"]),
        ("관측소", station, [STATION_NAMES[k] for k in sorted(STATION_NAMES)]),
    ]


def decompose(err_model, err_base, labels, order):
    """구간별 표본 수·MAE·전체 MAE 기여도·격차 기여도를 계산한다."""
    n_all = len(err_model)
    rows = []
    for lab in order:
        m = labels == lab
        k = int(m.sum())
        if k == 0:
            continue
        mae_m = float(err_model[m].mean())
        mae_b = float(err_base[m].mean())
        rows.append({
            "label": lab, "n": k, "share": k / n_all,
            "mae_model": mae_m, "mae_base": mae_b,
            "contrib_model": k / n_all * mae_m,      # 합 = 전체 모델 MAE
            "contrib_base": k / n_all * mae_b,       # 합 = 전체 기준선 MAE
            "contrib_gap": k / n_all * (mae_m - mae_b),   # 합 = 전체 격차
        })
    return rows


def print_table(title, rows, unit, total_gap):
    print(f"\n{'─' * 92}\n  {title}\n{'─' * 92}")
    print(f"  {'구간':<12}{'표본':>10}{'비중':>8}{'모델 MAE':>11}{'기준선':>11}"
          f"{'모델 기여':>11}{'격차 기여':>11}{'격차 점유':>10}")
    for r in rows:
        share_of_gap = (r["contrib_gap"] / total_gap * 100) if abs(total_gap) > 1e-12 else 0.0
        mark = "  ←" if r["contrib_gap"] > 0 and share_of_gap > 20 else ""
        print(f"  {r['label']:<12}{r['n']:>10,}{r['share']:>8.1%}"
              f"{r['mae_model']:>11.4f}{r['mae_base']:>11.4f}"
              f"{r['contrib_model']:>11.5f}{r['contrib_gap']:>+11.5f}"
              f"{share_of_gap:>9.1f}%{mark}")
    print(f"  {'합계':<12}{sum(r['n'] for r in rows):>10,}{'':>8}"
          f"{'':>11}{'':>11}{sum(r['contrib_model'] for r in rows):>11.5f}"
          f"{sum(r['contrib_gap'] for r in rows):>+11.5f}")


def clip_sweep(pp_raw, pt, seed=1234, n_grid=61):
    """후처리 클리핑 임계값을 재산출한다.

    무강수 구간이 격차의 대부분을 만든다면, 그 구간의 미세 출력을 0으로
    반올림하는 임계값이 얼마여야 하는지가 직접적인 처방이 된다. 다만 검증셋
    전체에서 고르고 같은 데이터로 채점하면 허수가 섞이므로, 임계값 선정
    규약과 동일하게 보정용/평가용을 나누고 순이득이 기준에 못 미치면
    채택하지 않는다(`threshold_validation.py` 참고).
    """
    rng = np.random.RandomState(seed)
    m = rng.rand(len(pt)) < 0.5
    grid = np.linspace(0.0, 3.0, n_grid)

    def mae(t, sel):
        return float(np.abs(np.where(pp_raw[sel] < t, 0.0, pp_raw[sel]) - pt[sel]).mean())

    def wet_f1(t, sel):
        """강수 발생 판정(0.1mm 기준)의 F1 — MAE 최적화가 이 능력을 없애는지 본다."""
        pred = np.where(pp_raw[sel] < t, 0.0, pp_raw[sel]) >= 0.1
        true = pt[sel] >= 0.1
        tp = int((pred & true).sum()); fp = int((pred & ~true).sum())
        fn = int((~pred & true).sum())
        p = tp / (tp + fp) if tp + fp else 0.0
        r = tp / (tp + fn) if tp + fn else 0.0
        return 2 * p * r / (p + r) if p + r else 0.0

    best_t = min(grid, key=lambda t: mae(t, m))
    base = float(np.abs(pt[~m]).mean())
    return {
        "current": PRECIP_CLIP_THRESH,
        "picked": float(best_t),
        "eval_mae_current": mae(PRECIP_CLIP_THRESH, ~m),
        "eval_mae_picked": mae(best_t, ~m),
        "eval_mae_raw": mae(0.0, ~m),
        "eval_baseline": base,
        # MAE 만 보면 "거의 항상 0" 이 이긴다 — 표본의 93.8% 가 무강수이기
        # 때문이다. 그 해가 강수 발생 판정을 없애버리는지 함께 잰다.
        "wet_f1_raw": wet_f1(0.0, ~m),
        "wet_f1_current": wet_f1(PRECIP_CLIP_THRESH, ~m),
        "wet_f1_picked": wet_f1(best_t, ~m),
        "n_calib": int(m.sum()), "n_eval": int((~m).sum()),
    }


def plot(precip_rows, temp_rows, totals, out_png):
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

    n = len(precip_rows)
    fig, axes = plt.subplots(2, n, figsize=(4.6 * n, 9.0))
    if n == 1:
        axes = axes.reshape(2, 1)

    for j, ((title, rows), (ttitle, trows)) in enumerate(zip(precip_rows, temp_rows)):
        # 윗줄 — 강수 격차 기여도
        ax = axes[0, j]
        labs = [r["label"] for r in rows]
        vals = [r["contrib_gap"] for r in rows]
        colors = ["#95372A" if v > 0 else "#2C6849" for v in vals]
        ax.barh(labs[::-1], vals[::-1], color=colors[::-1], alpha=0.88)
        ax.axvline(0, color="#444", lw=1.2)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v:+.4f}", va="center",
                    ha="left" if v >= 0 else "right", fontsize=9)
        ax.set_title(f"강수 격차 — {title}", fontsize=13, fontweight="bold")
        ax.set_xlabel("기준선 대비 격차 기여 (mm)", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(axis="x", alpha=0.25)

        # 아랫줄 — 기온 MAE 기여도
        ax = axes[1, j]
        labs = [r["label"] for r in trows]
        vals = [r["contrib_model"] for r in trows]
        ax.barh(labs[::-1], vals[::-1], color="#4C78A8", alpha=0.85)
        for i, v in enumerate(vals[::-1]):
            ax.text(v, i, f" {v:.3f}", va="center", ha="left", fontsize=9)
        ax.set_title(f"기온 MAE 기여 — {ttitle}", fontsize=13, fontweight="bold")
        ax.set_xlabel("전체 MAE에 대한 기여 (°C)", fontsize=11, fontweight="bold")
        ax.tick_params(labelsize=10)
        ax.grid(axis="x", alpha=0.25)

    fig.suptitle(
        f"오차 분해 — 강수 격차 {totals['precip_gap']:+.4f}mm "
        f"({totals['precip_gap_pct']:+.1f}%) 가 어느 구간에서 생기는가",
        fontsize=17, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.955])
    os.makedirs(os.path.dirname(out_png), exist_ok=True)
    fig.savefig(out_png, dpi=130, bbox_inches="tight")
    print(f"\n저장: {out_png}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--batch", type=int, default=eval_cache.DEFAULT_BATCH)
    ap.add_argument("--served", action="store_true",
                    help="서빙 후처리(0.2mm 미만 → 0)를 적용한 값으로 분해한다")
    args = ap.parse_args()

    d = eval_cache.load(args.ckpt, args.batch)
    pp = d["precip_pred"]
    if args.served:
        pp = np.where(pp < PRECIP_CLIP_THRESH, 0.0, pp)
    pt, tp_, tt = d["precip_true"], d["temp_pred"], d["temp_true"]

    err_p_model = np.abs(pp - pt)
    err_p_base = np.abs(pt)                      # 상시 0mm 산출
    err_t_model = np.abs(tp_ - tt)
    err_t_base = d["temp_persist_abs"]           # T(t+6) ≈ T(t)

    totals = {
        "n": int(len(pt)),
        "precip_mae": float(err_p_model.mean()),
        "precip_base": float(err_p_base.mean()),
        "precip_gap": float(err_p_model.mean() - err_p_base.mean()),
        "precip_gap_pct": float((err_p_model.mean() - err_p_base.mean())
                                / err_p_base.mean() * 100),
        "temp_mae": float(err_t_model.mean()),
        "temp_base": float(err_t_base.mean()),
        "served_postprocess": bool(args.served),
    }
    print(f"\n{'=' * 92}")
    print(f" 오차 분해 — 검증 표본 {totals['n']:,}개"
          f"{' · 서빙 후처리 적용' if args.served else ' · 후처리 전(문서 기준)'}")
    print(f"{'=' * 92}")
    print(f"  강수  모델 {totals['precip_mae']:.4f}mm  vs  기준선 "
          f"{totals['precip_base']:.4f}mm  →  격차 {totals['precip_gap']:+.4f}mm "
          f"({totals['precip_gap_pct']:+.1f}%)")
    print(f"  기온  모델 {totals['temp_mae']:.4f}°C  vs  기준선 "
          f"{totals['temp_base']:.4f}°C  →  개선 "
          f"{(totals['temp_mae'] - totals['temp_base']) / totals['temp_base'] * 100:+.1f}%")

    precip_rows, temp_rows, out = [], [], {"totals": totals, "segments": {}}
    for name, labels, order in segments(d):
        rp = decompose(err_p_model, err_p_base, labels, order)
        rt = decompose(err_t_model, err_t_base, labels, order)
        print_table(f"강수 · {name}", rp, "mm", totals["precip_gap"])
        precip_rows.append((name, rp))
        temp_rows.append((name, rt))
        out["segments"][name] = {"precip": rp, "temp": rt}

    # 검산 — 구간 기여도의 합이 전체 격차와 일치해야 한다.
    chk = sum(r["contrib_gap"] for r in precip_rows[0][1])
    print(f"\n  검산: 구간 기여도 합 {chk:+.6f} vs 전체 격차 "
          f"{totals['precip_gap']:+.6f} — "
          f"{'일치' if abs(chk - totals['precip_gap']) < 1e-6 else '★불일치★'}")

    # ── 후처리 임계값 재산출 ─────────────────────────────────
    sw = clip_sweep(d["precip_pred"], pt)
    gain = sw["eval_mae_current"] - sw["eval_mae_picked"]
    print(f"\n{'─' * 92}\n  후처리 클리핑 임계값 재산출 "
          f"(보정용 {sw['n_calib']:,} / 평가용 {sw['n_eval']:,})\n{'─' * 92}")
    print(f"  {'설정':<22}{'평가용 MAE':>12}{'기준선 대비':>12}{'강수 발생 F1':>14}")
    for tag, t, mae_v, f1_v in (
            ("후처리 없음(t=0)", 0.0, sw["eval_mae_raw"], sw["wet_f1_raw"]),
            (f"현행 t={sw['current']:.2f}", sw["current"],
             sw["eval_mae_current"], sw["wet_f1_current"]),
            (f"보정용 선정 t={sw['picked']:.2f}", sw["picked"],
             sw["eval_mae_picked"], sw["wet_f1_picked"])):
        print(f"  {tag:<22}{mae_v:>12.4f}"
              f"{(mae_v / sw['eval_baseline'] - 1) * 100:>11.1f}%{f1_v:>14.3f}")
    f1_drop = sw["wet_f1_current"] - sw["wet_f1_picked"]
    print(f"\n  MAE 순이득 {gain:+.5f}mm · 강수 발생 F1 변화 {-f1_drop:+.3f}")
    if gain > 1e-4 and f1_drop > 0.02:
        print("  판정: 채택하지 않는다 — MAE 이득이 '거의 항상 0'으로 기준선을")
        print("        흉내 내서 얻어진 것이고, 그 대가로 강수 발생 판정이 무너진다.")
        print("        표본의 93.8%가 무강수라 MAE 단독 최적화는 이 방향으로 끌린다.")
    elif gain > 1e-4:
        print("  판정: 검토 대상 — 발생 판정을 크게 해치지 않으면서 MAE가 개선된다.")
    else:
        print("  판정: 현행 유지.")
    out["clip_sweep"] = sw

    plot(precip_rows, temp_rows, totals, OUT_PNG)
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)
    print(f"저장: {OUT_JSON}")


if __name__ == "__main__":
    main()
