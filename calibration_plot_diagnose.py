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

사건 단위 보정(2026-08-16 추가): 처음엔 시간 표본 하나하나를 독립 시행으로
놓고 신뢰구간을 계산했는데, 한파·폭염은 며칠씩 이어지는 사건이라 같은
사건 안의 연속 시간 표본은 서로 강하게 연관돼 있다(독립이 아니다). 이러면
신뢰구간이 실제보다 좁게 나온다. `station_anomaly_investigate.py`로
확인해보니 대구·강릉은 시간 표본(216~384건)만 보면 적지 않아 보였지만
실제 사건 수는 8~10개뿐이었다(춘천 40개와 대조적). 그래서 신뢰구간은
"시간 표본 수"가 아니라 "사건 수"를 유효 표본 크기로 써서 계산한다
(점 추정값인 재현율·정밀도 자체는 그대로 시간 단위로 잰다 — 이 프로젝트의
다른 F1 표와 정의를 맞추기 위해서다. 넓어지는 건 신뢰구간뿐이다).

한글 라벨(2026-08-16 추가): 컨테이너 기본 이미지엔 한글 글리프가 있는
폰트가 전혀 없다(fc-list 0건 실측) — 라벨을 한글로 쓰면 네모(tofu)로
깨진다. 이 스크립트가 실행 시점에 `fonts-nanum`을 직접 설치하고
matplotlib 폰트로 등록한다(매번 재설치 — 이미지에 굽지 않고 컨테이너는
`--rm`으로 매번 새로 뜨므로).

서빙 판정선 기준으로 채점한다(2026-09-23 변경). 종전에는 보정 전 원본
확률 t=0.5 로 채점했는데, 화면에서 이 플롯 **바로 위**에 놓이는 극한기상
분류 성능표는 2026-09-07부터 서빙 판정선(보정 후 공간, 폭염 0.402 ·
한파 0.330) 기준이다. 같은 화면에서 "합산 표를 관측소별로 분해한 것"이라
설명하면서 둘이 서로 다른 동작점을 가리키고 있었다 — 실제로 부산 폭염이
표의 설명과 어긋나는 값으로 보였다. 채점 표본도 `*_mask_official` 로
맞춘다: 특보 비운영기간을 확정 음성으로 채운 체크포인트에서는 쉬운 음성이
대량으로 섞여 정밀도가 부풀고, 그러면 위 표와 **같은 질문에 답한 값**이
아니게 된다(`metrics_report.precision_block` 과 같은 이유).

산출물을 JSON 으로도 남긴다(2026-09-23). 종전에는 표를 표준출력으로만
찍어서, `app.py` 의 관측소별 성능 캡션을 사람이 손으로 옮겨 적었다 —
그 결과 모델을 교체해도 캡션만 이전 세대 값으로 남았다(부산 폭염
"재현율 63.4%·정밀도 54.0%"). 화면이 이 파일을 읽어 문장을 만들도록
바꿔 전사 고리를 없앤다. `docs/` 에 두는 이유는 배포가 읽어야 하기
때문이다(`cache/` 는 gitignore 대상이라 배포판에 실리지 않는다).

실행: python calibration_plot_diagnose.py [--out calibration.png]
"""
import argparse
import json
import math
import os
import subprocess
import sys

import numpy as np
import torch

import eval_cache
from train import STATION_NAMES, _parse_ts
from predict import CHECKPOINT, calibrate_prob, event_threshold

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 1024
OUT_JSON = "./docs/calibration_plot.json"

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


def count_events(timestamps, gap_hours=24):
    """gap_hours 이내로 이어지는 타임스탬프를 사건 하나로 묶어 센다.
    호출자가 이미 한 관측소·양성(또는 오탐) 표본만 걸러서 넘긴다고 가정한다."""
    times = sorted(_parse_ts(t) for t in timestamps)
    if not times:
        return 0
    n = 1
    for prev, cur in zip(times, times[1:]):
        if (cur - prev).total_seconds() > gap_hours * 3600:
            n += 1
    return n


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
    ap.add_argument("--extreme-gbm", default=None,
                    help="극한기상 확률을 이 GBM 에서 가져온다(서빙과 동일). "
                         "2026-09-26 부터 배포는 극한기상을 전용 GBM 이 낸다.")
    args = ap.parse_args()

    # 검증셋 추론은 `eval_cache` 가 만든 것을 재사용한다(2026-09-27 전환) —
    # 직접 `WeatherDataset` 을 구성하면 그 한 번이 RAM 약 22GiB 라, 같은
    # 계열 진단을 두 개만 겹쳐 돌려도 OOM 이 났다.
    ckpt = torch.load(CHECKPOINT, map_location="cpu", weights_only=True)
    d = eval_cache.load(CHECKPOINT)
    stns = np.asarray(d["stn"])
    tgt_ts = np.asarray(d["tgt_ts"])
    heat_p = np.asarray(d["heat_prob"]).astype(np.float64)
    cold_p = np.asarray(d["cold_prob"]).astype(np.float64)
    heat_y = np.asarray(d["y_heatwave"])
    cold_y = np.asarray(d["y_coldwave"])

    # 공식 라벨 전용 채점(2026-09-23) — 특보 비운영기간을 확정 음성으로 채운
    # 체크포인트는 채점 표본이 크게 늘어난다. 쉬운 음성이 섞이면 정밀도가
    # 부풀어, 화면에서 이 플롯 위에 놓이는 합산표와 다른 질문에 답한 값이
    # 된다(metrics_report.precision_block 과 같은 처리).
    def _official(base, off_key):
        mask = np.asarray(d[base]).astype(bool)
        if off_key not in d:
            return mask
        off = np.asarray(d[off_key]).astype(bool)
        if off.sum() and int(off.sum()) != int(mask.sum()):
            return mask & off
        return mask

    heat_m = _official("heat_mask", "heat_mask_official")
    cold_m = _official("cold_mask", "cold_mask_official")

    # 보정 후 공간에서, 서빙이 실제로 쓰는 판정선으로 채점한다(2026-09-23).
    # 관측소별 예외가 걸린 조합은 그 관측소만 임계값이 다르므로 판정선도
    # 관측소마다 조회한다 — 서빙과 같은 방식이어야 이 플롯이 위 표의
    # 분해로서 성립한다.
    gbm = None
    if args.extreme_gbm:
        # 서빙이 극한기상을 GBM 에서 내므로 이 플롯도 그래야 한다 — 신경망
        # 확률로 그리면 바로 위 합산표와 다른 동작점을 재게 되고, "합산표를
        # 관측소별로 분해한 것"이라는 설명이 성립하지 않는다(2026-09-23 에
        # 정확히 그 어긋남을 겪었다). 데이터셋은 이미 만들어져 있으므로
        # 그 입력 행렬을 그대로 쓴다 — 추가 구성을 하지 않는다.
        import extreme_gbm as _eg
        models, meta = _eg.load(args.extreme_gbm)
        if not models:
            raise SystemExit(f"극한기상 GBM 이 없다: {args.extreme_gbm}")
        if int(meta["meta_num_features"]) != ckpt["num_features"]:
            raise SystemExit("GBM 의 입력 차원이 체크포인트와 다르다.")
        # 표준화 입력 행렬은 특징 캐시에서 가져온다. 두 캐시가 같은 검증
        # 분할에서 나왔는지 **키로 대조한다** — 행 순서가 어긋나면 확률과
        # 라벨이 밀려 붙는데 지표는 그럴듯하게 나와 눈치챌 수 없다.
        fz = eval_cache.load_features(CHECKPOINT)
        xv = fz["x_val"].astype(np.float32)
        if len(xv) != len(stns):
            raise SystemExit("추론 캐시와 특징 캐시의 검증 표본 수가 다르다 — "
                             "eval_cache 를 --force 로 다시 만들 것")
        _a = np.asarray([int(v) for v in stns], dtype=np.int64)
        if not np.array_equal(_a, fz["stn_val"].astype(np.int64)):
            raise SystemExit("추론 캐시와 특징 캐시의 행 순서가 다르다 — "
                             "eval_cache 를 --force 로 다시 만들 것")
        heat_p = _eg.calibrated_batch(models, meta, "heatwave", xv)
        cold_p = _eg.calibrated_batch(models, meta, "coldwave", xv)
        gbm = (models, meta)
        print(f"극한기상 확률 출처: {args.extreme_gbm} (서빙과 동일)")
    else:
        heat_p = np.array([calibrate_prob(float(v), "heatwave", ckpt) for v in heat_p])
        cold_p = np.array([calibrate_prob(float(v), "coldwave", ckpt) for v in cold_p])

    results = {"heatwave": [], "coldwave": []}
    thresholds = {}
    for name, p, y, m in (("heatwave", heat_p, heat_y, heat_m),
                          ("coldwave", cold_p, cold_y, cold_m)):
        for stn_code in sorted(set(stns)):
            sel = m & (stns == stn_code)
            n = int(sel.sum())
            if n < 30:
                continue
            thr = (__import__("extreme_gbm").threshold(gbm[1], name) if gbm
                   else event_threshold(name, str(stn_code), ckpt))
            thresholds.setdefault(name, {})[str(stn_code)] = thr
            pred_pos = p[sel] >= thr
            labels = y[sel]
            ts_sel = tgt_ts[sel]
            precision, recall, tp, fp, fn = prf(pred_pos, labels)
            n_pos = int((labels == 1).sum())

            # 유효 표본 크기는 시간 표본이 아니라 사건 수로 잰다(위 docstring
            # 참고) — 재현율은 "실제 양성 사건" 수, 정밀도는 "모델이 양성으로
            # 예측한 사건" 수(오탐 사건도 며칠씩 뭉치므로 같은 보정이 필요).
            # 점 추정값(precision/recall)은 그대로 시간 단위를 쓴다.
            n_events_pos = count_events(ts_sel[labels == 1])
            n_events_predpos = count_events(ts_sel[pred_pos])
            _, r_lo, r_hi = wilson_ci(round(recall * n_events_pos), n_events_pos)
            _, p_lo, p_hi = wilson_ci(round(precision * n_events_predpos), n_events_predpos)

            results[name].append({
                "station": STATION_NAMES.get(stn_code, stn_code),
                "code": stn_code,
                "threshold": thr,
                "n": n, "n_pos": n_pos,
                "n_events_pos": n_events_pos, "n_events_predpos": n_events_predpos,
                "precision": precision, "p_lo": p_lo, "p_hi": p_hi,
                "recall": recall, "r_lo": r_lo, "r_hi": r_hi,
            })

    print("=" * 90)
    print(" 관측소별 재현율×정밀도 (서빙 판정선 기준, "
          "신뢰구간은 사건 수 기준 Wilson 95% CI)")
    print("=" * 90)
    for name, rows in results.items():
        print(f"\n[{name}]")
        print(f"{'관측소':<6} {'시간표본':>8} {'양성사건':>8} {'예측사건':>8} | "
              f"{'재현율':>7} {'CI':>16} | {'정밀도':>7} {'CI':>16}")
        for r in sorted(rows, key=lambda x: -x["n_pos"]):
            print(f"{r['station']:<6} {r['n']:>8,} {r['n_events_pos']:>8,} "
                  f"{r['n_events_predpos']:>8,} | "
                  f"{r['recall']:>6.1%} [{r['r_lo']:.1%},{r['r_hi']:.1%}] | "
                  f"{r['precision']:>6.1%} [{r['p_lo']:.1%},{r['p_hi']:.1%}]")

    # JSON 을 **그림보다 먼저** 쓴다 — 한글 폰트 설치나 matplotlib 쪽에서
    # 실패해도 화면 캡션이 쓰는 값은 남아야 한다. 종전에는 이 스크립트가
    # 승격 도중 죽으면 산출물이 통째로 이전 모델 시점에 머물렀다.
    _payload = {
        "checkpoint": {
            "lead_hours": ckpt.get("lead_hours"),
            "num_features": ckpt.get("num_features"),
            # 어느 모델로 잰 값인지 남긴다 — 화면이 옛 값을 새 값인 양
            # 보여주는 것을 막으려면 대조할 기준이 필요하다.
            "val_temp_naive_mae": ckpt.get("val_temp_naive_mae"),
        },
        "scored_at": "serving_threshold",
        "thresholds": thresholds,
        "stations": results,
    }
    os.makedirs(os.path.dirname(OUT_JSON), exist_ok=True)
    with open(OUT_JSON, "w", encoding="utf-8") as f:
        json.dump(_payload, f, ensure_ascii=False, indent=2)
    print(f"\n저장: {OUT_JSON}")

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

    fig, axes = plt.subplots(1, 2, figsize=(15.5, 7.6))
    colors = {"heatwave": "#E2954F", "coldwave": "#4C78A8"}
    labels_ko = {"heatwave": "폭염", "coldwave": "한파"}
    for ax, name in zip(axes, ["heatwave", "coldwave"]):
        rows = results[name]
        for r in rows:
            x, y = r["recall"], r["precision"]
            # 음수 방지(2026-09-07). 윌슨(Wilson) 구간은 중심이 0.5 쪽으로
            # 밀리므로 점추정이 0 이나 1 에 붙으면 **구간이 점추정을 포함하지
            # 않을 수 있다**(예: 표본 2개에서 정밀도 100%인데 상한 95.4%).
            # 그러면 오차막대 길이가 음수가 되어 matplotlib 이 예외를 던지고
            # 플롯 전체가 생성되지 않는다 — 실제로 수치예보 모델 승격에서
            # 이것 때문에 관측소별 플롯이 이전 모델 시점 그대로 남을 뻔했다.
            # 구간 자체는 그대로 두고 막대 길이만 0 으로 자른다.
            xerr = [[max(0.0, x - r["r_lo"])], [max(0.0, r["r_hi"] - x)]]
            yerr = [[max(0.0, y - r["p_lo"])], [max(0.0, r["p_hi"] - y)]]
            size = 20 + 4 * math.sqrt(r["n_events_pos"])
            ax.errorbar(x, y, xerr=xerr, yerr=yerr, fmt="+",
                        color=colors[name], markersize=size / 5,
                        markeredgewidth=2, capsize=4, elinewidth=1.5, alpha=0.85)
            label = f"{r['station']}({r['code']})" if korean_font else r["code"]
            ax.annotate(label, (x, y), textcoords="offset points",
                       xytext=(6, 6), fontsize=10)
        ax.set_xlabel("재현율 (Recall) — 실제 사건 중 잡아낸 비율",
                      fontsize=13, fontweight="bold")
        ax.set_ylabel("정밀도 (Precision) — 사건이라 판정한 것 중 맞은 비율",
                      fontsize=13, fontweight="bold")
        _thr = sorted({round(v, 3) for v in thresholds.get(name, {}).values()})
        _thr_txt = (f"판정선 {_thr[0]:.3f}" if len(_thr) == 1
                    else f"판정선 {_thr[0]:.3f}~{_thr[-1]:.3f}(관측소별 예외 있음)")
        ax.set_title(f"{labels_ko[name]} — 관측소별\n"
                     f"({_thr_txt} · 십자 = 사건 수 기준 Wilson 95% 신뢰구간)",
                     fontsize=15, fontweight="bold")
        ax.set_xlim(-0.05, 1.05)
        ax.set_ylim(-0.05, 1.05)
        ax.tick_params(labelsize=11)
        ax.grid(alpha=0.3)

    # 첫 번째 그래프(폭염) 왼쪽 위에 관측소 코드→이름 범례를 붙인다 — 점
    # 라벨에 이미 "이름(코드)"가 있지만, 코드만 봐도 바로 찾게 하려는
    # 목적이다(2026-08-16 사용자 요청). 12개를 한 줄씩 쌓으면 세로로 길어
    # 데이터를 가리므로 2열로 접는다(2026-08-17).
    legend_codes = sorted({r["code"] for rows in results.values() for r in rows})
    _pairs = [f"{c}: {STATION_NAMES.get(c, c)}" for c in legend_codes]
    _half = (len(_pairs) + 1) // 2
    _left, _right = _pairs[:_half], _pairs[_half:]
    _right += [""] * (len(_left) - len(_right))
    legend_text = "관측소 코드\n" + "\n".join(
        f"{a:<12}{b}" for a, b in zip(_left, _right))
    axes[0].text(
        0.02, 0.98, legend_text, transform=axes[0].transAxes,
        fontsize=11, va="top", ha="left", linespacing=1.5,
        family="monospace" if not korean_font else None,
        bbox=dict(boxstyle="round,pad=0.6", facecolor="white",
                  edgecolor="gray", alpha=0.9),
    )

    fig.suptitle("관측소별 신뢰도 분석 — 오른쪽 위로 갈수록 좋다"
                 "(정밀도·재현율이 함께 높다)",
                 fontsize=19, fontweight="bold")
    fig.tight_layout(rect=[0, 0, 1, 0.94])
    fig.savefig(args.out, dpi=130)
    print(f"\n저장: {args.out}")


if __name__ == "__main__":
    main()
