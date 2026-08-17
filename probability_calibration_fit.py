"""
probability_calibration_fit.py — 헤드별 확률 보정(isotonic) 적합·검증·체크포인트 반영.

`probability_calibration_check.py` 실측(2026-08-16): 네 헤드 모두 체계적으로
과신한다(한파는 "60~70%"라 해도 실제 빈도가 27%). 원인은 버그가 아니라 희소
사건 붕괴를 막으려고 BCE에 준 `pos_weight`가 양성 확률을 밀어올린 것이다 —
재현율을 얻은 대가로 보정을 내줬다.

이 스크립트는 재학습 없이 그 대가를 되돌린다. 단조 증가 함수로 확률을 다시
매핑하므로 **순위가 보존되어 분류 성능(F1·정밀도·재현율)은 그대로**이고,
확률의 의미만 바로잡힌다. 임계값도 같은 함수로 옮기면 판정 결과가 완전히
동일하다.

절차(임계값 선정과 같은 규율)
  1. 검증셋을 보정용/평가용 절반씩 무작위 분할(CALIB_SEED — 학습/검증 분할과
     독립적인 난수).
  2. 보정용에서만 isotonic 곡선을 적합한다.
  3. 한 번도 보지 않은 평가용에서 ECE 개선과 F1 보존을 확인한다.
  4. 평가용에서 개선이 확인된 헤드만 체크포인트에 기록한다.

sklearn을 쓰지 않는 이유 — 배포(Streamlit Cloud)는 `requirements.txt` 하나로
빌드되고 서빙 의존성을 최소로 유지한다. PAVA는 짧아서 직접 구현하고, 결과는
`np.interp`로 적용 가능한 조회표(x, y)로 저장한다. 추론 경로에 새 의존성이
전혀 생기지 않는다.

실행: python probability_calibration_fit.py [체크포인트] [--apply]
      --apply 없이 실행하면 측정만 하고 체크포인트를 건드리지 않는다.
"""
import os
import shutil
import sys
import tempfile

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import WeatherDataset, collect_historical, make_split, WET_THRESH
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from text_collector import SimulatedTextCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096
CALIB_SEED = 1234    # threshold_validation.py와 같은 값 — 같은 재분할을 쓴다
N_KNOTS = 200        # 조회표 크기. 200이면 곡선을 충분히 담으면서 체크포인트에
                     # 부담이 없다(헤드당 약 3KB).
MIN_ECE_GAIN = 0.005 # 평가용에서 이만큼은 좋아져야 채택한다
MIN_THRESH_GAIN = 0.01  # 임계값 재선정 채택 기준 — threshold_validation.py 와 동일


def _pava(y, w):
    """
    Pool Adjacent Violators — 가중 isotonic 회귀(단조 증가 적합).

    인접한 두 값이 순서를 어기면(앞이 뒤보다 크면) 둘을 가중평균으로 합친다.
    합친 결과가 다시 앞과 순서를 어기면 계속 합친다. 끝나면 전 구간이
    비감소가 된다.
    """
    vals, wts, cnts = [], [], []
    for yi, wi in zip(y, w):
        vals.append(float(yi)); wts.append(float(wi)); cnts.append(1)
        while len(vals) > 1 and vals[-2] > vals[-1]:
            v2, w2, c2 = vals.pop(), wts.pop(), cnts.pop()
            v1, w1, c1 = vals.pop(), wts.pop(), cnts.pop()
            nw = w1 + w2
            vals.append((v1 * w1 + v2 * w2) / nw)
            wts.append(nw); cnts.append(c1 + c2)
    out = np.empty(sum(cnts))
    pos = 0
    for v, c in zip(vals, cnts):
        out[pos:pos + c] = v
        pos += c
    return out


def fit_isotonic(probs, labels, n_knots=N_KNOTS):
    """
    확률 → 보정확률 조회표를 만든다.

    표본을 확률 순으로 등빈도 구간에 나눈 뒤 구간별 (평균확률, 실제빈도)를
    구하고, 그 빈도열에 PAVA를 걸어 단조로 만든다. 등간격이 아니라 등빈도로
    나누는 이유 — 확률이 0 근처에 몰려 있어 등간격으로 자르면 대부분의 구간이
    비고 정작 표본이 많은 구간의 해상도가 떨어진다.
    """
    order = np.argsort(probs, kind="mergesort")
    p, l = probs[order], labels[order]
    n = len(p)
    edges = np.linspace(0, n, min(n_knots, max(2, n // 50)) + 1).astype(int)
    xs, ys, ws = [], [], []
    for a, b in zip(edges[:-1], edges[1:]):
        if b <= a:
            continue
        xs.append(p[a:b].mean()); ys.append(l[a:b].mean()); ws.append(b - a)
    xs, ys, ws = np.array(xs), np.array(ys), np.array(ws, dtype=float)
    ys_iso = _pava(ys, ws)
    # np.interp는 x가 순증가여야 한다. 등빈도 구간의 평균확률은 이미 비감소지만
    # 동일값이 이어질 수 있어 아주 작은 증분을 넣어 강제로 순증가로 만든다.
    xs = np.maximum.accumulate(xs)
    for i in range(1, len(xs)):
        if xs[i] <= xs[i - 1]:
            xs[i] = xs[i - 1] + 1e-9
    return xs, ys_iso


def apply_calibration(probs, xs, ys):
    """조회표를 선형보간으로 적용. 양 끝은 끝값으로 고정된다(np.interp 기본)."""
    return np.interp(probs, xs, ys)


def ece(probs, labels, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    tot, acc = len(probs), 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            continue
        acc += abs(probs[m].mean() - labels[m].mean()) * m.sum()
    return acc / tot


def mce(probs, labels, n_bins=10):
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    worst = 0.0
    for lo, hi in zip(edges[:-1], edges[1:]):
        m = (probs >= lo) & (probs < hi if hi < 1.0 else probs <= hi)
        if not m.any():
            continue
        worst = max(worst, abs(probs[m].mean() - labels[m].mean()))
    return worst


def prf(probs, labels, t):
    pred = (probs >= t).astype(int)
    tp = int(((pred == 1) & (labels == 1)).sum())
    fp = int(((pred == 1) & (labels == 0)).sum())
    fn = int(((pred == 0) & (labels == 1)).sum())
    p = tp / (tp + fp) if tp + fp else 0.0
    r = tp / (tp + fn) if tp + fn else 0.0
    f = 2 * p * r / (p + r) if p + r else 0.0
    return p, r, f


def _best_threshold(probs, labels, n_grid=400):
    """주어진 확률 공간에서 F1 을 최대화하는 임계값을 격자 탐색으로 고른다."""
    if len(probs) == 0:
        return 0.5
    grid = np.linspace(float(probs.min()), float(probs.max()), n_grid)
    best_t, best_f1 = float(grid[0]), -1.0
    for t in grid:
        f1 = prf(probs, labels, float(t))[2]
        if f1 > best_f1:
            best_t, best_f1 = float(t), f1
    return best_t


def load_probs(ckpt_path):
    model, ckpt = load_model(ckpt_path, DEVICE)
    model.eval()
    records = collect_historical()
    txt_collector = (TendencyCollector(records) if ckpt.get("im_dim", 384) < 128
                     else SimulatedTextCollector())
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    _, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=False)
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
    heads = {
        "rain": (np.concatenate(rain_p), (precip_a >= WET_THRESH).astype(int)),
        "heatwave": (np.concatenate(heat_p)[hmask],
                     ds.y_heatwave[val_idx].numpy().astype(int)[hmask]),
        "coldwave": (np.concatenate(cold_p)[cmask],
                     ds.y_coldwave[val_idx].numpy().astype(int)[cmask]),
    }
    if dust_p:
        dmask = ds.dust_mask[val_idx].numpy().astype(bool)
        heads["dust"] = (np.concatenate(dust_p)[dmask],
                         ds.y_dust[val_idx].numpy().astype(int)[dmask])
    return heads, ckpt


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_apply = "--apply" in sys.argv
    ckpt_path = args[0] if args else CHECKPOINT

    heads, ckpt = load_probs(ckpt_path)
    from predict import EXTREME_EVENT_THRESH
    ko = {"rain": "강수", "heatwave": "폭염", "coldwave": "한파", "dust": "황사"}

    rng = np.random.RandomState(CALIB_SEED)
    calib_maps, summary = {}, []

    for key, (probs, labels) in heads.items():
        n = len(probs)
        m = rng.rand(n) < 0.5              # 보정용 절반
        pc, lc = probs[m], labels[m]       # calibration
        pe, le = probs[~m], labels[~m]     # evaluation

        xs, ys = fit_isotonic(pc, lc)
        pe_cal = apply_calibration(pe, xs, ys)
        pc_cal = apply_calibration(pc, xs, ys)   # 임계값 재선정용(보정용 절반)

        e0, e1 = ece(pe, le), ece(pe_cal, le)
        m0, m1 = mce(pe, le), mce(pe_cal, le)
        b0 = float(((pe - le) ** 2).mean())
        b1 = float(((pe_cal - le) ** 2).mean())

        # 순위 보존 확인 — 임계값을 같은 함수로 옮기면 판정이 동일해야 한다.
        t_raw = EXTREME_EVENT_THRESH.get(key, 0.5)
        t_cal = float(apply_calibration(np.array([t_raw]), xs, ys)[0])
        f_raw = prf(pe, le, t_raw)[2]
        f_cal = prf(pe_cal, le, t_cal)[2]

        # 옮긴 임계값이 판정을 보존하지 못할 때의 대비책(2026-08-17 추가).
        # 등장성 곡선은 평탄 구간을 가지므로 순증가가 아니다 — 임계값이 그
        # 구간 안에 떨어지면 구간 전체가 한쪽으로 몰려 실효 임계값이 어긋난다.
        # 실제로 한파 헤드 디커플링 직후 F1 이 0.4566→0.4484 로 떨어졌다
        # (calibrated_threshold_check.py 로 확인). 판정이 실제로 이뤄지는
        # 공간(보정 후)에서 임계값을 직접 고르되, 고르는 표본과 채점하는
        # 표본을 분리하고 순이득이 기준에 미달하면 채택하지 않는다.
        t_pick = _best_threshold(pc_cal, lc)
        f_pick = prf(pe_cal, le, t_pick)[2]
        repick = (f_pick - f_cal) >= MIN_THRESH_GAIN
        t_decision = t_pick if repick else t_cal

        gain = e0 - e1
        adopt = gain >= MIN_ECE_GAIN
        summary.append((key, e0, e1, m0, m1, b0, b1, t_raw, t_cal, f_raw, f_cal, gain, adopt))
        if adopt:
            calib_maps[key] = {"x": [float(v) for v in xs],
                               "y": [float(v) for v in ys],
                               "threshold_raw": t_raw,
                               "threshold_calibrated": t_cal,
                               # 서빙이 실제로 쓰는 판정선. 재선정을 채택하지
                               # 않으면 옮긴 값과 같다.
                               "threshold_decision": t_decision,
                               "threshold_repicked": bool(repick)}

        print(f"\n{'='*74}\n[{ko[key]}]  평가용 {len(pe):,}개 (보정용 {len(pc):,}개로 적합)")
        print(f"  ECE   {e0:.4f} → {e1:.4f}   (개선 {gain:+.4f})")
        print(f"  MCE   {m0:.4f} → {m1:.4f}")
        print(f"  Brier {b0:.4f} → {b1:.4f}")
        print(f"  임계값 {t_raw:.3f} → {t_cal:.3f} 로 이동 시 F1 {f_raw:.4f} → {f_cal:.4f}"
              f"  ({'보존됨' if abs(f_raw - f_cal) < 1e-6 else '차이 발생 — 재선정 검토'})")
        print(f"  보정 공간 재선정 t = {t_pick:.3f} → 평가용 F1 {f_pick:.4f} "
              f"(순이득 {f_pick - f_cal:+.4f}) — "
              f"{'채택' if repick else f'기각(기준 {MIN_THRESH_GAIN})'}")
        print(f"  서빙 판정선 = {t_decision:.3f}")
        print(f"  판정: {'채택' if adopt else f'기각(개선 {gain:+.4f} < 기준 {MIN_ECE_GAIN})'}")

    print(f"\n{'='*74}\n요약")
    print(f"  {'헤드':<8}{'ECE 전':>10}{'ECE 후':>10}{'F1 전':>10}{'F1 후':>10}  판정")
    for (key, e0, e1, _, _, _, _, _, _, f0, f1, _, adopt) in summary:
        print(f"  {ko[key]:<8}{e0:>10.4f}{e1:>10.4f}{f0:>10.4f}{f1:>10.4f}  "
              f"{'채택' if adopt else '기각'}")

    if not do_apply:
        print("\n(--apply 를 주지 않아 체크포인트는 그대로 둔다)")
        return

    if not calib_maps:
        print("\n채택된 헤드가 없어 체크포인트를 바꾸지 않는다.")
        return

    backup = ckpt_path.replace(".pt", "_before_probcal.pt")
    if not os.path.exists(backup):
        shutil.copy2(ckpt_path, backup)
        print(f"\n백업: {backup}")

    ckpt["prob_calibration"] = {
        "method": "isotonic(PAVA) on quantile bins",
        "calib_seed": CALIB_SEED,
        "min_ece_gain": MIN_ECE_GAIN,
        "heads": calib_maps,
    }
    # 공유 파일 저장은 고유 tmp + os.replace 로 원자적으로(CLAUDE.md 6절).
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(ckpt_path) or ".", suffix=".pt")
    os.close(fd)
    torch.save(ckpt, tmp)
    os.replace(tmp, ckpt_path)
    print(f"체크포인트에 prob_calibration 기록: {ckpt_path}")
    print(f"  채택 헤드: {', '.join(ko[k] for k in calib_maps)}")


if __name__ == "__main__":
    main()
