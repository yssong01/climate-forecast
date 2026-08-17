"""
neutral_input_check.py — '중립값'이 실제로 중립인지 잰다.

**배경.** 결측을 추정으로 채우지 않는다는 원칙에 따라, 이웃 관측이 없으면
Re축 보간장을 0.5로, 과거 관측이 없으면 Im축 경향을 0으로 채운다. 문서와
화면은 이를 "중립값"이라 부르는데, **중립임을 측정한 적이 없다.**

0.5 가 중립이라는 전제는 채널이 [0,1] 로 정규화돼 있다는 사실에서 왔다.
그러나 정규화 식이 채널마다 다르므로 0.5 가 뜻하는 물리량도 채널마다 다르다.

    기온   (v + 10) / 60   → 0.5 = 20 °C
    습도    v / 100        → 0.5 = 50 %
    기압   (v - 980) / 60  → 0.5 = 1010 hPa
    풍속    v / 20         → 0.5 = 10 m/s      ← 평상시 풍속과 거리가 멀다

중립이려면 0.5 가 그 채널의 실제 분포 중심 부근이어야 한다. 어긋나 있으면
결측이 잦은 관측소·시간대의 출력이 한 방향으로 치우친다 — 결측을 "모른다"로
처리하려던 설계가 "특정 값이라고 주장한다"로 바뀐다.

**무엇을 재는가.**
  1. 채널별 실제 분포(평균·중앙값)와 0.5 의 거리
  2. Re축을 통째로 중립값으로 바꿨을 때 출력이 얼마나, 어느 방향으로 움직이는지
  3. 같은 실험을 '데이터 평균'으로 채웠을 때와 비교 — 어느 쪽이 덜 움직이는가
  4. Im축(경향 0)에 대해서도 같은 점검

실행: python neutral_input_check.py [체크포인트] [--n 20000]
출력: 표준출력 + VERDICT 줄(승격 게이트에 붙일 수 있는 형식)
"""
import argparse

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import WeatherDataset, collect_historical, make_split
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 4096
CHANNELS = ["기온", "습도", "기압", "풍속"]
# interp_field_collector._CHANNEL_SPEC 의 역함수 — 0.5 가 뜻하는 물리량
INVERSE = [lambda x: x * 60 - 10, lambda x: x * 100,
           lambda x: x * 60 + 980, lambda x: x * 20]
UNITS = ["°C", "%", "hPa", "m/s"]

# 판정 기준 — 중립값이 채널 분포 중심에서 이만큼 벗어나면 경고한다.
# 0.5 는 [0,1] 구간의 정중앙이므로, 실제 평균이 0.5±0.15 밖이면 그 채널에서
# "정보 없음"이 사실상 특정 값 주장이 된다.
NEUTRAL_TOL = 0.15


def shift_stats(model, ds, idx, replace):
    """replace(x_img, x_txt) 로 입력을 바꿨을 때의 출력 변화량을 모은다."""
    d_temp, d_precip, d_heat, d_cold = [], [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), BATCH):
            sl = idx[b:b + BATCH].tolist()
            num = ds.X_num[sl].to(DEVICE)
            img = ds.X_img[sl].to(DEVICE).float()
            txt = ds.X_txt[sl].to(DEVICE)

            base = model(num_x=num, img_x=img, txt_x=txt)
            b_heat = torch.sigmoid(model._last_heatwave_logit).squeeze(-1)
            b_cold = torch.sigmoid(model._last_coldwave_logit).squeeze(-1)

            img2, txt2 = replace(img, txt)
            alt = model(num_x=num, img_x=img2, txt_x=txt2)
            a_heat = torch.sigmoid(model._last_heatwave_logit).squeeze(-1)
            a_cold = torch.sigmoid(model._last_coldwave_logit).squeeze(-1)

            d_temp.append((alt[:, 0] - base[:, 0]).cpu().numpy())
            d_precip.append((alt[:, 1] - base[:, 1]).cpu().numpy())
            d_heat.append((a_heat - b_heat).cpu().numpy())
            d_cold.append((a_cold - b_cold).cpu().numpy())
    return {k: np.concatenate(v) for k, v in
            (("temp", d_temp), ("precip", d_precip),
             ("heat", d_heat), ("cold", d_cold))}


def report(name, s):
    print(f"  {name:<22}"
          f"기온 {s['temp'].mean():+7.3f} °C (|Δ| {np.abs(s['temp']).mean():.3f})   "
          f"강수 {s['precip'].mean():+7.4f} mm   "
          f"폭염 {s['heat'].mean():+7.4f}   한파 {s['cold'].mean():+7.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--n", type=int, default=20000,
                    help="검증셋에서 무작위로 뽑을 표본 수(전체를 쓰면 느리다)")
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
    print(f"체크포인트: {args.ckpt} · 표본 {len(idx):,}개(검증셋에서 추출)")

    # ── 1. 채널별 실제 분포와 0.5 의 거리 ─────────────────────
    sub = ds.X_img[idx.tolist()].float()          # (N, 4, 32, 32)
    ch_mean = sub.mean(dim=(0, 2, 3)).numpy()
    ch_med = sub.flatten(2).median(dim=-1).values.median(dim=0).values.numpy()
    print(f"\n{'=' * 84}\n 1. Re축 채널별 실제 분포 vs 중립값 0.5\n{'=' * 84}")
    print(f"  {'채널':<8}{'실제 평균':>10}{'중앙값':>10}{'0.5 와 차이':>12}"
          f"{'0.5 의 물리적 의미':>22}{'실제 평균의 의미':>20}")
    worst = 0.0
    for c, (name, inv, u) in enumerate(zip(CHANNELS, INVERSE, UNITS)):
        gap = float(ch_mean[c]) - 0.5
        worst = max(worst, abs(gap))
        flag = "  ← 중립 아님" if abs(gap) > NEUTRAL_TOL else ""
        print(f"  {name:<8}{ch_mean[c]:>10.3f}{ch_med[c]:>10.3f}{gap:>+12.3f}"
              f"{inv(0.5):>18.1f} {u:<3}{inv(float(ch_mean[c])):>16.1f} {u}{flag}")

    # ── 2·3. 대체값별 출력 변화 ───────────────────────────────
    ch_mean_t = torch.tensor(ch_mean, device=DEVICE).view(1, -1, 1, 1)
    print(f"\n{'=' * 84}\n 2. 축을 통째로 대체했을 때 출력이 얼마나 움직이는가"
          f"\n    (평균 Δ가 0에서 멀수록 그 대체값이 한 방향으로 밀고 있다는 뜻)\n{'=' * 84}")
    cases = [
        ("Re → 0.5(현행 중립값)", lambda i, t: (torch.full_like(i, 0.5), t)),
        ("Re → 채널 평균",        lambda i, t: (ch_mean_t.expand_as(i).clone(), t)),
        ("Im → 0(현행 중립값)",   lambda i, t: (i, torch.zeros_like(t))),
        ("Im → 채널 평균",        lambda i, t: (i, ds.X_txt[idx.tolist()].mean(dim=0)
                                                .to(DEVICE).expand_as(t).clone())),
    ]
    results = {}
    for name, fn in cases:
        results[name] = shift_stats(model, ds, idx, fn)
        report(name, results[name])

    # ── 4. 판정 ──────────────────────────────────────────────
    re_neutral = np.abs(results["Re → 0.5(현행 중립값)"]["temp"]).mean()
    re_mean = np.abs(results["Re → 채널 평균"]["temp"]).mean()
    im_neutral = np.abs(results["Im → 0(현행 중립값)"]["temp"]).mean()
    im_mean = np.abs(results["Im → 채널 평균"]["temp"]).mean()

    print(f"\n{'=' * 84}\n 3. 판정\n{'=' * 84}")
    print(f"  Re축 — 0.5 대체 시 |Δ기온| {re_neutral:.3f}°C, "
          f"채널 평균 대체 시 {re_mean:.3f}°C")
    print(f"  Im축 — 0 대체 시   |Δ기온| {im_neutral:.3f}°C, "
          f"채널 평균 대체 시 {im_mean:.3f}°C")
    verdict = "PASS"
    if worst > NEUTRAL_TOL:
        verdict = "WARN"
        print(f"\n  ⚠ 0.5 가 채널 분포 중심에서 최대 {worst:.3f} 벗어나 있다 — "
              f"그 채널에서 '정보 없음'이 사실상 특정 값 주장이 된다.")
    if re_mean < re_neutral * 0.9:
        verdict = "WARN"
        print(f"  ⚠ 채널 평균으로 채우면 출력 변화가 {(1 - re_mean / re_neutral):.0%} "
              f"작다 — 그쪽이 더 중립에 가깝다.")
    if verdict == "PASS":
        print("  중립값이 분포 중심 부근이며, 대체 시 출력이 한 방향으로 밀리지 않는다.")
    print(f"\nVERDICT neutral_input {verdict} worst_channel_gap={worst:.4f} "
          f"re_shift={re_neutral:.4f} re_shift_meanfill={re_mean:.4f}")


if __name__ == "__main__":
    main()
