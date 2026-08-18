"""
head_decouple_finetune.py — 표현 학습과 재보정을 분리한다(Kang et al. 2020,
"Decoupling Representation and Classifier for Long-Tailed Recognition").

배경(2026-08-16): 표본 재가중(WeightedRandomSampler)을 트렁크까지 포함해서
그냥 학습에 걸었더니 강수 MAE가 기준선 대비 -17.9%에서 -40.8%(6배)까지
악화됐다(crop_oversample 실험). 원인으로 추정한 것 — ① pos_weight(자연
분포 기준으로 계산)가 재가중된 배치 분포와 안 맞아 확률 보정이 어긋났고,
② 강수·폭염·한파·황사 4개 헤드가 embed_dim=64 짜리 공유 트렁크(magnitude)
를 두고 그래디언트를 다투는데, 재가중으로 특정 헤드(사실상 강수)의 압력이
더 세져 트렁크 자체가 그쪽으로 끌려갔다.

이 스크립트는 문헌이 제안하는 정확한 처방을 시험한다: 트렁크(enc_z/enc_re/
enc_im/gate)는 자연 분포로 이미 잘 학습돼 있으니 그대로 얼리고, 재가중은
헤드(작은 MLP, 서로 파라미터 공유 없음)에만 건다. 트렁크가 안 움직이니
①②의 경로가 원리적으로 막힌다. pos_weight 는 재가중된 유효 분포에 다시
안 맞춰도 되도록 1.0(끔)으로 둔다 — 샘플러가 이미 균형을 잡아준다.

재가중 안 한 자연 분포로 헤드만 더 학습(oversample=1.0)하는 것도 대조군으로
같이 돌려, "얼리기 자체의 효과"와 "재가중의 효과"를 분리해서 본다.

실행:
    python head_decouple_finetune.py --oversample 3.0 --epochs 30
    python head_decouple_finetune.py --oversample 1.0 --epochs 30   # 대조군
"""
import argparse
import json
from datetime import timedelta

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, WeightedRandomSampler

from train import (
    collect_historical, WeatherDataset, make_split, _prf_metrics,
    WET_THRESH, DEVICE,
)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from predict import load_model, CHECKPOINT

# 트렁크가 동결돼 역전파 대상이 6천~수만 파라미터뿐이라 GPU 부담이 더 작다
# — VRAM 여유를 더 쓰도록 배치를 train.py 기본값(1024)보다 키운다.
BATCH = 2048
NUM_WORKERS = 8   # train.py 실측(perf_profile.py)과 동일 — CPU 코어 활용


_HEAD_ATTR = {
    "rain": ("head_rain", "head_precip"),
    "heatwave": ("head_heatwave",),
    "coldwave": ("head_coldwave",),
    "dust": ("head_dust",),
}


def freeze_trunk(model, heads):
    """트렁크(엔코더·게이트·기온 헤드)는 얼리고, heads 에 지정한 헤드만 연다.

    스모크 테스트(2026-08-16) 3종 비교 — 트렁크를 얼리고 재가중해도:
      강수 포함              : 강수 0.1786→0.238mm(악화), 한파 0.384→0.465(개선)
      강수 배제(폭염+한파+황사): 강수 0.1786(불변, 안 건드리므로), 한파 →0.464(개선),
                              폭염 →0.79(거의 그대로), 황사 →0.08(악화)
    한파만 뚜렷한 순수 이득이다. 황사는 표현 경쟁이 아니라 입력 신호 자체가
    부족한 문제(국외 PM10 장거리 수송)라 재가중으로도 안 풀린다 — 오히려
    적은 양성표본을 과대표집해 노이즈만 키운다. 그래서 기본은 한파 단독.
    """
    for p in model.parameters():
        p.requires_grad_(False)
    n_tunable = 0
    for h in heads:
        for attr in _HEAD_ATTR[h]:
            for p in getattr(model, attr).parameters():
                p.requires_grad_(True)
                n_tunable += p.numel()
    return n_tunable


def masked_bce(logit, target, mask, pos_weight=None):
    raw = nn.functional.binary_cross_entropy_with_logits(
        logit, target, pos_weight=pos_weight, reduction="none")
    return (raw * mask).sum() / mask.sum().clamp_min(1)


def evaluate(model, ds, idx, device):
    model.eval()
    temp_abs, precip_abs = [], []
    heat_p, cold_p, dust_p = [], [], []
    heat_y, cold_y, dust_y = [], [], []
    heat_m, cold_m, dust_m = [], [], []
    with torch.no_grad():
        for b in range(0, len(idx), BATCH):
            sl = idx[b:b + BATCH]
            pred = model(
                num_x=ds.X_num[sl].to(device),
                img_x=ds.X_img[sl].to(device).float(),
                txt_x=ds.X_txt[sl].to(device),
            )
            temp_abs.append((pred[:, 0].cpu() - ds.y[sl, 0]).abs())
            precip_abs.append((pred[:, 1].cpu() - ds.y[sl, 1]).abs())
            heat_p.append(torch.sigmoid(model._last_heatwave_logit).cpu().squeeze(-1))
            cold_p.append(torch.sigmoid(model._last_coldwave_logit).cpu().squeeze(-1))
            dust_p.append(torch.sigmoid(model._last_dust_logit).cpu().squeeze(-1))
            heat_y.append(ds.y_heatwave[sl]); cold_y.append(ds.y_coldwave[sl]); dust_y.append(ds.y_dust[sl])
            heat_m.append(ds.heat_mask[sl]); cold_m.append(ds.cold_mask[sl]); dust_m.append(ds.dust_mask[sl])

    temp_mae = torch.cat(temp_abs).mean().item()
    precip_mae = torch.cat(precip_abs).mean().item()
    metrics = {}
    for name, p, y, m in (("heatwave", heat_p, heat_y, heat_m),
                          ("coldwave", cold_p, cold_y, cold_m),
                          ("dust", dust_p, dust_y, dust_m)):
        mask = torch.cat(m).bool()
        metrics[name] = _prf_metrics(torch.cat(p)[mask], torch.cat(y)[mask])
    return temp_mae, precip_mae, metrics


_ALL_HEADS = ("rain", "heatwave", "coldwave", "dust")
_EVENT_KEY = {"rain": None, "heatwave": "y_heatwave", "coldwave": "y_coldwave",
             "dust": "y_dust"}  # rain 은 ds.y[:,1]>=WET_THRESH 로 별도 계산


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oversample", type=float, default=3.0)
    ap.add_argument("--epochs", type=int, default=30)
    ap.add_argument("--lr", type=float, default=5e-4)
    ap.add_argument("--out", default=None)
    ap.add_argument("--heads", default="coldwave",
                    help="쉼표로 구분한 미세조정 대상 헤드"
                         "(rain,heatwave,coldwave,dust 중). 기본은 한파 단독 "
                         "— 스모크 테스트(2026-08-16) 3종 비교에서 뚜렷한 순이득이 "
                         "확인된 유일한 헤드였다(강수는 오염, 황사는 표본 부족으로 악화).")
    # 위치 인자 없이 CHECKPOINT_PATH 환경변수에만 의존했다가, predict.py의
    # CHECKPOINT가 당시 하드코딩이라 조용히 무시된 사고가 있었다(2026-08-18,
    # 한파 중첩 전이학습 실험 무효화). predict.CHECKPOINT는 이제 그 환경변수를
    # 읽도록 고쳤지만, 여기서도 명시적 인자를 받아 이중으로 막는다 — 다음
    # 실행자가 어느 쪽에 의존하는지 헷갈릴 필요가 없게 한다.
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT,
                    help="기반 체크포인트 경로(생략 시 CHECKPOINT_PATH 환경변수 또는 배포 경로)")
    args = ap.parse_args()
    heads = [h.strip() for h in args.heads.split(",") if h.strip()]
    for h in heads:
        assert h in _ALL_HEADS, f"알 수 없는 헤드: {h}"

    model, ckpt = load_model(args.ckpt, DEVICE)
    print(f"기반 체크포인트 — {args.ckpt}")
    print(f"  split_mode={ckpt.get('split_mode')} "
          f"기온MAE {ckpt['val_temp_mae']:.4f} 강수MAE {ckpt['val_precip_mae']:.4f}")

    records = collect_historical()
    txt_collector = TendencyCollector(records)
    ds = WeatherDataset(
        records, sat_collector=InterpolatedFieldCollector(records, STATION_COORDS),
        txt_collector=txt_collector, lead_hours=ckpt["lead_hours"],
        mean=np.array(ckpt["mean"], dtype=np.float32),
        std=np.array(ckpt["std"], dtype=np.float32),
    )
    train_ds, val_ds = make_split(ds, ckpt.get("split_mode", "random"), verbose=True)
    train_idx = list(train_ds.indices)
    val_idx = list(val_ds.indices)

    n_tunable = freeze_trunk(model, heads)
    print(f"헤드만 미세조정 — 학습 가능 파라미터 {n_tunable:,}개 "
          f"(트렁크는 동결, 대상: {', '.join(heads)})")

    # ── 재가중 샘플러 — 미세조정 대상 헤드의 양성만으로 사건을 정의한다.
    # 안 건드리는 헤드(예: 강수)의 양성이 섞이면 재가중 예산이 낭비되고,
    # rain 처럼 표본 수가 압도적인 헤드가 끼면 다른 헤드의 신호가 묻힌다
    # (2026-08-16 스모크 테스트로 확인).
    is_event = torch.zeros(len(ds), dtype=torch.bool)
    for h in heads:
        if h == "rain":
            is_event |= (ds.y[:, 1] >= WET_THRESH)
        else:
            is_event |= getattr(ds, _EVENT_KEY[h]).bool()
    train_idx_t = torch.as_tensor(train_idx)
    train_event = is_event[train_idx_t]
    n_event = int(train_event.sum())
    print(f"학습셋 사건 표본 {n_event:,}개({100*n_event/len(train_idx):.1f}%) "
          f"— {args.oversample:.1f}배 오버샘플링")

    _dl = dict(num_workers=NUM_WORKERS, pin_memory=(DEVICE == "cuda"),
               persistent_workers=(NUM_WORKERS > 0))
    if args.oversample > 1.0:
        weights = torch.where(train_event, torch.tensor(args.oversample), torch.tensor(1.0))
        sampler = WeightedRandomSampler(weights.double(), num_samples=len(train_idx), replacement=True)
        train_loader = DataLoader(train_ds, batch_size=BATCH, sampler=sampler, **_dl)
    else:
        train_loader = DataLoader(train_ds, batch_size=BATCH, shuffle=True, **_dl)

    # pos_weight 를 안 쓴다 — 샘플러가 이미 균형을 잡아주므로 자연분포 기준
    # pos_weight 를 또 곱하면 이중보정이 된다(오늘 crop 실험 실패 원인 중 하나).
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=args.lr)

    print(f"\n{'에폭':>4} | {'손실':>8} | {'기온MAE':>8} | {'강수MAE':>8} | "
          f"{'폭염F1':>7} | {'한파F1':>7} | {'황사F1':>7}")
    print("-" * 70)

    best = None
    for epoch in range(1, args.epochs + 1):
        model.train()
        total_loss = 0.0
        n_batches = 0
        for x_num, x_img, x_txt, y_b, heat_b, cold_b, dust_b, hmask_b, cmask_b, dmask_b in train_loader:
            x_num, y_b = x_num.to(DEVICE), y_b.to(DEVICE)
            x_img = x_img.to(DEVICE).float()
            x_txt = x_txt.to(DEVICE)
            heat_b = heat_b.to(DEVICE).unsqueeze(-1)
            cold_b = cold_b.to(DEVICE).unsqueeze(-1)
            dust_b = dust_b.to(DEVICE).unsqueeze(-1)
            hmask_b = hmask_b.to(DEVICE).unsqueeze(-1)
            cmask_b = cmask_b.to(DEVICE).unsqueeze(-1)
            dmask_b = dmask_b.to(DEVICE).unsqueeze(-1)
            is_wet_b = (y_b[:, 1:2] >= WET_THRESH).float()

            optimizer.zero_grad()
            pred = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            loss = None

            def _add(term):
                nonlocal loss
                loss = term if loss is None else loss + term

            if "rain" in heads:
                _add(nn.functional.mse_loss(pred[:, 1:2], y_b[:, 1:2]))
                _add(nn.functional.binary_cross_entropy_with_logits(
                    model._last_rain_logit, is_wet_b))
            if "heatwave" in heads and hmask_b.sum() > 0:
                _add(masked_bce(model._last_heatwave_logit, heat_b, hmask_b))
            if "coldwave" in heads and cmask_b.sum() > 0:
                _add(masked_bce(model._last_coldwave_logit, cold_b, cmask_b))
            if "dust" in heads and dmask_b.sum() > 0:
                _add(masked_bce(model._last_dust_logit, dust_b, dmask_b))
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1

        temp_mae, precip_mae, m = evaluate(model, ds, val_idx, DEVICE)
        hf = m["heatwave"]["f1"]; cf = m["coldwave"]["f1"]; df = m["dust"]["f1"]
        print(f"{epoch:4d} | {total_loss/n_batches:8.4f} | {temp_mae:8.4f} | "
              f"{precip_mae:8.4f} | {hf:7.3f} | {cf:7.3f} | {df:7.3f}")

        # 선택 기준 — 미세조정 대상 헤드들의 F1(또는 강수 MAE)만 평균해
        # 고른다. 안 건드리는 헤드는 어차피 상수라 기준에 넣어도 무의미하다.
        parts = []
        if "rain" in heads:
            parts.append(-precip_mae)   # 낮을수록 좋음 → 부호 반전해 "높을수록 좋음"으로 통일
        if "heatwave" in heads:
            parts.append(hf)
        if "coldwave" in heads:
            parts.append(cf)
        if "dust" in heads:
            parts.append(df)
        score = sum(parts) / len(parts)
        if best is None or score > best[0]:
            # detach().clone() 이 필수다 — state_dict() 는 파라미터 텐서를 복사하지
            # 않고 그대로 돌려주므로, 그냥 담아두면 이후 에폭의 갱신이 그 텐서를
            # 제자리에서 덮어쓴다. 즉 "최선 에폭"을 골라놓고 실제로는 마지막 에폭을
            # 저장하게 되며, 마지막 줄에 출력하는 최선 지표도 저장된 모델의 값이
            # 아니게 된다. 2026-08-17 한파 디커플링에서 발견했다 — 최선 에폭 21의
            # 한파 F1 을 0.462 로 출력했으나 저장본을 patch_extreme_metrics.py 로
            # 재계산하니 마지막 에폭 값인 0.458 이 나와 드러났다.
            best = (score, epoch, precip_mae, temp_mae, hf, cf, df,
                    {attr: {k: v.detach().clone() for k, v in
                            getattr(model, attr).state_dict().items()}
                     for h in heads for attr in _HEAD_ATTR[h]})

    print(f"\n최선 에폭 {best[1]} — 강수MAE {best[2]:.4f} 기온MAE {best[3]:.4f} "
          f"폭염F1 {best[4]:.3f} 한파F1 {best[5]:.3f} 황사F1 {best[6]:.3f}")

    if args.out:
        for attr, sd in best[7].items():
            getattr(model, attr).load_state_dict(sd)
        ckpt_out = dict(ckpt)
        ckpt_out["model_state"] = model.state_dict()
        ckpt_out["val_precip_mae"] = best[2]
        ckpt_out["val_temp_mae"] = best[3]
        ckpt_out["head_decouple_heads"] = heads
        ckpt_out["head_decouple_oversample"] = args.oversample
        torch.save(ckpt_out, args.out)
        print(f"저장: {args.out}")


if __name__ == "__main__":
    main()
