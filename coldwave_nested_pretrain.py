"""
coldwave_nested_pretrain.py — 한파 헤드 중첩 극한사건 전이학습(사전학습 단계).

Jacques-Dumas et al.(arXiv:2103.09743)의 중첩 극한사건 전이학습을 적용한다
— 더 극단적인 사건은 덜 극단적인 사건의 부분집합이라는 성질을 이용해,
표본이 많은 "약한" 사건으로 먼저 학습한 뒤 표본이 극히 적은 진짜 사건으로
옮겨간다.

**왜 필요한가.** 한파는 공식 라벨 기간이 포털 제공 범위(2020년~)로 고정돼
못 늘리지만(README 검증 규약), 원시 기온 자체는 13.6년치가 있다. 이
스크립트는 학습셋(공식 학습/검증 분할과 동일 — 검증셋은 절대 보지 않는다)
기온 분포의 하위 백분위 3단계로 "약한 한파" 라벨을 만들어 head_coldwave를
순차적으로 사전학습한다. 트렁크는 동결한다(head_decouple_finetune.py 와
같은 원리 — Kang et al. 2020, Decoupling Representation and Classifier).

**마지막 단계 출력은 그 자체로 배포하지 않는다.** 이 스크립트가 만드는
체크포인트 경로를 `head_decouple_finetune.py`의 **위치 인자**로 넘기면
그 스크립트가 공식 라벨로 최종 미세조정을 수행한다 — 사전학습 따로,
공식 라벨 적응 따로다.

환경변수(`CHECKPOINT_PATH`)가 아니라 위치 인자를 쓸 것(2026-08-19 정정) —
`predict.py`의 `CHECKPOINT`가 그 환경변수를 하드코딩으로 무시하던 버그가
있었고, 그 때문에 2026-08-18 실험 두 번 모두 사전학습을 거치지 않은
프로덕션 체크포인트가 그대로 로드됐다. 지금은 predict.py 도 고쳐졌지만,
환경변수보다 명시적 인자가 더 안전하다.

**약한 라벨은 사전학습에만 쓴다 — 평가에는 절대 쓰지 않는다.** 검증은
언제나 공식 라벨(ds.y_coldwave/cold_mask)로만 수행한다(아래 스테이지별로
찍는 F1은 진행 상황 참고용일 뿐 스테이지 선택 기준이 아니다). 라벨 정의를
섞으면 과거 폭염 F1 사고(공식 0.842 vs 근사 0.195)가 되풀이된다
(EXTREME_LABEL_MASKING 도입 배경, README 참고).

백분위 임계값은 학습셋에서만 실측한다(하드코딩 금지, CLAUDE.md 4절) —
전역이나 검증셋에서 구하지 않는다.

head_coldwave 는 **커리큘럼 시작 전 한 번만** 초기화한다 — 직전 공식 라벨로
학습된 편향을 지우고 순수하게 약한 라벨 curriculum 으로만 쌓아올리기
위함이다(이미 공식 정의를 아는 헤드에 약한 신호를 더하면 잡음이 될 뿐이다).
스테이지 사이에는 재초기화하지 않는다 — p20 에서 배운 가중치 위에 p10 이
이어서 학습해야 "중첩(nested)"이 성립한다. 이걸 반복문 안에 두는 실수를
2026-08-18에 실제로 했다(README '기각한 시험' 정정 기록 참고) — 그러면
매 스테이지가 무작위 초기값에서 다시 시작해 앞 단계 학습이 통째로
버려지고, 결과가 "마지막 스테이지 하나만 돌린 것"과 같아진다. 이 파일
아래의 가중치 노름 로그가 그 회귀를 다시 조용히 들여오지 않았는지
매 실행마다 보여준다 — 스테이지 간 노름이 서로 다르면(수렴이 아니라
초기화로 리셋되면 항상 비슷한 값 근처에서 시작한다) 재발을 의심할 것.

실행: python coldwave_nested_pretrain.py --out checkpoints/numerical_trichef_coldwave_pretrain.pt
"""
import argparse
import math

import numpy as np
import torch
import torch.nn as nn

from head_decouple_finetune import freeze_trunk, evaluate, _HEAD_ATTR
from train import collect_historical, WeatherDataset, make_split
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector
from predict import load_model, CHECKPOINT

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH = 2048
# 하위 백분위 3단계 — 점점 좁혀 진짜 사건(포털 공식 정의)에 가까워진다.
# p20(가장 관대) → p10 → p5(공식 한파 비율 6.84%에 가장 근접).
WEAK_PERCENTILES = [20.0, 10.0, 5.0]
EPOCHS_PER_STAGE = 8
LR = 5e-4


def reinit_head(module):
    """Linear 층을 PyTorch 기본 초기화로 되돌린다 — 공식 라벨로 학습된 편향을
    지우고 커리큘럼을 무작위 초기값에서 시작한다(호출은 커리큘럼 시작 전 1회).

    주의: reset_parameters() 는 마지막 층의 편향까지 U(-1/√32, 1/√32)≈0 으로
    되돌린다. 그런데 `_binary_head`(pipeline_model.py)는 그 편향을 일부러
    logit(prior) 로 초기화해 둔다 — 희소 사건에서 "대부분 음성"을 학습 시작
    시점부터 반영하려는 것이다. 따라서 이 함수만 쓰면 헤드가 p≈0.5 에서
    출발하고, 이 스크립트의 train_stage 에는 pos_weight 도 없어 보정되지
    않는다. 반드시 _reinit_head_bias() 로 사전확률을 다시 심을 것.
    """
    for m in module.modules():
        if isinstance(m, nn.Linear):
            m.reset_parameters()


def _reinit_head_bias(module, temp_tgt_train, first_threshold):
    """마지막 Linear 의 편향을 첫 스테이지 약한 라벨의 사전확률 로짓으로 심는다.

    `_binary_head` 가 하던 일을 재초기화 후 복원하는 것이다. 비율은 학습
    표본에서 실측해 쓴다(하드코딩 금지 — CLAUDE.md 4절).
    """
    prior = float((temp_tgt_train <= first_threshold).mean())
    prior = min(max(prior, 1e-4), 1 - 1e-4)
    last = [m for m in module.modules() if isinstance(m, nn.Linear)][-1]
    with torch.no_grad():
        last.bias.fill_(math.log(prior / (1 - prior)))
    print(f"  헤드 편향 재초기화 — 1단계 사전확률 {prior:.4f} "
          f"(로짓 {math.log(prior/(1-prior)):+.3f})")


def head_checksum(module):
    """head_coldwave 가중치의 L2 노름 — 스테이지 간 warm start 를 실행마다
    직접 확인하기 위한 진단(2026-08-19 추가). 재초기화가 반복문 안으로
    되돌아가는 회귀가 생기면, 매 스테이지 학습 전 노름이 비슷한 무작위
    초기값 근처로 반복해서 리셋되는 게 여기서 바로 보인다 — 이전엔 코드
    리뷰로만 잡았던 문제라 실행 결과로도 볼 수 있게 남긴다."""
    with torch.no_grad():
        return math.sqrt(sum((p ** 2).sum().item() for p in module.parameters()))


def train_stage(model, ds, train_idx, weak_label_all, epochs, lr, device):
    optimizer = torch.optim.Adam(
        [p for p in model.parameters() if p.requires_grad], lr=lr)
    idx_t = torch.as_tensor(train_idx)
    weak_t = weak_label_all[idx_t]
    n = len(idx_t)
    for epoch in range(1, epochs + 1):
        model.train()
        perm = torch.randperm(n)
        total_loss, n_batches = 0.0, 0
        for b in range(0, n, BATCH):
            sel = perm[b:b + BATCH]
            sl = idx_t[sel]
            wl = weak_t[sel].to(device).unsqueeze(-1)
            x_num = ds.X_num[sl].to(device)
            x_img = ds.X_img[sl].to(device).float()
            x_txt = ds.X_txt[sl].to(device)
            optimizer.zero_grad()
            model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            loss = nn.functional.binary_cross_entropy_with_logits(
                model._last_coldwave_logit, wl)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            n_batches += 1
        print(f"      epoch {epoch:2d}/{epochs} | 약한라벨 BCE {total_loss/n_batches:.4f}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="checkpoints/numerical_trichef_coldwave_pretrain.pt")
    args = ap.parse_args()

    model, ckpt = load_model(CHECKPOINT, DEVICE)
    print(f"기반 체크포인트 — split_mode={ckpt.get('split_mode')} "
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

    n_tunable = freeze_trunk(model, ["coldwave"])
    print(f"트렁크 동결, head_coldwave만 학습 — {n_tunable:,}개 파라미터")

    temp_tgt_train = ds.y[torch.as_tensor(train_idx), 0].numpy()
    temp_tgt_all = ds.y[:, 0].numpy()

    print(f"\n학습셋 기온 분포(n={len(temp_tgt_train):,})에서 산출한 약한 한파 임계값:")
    thresholds = []
    for p in WEAK_PERCENTILES:
        t = float(np.percentile(temp_tgt_train, p))
        thresholds.append(t)
        n_pos = int((temp_tgt_train <= t).sum())
        print(f"  p{p:>4.1f} → {t:6.2f}°C 이하 ({n_pos:,}개, "
              f"{100*n_pos/len(temp_tgt_train):.1f}%)")

    # 재초기화는 **커리큘럼 시작 전 한 번만** 한다(2026-08-19 수정).
    #
    # 예전에는 이 호출이 for 문 **안**에 있어, 스테이지마다 직전 단계에서 배운
    # 가중치를 지우고 무작위 초기화로 되돌렸다. 그러면 p20·p10 단계는 계산만
    # 하고 버려져 결과가 "p5 한 단계만 돌린 것"과 비트 단위로 같아진다 —
    # 중첩 전이학습(더 쉬운 사건에서 warm start 해 더 극단적인 사건으로
    # 옮겨간다)이라는 방법 자체가 성립하지 않는다. 단계별 참고 F1 이 상승한
    # 것도 학습 누적이 아니라 임계값이 공식 한파 정의에 가까워진 효과였다.
    reinit_head(model.head_coldwave)
    _reinit_head_bias(model.head_coldwave, temp_tgt_train, thresholds[0])
    print(f"\n초기화 직후 head_coldwave 노름: {head_checksum(model.head_coldwave):.4f}")

    for stage, (p, t) in enumerate(zip(WEAK_PERCENTILES, thresholds), 1):
        print(f"\n{'='*70}\n스테이지 {stage}/{len(thresholds)} — p{p:.1f} 이하 "
              f"({t:.2f}°C) 를 양성으로 학습")
        norm_before = head_checksum(model.head_coldwave)
        weak_label_all = torch.tensor((temp_tgt_all <= t).astype(np.float32))
        train_stage(model, ds, train_idx, weak_label_all, EPOCHS_PER_STAGE, LR, DEVICE)
        norm_after = head_checksum(model.head_coldwave)
        print(f"  head_coldwave 노름: {norm_before:.4f} → {norm_after:.4f} "
              f"(직전 스테이지에서 이어졌으면 두 값이 가깝다; 초기화로 리셋됐으면 "
              f"norm_before 가 매번 초기화 직후 값 근처로 되돌아간다)")

        # 진행 상황 참고용 — 공식 라벨로 채점(스테이지 선택 기준 아님, 약한
        # 라벨로 학습한 헤드가 공식 정의와 얼마나 가까워지는지 관찰만 한다).
        _, _, m = evaluate(model, ds, val_idx, DEVICE)
        cf = m["coldwave"]["f1"]
        print(f"  [참고] 공식 라벨 기준 한파 F1(스테이지 선택에 미사용): {cf:.4f}")

    ckpt_out = dict(ckpt)
    ckpt_out["model_state"] = model.state_dict()
    ckpt_out["coldwave_nested_pretrain"] = {
        "weak_percentiles": WEAK_PERCENTILES,
        "weak_thresholds_c": thresholds,
        "epochs_per_stage": EPOCHS_PER_STAGE,
    }
    # val_temp_mae/val_precip_mae 등은 트렁크·다른 헤드가 안 바뀌었으므로
    # 그대로 둔다. head_coldwave 만 바뀌었고, 다음 단계(head_decouple_
    # finetune.py, 공식 라벨)가 한파 지표를 다시 재는 시점에 갱신된다.
    torch.save(ckpt_out, args.out)
    print(f"\n저장: {args.out}")
    print(f"다음 단계: CHECKPOINT_PATH={args.out} python head_decouple_finetune.py "
          f"--heads coldwave --out checkpoints/numerical_trichef_coldwave_nested.pt")


if __name__ == "__main__":
    main()
