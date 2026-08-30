"""
re_axis_channel_saliency.py — Re축 인코더가 입력 채널 중 무엇에 반응하는지
그래디언트로 잰다.

**왜 필요한가.** 게이트 축 배분(README)은 "Re축을 얼마나 쓰는가"만 알려주고
"Re축 안에서 어느 채널에 반응하는가"는 알려주지 않는다. 방금 기각한 5채널
(강수 채널 추가) 실험에서, 부트스트랩 CI 재검증으로 실제 유의한 변화가
기온 개선·황사 악화 둘뿐임이 드러났다(rejected-re-channel-precip 메모) —
새로 넣은 강수 채널이 애초에 그래디언트를 거의 못 받았는지(채널 자체가
무의미), 아니면 학습은 됐는데 트렁크·게이트 공유 경쟁 때문에 황사 헤드가
밀렸는지를 이 스크립트가 가른다. DenseNet_vs_Models 프로젝트의 Grad-CAM
활용 방식(모델이 "어디를 보는지" 사후 진단)을 참고해 이식했다
(cross-project-technique-survey 메모 1단계 항목 2).

**방법 — CAM이 아니라 채널별 vanilla gradient.** compact CNN(2.8만
파라미터, pipeline_model.SatelliteEncoder)은 최종 conv가 4×4 공간
해상도뿐이라 클래스 활성화 지도(CAM)로 공간 위치를 시각화하기엔 해상도가
너무 낮다. 대신 입력 이미지 x_img 에 대해 각 출력(기온·강수·폭염·한파·
황사 로짓)의 그래디언트를 직접 역전파해, 채널별 L2 노름으로 "이 출력이
어느 채널에 얼마나 민감한가"를 비교한다 — 공간 위치가 아니라 채널
중요도가 지금 필요한 질문에 더 직접적인 답이다.

**표본 구성 — eval_cache 를 쓰지 않는 이유.** 그래디언트가 필요해 저장된
예측값(eval_cache)만으론 부족하다. 소수 표본(수십 개)에 대해서만 원본
입력을 다시 구성하면 되므로, WeatherDataset처럼 전체 130만+건을 한꺼번에
텐서로 만들 필요가 없다 — get_image()/encode_single() 단일 표본 API를
그대로 쓴다(predict.py 실시간 추론 경로와 동일 함수, InterpolatedFieldCollector·
TendencyCollector 생성 자체는 가벼운 인덱스 구축이라 전체 records 를 넘겨도
안전하다).

실행 (GPU 컨테이너):
  python re_axis_channel_saliency.py <체크포인트> [--stn 108] [--n 8]
"""
import argparse
import sys

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from train import collect_historical, record_to_vec
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
# interp_field_collector._CHANNEL_SPEC_FULL 순서와 동일.
CHANNEL_NAMES = ["기온", "습도", "기압", "풍속", "강수"]
TARGETS = ["temp", "precip", "heat", "cold", "dust"]


def build_inputs(record, field_collector, txt_collector, ckpt, device):
    mean = np.asarray(ckpt["mean"], dtype=np.float32)
    std = np.asarray(ckpt["std"], dtype=np.float32)
    nf = ckpt.get("num_features", len(mean))
    num_vec = np.array(record_to_vec(record), dtype=np.float32)[:nf]
    num_norm = (num_vec - mean) / std
    x_num = torch.tensor(num_norm, dtype=torch.float32).unsqueeze(0).to(device)

    img = field_collector.get_image(record)
    x_img = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
    x_img.requires_grad_(True)

    im_dim = ckpt.get("im_dim", 384)
    if im_dim >= 128:
        from text_collector import SimulatedTextCollector
        txt_vec = SimulatedTextCollector().encode_single(record)
    else:
        txt_vec = txt_collector.encode_single(record)
    x_txt = torch.tensor(txt_vec, dtype=torch.float32).unsqueeze(0).to(device)

    return x_num, x_img, x_txt


def select_samples(records, stn, n, seed=0):
    stn_records = [r for r in records if str(r.get("stn")) == str(stn)]
    by_precip = sorted(stn_records, key=lambda r: r.get("precipitation", 0.0), reverse=True)[:n]
    by_heat = sorted(stn_records, key=lambda r: r.get("temperature", 0.0), reverse=True)[:n]
    by_cold = sorted(stn_records, key=lambda r: r.get("temperature", 0.0))[:n]
    rng = np.random.RandomState(seed)
    idx = rng.choice(len(stn_records), size=min(n, len(stn_records)), replace=False)
    calm = [stn_records[i] for i in idx]
    return {"강수최대": by_precip, "고온": by_heat, "저온": by_cold, "무작위(기준)": calm}


def target_logits(model):
    return {
        "temp": None,   # 아래서 out[0,0] 으로 별도 처리
        "precip": None,  # 아래서 out[0,1] 으로 별도 처리
        "heat": getattr(model, "_last_heatwave_logit", None),
        "cold": getattr(model, "_last_coldwave_logit", None),
        "dust": getattr(model, "_last_dust_logit", None),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=CHECKPOINT)
    ap.add_argument("--stn", default="108")
    ap.add_argument("--n", type=int, default=8)
    args = ap.parse_args()

    model, ckpt = load_model(args.ckpt, DEVICE)
    model.eval()
    re_channels = ckpt.get("re_channels", 4)
    names = CHANNEL_NAMES[:re_channels]
    print(f"체크포인트: {args.ckpt} (re_channels={re_channels}, 관측소={args.stn})")

    records = collect_historical()
    field_collector = InterpolatedFieldCollector(records, STATION_COORDS, n_bands=re_channels)
    txt_collector = TendencyCollector(records) if ckpt.get("im_dim", 384) < 128 else None

    groups = select_samples(records, args.stn, args.n)

    accum = {g: {t: np.zeros(re_channels) for t in TARGETS} for g in groups}
    counts = {g: 0 for g in groups}

    for gname, recs in groups.items():
        for r in recs:
            x_num, x_img, x_txt = build_inputs(r, field_collector, txt_collector, ckpt, DEVICE)
            out = model(num_x=x_num, img_x=x_img, txt_x=x_txt)
            logits = target_logits(model)
            logits["temp"] = out[0, 0]
            logits["precip"] = out[0, 1]

            for t in TARGETS:
                val = logits[t]
                if val is None:
                    continue
                if x_img.grad is not None:
                    x_img.grad.zero_()
                val.backward(retain_graph=True)
                if x_img.grad is None:
                    continue
                g = x_img.grad[0].detach().cpu().numpy()   # (C, H, W)
                per_ch = np.linalg.norm(g.reshape(g.shape[0], -1), axis=1)
                accum[gname][t] += per_ch
            counts[gname] += 1

    print(f"\n채널 순서: {names}")
    for gname in groups:
        n = max(counts[gname], 1)
        print(f"\n[{gname}] (표본 {counts[gname]}개 평균 |∂출력/∂입력채널|_2)")
        print(f"  {'타깃':<8}" + "".join(f"{c:>10}" for c in names))
        for t in TARGETS:
            vals = accum[gname][t] / n
            if np.allclose(vals, 0):
                continue
            print(f"  {t:<8}" + "".join(f"{v:>10.5f}" for v in vals))


if __name__ == "__main__":
    sys.exit(main())
