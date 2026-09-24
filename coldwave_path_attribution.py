"""
coldwave_path_attribution.py — 한파 헤드의 기온 비단조성이 **어느 경로**로
들어오는지 가른다(2026-09-24).

배경 — `coldwave_pathway_check.py` 가 재는 단조성은 지금까지 7개 구성 중 2개
(현행 배포 계열)에서만 PASS 했다. 입력·손실 수준의 개입을 두 번 시도해 두 번
다 빗나갔다:

  ① 한파 헤드 디커플링(2026-09-06) — 심각도만 0.710→0.448, 상관은 그대로 FAIL
  ② 계절 성분 중립화(2026-09-07) — 계절의존만 0.47→0.44, 기온진폭도 함께
     0.83→0.62 로 내려가 판정이 PASS→WARN 으로 **나빠졌다**

두 실패의 공통점은 "헤드가 기온을 더 보게" 만들지 못했다는 것이다. 그런데
**두 시도 모두 기온이 헤드에 어느 경로로 도달하는지 먼저 재지 않고** 처방부터
했다. 이 스크립트가 그 빠진 단계다 — 수치예보 승격을 성사시킨 것도 처방이
아니라 원인 귀속(대조군 측정)이 먼저였다.

## 기온이 한파 헤드에 닿는 두 경로

`TriCHEFPipeline.forward()` 에서 극한기상 헤드의 입력은

    _ext_in = cat([magnitude, v_z_ext])

이고, 기온(`num_x[:, 0]`)은 **두 갈래로** 여기 도달한다.

  (A) magnitude 경로 — `magnitude = √((w_re·v_re)² + (w_im·v_im)² + (w_z·v_z)²)`.
      기온은 `v_z = enc_z(num_x)` 와 게이트 `w = gate(num_x)` 를 통해 들어온다.
      **제곱이 부호를 없애므로 이 경로는 구조적으로 기온에 대해 단조가 아니다**
      — 학습 평균을 중심으로 V 자가 된다. 애초에 부호 표현 `v_z` 를 헤드에
      따로 넣은 이유가 이것이다(CLAUDE.md 5절).

  (B) 부호 경로 — `v_z_ext = enc_z(num_x)`(중립화 열은 0). 부호가 살아 있어
      원리적으로 단조가 가능한 유일한 경로다.

가설: 판정이 구성·seed 마다 뒤집히는 이유는 (A)가 (B)를 이길 때가 있기
때문이다. 그렇다면 처방은 "헤드가 기온을 더 보게" 가 아니라 **(A)에서 기온을
빼는 것**이며, 이는 `EXTREME_NWP_NEUTRAL` 로 이미 검증된 수단의 확장이다.

## 방법

`coldwave_pathway_check.py` 와 **완전히 같은 격자**(12달 × 기온 −15~35°C,
같은 기준 레코드 12개, 같은 `analyse()`)를 쓰되, 경로마다 다른 입력을 먹인다.

  both        : 두 경로 모두 흔든 기온 → 게이트가 재는 값 그대로(대조 기준)
  mag_frozen  : (A)에만 기준 기온을 고정, (B)는 흔든다 → 부호 경로 단독 성능
  sgn_frozen  : (B)에만 기준 기온을 고정, (A)는 흔든다 → magnitude 경로 단독
  both_frozen : 둘 다 고정 → 기온진폭이 0 이어야 한다(시험의 건전성 확인)

달은 네 조건 모두에서 함께 움직인다 — 얼리는 것은 기온뿐이다.

**읽는 법.** `mag_frozen` 에서 상관이 뚜렷한 음수로 돌아오면 (B)는 멀쩡하고
(A)가 판정을 뒤집고 있다는 뜻이다. `sgn_frozen` 에서 양의 상관(역전)이 크게
나오면 그 자체로 (A)가 역전의 원천임을 보인다. 둘 다 아니면 가설이 틀린
것이고, 그때는 부호 경로 자체에 단조 제약을 걸어야 한다.

실행:
    python coldwave_path_attribution.py [체크포인트 ...] [--head=coldwave]

인자를 안 주면 배포본과, 단조성이 FAIL 로 알려진 compact6 계열을 함께 잰다
— 한 구성만 보면 "그 체크포인트의 성질"과 "구조의 성질"을 구분할 수 없다.
"""
import sys

import numpy as np
import torch

from predict import CHECKPOINT, load_model
from coldwave_pathway_check import (
    DEVICE, TEMPS, MONTHS, make_x, pick_base_records, analyse,
)
from weather_collector import STATION_COORDS
from interp_field_collector import InterpolatedFieldCollector
from tendency_collector import TendencyCollector

# 인자를 안 줬을 때 함께 재는 구성 — 판정이 갈린 것으로 기록된 계열을 섞는다
# (`coldwave-monotonicity-is-fragile`: full14/42 PASS, compact6/42·43 FAIL).
DEFAULT_EXTRA = [
    "checkpoints/numerical_trichef_nwp_c6_20260907.pt",
    "checkpoints/numerical_trichef_nwp_compact6_s43.pt",
]

CONDITIONS = ("both", "mag_frozen", "sgn_frozen", "both_frozen")


def extreme_logit(model, x_mag, x_sgn, img, txt, head="coldwave"):
    """`forward()` 의 극한기상 경로만 떼어내 두 경로에 다른 입력을 먹인다.

    `x_mag` 는 magnitude 경로(인코더 3개 + 게이트)에, `x_sgn` 은 부호 경로
    (`enc_z` 재통과)에 들어간다. 둘이 같으면 `forward()` 와 정확히 같은 값이
    나와야 하며, `assert_faithful()` 이 그것을 실제로 확인한다 — 이 스크립트가
    측정하는 것이 모델의 실제 동작이 아니라 재구현의 동작이면 결론이 통째로
    무의미해진다.
    """
    # magnitude 경로가 실제로 받는 입력. `extreme_temp_neutral_idx` 가 켜진
    # 체크포인트는 그 열이 이미 0 이므로, 아래 `mag_frozen` 조건은 자연히
    # `both` 와 같아지고 `sgn_frozen` 은 기온 무반응이 된다 — 그 서명 자체가
    # 배선이 맞다는 확인이다.
    x_m = x_mag
    if model.extreme_temp_neutral_idx:
        x_m = x_mag.clone()
        x_m[:, model.extreme_temp_neutral_idx] = 0.0
    v_re, v_im, v_z = model.encode(x_m, img, txt)
    if model.orthogonalize:
        from pipeline_model import gram_schmidt_3axis
        v_re, v_im, v_z = gram_schmidt_3axis(v_re, v_im, v_z, model.gs_eps)
    if model.dynamic_gate:
        w = model.gate(x_m)
        w_re, w_im, w_z = w[:, 0:1], w[:, 1:2], w[:, 2:3]
    else:
        w_re, w_im, w_z = 1.0, model.alpha, model.phi
    magnitude = torch.sqrt(
        (w_re * v_re) ** 2 + (w_im * v_im) ** 2 + (w_z * v_z) ** 2 + 1e-7
    )
    if model.extreme_neutral_idx:
        num_neutral = x_sgn.clone()
        num_neutral[:, model.extreme_neutral_idx] = 0.0
        v_z_ext = model.enc_z(num_neutral)
    else:
        v_z_ext = model.enc_z(x_sgn)
    ext_in = (torch.cat([magnitude, v_z_ext], dim=-1)
              if model.signed_head_input else magnitude)
    return getattr(model, f"head_{head}")(ext_in)


def assert_faithful(model, x, img, txt, head):
    """재구현이 `forward()` 와 같은 값을 내는지 확인한다(경로 분리 전)."""
    with torch.no_grad():
        model(num_x=x, img_x=img, txt_x=txt)
        ref = getattr(model, f"_last_{head}_logit")
        got = extreme_logit(model, x, x, img, txt, head)
    delta = float((ref - got).abs().max())
    if delta > 1e-5:
        raise RuntimeError(
            f"재구현이 forward() 와 다르다 (최대 차이 {delta:.3e}) — "
            f"pipeline_model.forward() 가 바뀌었을 수 있다. 이 스크립트를 "
            f"고치기 전에는 결과를 신뢰하지 말 것.")
    return delta


def build_grids(model, ckpt, base, img, txt, head="coldwave"):
    """네 조건의 (달, 기온) 격자를 한 번에 만든다."""
    hh = str(base.get("timestamp", "202601011200"))[8:12] or "1200"
    nwp_fixed = None
    if ckpt.get("use_nwp", False):
        from nwp_collector import shared as nwp_shared
        nwp_fixed = nwp_shared().raw_forecast(base, ckpt["lead_hours"])
        if nwp_fixed is None:
            raise RuntimeError(
                f"탐침 기준 레코드({base.get('stn')} {base.get('timestamp')})의 "
                f"수치예보가 아카이브에 없다.")
    t_base = float(base["temperature"])

    grids = {c: np.zeros((len(MONTHS), len(TEMPS))) for c in CONDITIONS}
    with torch.no_grad():
        for i, (_, mmdd) in enumerate(MONTHS):
            # 같은 달에서 "흔든 기온"과 "고정 기온" 두 벡터를 만든다. 달은
            # 양쪽에서 똑같이 움직이므로, 두 벡터의 차이는 기온뿐이다 —
            # 기온에서 파생되는 수치예보 열(예보−현재 편차 등)까지 한꺼번에
            # 옮겨진다. 열 번호를 손으로 세지 않는 이유가 이것이다.
            r_fix = dict(base)
            r_fix["temperature"] = t_base
            r_fix["timestamp"] = f"2026{mmdd}{hh}"
            x_fix = make_x(ckpt, r_fix, nwp_fixed)
            for j, t in enumerate(TEMPS):
                r = dict(base)
                r["temperature"] = t
                r["timestamp"] = f"2026{mmdd}{hh}"
                x_var = make_x(ckpt, r, nwp_fixed)
                for cond, xm, xs in (
                    ("both",        x_var, x_var),
                    ("mag_frozen",  x_fix, x_var),
                    ("sgn_frozen",  x_var, x_fix),
                    ("both_frozen", x_fix, x_fix),
                ):
                    logit = extreme_logit(model, xm, xs, img, txt, head)
                    grids[cond][i, j] = float(torch.sigmoid(logit).item())
    return grids


def verdict_of(corr, expect_up):
    if np.isnan(corr):
        return "무반응"
    if (corr > 0.3) if expect_up else (corr < -0.3):
        return "정상"
    if (corr < -0.3) if expect_up else (corr > 0.3):
        return "★역전★"
    return "혼재"


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--head=")]
    head = next((a.split("=", 1)[1] for a in sys.argv[1:]
                 if a.startswith("--head=")), "coldwave")
    expect_up = (head == "heatwave")
    paths = args or ([CHECKPOINT] + DEFAULT_EXTRA)

    _meta = None
    for p in paths:
        try:
            c = torch.load(p, map_location="cpu", weights_only=True)
        except Exception:
            continue
        if c.get("use_nwp") or c.get("use_nwp_subset"):
            _meta = c
            break
    probes = pick_base_records(ckpt=_meta)
    print(f"기준 레코드 {len(probes)}개 (학습 캐시 균등 표집, 게이트와 동일)")
    print(f"  {probes[0][0]} ~ {probes[-1][0]}\n")

    prepared = []
    for ts, b, allr in probes:
        sat = InterpolatedFieldCollector(allr, STATION_COORDS)
        tnd = TendencyCollector(allr)
        prepared.append((
            ts, b,
            torch.tensor(sat.get_image(b), dtype=torch.float32).unsqueeze(0).to(DEVICE),
            torch.tensor(tnd.encode_single(b), dtype=torch.float32).unsqueeze(0).to(DEVICE),
        ))

    for p in paths:
        try:
            model, ckpt = load_model(p, DEVICE)
        except Exception as e:
            print(f"{p}  로드 실패: {str(e)[:60]}\n")
            continue
        model.eval()

        ts0, b0, img0, txt0 = prepared[0]
        fixed0 = None
        if ckpt.get("use_nwp", False):
            from nwp_collector import shared as nwp_shared
            fixed0 = nwp_shared().raw_forecast(b0, ckpt["lead_hours"])
        delta = assert_faithful(model, make_x(ckpt, b0, fixed0), img0, txt0, head)

        print(f"■ {p}")
        print(f"  재구현 대조: forward() 와 최대 차이 {delta:.2e} (통과)")
        print(f"  {'조건':<14}{'기온진폭':>10}{'계절진폭':>10}{'계절의존':>10}"
              f"{'최악상관':>10}{'심각도':>10}  판정")

        # 조건마다 탐침 12개 중 **최악**을 취한다 — 게이트와 같은 규칙이다.
        acc = {c: {"at": [], "as": [], "corr": [], "sev": []} for c in CONDITIONS}
        for ts, b, img, txt in prepared:
            try:
                grids = build_grids(model, ckpt, b, img, txt, head)
            except RuntimeError as e:
                print(f"  {ts}: {str(e)[:70]}")
                continue
            for c in CONDITIONS:
                at_i, as_i, corr_i, sev_i = analyse(grids[c])
                acc[c]["at"].append(at_i)
                acc[c]["as"].append(as_i)
                acc[c]["corr"].append(corr_i)
                acc[c]["sev"].append(sev_i[1] if expect_up else sev_i[0])

        for c in CONDITIONS:
            arr = np.array(acc[c]["corr"], dtype=float)
            at = float(np.mean(acc[c]["at"]))
            as_ = float(np.mean(acc[c]["as"]))
            ratio = as_ / at if at > 1e-9 else float("inf")
            if np.all(np.isnan(arr)):
                corr = float("nan")
            else:
                # 폭염은 최솟값이, 한파는 최댓값이 최악이다.
                k = int(np.nanargmin(arr) if expect_up else np.nanargmax(arr))
                corr = float(arr[k])
            sev = float(np.max(acc[c]["sev"])) if acc[c]["sev"] else float("nan")
            print(f"  {c:<14}{at:>10.4f}{as_:>10.4f}{ratio:>10.2f}"
                  f"{corr:>10.4f}{sev:>10.3f}  {verdict_of(corr, expect_up)}")

        # both_frozen 은 기온을 양쪽에서 얼렸으므로 기온진폭이 0 이어야 한다.
        # 0 이 아니면 얼리지 못한 기온 경로가 더 있다는 뜻이고, 그러면 위
        # 귀속이 성립하지 않는다 — 조용히 넘어가지 않고 크게 알린다.
        leak = float(np.mean(acc["both_frozen"]["at"]))
        if leak > 1e-6:
            print(f"  ⚠ both_frozen 기온진폭 {leak:.6f} ≠ 0 — 얼리지 못한 "
                  f"기온 경로가 남아 있다. 귀속 결과를 신뢰하지 말 것.")
        print()


if __name__ == "__main__":
    main()
