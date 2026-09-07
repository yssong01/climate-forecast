"""
test_diagnostics_smoke.py — 진단·게이트 스크립트가 "적어도 죽지는 않는지" 본다.

**왜 필요한가.** 이 저장소의 진단·게이트 스크립트는 각자 입력 벡터를 다시
만든다. 그래서 Z축 부가 특징을 하나 추가하면 그 사실을 모르는 스크립트가
`mean/std` 와 차원이 어긋나 **즉시 죽는다.** 2026-09-07 점검에서 이 계열의
잠복 결함이 세 건 한꺼번에 드러났다.

  · `seasonal_falsealarm_check.py`·`station_coverage_check.py`·
    `threshold_validation.py` 가 수치예보 체크포인트에서 shape 오류로 죽었다.
  · `coldwave_pathway_check.py` 는 탐침 기준 시각이 아카이브 범위 밖이라
    게이트가 **실행조차 되지 않았다** — 그대로 뒀으면 단조성이 "통과"가
    아니라 "미측정"인 채로 승격이 진행됐다.
  · `calibration_plot_diagnose.py` 는 승격 도중 죽어 관측소별 플롯이 이전
    모델 시점 그대로 남을 뻔했다.

세 건 모두 **한 번 실행해 보기만 했으면** 잡혔다. 도서 AWS·계절 아노말리
실험이 전부 게이트에 닿기 전에 기각돼 드러날 기회가 없었을 뿐, 구조적으로는
"입력 차원을 늘리는 실험이 성공하는 순간 승격 절차가 막히는" 상태였다.

**무엇을 검사하지 않는가.** 이 시험은 결과의 정확성을 보지 않는다 —
스크립트가 임포트되고, 체크포인트를 읽고, 입력 벡터를 만들어 모델을
한 번 통과시킬 수 있는지까지만 본다. 데이터셋 전체 구성(약 22GiB)은
돌리지 않으므로 CI 에서 수 초에 끝난다.

실행: python test_diagnostics_smoke.py
"""
import importlib
import sys
import traceback

import numpy as np
import torch

PASS, FAIL = "\033[92m[PASS]\033[0m", "\033[91m[FAIL]\033[0m"
results = []


def check(name, fn):
    try:
        detail = fn()
        results.append((True, name, detail))
        print(f"  {PASS} {name}" + (f" — {detail}" if detail else ""))
    except Exception as exc:                     # noqa: BLE001
        results.append((False, name, f"{type(exc).__name__}: {exc}"))
        print(f"  {FAIL} {name} — {type(exc).__name__}: {exc}")
        traceback.print_exc(limit=3)


def synthetic_ckpt(num_features, **extra):
    """최소 체크포인트 메타데이터. 가중치는 담지 않는다(로드하지 않으므로)."""
    ck = {
        "num_features": num_features,
        "mean": np.zeros(num_features, dtype=np.float32),
        "std": np.ones(num_features, dtype=np.float32),
        "lead_hours": 6, "embed_dim": 64, "im_dim": 12,
        "split_mode": "group", "split_algo": "hash",
        "temp_mean": 12.0, "temp_std": 10.0, "precip_mean": 0.16,
        "alpha_init": 1.0, "phi_init": 0.0, "orthogonalize": False,
        "persistence_residual": True, "dynamic_gate": True,
        "compact_satellite": True, "re_channels": 4,
        "signed_head_input": True, "wet_prior": 0.06,
    }
    ck.update(extra)
    return ck


# ── ① 모든 진단 스크립트가 임포트되는가 ──────────────────────────
DIAGNOSTIC_MODULES = [
    "eval_cache", "metrics_report", "error_breakdown", "threshold_validation",
    "probability_calibration_fit", "probability_calibration_check",
    "calibration_plot_diagnose", "station_threshold_check",
    "station_coverage_check", "seasonal_falsealarm_check",
    "coldwave_pathway_check", "station_anomaly_investigate",
    "precip_breakdown", "distribution_diagnostics", "gate_behavior_check",
    "neutral_input_check", "patch_extreme_metrics", "backtest_accuracy",
    "conformal_interval_fit", "bootstrap_ci_compare", "rebaseline_compare",
    "promote_checkpoint", "head_decouple_finetune", "coldwave_nested_pretrain",
    "nwp_collector", "collect_nwp_archive", "aerosol_feature_probe",
    "collect_aerosol_archive",
]


def _import_all():
    bad = []
    for m in DIAGNOSTIC_MODULES:
        try:
            importlib.import_module(m)
        except Exception as exc:                 # noqa: BLE001
            bad.append(f"{m}({type(exc).__name__})")
    if bad:
        raise RuntimeError("임포트 실패: " + ", ".join(bad))
    return f"{len(DIAGNOSTIC_MODULES)}개 모듈"


# ── ② 부가 특징 조합마다 aux_dataset_kwargs 가 일관된가 ──────────
def _aux_kwargs_consistency():
    from train import aux_dataset_kwargs
    cases = [
        ("구버전(부가 특징 없음)", {}, 14),
        ("수치예보 full14", {"use_nwp": True, "nwp_feature_set": "full14"}, 28),
        ("수치예보 compact6", {"use_nwp": True, "nwp_feature_set": "compact6"}, 20),
        ("수치예보 대조군", {"use_nwp_subset": True}, 14),
    ]
    out = []
    for label, extra, _nf in cases:
        kw = aux_dataset_kwargs(extra)
        # 키가 없는 구버전이 종전 동작(예외 없이 빈 설정)을 유지하는지
        assert "offseason_negative" in kw, f"{label}: offseason_negative 누락"
        if extra.get("use_nwp") or extra.get("use_nwp_subset"):
            assert kw.get("nwp_collector") is not None, f"{label}: 수집기 누락"
            assert kw["nwp_features"] == bool(extra.get("use_nwp")), label
        out.append(label)
    return " / ".join(out)


# ── ③ 부가 특징 차원별로 모델 forward 가 통과하는가 ──────────────
def _forward_all_dims():
    from pipeline_model import TriCHEFPipeline
    from nwp_collector import FEATURE_SETS, feature_dim
    shapes = []
    for fs in FEATURE_SETS:
        nf = 14 + feature_dim(fs)
        m = TriCHEFPipeline(
            num_features=nf, im_dim=12, signed_head_input=True,
            extreme_nwp_neutral_dims=feature_dim(fs),
            feat_mean=np.zeros(nf, dtype=np.float32),
            feat_std=np.ones(nf, dtype=np.float32)).eval()
        with torch.no_grad():
            o = m(num_x=torch.randn(3, nf), img_x=torch.randn(3, 4, 32, 32),
                  txt_x=torch.randn(3, 12))
        assert tuple(o.shape) == (3, 2), f"{fs}: 출력 shape {o.shape}"
        assert m._last_coldwave_logit is not None, f"{fs}: 한파 로짓 없음"
        shapes.append(f"{fs}(nf={nf})")
    # 계절 중립화도 같은 경로를 타는지
    m = TriCHEFPipeline(
        num_features=28, im_dim=12, signed_head_input=True,
        extreme_nwp_neutral_dims=14, extreme_neutral_idx=[12, 13],
        feat_mean=np.zeros(28, dtype=np.float32),
        feat_std=np.ones(28, dtype=np.float32)).eval()
    assert m.extreme_neutral_idx[:2] == [12, 13], "계절 인덱스 병합 실패"
    assert len(m.extreme_neutral_idx) == 16, "인덱스 개수 불일치"
    with torch.no_grad():
        m(num_x=torch.randn(3, 28), img_x=torch.randn(3, 4, 32, 32),
          txt_x=torch.randn(3, 12))
    shapes.append("계절중립화(16개)")
    return " / ".join(shapes)


# ── ④ 판정선 조회가 모든 체크포인트 형태에서 동작하는가 ──────────
def _threshold_paths():
    from predict import event_threshold, STATION_EVENT_THRESH_OVERRIDES
    legacy = synthetic_ckpt(14)                       # 키 없음 → 전역 예외 적용
    modern = synthetic_ckpt(28, station_thresh_overrides=[])
    (stn, ev) = next(iter(STATION_EVENT_THRESH_OVERRIDES))
    a, b = event_threshold(ev, stn, legacy), event_threshold(ev, stn, modern)
    assert abs(a - STATION_EVENT_THRESH_OVERRIDES[(stn, ev)]) < 1e-9, \
        "구버전에서 관측소 예외가 적용되지 않았다"
    assert abs(b - 0.5) < 1e-9, "신버전에서 예외가 잘못 적용됐다"
    return f"{stn}/{ev} 구버전 {a:.4f} · 신버전 {b:.4f}"


# ── ⑤ 배포 체크포인트가 실제로 로드되고 추론 가능한가 ────────────
def _deployed_checkpoints():
    import os
    from predict import load_model
    out = []
    for p in ("./checkpoints/numerical_trichef.pt",
              "./checkpoints/numerical_trichef_12h.pt"):
        if not os.path.exists(p):
            out.append(f"{os.path.basename(p)}(없음, 건너뜀)")
            continue
        m, c = load_model(p, "cpu")
        nf = c["num_features"]
        with torch.no_grad():
            o = m(num_x=torch.zeros(2, nf), img_x=torch.zeros(2, 4, 32, 32),
                  txt_x=torch.zeros(2, c.get("im_dim", 12)))
        assert tuple(o.shape) == (2, 2)
        out.append(f"{os.path.basename(p)}(nf={nf})")
    return " / ".join(out)


def main():
    print("=" * 70)
    print(" 진단·게이트 스크립트 연기 시험")
    print("=" * 70)
    print("\n[T1] 모든 진단 모듈 임포트")
    check("임포트", _import_all)
    print("\n[T2] 부가 특징 조합별 aux_dataset_kwargs 일관성")
    check("aux_dataset_kwargs", _aux_kwargs_consistency)
    print("\n[T3] 부가 특징 차원별 모델 forward")
    check("forward", _forward_all_dims)
    print("\n[T4] 판정선 조회(구버전/신버전 체크포인트)")
    check("event_threshold", _threshold_paths)
    print("\n[T5] 배포 체크포인트 로드·추론")
    check("배포 체크포인트", _deployed_checkpoints)

    n_ok = sum(1 for ok, _, _ in results if ok)
    print("\n" + "=" * 70)
    print(f" 결과: {n_ok}/{len(results)} 통과")
    print("=" * 70)
    if n_ok != len(results):
        sys.exit(1)


if __name__ == "__main__":
    main()
