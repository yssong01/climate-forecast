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
import os
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
    "coldwave_pathway_check", "coldwave_path_attribution",
    "station_anomaly_investigate",
    "precip_breakdown", "distribution_diagnostics", "gate_behavior_check",
    "neutral_input_check", "patch_extreme_metrics", "backtest_accuracy",
    "conformal_interval_fit", "bootstrap_ci_compare", "rebaseline_compare",
    "promote_checkpoint", "head_decouple_finetune", "coldwave_nested_pretrain",
    "nwp_collector", "collect_nwp_archive", "aerosol_feature_probe",
    "collect_aerosol_archive",
    # CI(refresh-data.yml)가 15분마다 호출한다 — 임포트가 깨지면 적중률
    # 기록이 통째로 멈춘다(2026-09-23 추가).
    "record_online_forecasts",
    # 승격 절차의 수동 작업 4번이 이 스크립트를 부른다(2026-09-23 추가).
    "capture_tab_screenshots",
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
    """부가 특징 조합마다 같은 키가 일관되게 채워지는지 본다.

    수집기 **객체**까지 만들려면 `cache/nwp_archive.json`(52MB)이 필요한데
    이 파일은 gitignore 대상이라 CI 체크아웃에는 없다. 그 사실을 몰라서
    이 시험은 신설 이후 CI 에서 한 번도 통과하지 못했다(2026-09-23 발견 —
    `NWPForecastCollector(required=True)` 가 `FileNotFoundError` 로 죽는다).

    아카이브가 없어도 확인할 수 있는 것(키 구성·특징 집합 전파·대조군
    플래그)은 그대로 보고, 수집기 구성만 아카이브가 있을 때 확인한다.
    "없으면 건너뛴다"를 조용히 하지 않고 결과 문구에 드러낸다.
    """
    from train import aux_dataset_kwargs
    from nwp_collector import ARCHIVE_PATH
    has_archive = os.path.exists(ARCHIVE_PATH)
    cases = [
        ("구버전(부가 특징 없음)", {}),
        ("수치예보 full14", {"use_nwp": True, "nwp_feature_set": "full14"}),
        ("수치예보 compact6", {"use_nwp": True, "nwp_feature_set": "compact6"}),
        ("수치예보 대조군", {"use_nwp_subset": True}),
    ]
    out = []
    for label, extra in cases:
        uses_nwp = bool(extra.get("use_nwp") or extra.get("use_nwp_subset"))
        if uses_nwp and not has_archive:
            # aux_dataset_kwargs() 호출 자체가 수집기를 만들므로 이 조합은
            # 통째로 건너뛴다 — 건너뛴 사실은 아래 결과 문구에 남긴다.
            out.append(f"{label}(아카이브 없음 — 확인 생략)")
            continue
        kw = aux_dataset_kwargs(extra)
        # 키가 없는 구버전이 종전 동작(예외 없이 빈 설정)을 유지하는지
        assert "offseason_negative" in kw, f"{label}: offseason_negative 누락"
        assert kw.get("nwp_feature_set") == extra.get("nwp_feature_set", "full14"), \
            f"{label}: nwp_feature_set 전파 실패"
        if uses_nwp:
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

    # magnitude 쪽 기온 중립화(2026-09-24). 인덱스 유도가 특징 집합을 따라
    # 달라지므로 두 집합 모두 확인한다 — compact6 에는 동시각 기온 편향 열이
    # 없어 Z축 0번 하나만 나와야 한다. 여기서 어긋나면 기온이 중립화되지
    # 않은 열로 새어 들어가 처방이 반쪽이 되는데, forward 는 멀쩡히 돌아
    # 단조성 게이트까지 가서야 드러난다.
    import os as _os
    _prev = _os.environ.get("USE_NWP")
    _os.environ["USE_NWP"] = "1"
    try:
        import importlib
        import train as _train
        importlib.reload(_train)
        for fs, want in (("full14", [0, 25]), ("compact6", [0])):
            _train.NWP_FEATURE_SET = fs
            nf = 14 + feature_dim(fs)
            got = _train.extreme_temp_neutral_index(nf)
            assert got == want, f"{fs}: 기온 중립화 인덱스 {got} ≠ {want}"
    finally:
        if _prev is None:
            _os.environ.pop("USE_NWP", None)
        else:
            _os.environ["USE_NWP"] = _prev
        importlib.reload(_train)

    x = torch.randn(3, 28)
    img, txt = torch.randn(3, 4, 32, 32), torch.randn(3, 12)
    x_t = x.clone(); x_t[:, 0] += 3.0                  # 기온만
    x_n = x.clone(); x_n[:, 25] += 3.0                 # 동시각 기온 편향만

    def _cold(model, xx):
        with torch.no_grad():
            model(num_x=xx, img_x=img, txt_x=txt)
            return model._last_coldwave_logit.clone()

    # (1) magnitude 만 보는 구성(signed_head_input=False)에서는 기온을 흔들어도
    #     극한기상 로짓이 **전혀** 움직이지 않아야 한다 — 이것이 이 처방의 핵심
    #     주장이고, 여기가 새면 단조성 게이트까지 가서야 드러난다.
    m_mag = TriCHEFPipeline(
        num_features=28, im_dim=12, signed_head_input=False,
        extreme_temp_neutral_idx=[0, 25],
        feat_mean=np.zeros(28, dtype=np.float32),
        feat_std=np.ones(28, dtype=np.float32)).eval()
    leak = float((_cold(m_mag, x_t) - _cold(m_mag, x)).abs().max())
    assert leak < 1e-6, f"magnitude 경로로 기온이 새고 있다 ({leak:.2e})"

    # (2) 배포 구성(부호 경로 있음)에서는 **기온이 계속 헤드에 닿아야** 한다.
    #     이 처방은 기온을 빼는 것이 아니라, 단조가 불가능한 경로에서만 빼고
    #     부호가 살아 있는 경로로 몰아주는 것이다. 여기가 0 이면 헤드가 기온을
    #     아예 못 보게 된 것이므로 실패다.
    m_full = TriCHEFPipeline(
        num_features=28, im_dim=12, signed_head_input=True,
        extreme_nwp_neutral_dims=14, extreme_temp_neutral_idx=[0, 25],
        feat_mean=np.zeros(28, dtype=np.float32),
        feat_std=np.ones(28, dtype=np.float32)).eval()
    base = _cold(m_full, x)
    assert float((_cold(m_full, x_t) - base).abs().max()) > 1e-6, \
        "부호 경로까지 막혀 헤드가 기온을 못 본다"
    # 동시각 기온 편향(25번)은 두 경로 모두에서 중립화되므로 아무 영향이 없다.
    assert float((_cold(m_full, x_n) - base).abs().max()) < 1e-6, \
        "수치예보 편향 열이 극한기상 헤드로 새고 있다"
    # 회귀 경로는 원래 magnitude 를 그대로 쓰므로 기온에 반응해야 한다.
    with torch.no_grad():
        r1 = m_full(num_x=x, img_x=img, txt_x=txt).clone()
        r2 = m_full(num_x=x_t, img_x=img, txt_x=txt).clone()
    assert float((r2 - r1).abs().max()) > 1e-6, "회귀 경로까지 중립화됐다"
    shapes.append("기온중립화(magnitude)")
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
