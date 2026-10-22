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
    "baseline_suite", "precip_gbm", "extreme_gbm",
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
    # 같은 워크플로의 관측 창 갱신 — 임포트가 깨지면 창이 멈춘다(2026-10-21 추가).
    "refresh_deploy_data",
    # 2026-09-27 신설 — 남아 있던 세 항목을 닫은 측정 도구들.
    "aerosol_gbm_check", "precip_floor_check", "migrate_training_wind_dir",
    # 2026-10-21 신설 — 관측소별 강수 F1 을 검증셋·온라인에서 같은 정의로 잰다.
    "station_precip_check",
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

    # 융합 모드 3종(2026-09-24). 파라미터 수가 **완전히 같아야** 대조 실험이
    # 아키텍처 규모에 교란되지 않는다. `zonly` 는 Re·Im 을 실제로 무시하는지,
    # `linear` 는 부호를 실제로 보존하는지도 함께 확인한다.
    npar = {}
    for mode in ("modulus", "linear", "zonly"):
        m = TriCHEFPipeline(
            num_features=28, im_dim=12, signed_head_input=True, fusion=mode,
            feat_mean=np.zeros(28, dtype=np.float32),
            feat_std=np.ones(28, dtype=np.float32)).eval()
        npar[mode] = sum(p.numel() for p in m.parameters())
        with torch.no_grad():
            o = m(num_x=torch.randn(3, 28), img_x=torch.randn(3, 4, 32, 32),
                  txt_x=torch.randn(3, 12))
        assert tuple(o.shape) == (3, 2), f"{mode}: 출력 shape {o.shape}"
    assert len(set(npar.values())) == 1, f"모드별 파라미터 수가 다르다: {npar}"

    x, img, txt = torch.randn(3, 28), torch.randn(3, 4, 32, 32), torch.randn(3, 12)
    mz = TriCHEFPipeline(
        num_features=28, im_dim=12, signed_head_input=True, fusion="zonly",
        feat_mean=np.zeros(28, dtype=np.float32),
        feat_std=np.ones(28, dtype=np.float32)).eval()
    with torch.no_grad():
        a = mz(num_x=x, img_x=img, txt_x=txt).clone()
        b = mz(num_x=x, img_x=torch.randn(3, 4, 32, 32),
               txt_x=torch.randn(3, 12)).clone()
    assert float((a - b).abs().max()) < 1e-6, "zonly 인데 Re·Im 이 출력을 바꾼다"

    # 게이트를 켠 구성으로 만든다 — 배포 설정이 `dynamic_gate=True` 이고,
    # 꺼져 있으면 `self.gate` 자체가 없다.
    ml = TriCHEFPipeline(
        num_features=28, im_dim=12, signed_head_input=True, fusion="linear",
        dynamic_gate=True,
        feat_mean=np.zeros(28, dtype=np.float32),
        feat_std=np.ones(28, dtype=np.float32)).eval()
    with torch.no_grad():
        ml(num_x=x, img_x=img, txt_x=txt)
        v_re, v_im, v_z = ml.encode(x, img, txt)
        w = ml.gate(x)
        s = ml._fuse(v_re, v_im, v_z, w[:, 0:1], w[:, 1:2], w[:, 2:3])
    assert float(s.min()) < 0, "linear 인데 융합값이 전부 비음수다(부호 미보존)"
    shapes.append(f"융합 3모드(파라미터 {next(iter(npar.values())):,} 동일)")
    return " / ".join(shapes)


# ── ③-2 기온 전용 보조 체크포인트 경로 ──────────────────────────
def _temp_checkpoint_path():
    """배포 경로가 **절대경로**이고 `accuracy` 와 **같은 값**인지 확인한다.

    왜 시험으로 고정하는가(2026-09-25) — 이 상수를 상대경로로 넣었다가
    `refresh-data.yml` 의 레이아웃(코드와 데이터를 다른 디렉터리에 체크아웃하고
    데이터 쪽을 작업 디렉터리로 사용)에서 12개 관측소 전부가
    FileNotFoundError 로 실패했다. `record_online_forecasts.DEFAULT_CHECKPOINT`
    가 같은 이유로 이미 절대경로를 쓰고 있었는데 그 함정을 다시 만든 것이다.

    두 모듈이 기본값을 따로 갖는 이유는 `accuracy` 가 predict 를 임포트하면
    torch·train 까지 끌고 오기 때문이다(수집 스크립트도 이 모듈을 쓴다).
    한 곳으로 합칠 수 없으므로 **일치를 여기서 강제한다.**
    """
    import os
    import importlib
    import predict
    import accuracy as _acc

    assert os.path.isabs(predict.TEMP_CHECKPOINT), (
        f"TEMP_CHECKPOINT 가 상대경로다: {predict.TEMP_CHECKPOINT}")

    # accuracy 의 기본값을 꺼내 비교한다 — 환경변수가 없을 때의 경로.
    prev = os.environ.pop("TEMP_CHECKPOINT_PATH", None)
    try:
        importlib.reload(_acc)
        acc_default = os.path.join(
            os.path.dirname(os.path.abspath(_acc.__file__)),
            "checkpoints", "numerical_trichef_temp.pt")
        assert os.path.abspath(acc_default) == os.path.abspath(
            predict.TEMP_CHECKPOINT), (
            f"predict 와 accuracy 의 기본 경로가 다르다:\n"
            f"  predict  {predict.TEMP_CHECKPOINT}\n  accuracy {acc_default}")
        # 실제로 존재해야 배포가 성립한다(gitignore 부정 규칙으로 추적 중).
        assert os.path.exists(predict.TEMP_CHECKPOINT), (
            f"배포 경로에 기온 전용 체크포인트가 없다: {predict.TEMP_CHECKPOINT}")
        # 신원에 보조 모델이 실제로 섞이는가 — 빠뜨리면 서로 다른 세대의
        # 적중률이 한 줄로 뭉친다.
        main = os.path.join(os.path.dirname(os.path.abspath(_acc.__file__)),
                            "checkpoints", "numerical_trichef.pt")
        with_temp = _acc.model_fingerprint(main)
        os.environ["TEMP_CHECKPOINT_PATH"] = "/nonexistent.pt"
        importlib.reload(_acc)
        without = _acc.model_fingerprint(main)
        assert with_temp != without, "model_id 가 보조 모델을 반영하지 않는다"
    finally:
        os.environ.pop("TEMP_CHECKPOINT_PATH", None)
        if prev is not None:
            os.environ["TEMP_CHECKPOINT_PATH"] = prev
        importlib.reload(_acc)
    return f"절대경로 · predict≡accuracy · 신원 반영"


# ── ③-3 강수 전용 GBM ───────────────────────────────────────────
def _precip_gbm_path():
    """배포 GBM 이 있고, numpy 추론기가 동작하며, 신원·구간 출처가 맞는가.

    셋을 함께 본다(2026-09-26).
      ① 경로가 절대경로이고 `predict` 와 `accuracy` 의 기본값이 같은가 —
         상대경로는 CI 레이아웃에서 못 찾는다(같은 함정을 이미 한 번 겪었다).
      ② 체크포인트의 강수 예측구간이 **GBM 기준으로 적합됐다는 표시**와
         실제 강수 출처가 맞는가 — 어긋나면 다른 모델의 오차 분포로 구간을
         그리게 되고 화면은 멀쩡해 보인다.
      ③ 입력 차원·리드타임이 어긋나는 조합을 막는가.
    """
    import os
    import torch
    import predict
    import precip_gbm as pg

    assert os.path.isabs(predict.PRECIP_GBM), \
        f"PRECIP_GBM 이 상대경로다: {predict.PRECIP_GBM}"
    acc_default = os.path.join(
        os.path.dirname(os.path.abspath(predict.__file__)),
        "checkpoints", "precip_gbm.npz")
    assert os.path.abspath(acc_default) == os.path.abspath(predict.PRECIP_GBM), \
        "predict 와 accuracy 의 GBM 기본 경로가 다르다"
    assert os.path.exists(predict.PRECIP_GBM), \
        f"배포 경로에 강수 GBM 이 없다: {predict.PRECIP_GBM}"

    amt, occ, meta = pg.load(predict.PRECIP_GBM)
    assert amt is not None, "GBM 로드 실패"
    nf = int(meta["meta_num_features"])
    x = np.zeros((3, nf), dtype=np.float32)
    a, o = amt.predict(x), occ.predict_proba1(x)
    assert a.shape == (3,) and o.shape == (3,), "추론 출력 shape 불일치"
    assert np.all((o >= 0) & (o <= 1)), "확률이 [0,1] 밖이다"

    ck = torch.load(os.path.join(os.path.dirname(os.path.abspath(predict.__file__)),
                                 "checkpoints", "numerical_trichef.pt"),
                    map_location="cpu", weights_only=True)
    src = (ck.get("conformal_interval") or {}).get("precip_source")
    assert src and src != "model", (
        "강수 예측구간이 GBM 기준으로 적합되지 않았다 — "
        "conformal_interval_fit.py --precip-gbm 을 돌릴 것")
    assert predict.load_precip_gbm(ck, predict.PRECIP_GBM) is not None

    # 리드타임이 다른 조합은 막아야 한다.
    ck12 = torch.load(os.path.join(os.path.dirname(os.path.abspath(predict.__file__)),
                                   "checkpoints", "numerical_trichef_12h.pt"),
                      map_location="cpu", weights_only=True)
    try:
        predict.load_precip_gbm(ck12, predict.PRECIP_GBM)
        raise AssertionError("리드타임 불일치를 통과시켰다")
    except RuntimeError:
        pass
    # ④ 화면이 판정선 상수를 따로 읽지 않는가(2026-09-26 추가).
    #
    # 서빙은 모델 파일의 `meta_gate_tau`(0.310)를 쓰는데 화면 서술 여덟
    # 군데가 신경망 시절 상수(0.85/0.805)를 계속 읽고 있었다 — 화면이
    # "85% 미만이면 0으로 처리한다"고 설명하는 동안 모델은 31% 를 기준으로
    # 잘랐다. 같은 유형이 2026-09-23 에 한 번 났고, 그때 상수를 남겨 둔 채
    # 서술만 고쳐 재발했다. 그래서 **소스 수준으로** 막는다: `app.py` 에서
    # 그 상수를 읽는 곳은 `_gate()` 정의부(임포트·폴백·주석)뿐이어야 한다.
    app_src = os.path.join(os.path.dirname(os.path.abspath(predict.__file__)),
                           "app.py")
    with open(app_src, encoding="utf-8") as fh:
        hits = [(i, ln.strip()) for i, ln in enumerate(fh, 1)
                if "PRECIP_PROB_GATE" in ln]
    # 허용: 임포트 1줄 · `_gate()` 안의 폴백 1줄 · 그 docstring 의 언급.
    stray = [h for h in hits
             if not (h[1].startswith("PRECIP_PROB_GATE")
                     or h[1].startswith("return PRECIP_PROB_GATE_BY_LEAD")
                     or h[1].startswith("근접 판정 표시만"))]
    assert not stray, (
        "app.py 가 강수 판정선 상수를 `_gate()` 밖에서 읽는다 — 서빙과 "
        f"어긋날 수 있다: {stray}")
    assert any("def _gate(" in ln for ln in open(app_src, encoding="utf-8")), \
        "app.py 에 강수 판정선 라우터 `_gate()` 가 없다"

    return (f"τ={float(meta['meta_gate_tau']):.3f} · "
            f"F1 {float(meta['meta_val_precip_wet_f1']):.4f} · 구간 출처 확인 · "
            f"화면 상수 누출 0")


def _accuracy_wet_stats():
    """온라인 적중률의 강수 항목이 **기저율과 함께** 나오는가(2026-09-27).

    왜 필요한가 — 강수 적중률은 모델이 아니라 **날씨**에 따라 움직인다.
    배포 로그 실측: 건조한 구간(실제 강수 2/396)에서 99.5%, 비 온 구간
    (91/264)에서 65.2% 였는데, 후자의 F1 은 0.562 로 그 세대 검증값
    (0.588)과 사실상 같았다. 숫자만 보면 성능이 무너진 것처럼 읽힌다.
    저장소 규약(§2 양성이 희박하면 accuracy 대신 precision/recall/F1)을
    정작 이 화면이 지키지 않고 있었다.

    세 가지 경계를 모두 시험한다 — 비가 온 구간, 비가 없는 구간, 그리고
    **비가 왔는데 강수를 한 번도 예측하지 않은 구간**(정밀도의 분모가 0).
    셋째가 실제 로그에 있었고, `None` 을 f-string 에 꽂으면 화면이 죽는다.
    """
    import json
    import tempfile
    import accuracy

    def _mk(rows):
        fd, p = tempfile.mkstemp(suffix=".json")
        with os.fdopen(fd, "w", encoding="utf-8") as fh:
            json.dump(rows, fh)
        return p

    def _row(pred, act):
        return {"station": "108", "made_at": "202609270000",
                "target_time": "202609270600", "pred_temp": 20.0,
                "pred_precip": pred, "actual_temp": 20.0, "actual_precip": act,
                "hit_temp": True,
                "hit_precip": (pred >= accuracy.PRECIP_THRESH) ==
                              (act >= accuracy.PRECIP_THRESH),
                "source": "ci", "model_id": "sha256:test"}

    # ① 비가 온 구간 — TP 2 · FP 1 · FN 1 → 정밀도 2/3 · 재현율 2/3 · F1 2/3
    p1 = _mk([_row(1.0, 1.0), _row(1.0, 2.0), _row(1.0, 0.0), _row(0.0, 3.0)])
    w1 = accuracy.stats(path=p1, model_id="sha256:test")["cum_wet"]
    assert w1["n_pos"] == 3, w1
    assert abs(w1["precision"] - 2 / 3) < 1e-9, w1
    assert abs(w1["recall"] - 2 / 3) < 1e-9, w1
    assert abs(w1["f1"] - 2 / 3) < 1e-9, w1

    # ② 비가 전혀 없는 구간 — 적중률은 100% 인데 판별력은 잰 적이 없다
    p2 = _mk([_row(0.0, 0.0) for _ in range(5)])
    s2 = accuracy.stats(path=p2, model_id="sha256:test")
    assert s2["cum_precip"] == 1.0, s2
    assert s2["cum_wet"]["n_pos"] == 0, s2
    assert s2["cum_wet"]["f1"] is None, "양성이 없으면 F1 은 정의되지 않는다"

    # ③ 비가 왔는데 강수를 한 번도 예측하지 않음 — 정밀도 분모 0
    p3 = _mk([_row(0.0, 1.0), _row(0.0, 0.0)])
    w3 = accuracy.stats(path=p3, model_id="sha256:test")["cum_wet"]
    assert w3["n_pos"] == 1 and w3["precision"] is None and w3["recall"] == 0.0, w3
    assert w3["f1"] == 0.0, w3

    # 화면이 그 None 을 그대로 포맷하지 않는지 — 소스 수준으로 막는다.
    app_src = os.path.join(os.path.dirname(os.path.abspath(accuracy.__file__)),
                           "app.py")
    src = open(app_src, encoding="utf-8").read()
    assert "_w['precision']:.1%" not in src and '_w["precision"]:.1%' not in src, \
        "app.py 가 None 일 수 있는 정밀도를 직접 포맷한다 — 화면이 죽는다"
    for p in (p1, p2, p3):
        os.unlink(p)
    return "기저율·정밀도·재현율·F1 · 분모 0 경계 · 화면 포맷 보호"


def _wind_dir_units():
    """서빙 입력의 풍향이 학습과 같은 도 단위인가(2026-10-21 사고).

    API허브 WD 는 10도 단위인데 파서가 그대로 저장해, 실시간 입력의 풍향
    특징이 학습 분포 밖에 있었다(검증셋 재현: 폭염 F1 0.837→0.641). 연기
    시험이 서빙 입력의 **단위**를 본 적이 없어서 수개월 동안 드러나지 않았다.
    파서·이행 규칙·저장소 창 폴백 세 경로를 모두 확인한다.
    """
    import json
    import weather_collector as wc
    from train import record_to_vec

    # ① 파서 — WD 27(10도 단위) → 270°
    text = ("#START7777\n"
            "# YYYYMMDDHHMM STN WD WS GST_WD GST_WS GST_TM PA PS PT PR TA TD HM PV RN\n"
            "202610210900 108 27 2.3 -9 -9 -9 1010.0 1015.2 -9 -9 15.4 8.0 65 -9 0.0\n"
            "#7777END\n")
    rec = wc.RobustWeatherCollector(api_key="", stn="108")._parse_asos(
        text, "202610210900")
    assert rec["wind_dir"] == 270.0 and rec["wind_dir_unit"] == wc.WIND_DIR_UNIT, rec
    vec = record_to_vec(rec)
    assert abs(vec[4] + 1.0) < 1e-9 and abs(vec[5]) < 1e-9, \
        f"서풍(270°)의 sin/cos 가 (-1, 0)이 아니다: {vec[4:6]}"

    # ② 이행 규칙 — 표식 없는 10도 단위만 바꾸고, 멱등이다
    old = {"wind_dir": 27.0, "stn": "108", "timestamp": "202610210900",
           "status": "SUCCESS_LIVE"}
    m = wc.migrate_api_wind_dir(old)
    assert m["wind_dir"] == 270.0 and old["wind_dir"] == 27.0, "원본을 바꾸면 안 된다"
    assert wc.migrate_api_wind_dir(m) == m, "두 번 적용하면 열 배가 된다"
    assert wc.migrate_api_wind_dir(dict(old, wind_dir=270.0))["wind_dir"] == 270.0

    # ③ 저장소 창 폴백 — 이행 전 창이 주입돼도 도 단위로 나간다
    wc.set_offline_fallback([old])
    try:
        assert wc._offline_record("108", "202610210900")["wind_dir"] == 270.0
    finally:
        wc.set_offline_fallback([])

    # ④ 로컬 저장소 창(main 의 최후 폴백)을 이행 규칙에 통과시키면 도 단위
    #    범위(최댓값 36 초과·360 이하)에 드는가 — 실제 자료로 규칙을 확인한다
    note = "로컬 창 없음"
    path = "./cache/recent_window.json"
    if os.path.exists(path):
        with open(path, encoding="utf-8") as fh:
            win = json.load(fh)
        vals = [wc.migrate_api_wind_dir(r)["wind_dir"] for r in win]
        assert max(vals) > 36.0 and max(vals) <= 360.0, \
            f"창 풍향 최댓값 {max(vals)} — 10도 단위가 남아 있다"
        note = f"로컬 창 {len(win):,}건 최댓값 {max(vals):.0f}°"
    return f"파서 27→270° · sin/cos(-1,0) · 멱등 · 폴백 이행 · {note}"


def _training_wd_guard():
    """학습 캐시의 10도 단위 풍향이 학습을 막는가(2026-10-22).

    `train.train()` 이 부르는 가드다. 포털 구간(2026 이전)·표식 있는 레코드는
    통과하고, API 구간의 표식 없는 레코드는 막아야 한다.
    """
    import migrate_training_wind_dir as m
    ok = [{"timestamp": "202512312300", "wind_dir": 20.0},
          {"timestamp": "202603010000", "wind_dir": 270.0, "wind_dir_unit": "deg"}]
    m.assert_migrated(ok)
    try:
        m.assert_migrated(ok + [{"timestamp": "202603010100", "wind_dir": 27.0}])
        raise AssertionError("표식 없는 API 구간 레코드를 통과시켰다")
    except SystemExit:
        pass
    return "포털 구간·이행 레코드 통과 · 미이행 API 레코드 차단"


def _api_budget_guards():
    """API 호출이 결측·장애 때 폭주하지 않는가(2026-10-21).

    실측 2026-08-28 하루 2,466건(평시 약 300건). 결측 관측소의 공백 전체를
    매 실행 재요청했고, 일일 상한은 프로세스 계수라 실행마다 초기화됐다.
    앱은 수치예보가 없을 때도 API 15회를 다 쓴 뒤에야 예외를 올렸다.
    """
    from datetime import datetime, timedelta
    import weather_collector as wc
    import refresh_deploy_data as rd
    import predict as pr

    # ① 계획 — 정상(공백 1시간)·결측(10시간)·창에 없음
    ceiling = datetime(2026, 10, 21, 21, 0)
    ts = lambda dt: dt.strftime("%Y%m%d%H00")
    existing = {("108", ts(ceiling - timedelta(hours=1))): {},
                ("112", ts(ceiling - timedelta(hours=10))): {}}
    targets, backfill = rd._plan_targets(existing, ["108", "112", "119"],
                                         ceiling, {})
    assert sorted(targets) == sorted([("108", ts(ceiling)), ("112", ts(ceiling)),
                                      ("119", ts(ceiling))]), targets
    assert len(backfill["112"]) == 9 and len(backfill["119"]) == rd.SEED_HOURS - 1
    assert "108" not in backfill and rd.FETCH_RETRIES == 1

    # ② 일일 상한 — 저장소 누계로 계수를 시작하면 상한이 하루 단위로 걸린다
    saved = dict(wc._call_stats)
    try:
        today = datetime.now(wc.KST).strftime("%Y%m%d")
        wc.preload_call_count(today, wc.DAILY_CALL_BUDGET)
        assert not wc._reserve_call(), "누계가 상한이면 더 부르면 안 된다"
        wc.preload_call_count("19990101", 0)            # 다른 날짜는 무시
        assert wc.api_call_stats()["count"] == wc.DAILY_CALL_BUDGET
    finally:
        wc._call_stats.clear()
        wc._call_stats.update(saved)

    # ③ 앱 — 수치예보가 없으면 대상 관측소 1회만 조회하고 멈춘다
    calls = []

    def _fake_fetch(self, *a, **k):
        calls.append(self.stn)
        return {"timestamp": "202610212100", "temperature": 10.0,
                "precipitation": 0.0, "humidity": 50.0, "wind_speed": 1.0,
                "wind_dir": 270.0, "wind_dir_unit": "deg", "pressure": 1015.0,
                "precip_type": 0, "stn": self.stn, "status": "SUCCESS_LIVE"}

    class _NoNWP:
        def encode(self, *a, **k):
            return None

    orig_fetch, orig_nwp = pr.RobustWeatherCollector.fetch, pr._nwp_collector
    pr.RobustWeatherCollector.fetch = _fake_fetch
    pr._nwp_collector = lambda: _NoNWP()
    try:
        m, c = pr.load_model("./checkpoints/numerical_trichef.pt", "cpu")
        assert c.get("use_nwp"), "배포 체크포인트가 수치예보를 쓰지 않는다 — 시험 전제 확인"
        try:
            pr.predict(stn="108", model=m, ckpt=c)
            raise AssertionError("수치예보가 없는데 예외가 나지 않았다")
        except pr.NWPUnavailable:
            pass
        assert calls == ["108"], f"수치예보 확인 전에 {len(calls)}회 조회했다: {calls}"

        # ④ Z축 전용 구성(2026-10-22) — 보조 모델 셋이 다 있으면 이웃·시차를
        #    조회하지 않는다(출력 1건에 API 1회). 축 배분도 내보내지 않는다.
        class _ZeroNWP:
            def encode(self, *a, **k):
                return np.zeros(14, dtype=np.float32)

        calls.clear()
        orig_at = pr.RobustWeatherCollector.fetch_at
        pr.RobustWeatherCollector.fetch_at = lambda self, tm, *a, **k: (
            calls.append((self.stn, tm)) or _fake_fetch(self))
        pr._nwp_collector = lambda: _ZeroNWP()
        try:
            res = pr.predict(stn="108", model=m, ckpt=c)
        finally:
            pr.RobustWeatherCollector.fetch_at = orig_at
        cov = res["input_coverage"]
        assert cov["z_only"] and calls == ["108"], f"Z축 전용인데 {len(calls)}회 조회: {calls}"
        assert res["gate_weights"] is None and cov["neighbors_live"] is None
        assert res["forecast"]["temperature"] is not None
    finally:
        pr.RobustWeatherCollector.fetch = orig_fetch
        pr._nwp_collector = orig_nwp
    return ("탐침 계획(정상 1·결측 1+9·신규 1+23) · 재시도 1 · 일일 상한 · "
            "수치예보 선확인(조회 1회) · Z축 전용 출력 1건 = 조회 1회")


def _window_gap_upkeep():
    """창 내부 공백 재요청과 대조 불가 대기 항목 정리(2026-10-22).

    ① 최신 관측 앞의 구멍을 최신 쪽부터 찾고, 탐침 중 관측소는 빼며, 실행당
       상한을 지킨다. ② 창보다 오래된 **대기** 항목만 지우고 대조 완료 항목·
       창 안의 대기 항목은 남긴다.
    """
    import tempfile
    import accuracy
    import refresh_deploy_data as rd
    rec = lambda s, t: {"stn": s, "timestamp": t}
    hours = [f"20261020{h:02d}00" for h in range(0, 24)]
    existing = {(s, t): rec(s, t) for s in ("119", "108") for t in hours}
    for t in ("202610201200", "202610201300", "202610201500"):
        del existing[("119", t)]
    got = rd._interior_holes(existing, ["119", "108"], set())
    assert got == [("119", "202610201500"), ("119", "202610201300"),
                   ("119", "202610201200")], got
    assert rd._interior_holes(existing, ["119", "108"], {"119"}) == []
    for t in hours[2:10]:
        existing.pop(("108", t), None)
    assert len(rd._interior_holes(existing, ["119", "108"], set())) == rd.INTERIOR_MAX_PER_RUN

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        e = lambda tt, done: {"station": "119", "target_time": tt, "source": "ci",
                              "actual_temp": 1.0 if done else None}
        accuracy._save([e("202610100000", False), e("202610100000", True),
                        e("202610150000", False)], path)
        assert accuracy.expire_unresolvable("202610120000", path) == 1
        left = accuracy._load(path)
        assert len(left) == 2 and all(x["target_time"] == "202610150000" or
                                      x["actual_temp"] is not None for x in left)
    finally:
        os.remove(path)
    return "내부 구멍 최신순·탐침 관측소 제외·실행당 상한 · 창 밖 대기 항목만 정리"


def _pipeline_units():
    """CI 파이프라인 단위 시험(2026-10-22, NEXT.md P3-2).

    ① 수치예보 최신성 게이트 — 여유가 충분하면 통과, 관측소가 통째로 빠지면
       실패(종전에는 있는 관측소만 세서 빠진 관측소를 놓쳤다).
    ② `accuracy.trim` — 기간이 지난 대조 완료 항목만 지우고 대기 항목은 남긴다.
    ③ 극한기상 GBM 두 파일 — 헤드마다 저장된 보정 방법(베타·등온·안 함)으로
       확률이 [0, 1] 안에 나오고, 기온 단조 제약 헤드는 기온을 올리면 폭염이
       내려가지 않는다.
    """
    import tempfile
    from datetime import datetime, timedelta
    import accuracy
    import collect_nwp_archive as cna
    import extreme_gbm as eg
    from weather_collector import KST, STATION_COORDS

    now = datetime.now(KST).replace(tzinfo=None)
    keys = [(now + timedelta(hours=h)).strftime("%Y%m%d%H00") for h in range(-5, 30)]
    full = {s_: {k: [0.0] * 7 for k in keys} for s_ in STATION_COORDS}
    cna._freshness_gate(full)
    short = dict(full)
    short.pop(next(iter(STATION_COORDS)))
    try:
        cna._freshness_gate(short)
        raise AssertionError("관측소가 빠진 수치예보 창을 통과시켰다")
    except SystemExit:
        pass

    fd, path = tempfile.mkstemp(suffix=".json")
    os.close(fd)
    try:
        e = lambda tt, done: {"station": "108", "target_time": tt, "source": "ci",
                              "actual_temp": 1.0 if done else None}
        accuracy._save([e("202609010000", True), e("202609010000", False),
                        e("202610200000", True)], path)
        assert accuracy.trim(30, path) == 1
        left = accuracy._load(path)
        assert len(left) == 2 and any(x["actual_temp"] is None for x in left)
    finally:
        os.remove(path)

    root = os.path.dirname(os.path.abspath(__file__))
    methods = set()
    for name in ("extreme_gbm.npz", "extreme_gbm_12h.npz"):
        models, meta = eg.load(os.path.join(root, "checkpoints", name))
        assert set(models) == {"heatwave", "coldwave", "dust"}, (name, set(models))
        nf = int(meta["meta_num_features"])
        x = np.zeros((5, nf), dtype=np.float32)
        x[:, 0] = np.linspace(-2, 2, 5)          # 표준화된 관측기온만 올린다
        for head in models:
            methods.add(str(meta[f"meta_{head}_cal_method"]))
            p = eg.calibrated_batch(models, meta, head, x)
            assert np.all(np.isfinite(p)) and np.all((p >= 0) & (p <= 1)), (name, head)
        heat = eg.calibrated_batch(models, meta, "heatwave", x)
        assert np.all(np.diff(heat) >= -1e-12), f"{name} 폭염이 기온에 단조가 아니다: {heat}"
    return f"최신성 게이트 · trim · 극한 GBM 2개(보정 {sorted(methods)})"


def _record_online_e2e():
    """`record_online_forecasts.py --dry-run` 소형 종단 시험(2026-10-22, P3-2).

    CI 가 15분마다 부르는 배포 경로다. 임시 작업 디렉터리에 main 의 저장소 창을
    복사하고, **시험 전용 합성 수치예보 창**(상수값 — 출력의 정확성이 아니라
    배선을 본다)을 만들어 API 0건으로 12곳 전부 예측이 나오는지 본다. 이 경로가
    죽으면 화면의 적중률이 조용히 비어 간다(CLAUDE.md 7절).
    """
    import json
    import shutil
    import subprocess
    import tempfile
    root = os.path.dirname(os.path.abspath(__file__))
    src = os.path.join(root, "cache", "recent_window.json")
    if not os.path.exists(src):
        return "저장소 창 없음 — 건너뜀"
    with open(src, encoding="utf-8") as fh:
        window = json.load(fh)
    latest = max(str(r["timestamp"])[:12] for r in window)
    from datetime import datetime, timedelta
    t0 = datetime.strptime(latest, "%Y%m%d%H%M")
    keys = [(t0 + timedelta(hours=h)).strftime("%Y%m%d%H00") for h in range(-48, 49)]
    row = [0.0, 15.0, 60.0, 50.0, 2.0, 180.0, 1010.0]
    stns = sorted({str(r["stn"]) for r in window})
    tmp = tempfile.mkdtemp()
    try:
        os.makedirs(os.path.join(tmp, "cache"))
        shutil.copy(src, os.path.join(tmp, "cache", "recent_window.json"))
        with open(os.path.join(tmp, "cache", "nwp_recent.json"), "w") as fh:
            json.dump({s_: {k: row for k in keys} for s_ in stns}, fh)
        env = dict(os.environ, KMA_API_KEY="", PYTHONPATH=root)
        out = subprocess.run(
            [sys.executable, os.path.join(root, "record_online_forecasts.py"), "--dry-run"],
            cwd=tmp, env=env, capture_output=True, text=True, timeout=600)
        txt = out.stdout + out.stderr
        assert out.returncode == 0, txt[-800:]
        assert "실패 0건" in txt, txt[-800:]
        n = sum(1 for line in txt.splitlines() if " → " in line and "기온" in line)
        assert n == len(stns), f"예측 {n}건 / 관측소 {len(stns)}곳\n{txt[-800:]}"
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
    return f"관측소 {n}곳 예측 · 실패 0 · API 키 없음"


def _promotion_guards():
    """승격 절차의 두 안전장치가 동작하는가(2026-10-21).

    ① 죽은 게이트를 통과로 세지 않는다 — 판정 누락 판별.
    ② 입력 계약 — 배포 짝(주 체크포인트·기온 모델)은 일치, 리드타임이 다른
       짝은 불일치로 잡는가. 키가 없을 때의 기본값을 코드와 같게 해석하지
       않으면 같은 입력인데도 불일치로 판정된다(시험 중 실제로 그랬다).
    """
    import promote_checkpoint as pc
    found = {"seasonal_false_alarm": ("PASS", ""),
             "station_coverage_heatwave": ("PASS", "")}
    assert pc.missing_verdicts(found, ["seasonal_false_alarm", "station_coverage_*"]) == []
    assert pc.missing_verdicts(found, ["monotonicity_coldwave"]) == ["monotonicity_coldwave"]
    assert pc.missing_verdicts({}, ["station_coverage_*"]) == ["station_coverage_*"]

    load = lambda p: torch.load(p, map_location="cpu", weights_only=True)
    pairs = [(pc.PRODUCTION_TARGETS[6], pc.PRODUCTION_TARGETS["temp6"]),
             (pc.PRODUCTION_TARGETS[12], pc.PRODUCTION_TARGETS["temp12"])]
    for main_p, temp_p in pairs:
        if os.path.exists(main_p) and os.path.exists(temp_p):
            d = pc.input_contract_diff(load(main_p), load(temp_p))
            assert d == [], f"배포 짝이 입력 계약 불일치로 판정됐다: {main_p} {d}"
    if all(os.path.exists(pc.PRODUCTION_TARGETS[k]) for k in (6, 12)):
        d = pc.input_contract_diff(load(pc.PRODUCTION_TARGETS[6]),
                                   load(pc.PRODUCTION_TARGETS[12]))
        assert "lead_hours" in d, d
    return "판정 누락 판별 · 배포 짝 계약 일치 · 리드타임 다른 짝 불일치"


def _gbm_norm_binding():
    """GBM 이 학습 때의 정규화 통계·출처에 묶여 있는가(2026-10-22).

    ① 배포 GBM 넷이 각자의 주 체크포인트와 대조를 통과하고, 통계를 한 칸만
       바꾼 체크포인트·통계가 없는 파일은 막는다(종전에는 차원만 봤다).
    ② 출처 대조가 양방향이다 — 기록만 있음·모델만 있음·다른 파일을 막고,
       경로 표기 차이(`./`)는 같은 파일로 본다.
    ③ 앱의 보조 모델 캐시가 주 체크포인트 지문을 키로 받는다.
    """
    import copy
    import predict
    from precip_gbm import check_input_contract
    root = os.path.dirname(os.path.abspath(predict.__file__))
    ck_of = lambda n: torch.load(os.path.join(root, "checkpoints", n),
                                 map_location="cpu", weights_only=True)
    combos = [("numerical_trichef.pt", "precip_gbm.npz", "extreme_gbm.npz"),
              ("numerical_trichef_12h.pt", "precip_gbm_12h.npz", "extreme_gbm_12h.npz")]
    for ck_n, pg_n, eg_n in combos:
        ck = ck_of(ck_n)
        pg_p, eg_p = (os.path.join(root, "checkpoints", n) for n in (pg_n, eg_n))
        g = predict.load_precip_gbm(ck, pg_p)
        e = predict.load_extreme_gbm(ck, eg_p)
        assert g is not None and e is not None, ck_n
        # 배포 체크포인트의 기록과 실제 경로가 일치해야 한다(양방향 대조 통과).
        predict._check_source("강수", ck.get("precip_source"), g, lambda t: t[2], "")
        predict._check_source("극한기상", ck.get("extreme_source"), e, lambda t: t[1], "")
        bad = copy.deepcopy(ck)
        bad["mean"] = list(bad["mean"])
        bad["mean"][3] = float(bad["mean"][3]) + 0.5
        for fn, path in ((predict.load_precip_gbm, pg_p), (predict.load_extreme_gbm, eg_p)):
            try:
                fn(bad, path)
                raise AssertionError(f"정규화 통계 불일치를 통과시켰다: {path}")
            except RuntimeError:
                pass
    legacy = {k: v for k, v in g[2].items() if k not in ("meta_mean", "meta_std")}
    try:
        check_input_contract(legacy, ck, "legacy.npz")
        raise AssertionError("통계가 없는 GBM 파일을 통과시켰다")
    except RuntimeError:
        pass

    fake = (None, None, {"source_path": os.path.join(root, "checkpoints", "precip_gbm.npz")})
    m = lambda t: t[2]
    # 기록은 저장소 루트 기준이다 — CI 처럼 다른 디렉터리에서 돌아도 같은 파일로 본다.
    _cwd = os.getcwd()
    try:
        os.chdir(os.path.dirname(root) or "/")
        predict._check_source("강수", "./checkpoints/precip_gbm.npz", fake, m, "")
    finally:
        os.chdir(_cwd)
    for rec, gbm in (("./checkpoints/precip_gbm.npz", None), (None, fake),
                     ("./checkpoints/precip_gbm_12h.npz", fake)):
        try:
            predict._check_source("강수", rec, gbm, m, "")
            raise AssertionError(f"출처 불일치를 통과시켰다: {rec} / {gbm}")
        except RuntimeError:
            pass

    with open(os.path.join(root, "app.py"), encoding="utf-8") as fh:
        app_src = fh.read()
    for fn in ("get_extreme_gbm_at", "get_precip_gbm_at", "get_temp_model_at"):
        assert f"def {fn}(path: str, fingerprint: str, main_fp: str," in app_src, \
            f"{fn} 캐시 키에 주 체크포인트 지문이 없다"
    return "배포 GBM 4개 통계·출처 일치 · 통계 변조·통계 없음·출처 3방향 불일치 차단 · 캐시 키"


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
    print("\n[T3-2] 기온 전용 보조 체크포인트 경로·신원")
    check("temp_checkpoint", _temp_checkpoint_path)
    print("\n[T3-3] 강수 전용 GBM 경로·추론·구간 출처")
    check("precip_gbm", _precip_gbm_path)
    print("\n[T3-4] 온라인 적중률 — 강수는 기저율과 함께 읽는가")
    check("accuracy_wet", _accuracy_wet_stats)
    print("\n[T3-5] 서빙 입력의 풍향 단위(API 10도 단위 → 도)")
    check("wind_dir_units", _wind_dir_units)
    print("\n[T3-5b] 학습 캐시 풍향 단위 가드")
    check("training_wd_guard", _training_wd_guard)
    print("\n[T3-6] API 호출 폭주 방지(탐침·일일 상한·수치예보 선확인)")
    check("api_budget", _api_budget_guards)
    print("\n[T3-6b] 창 내부 공백·대조 불가 대기 항목")
    check("window_upkeep", _window_gap_upkeep)
    print("\n[T3-6c] CI 파이프라인 단위(최신성 게이트·trim·극한 GBM)")
    check("pipeline_units", _pipeline_units)
    print("\n[T3-6d] 온라인 기록 종단 시험(dry-run)")
    check("record_online_e2e", _record_online_e2e)
    print("\n[T3-7] 승격 절차 — 죽은 게이트·입력 계약")
    check("promotion", _promotion_guards)
    print("\n[T3-8] GBM 정규화 통계·출처 결속, 앱 캐시 키")
    check("gbm_binding", _gbm_norm_binding)
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
