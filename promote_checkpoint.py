"""
promote_checkpoint.py — 후보 체크포인트를 배포로 승격하는 절차를 한 번에 실행한다.

**왜 필요한가.** 재학습 후에는 점검·재보정·산출물 재생성을 순서대로 여러 번
돌려야 하는데, 이 절차가 사람의 기억에 의존하고 있었다. 실제로 2026-08-17
승격에서 두 번 누락이 발생했다 — 신뢰도 곡선 그림이 이전 모델 시점 그대로
남아 README 표와 어긋났고, 제거된 관측소별 예외가 화면 설명에는 계속
"적용 중"으로 남아 있었다. 둘 다 조용히 틀린 값을 보여주는 종류의 사고다.

빠뜨려도 오류가 나지 않는 절차는 결국 빠뜨리게 된다. 순서를 코드로 고정하고,
통과하지 못하면 승격을 중단한다.

**게이트(하나라도 FAIL이면 중단)**
  1. 계절 오탐 — 라벨 마스크가 가리는 구간을 직접 본다. 폭염 헤드가 한겨울에,
     한파 헤드가 한여름에 작동하지 않는지. 집계 지표(F1)로는 잡히지 않는다.
  2. 단조성 — 기온을 흔들었을 때 두 헤드가 물리적으로 옳은 방향으로 반응하는지.
  3. 회귀 성능 — 현행 배포본 대비 기온·강수가 뒷걸음질 치지 않는지.

**게이트 통과 후 자동 수행**
  4. 확률 보정 재적합(`probability_calibration_fit.py --apply`)
  5. 신뢰도 곡선·관측소별 플롯 재생성

**여전히 사람이 해야 하는 일**은 마지막에 목록으로 출력한다. 임계값 상수
(`predict.py`)와 README 수치는 판단이 필요해 자동화하지 않는다 — 자동으로
고치면 검증 없이 문서가 바뀌는 더 나쁜 문제가 생긴다.

실행:
  python promote_checkpoint.py ./checkpoints/<후보>.pt            # 점검만
  python promote_checkpoint.py ./checkpoints/<후보>.pt --promote  # 통과 시 승격
"""
import os
import re
import shutil
import subprocess
import sys
import time

# 배포 경로는 리터럴로 못박는다 — `predict.CHECKPOINT` 를 쓰지 않는다.
#
# 왜(2026-08-19 수정): predict.CHECKPOINT 는 `CHECKPOINT_PATH` 환경변수를
# 읽는다. 그런데 CLAUDE.md 3절은 대조 실행 시 그 변수를 지정하라고 지시하고,
# coldwave_nested_pretrain.py 는 아예 `CHECKPOINT_PATH=... python ...` 명령을
# 출력한다. 그 변수를 export 한 셸에서 이 스크립트를 돌리면 "현행 배포본"이
# 통째로 실험 파일로 바뀐다 — 게이트 3(회귀 성능)이 후보를 엉뚱한 기준선과
# 비교하고(변수가 후보와 같으면 자기 자신과 비교해 무조건 통과),
# shutil.copy2 가 배포 경로가 아닌 실험 파일을 덮어쓰면서 화면에는
# "배포 경로 교체 완료"라고 찍힌다. 자식 프로세스(확률 보정 재적합 등)도
# 환경을 상속하므로 보정 곡선과 그림까지 배포되지 않는 파일 기준으로
# 다시 만들어진다. 이 스크립트가 막으려던 "조용히 틀린 값" 사고 그 자체다.
# 승격 대상만은 환경에 흔들리지 않아야 한다.
PRODUCTION_CHECKPOINT = "./checkpoints/numerical_trichef.pt"
CHECKPOINT = PRODUCTION_CHECKPOINT   # 이 파일 안의 기존 참조를 그대로 둔다

# 게이트 임계 — seasonal_falsealarm_check.py / coldwave_pathway_check.py 와
# 같은 기준을 쓴다. 값을 바꾸려면 두 곳을 함께 고쳐야 한다.
# 회귀 성능이 현행 대비 이 비율 이상 나빠지면 WARN 으로 기록한다.
#
# 주석은 오랫동안 "FAIL" 이라고 적혀 있었으나 코드는 warnings 에만 담았고,
# --promote 가 주어지면 WARN 을 무시하고 진행한다 — 즉 이 게이트는 실제로
# 승격을 막은 적이 없다. 문서와 동작을 일치시키되(2026-08-19), 판정 자체는
# WARN 으로 유지한다: 헤드 하나를 개선하면 다른 지표가 소폭 나빠지는 것이
# 이 프로젝트에서는 정상 범위이고(공유 표현 경쟁), 그때마다 자동 차단하면
# 개선을 아예 승격할 수 없기 때문이다. 대신 사람이 반드시 보도록 남긴다.
# 진짜 차단이 필요한 항목(계절 오탐·단조성·시계 불일치)은 failures 로 간다.
REGRESSION_TOL = 0.02


def run(cmd, title):
    """스크립트를 돌리며 출력을 그대로 흘려보내고, 전체 stdout을 돌려준다."""
    print(f"\n{'='*78}\n▶ {title}\n{'='*78}")
    t0 = time.time()
    proc = subprocess.Popen([sys.executable] + cmd, stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT, text=True, bufsize=1)
    lines = []
    for line in proc.stdout:
        print(line, end="")
        lines.append(line)
    proc.wait()
    print(f"  ({time.time()-t0:.0f}초, 종료코드 {proc.returncode})")
    return "".join(lines), proc.returncode


def verdicts(out):
    """`VERDICT <이름> <PASS|WARN|FAIL> ...` 줄을 모은다."""
    found = {}
    for m in re.finditer(r"^VERDICT (\S+) (PASS|WARN|FAIL)(.*)$", out, re.M):
        found[m.group(1)] = (m.group(2), m.group(3).strip())
    return found


def ckpt_metrics(path):
    import torch
    c = torch.load(path, map_location="cpu", weights_only=True)
    em = c.get("extreme_metrics") or {}
    return {
        "num_features": c.get("num_features"),
        "signed": c.get("signed_head_input", False),
        "lead_hours": c.get("lead_hours"),
        "temp": c.get("val_temp_mae"),
        "precip": c.get("val_precip_mae"),
        "temp_naive": c.get("val_temp_naive_mae"),
        "precip_naive": c.get("val_precip_naive_mae"),
        "heat_f1": (em.get("heatwave") or {}).get("f1"),
        "cold_f1": (em.get("coldwave") or {}).get("f1"),
        "has_calibration": bool(c.get("prob_calibration")),
    }


def main():
    args = [a for a in sys.argv[1:] if not a.startswith("--")]
    do_promote = "--promote" in sys.argv
    if not args:
        print("사용법: python promote_checkpoint.py <후보 체크포인트> [--promote]")
        sys.exit(2)
    cand = args[0]
    if not os.path.exists(cand):
        print(f"후보 체크포인트를 찾을 수 없다: {cand}")
        sys.exit(2)

    print(f"후보  : {cand}")
    print(f"현행  : {CHECKPOINT}")
    cm = ckpt_metrics(cand)
    print(f"        후보 num_features={cm['num_features']} signed={cm['signed']}")

    failures, warnings = [], []

    # ── 게이트 1·2 — 후보를 직접 지정해 점검한다 ──────────────────
    def collect(out):
        for name, (code, extra) in verdicts(out).items():
            if code == "FAIL":
                failures.append(f"{name}: FAIL {extra}")
            elif code == "WARN":
                warnings.append(f"{name}: WARN {extra}")

    out, _ = run(["seasonal_falsealarm_check.py", cand], "게이트 1 — 계절 오탐")
    collect(out)

    for head in ("coldwave", "heatwave"):
        out, _ = run(["coldwave_pathway_check.py", f"--head={head}", cand],
                     f"게이트 2 — 단조성({head})")
        collect(out)

    # 계절 축과 같은 사각지대가 관측소 축에도 있는지 본다. 라벨이 없는
    # 관측소의 출력은 채점된 적이 없으므로, 경보를 얼마나 내는지 확인한다.
    out, _ = run(["station_coverage_check.py", cand], "게이트 2-1 — 관측소 라벨 사각지대")
    collect(out)

    # ── 게이트 3 — 회귀 성능이 뒷걸음질 치지 않는지 ────────────────
    print(f"\n{'='*78}\n▶ 게이트 3 — 회귀 성능 비교\n{'='*78}")
    try:
        pm = ckpt_metrics(CHECKPOINT)
    except Exception as e:
        pm = None
        print(f"  현행 체크포인트를 읽지 못했다({e}) — 비교를 건너뛴다.")
    if pm:
        # 비교가 성립하는지 먼저 확인한다(2026-08-19 추가).
        #
        # ① 예보 시계가 다르면 절대 MAE 는 비교 대상이 아니다. LEAD_HOURS 가
        #    환경변수로 열린 뒤로 실제 가능해진 사고다 — 짧은 시계(예 +3h)는
        #    퍼시스턴스가 워낙 강해 절대 MAE 가 낮게 나오므로, 그대로 비교하면
        #    모든 항목이 OK 로 통과하고 배포 경로가 조용히 다른 시계의 모델로
        #    바뀐다(화면은 "+3시간 후 모델 출력값"을 그리게 된다).
        # ② 기준선(naive MAE)이 다르면 검증셋이 다르다는 뜻이므로 비교 자체가
        #    성립하지 않는다 — CLAUDE.md 2절이 명시한 확인 절차다.
        if pm.get("lead_hours") != cm.get("lead_hours"):
            failures.append(
                f"예보 시계 불일치: 현행 +{pm.get('lead_hours')}h vs "
                f"후보 +{cm.get('lead_hours')}h — 절대 MAE 비교가 성립하지 않는다. "
                f"다른 시계의 모델은 배포 경로를 공유하지 않는다(별도 경로로 서빙할 것)")
        for _k, _lbl in (("temp_naive", "기온 기준선"), ("precip_naive", "강수 기준선")):
            _a, _b = pm.get(_k), cm.get(_k)
            if _a is not None and _b is not None and abs(_a - _b) > 1e-4:
                failures.append(
                    f"{_lbl}(naive MAE) 불일치: {_a:.4f} vs {_b:.4f} — "
                    f"검증셋이 다르다는 뜻이라 성능 비교가 성립하지 않는다")

        print(f"  {'지표':<12}{'현행':>12}{'후보':>12}   판정")
        for key, label, lower_better in (("temp", "기온 MAE", True),
                                         ("precip", "강수 MAE", True),
                                         ("heat_f1", "폭염 F1", False),
                                         ("cold_f1", "한파 F1", False)):
            a, b = pm.get(key), cm.get(key)
            if a is None or b is None:
                continue
            worse = (b > a * (1 + REGRESSION_TOL)) if lower_better \
                else (b < a * (1 - REGRESSION_TOL))
            mark = "★악화★" if worse else "OK"
            if worse:
                warnings.append(f"{label} 악화: {a:.4f} → {b:.4f}")
            print(f"  {label:<12}{a:>12.4f}{b:>12.4f}   {mark}")

    # ── 판정 ──────────────────────────────────────────────────────
    print(f"\n{'='*78}\n 게이트 종합\n{'='*78}")
    if failures:
        for f in failures:
            print(f"  ★FAIL★ {f}")
    if warnings:
        for w in warnings:
            print(f"  ⚠ WARN  {w}")
    if not failures and not warnings:
        print("  전부 통과")

    if failures:
        print("\n승격을 중단한다 — FAIL 항목을 해결해야 한다.")
        sys.exit(1)
    if not do_promote:
        print("\n(--promote 를 주지 않아 여기서 멈춘다. 배포 경로는 그대로다.)")
        sys.exit(0)
    if warnings:
        print("\nWARN 이 있으나 --promote 가 지정되어 진행한다 — 위 항목을 확인할 것.")

    # ── 승격 ──────────────────────────────────────────────────────
    # 후보가 이미 배포 경로 자신이면 복사할 것이 없다 — shutil.copy2 는
    # SameFileError 를 던지는데, 그 전에 백업이 이미 만들어져 쓰레기 파일만
    # 남는다. 먼저 걸러낸다(2026-08-19).
    if os.path.abspath(cand) == os.path.abspath(CHECKPOINT):
        print(f"\n후보가 배포 경로와 같은 파일이다({CHECKPOINT}) — 복사를 건너뛴다.")
    else:
        stamp = time.strftime("%Y%m%d_%H%M%S")
        backup = CHECKPOINT.replace(".pt", f"_before_{stamp}.pt")
        shutil.copy2(CHECKPOINT, backup)
        print(f"\n현행 백업: {backup}")
        shutil.copy2(cand, CHECKPOINT)
        print(f"배포 경로 교체 완료: {CHECKPOINT}")

    # ── 후속 자동 수행 ────────────────────────────────────────────
    # 자식에게 배포 경로를 **명시적으로** 넘긴다. 인자를 생략하면 자식이
    # predict.CHECKPOINT(=CHECKPOINT_PATH 환경변수)로 기본값을 잡으므로,
    # 환경이 오염된 셸에서는 배포되지 않은 파일에 보정을 적합하게 된다
    # (위 PRODUCTION_CHECKPOINT 주석 참고). 환경변수 자체도 제거해
    # 경로 인자를 받지 않는 자식(calibration_plot_diagnose.py)까지 보호한다.
    os.environ.pop("CHECKPOINT_PATH", None)
    run(["probability_calibration_fit.py", CHECKPOINT, "--apply"], "확률 보정 재적합")
    run(["probability_calibration_check.py", CHECKPOINT], "신뢰도 곡선 재생성")
    run(["calibration_plot_diagnose.py"], "관측소별 플롯 재생성")

    # ── 사람이 해야 할 일 ─────────────────────────────────────────
    print(f"\n{'='*78}\n 남은 수동 작업 — 자동화하지 않는다(판단이 필요하다)\n{'='*78}")
    print("""  1. threshold_validation.py 를 돌려 임계값을 재산출하고, 채택 기준(순이득
     0.01)을 넘은 것만 predict.py 의 EXTREME_EVENT_THRESH 에 반영한다.
  2. 관측소별 예외(STATION_EVENT_THRESH_OVERRIDES)가 남아 있다면
     station_threshold_check.py 로 이 모델에서도 유효한지 재검증한다 —
     이전 모델에서 유효했다는 이유만으로 유지하지 않는다.
  3. README.md 와 app.py 의 수치·서술을 새 모델 기준으로 갱신한다:
       · 최상단 요약표, 재학습 결과표, 극한기상 성능표
       · 확률 보정 표(ECE)
       · 관측소별 성능 캡션(플롯이 바뀌었으므로 서술도 바뀐다)
       · 임계값·예외에 관한 설명
  4. 탭 스크린샷(docs/images/tab-*.png)을 다시 찍는다.
  5. CI(test_phase34.py · test_phase36.py)를 돌린다.
  6. 배포 후, import 모듈(predict.py·pipeline_model.py)의 시그니처가 바뀌었다면
     Streamlit Cloud 에서 앱을 재시작한다(모듈 캐시 때문에 재배포만으로는
     반영되지 않는다).""")


if __name__ == "__main__":
    main()
