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
  0. 입력 계약 — 이 체크포인트와 **같은 표준화 입력 벡터를 공유하는 모델**
     (기온 전용 모델·강수 GBM·극한기상 GBM)이 있으면 정규화 통계·입력 차원이
     같아야 한다. 다르면 그 모델들이 조용히 어긋나거나(GBM) 서빙이 시작 시
     멈춘다(기온 모델). 이 절차는 그 모델들을 다시 만들지 않는다. GBM 은
     파일에 적힌 학습 시 통계(`meta_mean`/`meta_std`)와 직접 대조한다.
  1. 계절 오탐 — 라벨 마스크가 가리는 구간을 직접 본다. 폭염 헤드가 한겨울에,
     한파 헤드가 한여름에 작동하지 않는지. 집계 지표(F1)로는 잡히지 않는다.
  2. 단조성 — 기온을 흔들었을 때 두 헤드가 물리적으로 옳은 방향으로 반응하는지.
  3. 회귀 성능 — 현행 배포본 대비 **서빙에 쓰이는** 출력이 뒷걸음질 치지 않는지.

**죽은 게이트는 통과가 아니다(2026-10-21).** 게이트 스크립트가 0이 아닌 코드로
끝나거나 기대한 `VERDICT` 줄을 내지 않으면 FAIL 이다. 종전에는 판정 줄만 모아,
게이트가 OOM 이나 예외로 죽으면 실패 목록에 아무것도 더해지지 않아 "전부
통과"로 찍혔다.

**스테이징(2026-10-21).** 후보를 배포 경로 옆의 스테이징 파일로 복사해 출처
키 이월·게이트 기록·후속 재적합을 **그 파일에** 끝낸 뒤에야 배포 경로와
교체한다. 종전에는 배포 경로를 먼저 덮어쓰고 후속 단계를 돌렸는데, 바로 다음
줄의 `NameError`(`gate_records` 미정의) 때문에 `--promote` 는 **언제나** 배포
경로를 날 후보로 바꾼 채 죽었다 — 보정·예측구간·화면 지표가 하나도 따라오지
않는 상태다. 이제 후속 단계 하나라도 실패하면 배포 경로는 그대로 남는다.

**게이트 통과 후 자동 수행** — 배포 구성에 따라 다르다.
  · 전용 GBM 구성(배포본에 `extreme_source`/`precip_source`): 출처 키와 GBM
    보정 기록을 나가는 배포본에서 이월하고, 예측구간·화면 지표를 GBM·기온
    모델 플래그와 함께 재계산한다. 신경망 확률 보정 재적합은 하지 않는다 —
    `prob_calibration` 을 통째로 신경망 값으로 바꿔 GBM 판정선을 지운다.
  · 기온 전용 후보: 예측구간·화면 지표만 재계산한다.
  · 신경망 구성(종전): 확률 보정 재적합·예측구간·화면 지표.
  그림(docs/images)은 교체 **뒤에** +6h 배포본 기준으로만 다시 그린다.

**여전히 사람이 해야 하는 일**은 마지막에 목록으로 출력한다 — README 수치와
화면 서술은 판단이 필요해 자동화하지 않는다.

**동반 후보(2026-10-22).** 정규화 통계가 바뀌는 재학습(자료 정정·추가)은 그
통계를 공유하는 기온 모델·두 GBM 을 새 통계로 다시 만들어 함께 넘긴다 —
`--temp-cand=` · `--precip-gbm-cand=` · `--extreme-gbm-cand=`. 게이트 0 은 함께
들어갈 파일끼리 대조하고, 게이트는 후보 극한 GBM 으로 재며, 스테이징에서 전부
후속 처리한 뒤 동반 파일·주 체크포인트 순으로 교체한다(각각 `_before_` 백업).

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
# 리드타임마다 배포 경로가 다르다(app.py 의 CHECKPOINT / CHECKPOINT_12H).
# `--target` 으로 고를 수 있게 하되, 목록에 있는 경로만 허용한다 — 임의
# 경로를 받으면 실험 파일을 배포 경로로 착각해 덮어쓰는 사고가 난다
# (2026-08-19 에 실제로 겪은 유형이다).
PRODUCTION_TARGETS = {
    6:  "./checkpoints/numerical_trichef.pt",
    12: "./checkpoints/numerical_trichef_12h.pt",
    # 기온 전용 보조 모델(2026-09-25). 기온과 그 예측구간만 낸다 —
    # predict.TEMP_CHECKPOINT / app.TEMP_CHECKPOINT_12H 가 읽는 경로다.
    "temp6":  "./checkpoints/numerical_trichef_temp.pt",
    "temp12": "./checkpoints/numerical_trichef_temp_12h.pt",
}
# 기온 전용 후보는 극한기상 헤드가 경사를 받지 않아 그 게이트가 의미가 없다
# (확률이 초기값 근처에 머물러 FAIL 로 잡히지만, 그 헤드는 **서빙에서 쓰이지
# 않는다**). 대신 **주 모델을 건드리지 않으므로 그 게이트들의 결과가 정의상
# 불변**이다 — 건너뛰는 것이 아니라 해당 사항이 없는 것이다. 회귀 성능
# 게이트와 예측구간 재적합은 그대로 받는다.
TEMP_ONLY_TARGETS = (PRODUCTION_TARGETS["temp6"], PRODUCTION_TARGETS["temp12"])
# 주 체크포인트 ↔ 그와 표준화 입력을 공유하는 기온 전용 모델.
# `predict.load_temp_model` 이 서빙 시점에 두 파일의 정규화 통계 일치를 요구한다.
TEMP_OF_MAIN = {PRODUCTION_TARGETS[6]: PRODUCTION_TARGETS["temp6"],
                PRODUCTION_TARGETS[12]: PRODUCTION_TARGETS["temp12"]}
MAIN_OF_TEMP = {v: k for k, v in TEMP_OF_MAIN.items()}
CHECKPOINT = PRODUCTION_CHECKPOINT   # 이 파일 안의 기존 참조를 그대로 둔다

# 같은 표준화 입력 벡터를 만드는지 가르는 항목과, **키가 없을 때 코드가 쓰는
# 기본값**(`train.aux_dataset_kwargs`·`predict.load_model` 과 같아야 한다).
# 기본값 없이 비교하면 키가 없는 구버전과 키를 적은 신버전이 같은 입력인데도
# 다르다고 판정된다 — 실제로 배포 주 체크포인트에는 `nwp_feature_set` 키가
# 없고 기온 모델에는 "full14" 가 적혀 있다(2026-10-21 시험에서 발견).
INPUT_CONTRACT_DEFAULTS = {
    "num_features": None, "lead_hours": None,
    "im_dim": 384, "re_channels": 4,
    "use_nwp": False, "use_nwp_subset": False, "nwp_feature_set": "full14",
    "use_island": False, "use_climatology_anomaly": False,
}

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
    """스크립트를 돌리며 출력을 그대로 흘려보내고, (전체 stdout, 종료코드)를 돌려준다."""
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


def missing_verdicts(found, expect):
    """기대한 판정 중 나오지 않은 것. 끝이 `*` 이면 그 접두사로 하나 이상."""
    miss = []
    for e in expect:
        if e.endswith("*"):
            if not any(n.startswith(e[:-1]) for n in found):
                miss.append(e)
        elif e not in found:
            miss.append(e)
    return miss


def load_ckpt(path):
    import torch
    return torch.load(path, map_location="cpu", weights_only=True)


def save_ckpt_atomic(ck, path):
    """고유 tmp + os.replace — 도중에 죽어도 잘린 파일이 남지 않는다(1절 6항)."""
    import tempfile
    import torch
    fd, tmp = tempfile.mkstemp(dir=os.path.dirname(path) or ".", suffix=".pt")
    os.close(fd)
    try:
        torch.save(ck, tmp)
        os.replace(tmp, path)
    except BaseException:
        if os.path.exists(tmp):
            os.remove(tmp)
        raise


def input_contract_diff(a, b):
    """두 체크포인트가 같은 표준화 입력 벡터를 만드는가 — 다른 항목 목록."""
    diff = [k for k, dflt in INPUT_CONTRACT_DEFAULTS.items()
            if a.get(k, dflt) != b.get(k, dflt)]
    for k in ("mean", "std"):
        ra = [round(float(v), 6) for v in (a.get(k) or [])]
        rb = [round(float(v), 6) for v in (b.get(k) or [])]
        if ra != rb:
            diff.append(k)
    return diff


def gbm_contract(path, ck):
    """GBM 파일의 학습 시 정규화 통계·차원이 체크포인트와 같은지 — 다르면 RuntimeError."""
    import numpy as np
    from precip_gbm import check_input_contract
    with np.load(path, allow_pickle=False) as z:
        meta = {k: z[k] for k in z.files if k.startswith("meta_")}
    check_input_contract(meta, ck, path)


def _npz_meta(path, key):
    import numpy as np
    with np.load(path, allow_pickle=False) as z:
        return float(z[key]) if key in z.files else None


def ckpt_metrics(c):
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
    # 검증셋이 다른 후보를 위한 경로(2026-09-07). 게이트 3 의 "기준선 불일치"
    # FAIL 은 표본 구성이 바뀐 재학습을 **원리적으로** 막는다 — 실제로 세 번
    # 연속 승격이 막혔다. 이 플래그를 주면 저장된 지표를 나란히 놓는 대신
    # `rebaseline_compare.py` 로 **두 검증셋의 교집합**(어느 쪽도 학습에 쓰지
    # 않은 표본)에서 직접 비교한다. 게이트를 끄는 것이 아니라 **더 엄격한
    # 비교로 대체**하는 것이며, 유의한 회귀가 하나라도 있으면 FAIL 한다.
    do_rebaseline = "--rebaseline" in sys.argv
    # 승격 대상 배포 경로. 주지 않으면 +6h 경로다(종전 동작).
    global CHECKPOINT
    _tgt = next((a.split("=", 1)[1] for a in sys.argv[1:]
                 if a.startswith("--target=")), None)
    if _tgt is not None:
        if _tgt not in PRODUCTION_TARGETS.values():
            print(f"허용되지 않은 승격 경로: {_tgt}\n"
                  f"  가능한 값: {', '.join(PRODUCTION_TARGETS.values())}")
            sys.exit(2)
        CHECKPOINT = _tgt
    if not args:
        print("사용법: python promote_checkpoint.py <후보 체크포인트> [--promote]")
        sys.exit(2)
    cand = args[0]
    if not os.path.exists(cand):
        print(f"후보 체크포인트를 찾을 수 없다: {cand}")
        sys.exit(2)

    print(f"후보  : {cand}")
    print(f"현행  : {CHECKPOINT}")
    cand_ck = load_ckpt(cand)
    cm = ckpt_metrics(cand_ck)
    print(f"        후보 num_features={cm['num_features']} signed={cm['signed']}")

    failures, warnings = [], []
    # 게이트 판정 — 승격 시 체크포인트의 `promotion_gates` 에 남는다(화면이 읽는다).
    gate_records = {}

    def run_gate(cmd, title, expect):
        """게이트를 돌려 판정을 모은다. **죽은 게이트는 FAIL 이다**(모듈 docstring)."""
        out, code = run(cmd, title)
        found = verdicts(out)
        for name, (verdict, extra) in found.items():
            gate_records[name] = {"verdict": verdict, "detail": extra}
            if verdict == "FAIL":
                failures.append(f"{name}: FAIL {extra}")
            elif verdict == "WARN":
                warnings.append(f"{name}: WARN {extra}")
        miss = missing_verdicts(found, expect)
        if code != 0 or miss:
            why = f"종료코드 {code}" if code != 0 else f"판정 누락 {', '.join(miss)}"
            failures.append(f"{title}: 게이트가 판정을 내지 못했다({why}) — "
                            f"죽은 게이트를 통과로 세지 않는다")

    # ── 배포 구성 — 출처 키는 **나가는 배포본**에서 읽는다 ─────────────
    # 갓 학습한 후보에는 출처 키가 없다. 후보에서 읽으면 GBM 구성을 모르는
    # 채로 신경망 기준 후속 단계를 돌려 GBM 출처·보정 기록을 지운다.
    _temp_only = CHECKPOINT in TEMP_ONLY_TARGETS
    deployed = load_ckpt(CHECKPOINT) if os.path.exists(CHECKPOINT) else None
    _ex_gbm = _pr_gbm = None
    if deployed is not None and not _temp_only:
        _ex_gbm = deployed.get("extreme_source")
        _pr_gbm = deployed.get("precip_source")
    for _p in (_ex_gbm, _pr_gbm):
        if _p and not os.path.exists(_p):
            print(f"배포 체크포인트가 전용 모델({_p})을 가리키는데 파일이 없다 — "
                  f"게이트를 신뢰할 수 없다.")
            sys.exit(2)
    _temp_dep = TEMP_OF_MAIN.get(CHECKPOINT)
    if _temp_dep and not os.path.exists(_temp_dep):
        _temp_dep = None
    _gbm_mode = bool(_ex_gbm or _pr_gbm)
    if _gbm_mode:
        print(f"\n배포 구성: 전용 GBM — 극한기상 {_ex_gbm or '없음'} · 강수 {_pr_gbm or '없음'}"
              + (f" · 기온 {_temp_dep}" if _temp_dep else ""))

    # ── 동반 후보(2026-10-22) — 입력 계약이 바뀌는 재학습 ──────────────
    # 주 체크포인트를 재학습하면 정규화 통계가 바뀌어, 그 표준화 입력을 공유하는
    # 기온 모델·두 GBM 도 **새 통계로 다시 만든 것과 함께** 들어가야 한다. 종전에는
    # 게이트 0 이 그런 후보를 막기만 했다(교체할 방법이 없었다). 동반 후보를 주면
    # 게이트 0 은 **함께 들어갈 파일들끼리** 대조하고, 게이트 1·2 는 후보 극한기상
    # GBM 으로 재며, 스테이징에서 전부 후속 처리한 뒤 한꺼번에 교체한다.
    def _opt(name):
        return next((a.split("=", 1)[1] for a in sys.argv[1:]
                     if a.startswith(f"--{name}=")), None)
    comp = {"temp": _opt("temp-cand"), "precip": _opt("precip-gbm-cand"),
            "extreme": _opt("extreme-gbm-cand")}
    _final = {"temp": _temp_dep, "precip": _pr_gbm, "extreme": _ex_gbm}
    for _k, _p in comp.items():
        if not _p:
            continue
        if _temp_only or not _gbm_mode:
            print("동반 후보는 전용 GBM 구성의 주 체크포인트 승격에서만 쓴다.")
            sys.exit(2)
        if not os.path.exists(_p):
            print(f"동반 후보 파일이 없다: {_p}")
            sys.exit(2)
        if not _final[_k]:
            print(f"동반 후보({_k})가 교체할 배포 파일이 이 구성에 없다.")
            sys.exit(2)
    _temp_use = comp["temp"] or _temp_dep
    _pr_use = comp["precip"] or _pr_gbm
    _ex_use = comp["extreme"] or _ex_gbm
    if any(comp.values()):
        print("동반 후보: " + " · ".join(f"{k} {p} → {_final[k]}" for k, p in comp.items() if p))

    # ── 게이트 0 — 입력 계약 ─────────────────────────────────────
    print(f"\n{'='*78}\n▶ 게이트 0 — 입력 계약(같은 표준화 입력을 공유하는 모델)\n{'='*78}")
    if _temp_only:
        _main = MAIN_OF_TEMP[CHECKPOINT]
        _diff = input_contract_diff(load_ckpt(_main), cand_ck)
        _who = f"주 체크포인트 {_main}"
    else:
        # **함께 쓰일 파일과 직접** 대조한다(2026-10-22) — 기온 모델은 그 체크포인트,
        # GBM 은 자기 파일에 적힌 학습 시 통계. 배포본을 거쳐 비교하면 배포본 자신이
        # 이미 어긋나 있을 때 후보도 같은 통계라는 이유로 통과하고, 동반 후보로
        # 함께 교체할 때는 나가는 배포본이 기준이 될 수 없다.
        _shared = [p for p in (_temp_use, _ex_use, _pr_use) if p]
        _diff = ([f"기온 모델 {k}" for k in input_contract_diff(load_ckpt(_temp_use), cand_ck)]
                 if _temp_use else [])
        for _p, _lab in ((_pr_use, "강수 GBM"), (_ex_use, "극한기상 GBM")):
            if _p:
                try:
                    gbm_contract(_p, cand_ck)
                except RuntimeError as e:
                    _diff.append(f"{_lab}({e})")
        _who = "이 체크포인트의 입력을 공유하는 " + (", ".join(_shared) or "모델 없음")
    if _diff:
        failures.append(
            f"입력 계약 불일치({', '.join(_diff)}) — {_who} 와 표준화 입력이 달라진다. "
            f"그 모델들을 이 통계로 다시 만들어 동반 후보(--temp-cand·--precip-gbm-cand·"
            f"--extreme-gbm-cand)로 함께 넘겨야 한다(GBM 은 조용히 어긋나고, 기온 모델은 "
            f"서빙 시작 시 멈춘다)")
        gate_records["input_contract"] = {"verdict": "FAIL", "detail": ",".join(_diff)}
        print(f"  ★FAIL★ {', '.join(_diff)} 가 다르다 — 나머지 게이트를 돌리지 않는다.")
    else:
        gate_records["input_contract"] = {"verdict": "PASS", "detail": ""}
        print(f"  PASS — {_who} 와 같은 입력 벡터를 만든다.")

    # ── 게이트 1·2 — 후보를 직접 지정해 점검한다 ──────────────────
    # 극한기상을 전용 GBM 이 내면 게이트도 **그 모델**을 재야 한다
    # (2026-09-26). 신경망 헤드는 배포에서 쓰이지 않으므로, 그것을
    # 검사하는 게이트는 배포와 무관한 것을 재고 통과시킨다 — 절차가
    # 아무것도 지키지 못하는 상태가 된다.
    _gopt = ([f"--extreme-gbm={_ex_use}"] if _ex_use else [])
    if failures:
        pass                                    # 입력 계약이 깨지면 더 재지 않는다
    elif _temp_only:
        print(f"\n{'='*78}\n▶ 게이트 1·2·2-1 — 해당 없음(기온 전용 후보)\n{'='*78}")
        print("  이 후보는 기온과 그 예측구간만 낸다. 극한기상 헤드는 손실을 받지\n"
              "  않아 확률이 초기값 근처이고, **서빙에서 쓰이지 않는다** — 주 모델을\n"
              "  건드리지 않으므로 계절 오탐·단조성·관측소 사각지대 판정은 정의상\n"
              "  불변이다. 회귀 성능 게이트는 그대로 적용한다.")
    else:
        if _ex_use:
            print(f"\n극한기상 게이트 대상: {_ex_use} "
                  + ("(동반 후보 — 승격되면 서빙이 쓴다)" if comp["extreme"] else "(서빙과 동일)"))
        run_gate(["seasonal_falsealarm_check.py", cand] + _gopt,
                 "게이트 1 — 계절 오탐", ["seasonal_false_alarm"])
        for head in ("coldwave", "heatwave"):
            run_gate(["coldwave_pathway_check.py", f"--head={head}", cand] + _gopt,
                     f"게이트 2 — 단조성({head})", [f"monotonicity_{head}"])
        # 계절 축과 같은 사각지대가 관측소 축에도 있는지 본다. 라벨이 없는
        # 관측소의 출력은 채점된 적이 없으므로, 경보를 얼마나 내는지 확인한다.
        run_gate(["station_coverage_check.py", cand] + _gopt,
                 "게이트 2-1 — 관측소 라벨 사각지대", ["station_coverage_*"])

    # ── 게이트 3 — 회귀 성능이 뒷걸음질 치지 않는지 ────────────────
    print(f"\n{'='*78}\n▶ 게이트 3 — 회귀 성능 비교\n{'='*78}")
    pm = ckpt_metrics(deployed) if (deployed is not None and not failures) else None
    if deployed is None:
        print("  현행 체크포인트가 없다 — 비교를 건너뛴다.")
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
        _mismatch = [
            (_lbl, pm.get(_k), cm.get(_k))
            for _k, _lbl in (("temp_naive", "기온 기준선"), ("precip_naive", "강수 기준선"))
            if pm.get(_k) is not None and cm.get(_k) is not None
            and abs(pm[_k] - cm[_k]) > 1e-4
        ]
        for _lbl, _a, _b in _mismatch:
            msg = (f"{_lbl}(naive MAE) 불일치: {_a:.4f} vs {_b:.4f} — "
                   f"검증셋이 다르다는 뜻이라 저장된 지표의 직접 비교가 성립하지 않는다")
            if do_rebaseline:
                warnings.append(msg + " (교집합 비교로 대체함)")
            else:
                failures.append(msg + " — --rebaseline 으로 교집합 비교를 쓸 수 있다")
        if _mismatch and do_rebaseline:
            run_gate(["rebaseline_compare.py", CHECKPOINT, cand],
                     "게이트 3′ — 교집합 재기준선 비교(학습 누수 없음)", ["rebaseline"])
            print("  저장된 지표의 직접 비교는 생략한다 — 검증셋이 다르고, 확률 눈금이")
            print("  이동하면 t=0.5 로 잰 F1 은 서로 다른 동작점을 뜻해 비교가 성립하지")
            print("  않는다. 위 게이트 3′(교집합 비교)의 결과를 판단 근거로 쓴다.")

        # **서빙에 쓰이는 출력만 비교한다(2026-10-21).** 전용 모델 구성에서
        # 배포본의 강수·극한기상 지표는 GBM 값이고(metrics_report 가 패치),
        # 후보의 값은 신경망 값이다 — 나란히 놓으면 모델 계열을 비교하게 된다
        # (강수는 신경망 MAE 가 낮아 늘 OK, 한파는 t=0.5 신경망 F1 이 낮아 늘
        # 악화). 게이트 0 이 입력 계약 동일을 보장하므로 그 출력들은 이 승격으로
        # 바뀌지 않는다.
        served_by = {}
        if not _temp_only:
            if _temp_dep:
                served_by["temp"] = _temp_dep
            if _pr_gbm:
                served_by["precip"] = _pr_gbm
            if _ex_gbm:
                served_by["heat_f1"] = served_by["cold_f1"] = _ex_gbm
        rows = (("temp", "기온 MAE", True), ("precip", "강수 MAE", True),
                ("heat_f1", "폭염 F1", False), ("cold_f1", "한파 F1", False))
        if _temp_only:
            rows = rows[:1]     # 기온 전용 후보가 서빙에 내는 것은 기온뿐이다
        # 동반 후보가 있는 출력은 그 모델끼리 저장된 값을 비교한다 — 같은 검증
        # 표본에서 각자 학습 시점에 잰 값이다(기온: 검증셋 MAE, GBM: 평가용 절반).
        _cmp = {}
        if comp["temp"]:
            _cmp["temp"] = (load_ckpt(_temp_dep).get("val_temp_mae"),
                            load_ckpt(comp["temp"]).get("val_temp_mae"), "기온 모델 검증 MAE")
        if comp["precip"]:
            _cmp["precip"] = (_npz_meta(_pr_gbm, "meta_val_precip_wet_f1"),
                              _npz_meta(comp["precip"], "meta_val_precip_wet_f1"),
                              "강수 GBM 발생 F1(평가용 절반)")
        if comp["extreme"]:
            for _k, _h in (("heat_f1", "heatwave"), ("cold_f1", "coldwave")):
                _cmp[_k] = (_npz_meta(_ex_gbm, f"meta_{_h}_f1_cal"),
                            _npz_meta(comp["extreme"], f"meta_{_h}_f1_cal"),
                            f"극한 GBM {_h} F1(평가용 절반)")
        print(f"  {'지표':<12}{'현행':>12}{'후보':>12}   판정")
        for key, label, lower_better in rows:
            if key in _cmp:
                a, b, what = _cmp[key]
                worse = ((b > a * (1 + REGRESSION_TOL)) if key == "temp"
                         else (b < a * (1 - REGRESSION_TOL)))
                if worse:
                    warnings.append(f"{what} 악화: {a:.4f} → {b:.4f}")
                # 라벨은 실제로 비교한 지표로 적는다 — 동반 후보 행은 MAE 자리에
                # F1 이 들어가기도 한다(강수 GBM 은 발생 F1 로 비교한다).
                print(f"  {what:<28}{a:>10.4f}{b:>10.4f}   {'★악화★' if worse else 'OK'}"
                      f"  (동반 후보)")
                continue
            if key in served_by:
                print(f"  {label:<12}{'—':>12}{'—':>12}   서빙은 {served_by[key]} 가 낸다"
                      f"(입력 계약 동일 → 출력 불변)")
                continue
            a, b = pm.get(key), cm.get(key)
            if a is None or b is None:
                continue
            worse = (b > a * (1 + REGRESSION_TOL)) if lower_better \
                else (b < a * (1 - REGRESSION_TOL))
            mark = "★악화★" if worse else "OK"
            if worse and not (_mismatch and do_rebaseline):
                warnings.append(f"{label} 악화: {a:.4f} → {b:.4f}")
            elif worse:
                mark = "(비교 불가)"
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

    # ── 스테이징 — 배포 경로를 건드리기 전에 후속 처리를 끝낸다 ──────
    stamp = time.strftime("%Y%m%d_%H%M%S")
    staging = CHECKPOINT.replace(".pt", f"_staging_{stamp}.pt")
    shutil.copy2(cand, staging)
    print(f"\n스테이징: {staging}")

    # 자식에게 경로를 **명시적으로** 넘긴다. 인자를 생략하면 자식이
    # predict.CHECKPOINT(=CHECKPOINT_PATH 환경변수)로 기본값을 잡으므로,
    # 환경이 오염된 셸에서는 배포되지 않은 파일에 보정을 적합하게 된다
    # (위 PRODUCTION_CHECKPOINT 주석 참고).
    os.environ.pop("CHECKPOINT_PATH", None)

    comp_staging = {}

    def chain(cmd, title):
        out, code = run(cmd, title)
        if code != 0:
            raise RuntimeError(f"{title} 실패(종료코드 {code})")
        return out

    try:
        st = load_ckpt(staging)
        # 출처 키와 GBM 보정 기록을 나가는 배포본에서 이월한다. 게이트 0 이
        # 입력 계약 동일을 보장하므로 그 모델들은 이 체크포인트에서도 유효하다.
        carried = []
        if _ex_gbm:
            st["extreme_source"] = _ex_gbm
            # 판정선·보정 방법·곡선 메타는 GBM 의 것이다(extreme_gbm
            # --patch-checkpoint 가 적었다). 후보 자신의 값은 신경망 헤드용이다.
            # 극한 GBM 동반 후보가 있으면 아래에서 그 파일로 다시 적는다.
            if deployed.get("prob_calibration") is not None and not comp["extreme"]:
                st["prob_calibration"] = deployed["prob_calibration"]
            carried += ["extreme_source"] + ([] if comp["extreme"] else ["prob_calibration"])
        if _pr_gbm:
            st["precip_source"] = _pr_gbm
            carried.append("precip_source")
        st["promotion_gates"] = {"checked_at": time.strftime("%Y-%m-%d"),
                                 "results": gate_records}
        save_ckpt_atomic(st, staging)
        if carried:
            print(f"이월: {', '.join(carried)} (나가는 배포본 → 스테이징)")
        print("게이트 판정 기록:")
        for name, r in sorted(gate_records.items()):
            print(f"  {name}: {r['verdict']} {r['detail']}")

        if _temp_only:
            chain(["conformal_interval_fit.py", staging, "--apply"], "예측구간 재적합")
            chain(["metrics_report.py", staging, "--patch-checkpoint", "--no-plot"],
                  "화면용 지표 기록")
        elif _gbm_mode:
            print("\n(확률 보정 재적합은 하지 않는다 — prob_calibration 을 신경망 값으로\n"
                  " 통째로 바꿔 GBM 판정선을 지운다. GBM 의 보정은 모델 파일에 있다.)")
            # 동반 후보를 배포 경로 옆 스테이징 이름으로 복사한다 — 교체는 같은
            # 디렉터리 안의 os.replace 로 한다.
            for _k, _p in comp.items():
                if _p:
                    _ext = os.path.splitext(_final[_k])[1]
                    comp_staging[_k] = _final[_k].replace(_ext, f"_staging_{stamp}{_ext}")
                    shutil.copy2(_p, comp_staging[_k])
            if comp["extreme"]:
                chain(["extreme_gbm.py", "--patch-only", comp_staging["extreme"],
                       "--checkpoint", staging, "--out", _ex_gbm],
                      "극한기상 GBM 성질 기록(동반 후보)")
            _temp_for_mr = _temp_dep
            if comp["temp"]:
                chain(["conformal_interval_fit.py", comp_staging["temp"], "--apply"],
                      "기온 모델 예측구간 재적합(동반 후보)")
                chain(["metrics_report.py", comp_staging["temp"], "--patch-checkpoint",
                       "--no-plot"], "기온 모델 지표 기록(동반 후보)")
                _temp_for_mr = comp_staging["temp"]
            _pr_stage = comp_staging.get("precip") or _pr_gbm
            _ex_stage = comp_staging.get("extreme") or _ex_gbm
            chain(["conformal_interval_fit.py", staging, "--apply"]
                  + ([f"--precip-gbm={_pr_stage}"] if _pr_stage else []), "예측구간 재적합")
            _mr = ["metrics_report.py", staging, "--patch-checkpoint", "--no-plot"]
            if _pr_stage:
                _mr += ["--precip-gbm", _pr_stage]
            if _ex_stage:
                _mr += ["--extreme-gbm", _ex_stage]
            if _temp_for_mr:
                _mr += ["--temp-checkpoint", _temp_for_mr]
            chain(_mr, "화면용 지표 기록(서빙 구성)")
            # 후속 단계가 **읽은 경로**(스테이징 이름)를 출처로 적었다 — 배포 경로로
            # 바꿔 적는다. 서빙이 이 기록과 실제 로드 경로를 대조하므로(predict.
            # _check_source) 그대로 두면 교체 직후 화면이 멈춘다.
            if comp["precip"]:
                st = load_ckpt(staging)
                st["precip_source"] = _pr_gbm
                if st.get("conformal_interval"):
                    st["conformal_interval"]["precip_source"] = _pr_gbm
                save_ckpt_atomic(st, staging)
                print(f"출처 경로 정정: {comp_staging['precip']} → {_pr_gbm}")
        else:
            chain(["probability_calibration_fit.py", staging, "--apply"], "확률 보정 재적합")
            # 정합적 예측구간도 반드시 다시 적합한다(2026-09-07 추가). 빠뜨리면
            # 체크포인트에 `conformal_interval` 이 없어 화면의 90% 구간이 **조용히
            # 사라진다** — 오류도 경고도 없이 기능 하나가 없어지는 유형이다.
            chain(["conformal_interval_fit.py", staging, "--apply"], "예측구간 재적합")
            _mr = ["metrics_report.py", staging, "--patch-checkpoint", "--no-plot"]
            if _temp_dep:
                _mr += ["--temp-checkpoint", _temp_dep]
            chain(_mr, "화면용 지표 기록(서빙 MAE·극한기상 서빙 판정선 기준)")
    except BaseException as e:                       # noqa: BLE001
        for _p in [staging] + list(comp_staging.values()):
            if os.path.exists(_p):
                os.remove(_p)
        print(f"\n★후속 처리 실패★ {type(e).__name__}: {e}")
        print(f"배포 경로는 그대로다({CHECKPOINT}). 스테이징 파일은 지웠다.")
        sys.exit(1)

    # ── 교체 ──────────────────────────────────────────────────────
    # 동반 후보부터 바꾸고 주 체크포인트를 마지막에 바꾼다. 어느 순서든 교체
    # 사이의 짧은 순간에는 짝이 어긋나지만, 서빙은 그때 조용히 틀리지 않고
    # 계약 대조(load_* · _check_source)로 멈췄다가 다음 재실행에서 맞춰진다.
    for _k, _sp in comp_staging.items():
        _ext = os.path.splitext(_final[_k])[1]
        _bk = _final[_k].replace(_ext, f"_before_{stamp}{_ext}")
        shutil.copy2(_final[_k], _bk)
        os.replace(_sp, _final[_k])
        print(f"동반 교체: {_final[_k]} (백업 {_bk})")
    if os.path.exists(CHECKPOINT):
        backup = CHECKPOINT.replace(".pt", f"_before_{stamp}.pt")
        shutil.copy2(CHECKPOINT, backup)
        print(f"\n현행 백업: {backup}")
    os.replace(staging, CHECKPOINT)
    print(f"배포 경로 교체 완료: {CHECKPOINT}")
    # 후속 단계가 스테이징 파일 이름으로 남긴 부산물을 지운다 — 예측구간
    # 재적합의 자체 백업(`*_before_conformal.pt`)과 비배포 경로용 지표 JSON.
    # 교체가 끝났으므로 원본은 위 현행 백업이 보존한다.
    import glob
    for _sp in [staging] + ([comp_staging["temp"]] if "temp" in comp_staging else []):
        _stem = os.path.splitext(os.path.basename(_sp))[0]
        for _p in (glob.glob(os.path.join(os.path.dirname(_sp) or ".", f"{_stem}_before_*.pt"))
                   + glob.glob(f"./cache/metrics_report_{_stem}.json")):
            os.remove(_p)
            print(f"  부산물 정리: {_p}")

    # ── 그림 — 교체 뒤, +6h 배포본 기준으로만 ──────────────────────
    # 화면에 쓰이는 그림(docs/images/*.png)은 **+6h 배포본 기준의 공용
    # 산출물**이다. 다른 리드타임을 승격하면서 이걸 다시 그리면 화면이
    # 엉뚱한 모델의 그림을 보여주게 된다 — 2026-09-07 +12h 승격에서 실제로
    # calibration_plot.png·probability_calibration.png 가 12h 기준으로
    # 덮어써졌다. 그림 실패는 배포를 되돌릴 사유가 아니므로 남은 작업으로 알린다.
    plot_fail = []
    _is_primary = os.path.abspath(CHECKPOINT) == os.path.abspath(PRODUCTION_CHECKPOINT)
    if _is_primary:
        os.environ["CHECKPOINT_PATH"] = CHECKPOINT      # calibration_plot_diagnose 가 읽는다
        _mr = ["metrics_report.py", CHECKPOINT]
        if _pr_gbm:
            _mr += ["--precip-gbm", _pr_gbm]
        if _ex_gbm:
            _mr += ["--extreme-gbm", _ex_gbm]
        if _temp_dep:
            _mr += ["--temp-checkpoint", _temp_dep]
        steps = [(_mr, "지표 그림 재생성"),
                 (["calibration_plot_diagnose.py"] + _gopt, "관측소별 플롯 재생성")]
        if not _gbm_mode:
            steps.insert(0, (["probability_calibration_check.py", CHECKPOINT],
                             "신뢰도 곡선 재생성"))
        for cmd, title in steps:
            _, code = run(cmd, title)
            if code != 0:
                plot_fail.append(title)
    else:
        print(f"\n{'='*78}\n▶ 그림 재생성 — 건너뜀(+6h 배포본 기준 공용 산출물)\n{'='*78}")
        print(f"  필요하면 +6h 로 따로 실행할 것: "
              f"CHECKPOINT_PATH={PRODUCTION_CHECKPOINT} python calibration_plot_diagnose.py")

    # ── 사람이 해야 할 일 ─────────────────────────────────────────
    print(f"\n{'='*78}\n 남은 수동 작업 — 자동화하지 않는다(판단이 필요하다)\n{'='*78}")
    if plot_fail:
        print(f"  ★ 그림 재생성 실패: {', '.join(plot_fail)} — 배포는 교체됐다. 다시 돌릴 것.")
    if _gbm_mode:
        print("""  0. 판정선은 상수가 아니라 **모델 파일**에 있다(강수 meta_gate_tau ·
     극한기상 tau_cal). 이 승격은 GBM 을 바꾸지 않으므로 판정선도 그대로다.
     GBM 을 다시 만들었다면(precip_gbm.py · extreme_gbm.py) 그 시점 표본에서
     재선정된 값이 파일에 들어간다 — predict.py 의 상수를 고치지 않는다.
  1. 신뢰도 곡선 그림(docs/images/probability_calibration.png)은 GBM 모드가
     없어 다시 그리지 않았다 — 화면이 이 그림을 GBM 의 것처럼 설명하지
     않는지 확인한다(NEXT.md 2절 P2-1).""")
    else:
        print("""  0. 강수 게이팅 임계값(predict.PRECIP_PROB_GATE_BY_LEAD)을 재선정한다 —
     규칙 9 는 극한기상 판정선만이 아니라 이 게이트에도 적용된다.
  1. threshold_validation.py 를 돌려 임계값을 재산출하고, 채택 기준(순이득
     0.01)을 넘은 것만 predict.py 의 EXTREME_EVENT_THRESH 에 반영한다.
  2. 관측소별 예외(STATION_EVENT_THRESH_OVERRIDES)가 남아 있다면
     station_threshold_check.py 로 이 모델에서도 유효한지 재검증한다 —
     이전 모델에서 유효했다는 이유만으로 유지하지 않는다.""")
    print("""  3. README.md 와 app.py 의 수치·서술을 새 모델 기준으로 갱신한다:
       · 최상단 요약표, 재학습 결과표, 극한기상 성능표
       · 확률 보정 표(ECE)
       · 관측소별 성능 캡션(플롯이 바뀌었으므로 서술도 바뀐다)
  3-1. **어떤 출력의 출처(모델)가 바뀌었다면, 화면에서 옛 출처 이름을
     직접 grep 한다** — 예: 극한기상을 GBM 으로 옮겼다면
       grep -n "다중과제 신경망\\|신경망 헤드" app.py
     강수·융합 서술도 본다(2026-10-22 — GBM 전환 뒤에도 남아 있던 문구들):
       grep -n "곱한 값\\|곱하여 산출\\|확률과 강도의 곱\\|허들\\|3축을 논문 Eq.1로\\|기온·습도·기압·풍속에 한정" app.py
     가능하면 `precip_source`·`extreme_source` 로 분기한다(`app._precip_how` 처럼).
     지표는 체크포인트 조회로 따라오지만 **"무엇이 이 값을 낸다"는 서술은
     사람이 고쳐야 한다.** 2026-09-26 에 같은 유형(화면이 서빙과 다른 것을
     가리킴)이 하루에 세 번 났다 — 판정선 상수·지표 키·출처 서술.
  4. 탭 스크린샷을 다시 찍는다: python capture_tab_screenshots.py
     (로컬 docker 필요. 화면 문구·지표를 고쳤을 때도 함께 돌릴 것.)
  5. 시험을 돌린다: test_diagnostics_smoke.py(GBM 경로를 보는 유일한 시험) ·
     test_phase34.py · test_phase36.py.
  6. 배포 후 Streamlit Cloud 에서 앱을 Reboot 한다 — import 모듈이 캐시돼
     재배포만으로는 새 모델 파일·코드가 반영되지 않을 수 있다.""")
    if plot_fail:
        sys.exit(3)


if __name__ == "__main__":
    main()
