"""
baseline_suite.py — Tri-CHEF 를 **제대로 된 기준선들과** 같은 표본에서 비교한다
(2026-09-24 신설).

## 왜 필요한가

지금까지 이 저장소가 써 온 기준선은 두 개뿐이다 — 기온은 퍼시스턴스
(T_{t+L} ≈ T_t), 강수는 상시 0 예측. 그런데 2026-09-07 배포부터 모델은
`jma_gsm` **수치예보 예보값을 입력으로 받는다.** 절제 실험은 기온 이득의
94~95%를 `예보기온` 하나가 나른다고 이미 밝혔다.

그렇다면 물어야 할 질문은 "퍼시스턴스보다 나은가"가 아니라 **"예보값을
그대로 쓰는 것보다 나은가"** 이고, 그 질문은 한 번도 측정된 적이 없다.
퍼시스턴스 대비 +64.6%는 모델이 이제 쓰지 않는 기준선에 대한 개선률이다.

기상 분야에서 이 질문의 표준 비교군은 셋이다.

  ① **원시 수치예보(raw NWP)** — 후처리 없이 예보값 그대로. 리드타임이
     보존된 아카이브(`_previous_day1`)라 공정한 예보 대 예보 비교다.
  ② **MOS(Model Output Statistics)** — 예보값·퍼시스턴스·시각·관측소로
     선형회귀. 운영 기상의 표준 후처리이며 파라미터가 수십 개다.
  ③ **기후값(climatology)** — 관측소×월×시각 평균. 계절 성분만으로 얼마나
     설명되는지를 분리한다.

여기에 표형 데이터의 강한 기준선인 **경사부스팅(GBM)** 을 더한다 — 모델과
**완전히 같은 28개 특징**을 받으므로, 차이가 나면 그것은 특징이 아니라
아키텍처의 차이다.

## 공정성을 위해 지키는 것

- **같은 분할·같은 표본.** 입력 행렬은 `eval_cache.load_features()` 에서
  받으며, 그 함수가 체크포인트의 검증셋 대조를 통과시킨다. 모델 예측은
  `eval_cache.load()` 에서 받고, 두 캐시의 서명이 같은지 직접 대조한다.
- **학습 분할에서만 적합한다.** ②③④는 학습 인덱스만 보고, 검증에서는
  예측만 한다.
- **판정선은 모든 방법에 같은 방식으로 준다.** 강수 발생 F1 은 검증셋에서
  최댓값을 취하는 **낙관적 상한**이며 — 방법마다 유리하게 고르는 것을 막기
  위해 전부 같은 규칙을 쓴다. 배포 판정선으로 잰 서빙 F1 은 따로 표기한다.

실행:
    python baseline_suite.py [체크포인트] [--gbm-iter 200]
"""
import argparse
import sys

import numpy as np

import eval_cache
from nwp_collector import FEATURE_SETS

# Z축 표준 열 번호(`train.record_to_vec` docstring 이 위치 고정을 명시한다).
I_TEMP, I_HSIN, I_HCOS, I_LAT, I_LON, I_YSIN, I_YCOS = 0, 8, 9, 10, 11, 12, 13
# 수치예보 14차원 원본 열 번호(`nwp_collector._encode`).
NWP_RAW_PRECIP, NWP_RAW_TEMP = 0, 1


def _destd(x, mean, std, col):
    return x[:, col] * std[col] + mean[col]


def _nwp_cols(nf: int, feature_set: str):
    """(예보강수 열, 예보기온 열) — 없으면 None. 수치예보 블록은 벡터 뒤에 붙는다."""
    cols = FEATURE_SETS.get(feature_set)
    if not cols:
        return None, None
    base = nf - len(cols)
    def pos(raw):
        return base + cols.index(raw) if raw in cols else None
    return pos(NWP_RAW_PRECIP), pos(NWP_RAW_TEMP)


def mae(pred, true):
    return float(np.mean(np.abs(pred - true)))


def max_f1(score, y):
    """점수 → (최대 F1, 그 판정선). 모든 방법에 같은 규칙으로 적용한다."""
    order = np.argsort(-score)
    tp = np.cumsum(y[order])
    fp = np.cumsum(1 - y[order])
    fn = y.sum() - tp
    f1 = 2 * tp / np.maximum(2 * tp + fp + fn, 1e-9)
    k = int(np.argmax(f1))
    return float(f1[k]), float(score[order][k])


def f1_at(score, y, thr):
    pred = score >= thr
    tp = float(np.sum(pred & (y > 0)))
    fp = float(np.sum(pred & (y == 0)))
    fn = float(np.sum(~pred & (y > 0)))
    return 2 * tp / max(2 * tp + fp + fn, 1e-9)


def honest_f1(score, y, half):
    """보정용 절반에서 판정선을 고르고 **평가용 절반에서만** 채점한다.

    검증셋 전체에서 최댓값을 취하면 자유도 하나만큼 낙관 편향이 섞인다.
    저장소 규약(§2 "임계값은 보정용/평가용을 분리해서 고른다")과 같은 방식이며,
    모든 방법에 동일하게 적용해야 비교가 성립한다.
    """
    thr = max_f1(score[half], y[half])[1]
    return f1_at(score[~half], y[~half], thr)


def conditional_mae(pred, true, y_wet):
    """실제 강수 구간에서만 잰 MAE — 이 저장소의 강수 주 지표."""
    m = y_wet.astype(bool)
    return float(np.mean(np.abs(pred[m] - true[m])))


def climatology(stn_tr, ts_tr, y_tr, stn_va, ts_va):
    """관측소 × 월 × 시각 평균. 학습 분할에서만 만든다."""
    def key(stn, ts):
        # ts 는 YYYYMMDDHHmm(12자리) 이므로 월은 10**6 으로 나눠야 나온다 —
        # 10**4 로 나누면 '일'이 잡힌다(첫 실행에서 실제로 그랬고, 기후값
        # MAE 가 8.57°C 로 퍼시스턴스보다 나빠 드러났다).
        month = (ts // 10 ** 6) % 100
        hour = (ts // 100) % 100
        return stn.astype(np.int64) * 10000 + month * 100 + hour
    k_tr, k_va = key(stn_tr, ts_tr), key(stn_va, ts_va)
    uniq, inv = np.unique(k_tr, return_inverse=True)
    tot = np.bincount(inv, weights=y_tr, minlength=len(uniq))
    cnt = np.bincount(inv, minlength=len(uniq))
    table = tot / np.maximum(cnt, 1)
    glob = float(y_tr.mean())
    pos = np.searchsorted(uniq, k_va)
    pos = np.clip(pos, 0, len(uniq) - 1)
    hit = uniq[pos] == k_va
    return np.where(hit, table[pos], glob)


def station_dummies(stn, levels):
    """관측소 원핫(첫 수준 제외 — 절편과 공선이 되지 않도록)."""
    return np.stack([(stn == s).astype(np.float32) for s in levels[1:]], axis=1)


def mos(x_tr, y_tr, x_va, stn_tr, stn_va, cols, levels):
    """선형 MOS — 예보값·퍼시스턴스·시각·계절·관측소. 학습 분할에서 적합."""
    def design(x, stn):
        base = [np.ones(len(x), dtype=np.float32)] + [x[:, c] for c in cols]
        return np.concatenate(
            [np.stack(base, axis=1), station_dummies(stn, levels)], axis=1)
    A = design(x_tr, stn_tr)
    beta, *_ = np.linalg.lstsq(A, y_tr, rcond=None)
    return design(x_va, stn_va) @ beta


def gbm_regress(x_tr, y_tr, x_va, iters):
    from sklearn.ensemble import HistGradientBoostingRegressor
    m = HistGradientBoostingRegressor(max_iter=iters, learning_rate=0.1,
                                      random_state=42)
    m.fit(x_tr, y_tr)
    return m.predict(x_va)


def gbm_classify(x_tr, y_tr, x_va, iters):
    from sklearn.ensemble import HistGradientBoostingClassifier
    m = HistGradientBoostingClassifier(max_iter=iters, learning_rate=0.1,
                                       random_state=42)
    m.fit(x_tr, y_tr)
    return m.predict_proba(x_va)[:, 1]


def check_same_split(feat, ev):
    """두 캐시가 같은 체크포인트·같은 분할을 가리키는지 대조한다.

    따로 만들어지는 캐시라 한쪽만 낡을 수 있다. 어긋난 채 비교하면 모델과
    기준선이 다른 표본에서 채점되는데 지표는 멀쩡해 보인다 —
    `_assert_split_matches_checkpoint` 가 막으려는 것과 같은 종류의 사고다.
    """
    keys = ["ckpt_path", "ckpt_mtime", "ckpt_size", "split_mode", "split_algo",
            "num_features", "lead_hours", "data_mtime", "data_size"]
    bad = [k for k in keys
           if k in feat and k in ev and str(feat[k]) != str(ev[k])]
    if bad:
        raise SystemExit(
            f"두 캐시의 서명이 다르다({', '.join(bad)}) — "
            f"`python eval_cache.py --force` 로 둘 다 다시 만들 것.")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt", nargs="?", default=eval_cache.CHECKPOINT)
    ap.add_argument("--gbm-iter", type=int, default=200)
    args = ap.parse_args()

    feat = eval_cache.load_features(args.ckpt)
    ev = eval_cache.load(args.ckpt)
    check_same_split(feat, ev)

    mean, std = feat["mean"], feat["std"]
    x_tr, x_va = feat["x_train"], feat["x_val"]
    nf = x_va.shape[1]
    t_tr, t_va = feat["temp_true_train"], feat["temp_true_val"]
    p_tr, p_va = feat["precip_true_train"], feat["precip_true_val"]
    stn_tr, stn_va = feat["stn_train"], feat["stn_val"]
    ts_tr, ts_va = feat["tgt_ts_train"], feat["tgt_ts_val"]
    wet = float(feat["wet_thresh"])
    y_wet_tr = (p_tr >= wet).astype(np.int8)
    y_wet_va = (p_va >= wet).astype(np.int8)
    levels = np.unique(np.concatenate([stn_tr, stn_va]))
    im_tr, im_va = feat["im_train"], feat["im_val"]
    # 판정선 선정용 절반을 **날짜로** 가른다 — 표본을 무작위로 쪼개면 같은 날
    # 다른 시각이 양쪽에 들어가 판정선 선정이 채점 표본을 엿본다(§2 규약이
    # 검증 분할에 요구하는 것과 같은 이유다).
    half = ((ts_va // 10 ** 4) % 2 == 0)
    print(f"판정선 보정용 {half.sum():,} · 평가용 {(~half).sum():,}"
          f" (목표일 홀짝으로 분리)")

    fs = str(feat["nwp_feature_set"]) if bool(feat["use_nwp"]) else ""
    c_pr, c_tp = _nwp_cols(nf, fs)

    print(f"체크포인트: {args.ckpt}")
    print(f"학습 {len(t_tr):,} · 검증 {len(t_va):,} · 특징 {nf} · "
          f"수치예보 {fs or '없음'} · 습윤 기준 {wet}mm")
    print(f"검증 양성률(강수) {y_wet_va.mean()*100:.2f}%\n")

    # ── 기온 ───────────────────────────────────────────────────
    persist_va = _destd(x_va, mean, std, I_TEMP)
    rows = [("퍼시스턴스(현행 기준선)", mae(persist_va, t_va))]
    rows.append(("기후값(관측소×월×시각)",
                 mae(climatology(stn_tr, ts_tr, t_tr, stn_va, ts_va), t_va)))
    if c_tp is not None:
        nwp_t_va = _destd(x_va, mean, std, c_tp)
        rows.append(("**원시 수치예보 예보기온**", mae(nwp_t_va, t_va)))
        mos_cols = [I_TEMP, c_tp, I_HSIN, I_HCOS, I_YSIN, I_YCOS]
    else:
        mos_cols = [I_TEMP, I_HSIN, I_HCOS, I_YSIN, I_YCOS]
    rows.append(("MOS(선형, 예보+퍼시스턴스+시각+관측소)",
                 mae(mos(x_tr, t_tr, x_va, stn_tr, stn_va, mos_cols, levels), t_va)))
    print("GBM 기온 적합 중...", flush=True)
    gbm_t = gbm_regress(x_tr, t_tr, x_va, args.gbm_iter)
    rows.append((f"GBM(Z축 {nf}특징, HistGB {args.gbm_iter}회)", mae(gbm_t, t_va)))
    xi_tr = np.concatenate([x_tr, im_tr], axis=1)
    xi_va = np.concatenate([x_va, im_va], axis=1)
    rows.append((f"GBM(Z축+Im축 {xi_va.shape[1]}특징)",
                 mae(gbm_regress(xi_tr, t_tr, xi_va, args.gbm_iter), t_va)))
    rows.append(("**Tri-CHEF(배포본, 3축)**", mae(ev["temp_pred"], t_va)))

    print(f"\n{'기온 — 평균절대오차(MAE, °C)':<44}{'MAE':>9}{'퍼시스턴스 대비':>16}")
    print("-" * 70)
    base = rows[0][1]
    for name, v in rows:
        print(f"{name:<44}{v:>9.4f}{(base - v) / base * 100:>15.1f}%")

    # ── 강수 ───────────────────────────────────────────────────
    print(f"\nGBM 강수 적합 중...", flush=True)
    gbm_amt = np.clip(gbm_regress(x_tr, p_tr, x_va, args.gbm_iter), 0, None)
    gbm_occ = gbm_classify(x_tr, y_wet_tr, x_va, args.gbm_iter)
    gbm_amt_im = np.clip(gbm_regress(xi_tr, p_tr, xi_va, args.gbm_iter), 0, None)
    gbm_occ_im = gbm_classify(xi_tr, y_wet_tr, xi_va, args.gbm_iter)

    prows = [("상시 0 예측(현행 기준선)", np.zeros_like(p_va), None)]
    if c_pr is not None:
        nwp_p_va = np.clip(_destd(x_va, mean, std, c_pr), 0, None)
        prows.append(("**원시 수치예보 예보강수**", nwp_p_va, nwp_p_va))
    prows.append((f"GBM(Z축 {nf}특징)", gbm_amt, gbm_occ))
    prows.append((f"GBM(Z축+Im축 {nf + im_va.shape[1]}특징)", gbm_amt_im, gbm_occ_im))
    prows.append(("**Tri-CHEF(배포본, 원본 출력)**", ev["precip_pred"],
                  ev["rain_prob"]))

    print(f"\n{'강수':<40}{'MAE(mm)':>10}{'강수구간MAE':>12}"
          f"{'발생F1':>9}{'(상한)':>9}")
    print("-" * 82)
    for name, amt, score in prows:
        if score is None:
            f1s, f1m = "—", "—"
        else:
            f1s = f"{honest_f1(score, y_wet_va, half):.4f}"
            f1m = f"{max_f1(score, y_wet_va)[0]:.4f}"
        print(f"{name:<40}{mae(amt, p_va):>10.4f}"
              f"{conditional_mae(amt, p_va, y_wet_va):>12.4f}{f1s:>9}{f1m:>9}")
    print("\n※ 발생 F1 — 검증셋 절반에서 판정선을 고르고 나머지 절반에서만 채점한다"
          "(모든 방법 동일). 괄호는 전체에서 최댓값을 취한 낙관적 상한이다.")
    print("※ 강수구간 MAE 는 실측 강수 ≥ 습윤기준 표본에서만 잰 값이다"
          "(이 저장소의 강수 주 지표).")

    # ── 극한기상 3종 ───────────────────────────────────────────
    print("\nGBM 극한기상 적합 중...", flush=True)
    print(f"\n{'극한기상 — F1(보정/평가 분리)':<40}{'GBM(Z축)':>12}"
          f"{'Tri-CHEF':>12}{'표본':>10}")
    print("-" * 76)
    for lab, mkey, pkey in (("heatwave", "heat_mask", "heat_prob"),
                            ("coldwave", "cold_mask", "cold_prob"),
                            ("dust", "dust_mask", "dust_prob")):
        # 학습은 모델이 본 것과 같은 마스크·라벨로(비운영기간 채움 포함),
        # 채점은 공식 라벨 표본으로만 — 모델 지표와 같은 조건이다.
        mtr = feat[f"{mkey}_train"].astype(bool)
        off_key = f"{mkey.split('_')[0]}_mask_official_val"
        mva = (feat[off_key] if off_key in feat
               else feat[f"{mkey}_val"]).astype(bool)
        ytr = feat[f"y_{lab}_train"][mtr]
        yva = feat[f"y_{lab}_val"][mva]
        if ytr.sum() < 10 or yva.sum() < 10:
            print(f"{lab:<40}{'표본 부족':>12}")
            continue
        g = gbm_classify(x_tr[mtr], ytr.astype(int), x_va[mva], args.gbm_iter)
        h = half[mva]
        print(f"{lab:<40}{honest_f1(g, yva, h):>12.4f}"
              f"{honest_f1(ev[pkey][mva], yva, h):>12.4f}{int(mva.sum()):>10,}")

    # ── 판정 ───────────────────────────────────────────────────
    print()
    if c_tp is not None:
        m_model, m_nwp = mae(ev["temp_pred"], t_va), mae(nwp_t_va, t_va)
        print(f"[판정 1] 기온 — 원시 수치예보 {m_nwp:.4f} vs Tri-CHEF {m_model:.4f} "
              f"→ 모델 순이득 {m_nwp - m_model:+.4f}°C "
              f"({(m_nwp - m_model) / m_nwp * 100:+.1f}%)")
    m_gbm = mae(gbm_t, t_va)
    m_model = mae(ev["temp_pred"], t_va)
    print(f"[판정 2] 기온 — GBM(Z축만) {m_gbm:.4f} vs Tri-CHEF {m_model:.4f} "
          f"→ 격차 {m_model - m_gbm:+.4f}°C "
          f"({(m_model - m_gbm) / m_gbm * 100:+.1f}%)")
    print("         GBM 은 Re축(공간 격자)·Im축(경향)을 **보지 않는다** — "
          "Z축 단독 표형 기준선이다.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
