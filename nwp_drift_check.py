"""
nwp_drift_check.py — 수치예보 `previous_day1` 창의 **미래 유효시각 값이 실시간으로 얼마나
바뀌는지** data 브랜치 커밋 이력으로 잰다(2026-10-22).

## 왜 필요한가

학습은 유효시각이 지난 뒤 굳은 값(아카이브)을 쓰고, 서빙은 예측 시점에 받은 값을 쓴다.
둘이 같아야 학습·서빙이 같은 입력을 본다(CLAUDE.md 4절). 2026-10-22 조사에서 GSM 의
`previous_day1` 은 유효시각까지 5~7h 면 85% 가 같지만 11~13h 면 34%, 17h 이상이면 0% 로,
**미래 값이 잠정값**임이 드러났다 — +12h 모델 입력에 작은 불일치가 있다는 뜻이다. 새 자료원
(`jma_msm`, 측정용 스냅숏 `cache/nwp_msm_recent.json`)을 채택하기 전에 같은 질문에 답한다.

## 재는 방법

data 브랜치에서 그 파일을 바꾼 커밋들을 시간순으로 읽는다. 각 (관측소, 유효시각)의
**최종값**은 유효시각 + 1h 이후에 받은 마지막 스냅숏의 값이다(조사에서 이 최종값이 학습
아카이브와 42,756/42,756 일치함을 확인했다). 스냅숏마다 "유효시각까지 남은 시간" 구간별로
최종값과 같은 비율, 기온·강수의 평균 절대 차를 낸다. 스냅숏 시각은 커밋 시각이다(수집 직후
커밋되므로 수 분 이내 오차).

실행:
    python nwp_drift_check.py                              # GSM(cache/nwp_recent.json)
    python nwp_drift_check.py --file cache/nwp_msm_recent.json
"""
import argparse
import collections
import json
import subprocess
import sys
from datetime import datetime, timedelta, timezone

KST = timezone(timedelta(hours=9))


def snapshots(path, branch, max_n):
    log = subprocess.run(["git", "log", branch, "--format=%H %ct", "--", path],
                         capture_output=True, text=True, check=True).stdout.split("\n")
    commits = [(l.split()[0], int(l.split()[1])) for l in log if l.strip()][::-1]
    step = max(1, len(commits) // max_n)
    out = []
    for h, ct in commits[::step]:
        raw = subprocess.run(["git", "show", f"{h}:{path}"], capture_output=True, text=True).stdout
        try:
            d = json.loads(raw)
        except ValueError:
            continue
        out.append((ct, d.get("rows", d) if isinstance(d, dict) else d))
    return out


def vt(key):
    return datetime.strptime(key, "%Y%m%d%H%M").replace(tzinfo=KST).timestamp()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--file", default="cache/nwp_recent.json")
    ap.add_argument("--branch", default="origin/data")
    ap.add_argument("--max-snapshots", type=int, default=400)
    ap.add_argument("--temp-index", type=int, default=None,
                    help="행 안 기온 위치(기본: GSM 1, MSM 1)")
    args = ap.parse_args()
    ti = 1 if args.temp_index is None else args.temp_index
    snaps = snapshots(args.file, args.branch, args.max_snapshots)
    if len(snaps) < 2:
        print("스냅숏이 부족하다 — 수집이 더 쌓인 뒤 다시 잴 것.")
        return 1
    final = {}
    for ct, rows in snaps:
        for stn, r in rows.items():
            for k, v in r.items():
                if v is not None and ct >= vt(k) + 3600:
                    final[(stn, k)] = v
    same = collections.defaultdict(lambda: [0, 0])
    dT = collections.defaultdict(list)
    dP = collections.defaultdict(list)
    for ct, rows in snaps:
        for stn, r in rows.items():
            for k, v in r.items():
                f = final.get((stn, k))
                if v is None or f is None:
                    continue
                ahead = (vt(k) - ct) / 3600
                if ahead < 0:
                    continue
                b = int(ahead // 3) * 3
                same[b][1] += 1
                same[b][0] += v == f
                dT[b].append(abs(v[ti] - f[ti]))
                dP[b].append(abs(v[0] - f[0]))
    t0, t1 = datetime.fromtimestamp(snaps[0][0], KST), datetime.fromtimestamp(snaps[-1][0], KST)
    print(f"{args.file} · 스냅숏 {len(snaps)}개 ({t0:%Y-%m-%d %H:%M} ~ {t1:%Y-%m-%d %H:%M} KST) · "
          f"최종값 확정 {len(final):,}개")
    print(f"{'유효시각까지':>12}{'최종과 같음':>12}{'n':>9}{'기온 |Δ| 평균':>14}{'강수 |Δ| 평균':>14}")
    for b in sorted(same):
        n = same[b][1]
        print(f"{f'{b}~{b + 3}h':>12}{same[b][0] / n:>12.3f}{n:>9,}"
              f"{sum(dT[b]) / n:>14.3f}{sum(dP[b]) / n:>14.3f}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
