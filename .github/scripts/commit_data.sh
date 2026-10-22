#!/usr/bin/env bash
# data 브랜치에 수집분을 커밋·푸시한다(refresh-data.yml 이 두 번 부른다).
#
#   commit_data.sh "<커밋 메시지>" <파일>...      # `-f:<파일>` 은 git add -f
#
# 2026-10-22 에 워크플로의 커밋 스텝에서 떼어 냈다 — 관측을 수치예보 수집
# **전에** 한 번 커밋하려면 같은 절차가 두 번 필요한데, 아래 rebase 처리는
# 두 번 적으면 한쪽만 고치게 되는 종류다.
set -u
# CI 밖에서는 돌지 않는다 — 개발 저장소에서 잘못 부르면 그 저장소의 HEAD 를
# data 브랜치로 push 하려 든다(2026-10-22 시험 중 실제로 개발 저장소에서
# 한 번 실행됐다. 변경이 없어 커밋 전에 멈췄지만 신원 설정을 덮어썼다).
if [ "${GITHUB_ACTIONS:-}" != "true" ] && [ "${COMMIT_DATA_ALLOW_LOCAL:-}" != "1" ]; then
  echo "::error::commit_data.sh 는 CI 전용이다(시험은 COMMIT_DATA_ALLOW_LOCAL=1)"
  exit 2
fi
# 신원은 저장소 설정에 쓰지 않고 이 커밋에만 준다.
export GIT_AUTHOR_NAME="github-actions[bot]" GIT_COMMITTER_NAME="github-actions[bot]"
export GIT_AUTHOR_EMAIL="41898282+github-actions[bot]@users.noreply.github.com"
export GIT_COMMITTER_EMAIL="$GIT_AUTHOR_EMAIL"

msg="$1"; shift
for f in "$@"; do
  case "$f" in
    # data 브랜치는 **자체 .gitignore 스냅숏**을 갖고 있어 main 의 예외
    # 규칙(`!cache/nwp_recent.json`)이 반영돼 있지 않다. 그래서 평범한
    # `git add` 는 이 파일을 조용히 무시한다 — 2026-09-07 에 실제로
    # 이것 때문에 워크플로가 계속 초록불인데 파일만 안 올라왔다.
    # 이 브랜치는 캐시 파일 운반 전용이므로 -f 로 강제하는 것이 맞고,
    # 실패를 `2>/dev/null` 로 삼키지 않는다(그 침묵이 원인 규명을
    # 늦췄다). 파일이 없으면 경고로 드러낸다.
    -f:*)
      p="${f#-f:}"
      if [ -f "$p" ]; then git add -f "$p"
      else echo "::warning::$p 이 없다 — 그 파일을 만드는 스텝을 확인할 것"; fi ;;
    *)
      if [ -f "$f" ]; then git add "$f"
      else echo "::warning::$f 이 없다"; fi ;;
  esac
done

if git diff --cached --quiet; then
  echo "변경 없음 — 커밋 생략 (이미 최신이거나 신규 수집 0건)"
  exit 0
fi
git commit -q -m "$msg"

# 커밋과 푸시 사이에 다른 실행이 먼저 푸시하면 non-fast-forward 로
# 거부되므로 rebase 로 다시 얹는다.
#
# rebase --abort 가 반드시 먼저 와야 한다(2026-08-19 수정). 이 잡이
# 건드리는 cache/*.json 은 파일 전체가 한 줄로 재생성되므로, 원격이
# 같은 파일을 건드렸으면 충돌은 가능성이 아니라 **필연**이다. 그때
# 러너는 rebase 진행 중 상태로 남고, 뒤이은 재시도는 네트워크에
# 닿지도 못한 채 "Pulling is not possible... unmerged files" 로
# 죽으면서 화면에는 엉뚱하게 "원격이 앞서 있음"만 찍힌다.
#
# 충돌 시 러너가 방금 수집한 쪽을 택한다 — 이 파일들은 관측을 다시
# 받아 통째로 재생성한 결과라 우리 것이 더 최신이다.
#
# 여기서 `-X theirs` 인 것이 맞다. rebase 는 upstream 을 체크아웃한 뒤
# 내 커밋을 그 위에 replay 하므로 ours/theirs 가 뒤집힌다 —
# ours=원격(upstream), theirs=내 커밋. 실측(2026-08-19)으로 확인:
#   -X ours   → 결과 REMOTE      (수집분이 버려진다. 틀림)
#   -X theirs → 결과 LOCAL-fresh (수집분이 남는다. 맞음)
#
# push/pull 대상은 `origin data` 로 명시한다 — actions/checkout 은
# 브랜치 ref 를 detached HEAD 로 체크아웃하므로, 인자 없는
# `git push`/`git pull` 은 추적 브랜치가 없어 실패한다.
for i in 1 2 3; do
  git rebase --abort 2>/dev/null || true
  git pull -q --rebase -X theirs --autostash origin data \
    && git push -q origin HEAD:data && exit 0
  echo "push 재시도 $i/3 (원격이 앞서 있거나 rebase 충돌)"
  sleep 5
done
git rebase --abort 2>/dev/null || true
echo "::error::3회 시도 후에도 push 실패"
exit 1
