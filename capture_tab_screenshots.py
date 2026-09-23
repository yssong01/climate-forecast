"""
capture_tab_screenshots.py — 배포 화면 4개 탭을 실제로 렌더링해 `docs/images/tab-*.png`
를 다시 찍는다.

**왜 스크립트로 두는가.** README 와 발표 자료가 이 네 장을 싣고 있어서, 화면
문구나 지표를 고칠 때마다 함께 갱신해야 한다. 그런데 지금까지 이 절차는
사람의 기억에만 있었고(`promote_checkpoint.py` 가 출력하는 "남은 수동 작업"
목록의 4번), 실제로 두 번 연속 빠뜨렸다 — 2026-09-23 하루에만 캡션을 고친 뒤
찍지 않아 한 번, 그 다음 커밋에서 또 한 번. 화면 수치를 체크포인트 조회로
바꿔 전사 고리를 없애는 작업을 하면서 정작 이 고리를 남겨 둘 이유가 없다.

**왜 이렇게 복잡한가 — 겪은 함정들.**

  · **`WebFetch` 로는 안 된다.** Streamlit Community Cloud 가 뷰어 인증
    리다이렉트를 걸어 세션 없는 단발 요청은 콘텐츠를 못 받는다. 그래서 배포
    URL 이 아니라 **같은 코드를 로컬 컨테이너로 띄워** 찍는다.
  · **탭은 역할 선택자로 잡는다**(`[role="tab"]`). 좌표 클릭은 빗나가도
    스크린샷이 정상으로 찍혀서, 같은 화면 넉 장을 얻고도 한참 모른다.
  · **playwright 파이썬 패키지를 이미지 버전에 고정**해야 한다. 그냥 설치하면
    최신이 깔려 이미지의 브라우저 빌드와 안 맞아 즉시 죽는다.
  · **저장 경로는 컨테이너 관점**이어야 한다. 호스트 경로를 쓰면 컨테이너
    안 아무 데나 파일이 생기고 종료 코드는 0 이다.
  · **뷰포트를 크게 잡는다**(5200px). `full_page=True` 는 Streamlit 내부
    스크롤 구조 때문에 뷰포트 높이만큼만 잡힌다.
  · **CPU 추론은 느리다.** 첫 화면은 탭이 나타날 때까지 기다린다(실측
    80초까지 걸렸고, 예측 캐시가 살아 있으면 20초면 된다). 탭 전환
    뒤에는 16초를 기다린다.
  · **적중률 로그가 오염된다.** 로컬로 띄운 앱도 실제로 예측하며
    `cache/accuracy_log.json`(git 추적 대상)에 항목을 추가한다. 끝나고
    되돌린다.

실행: python capture_tab_screenshots.py [--port 8599] [--keep]
      로컬에 docker 가 필요하다(이 스크립트 자체는 컨테이너 밖에서 돈다).
"""
import argparse
import os
import shutil
import subprocess
import sys
import tempfile

APP_IMAGE = "tri-chef-app:latest"
PLAYWRIGHT_IMAGE = "mcr.microsoft.com/playwright/python:v1.48.0-jammy"
PLAYWRIGHT_PIN = "playwright==1.48.0"
OUT_DIR = "./docs/images"
TABS = ["tab-trend", "tab-extreme", "tab-performance", "tab-model"]
ACCURACY_LOG = "./cache/accuracy_log.json"

SHOT_SCRIPT = '''
import time
from playwright.sync_api import sync_playwright

NAMES = {names!r}
with sync_playwright() as p:
    b = p.chromium.launch(args=["--no-sandbox"])
    # 다크 모드 — 기존 네 장과 같은 조건이어야 문서에서 섞이지 않는다.
    # 앱은 테마를 고정하지 않으므로 브라우저 설정을 그대로 따른다.
    pg = b.new_page(viewport={{"width": 1500, "height": 5200}},
                    color_scheme="dark")
    pg.goto("http://localhost:{port}", wait_until="networkidle", timeout=120000)
    # **고정 대기로는 안 된다.** 첫 실행은 관측소 12곳 × 리드타임 2개를
    # 실제로 조회하며 CPU 추론까지 해서 탭이 뜨기까지 80초가 걸렸다(실측).
    # 같은 정시 안에 다시 돌리면 예측 캐시가 살아 있어 20초면 끝난다 —
    # 편차가 4배라 넉넉한 상수를 잡는 것보다 **나타날 때까지 기다리는** 편이
    # 맞다. 고정 45초를 쓰던 판은 캐시가 식은 뒤 실제로 빈 화면을 찍었다.
    pg.wait_for_selector('[role="tab"]', timeout=240000)
    time.sleep(8)          # 탭 줄이 뜬 뒤 본문이 자리를 잡을 여유
    tabs = pg.locator('[role="tab"]')
    n = tabs.count()
    if n != len(NAMES):
        raise SystemExit(f"탭이 {{n}}개다(기대 {{len(NAMES)}}개) — 화면 구성이 바뀌었는지 확인할 것")
    for i, name in enumerate(NAMES):
        tabs.nth(i).click()
        time.sleep(16)
        pg.screenshot(path=f"/out/{{name}}.png")
        print("찍음:", name, "—", tabs.nth(i).inner_text().strip())
    b.close()
'''


def run(cmd, **kw):
    return subprocess.run(cmd, shell=isinstance(cmd, str), **kw)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", type=int, default=8599)
    ap.add_argument("--keep", action="store_true",
                    help="앱 컨테이너를 끝나고 지우지 않는다(디버깅용)")
    args = ap.parse_args()

    if shutil.which("docker") is None:
        print("docker 를 찾을 수 없다 — Docker Desktop 의 WSL 통합을 확인할 것.")
        return 1
    if not os.path.exists(".env"):
        print(".env 가 없다 — 앱이 관측을 조회하지 못해 화면이 폴백으로 찍힌다.")
        return 1

    cwd = os.path.abspath(".")
    tmp = tempfile.mkdtemp(prefix="tabshots_")
    os.chmod(tmp, 0o777)
    with open(os.path.join(tmp, "shot.py"), "w", encoding="utf-8") as f:
        f.write(SHOT_SCRIPT.format(names=TABS, port=args.port))

    # 적중률 로그는 로컬 실행이 오염시키므로 미리 백업해 둔다.
    backup = None
    if os.path.exists(ACCURACY_LOG):
        backup = os.path.join(tmp, "accuracy_log.json")
        shutil.copy2(ACCURACY_LOG, backup)

    name = "tabshot_app"
    run(f"docker rm -f {name}", stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    print(f"앱 컨테이너 기동({APP_IMAGE}, 포트 {args.port})…")
    up = run(["docker", "run", "-d", "--name", name,
              "-p", f"{args.port}:8501", "-v", f"{cwd}:/app",
              "--env-file", ".env", APP_IMAGE],
             stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
    if up.returncode != 0:
        print("앱 컨테이너 기동 실패:", up.stderr.decode()[:300])
        return 1

    try:
        import time as _t
        _t.sleep(30)
        print("렌더링·촬영…(CPU 추론이라 2분 안팎 걸린다)")
        shot = run(["docker", "run", "--rm", "--network", "host",
                    "-v", f"{tmp}:/work", "-v", f"{os.path.abspath(OUT_DIR)}:/out",
                    "-w", "/work", PLAYWRIGHT_IMAGE,
                    "bash", "-c",
                    f"pip install -q '{PLAYWRIGHT_PIN}' >/dev/null 2>&1 && python shot.py"])
        if shot.returncode != 0:
            print("촬영 실패 — 위 메시지를 확인할 것.")
            return 1
    finally:
        if not args.keep:
            run(f"docker rm -f {name}", stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL)
        # 오염된 적중률 로그를 되돌린다. 조용히 넘기지 않고 바뀌었는지 밝힌다.
        if backup and os.path.exists(ACCURACY_LOG):
            if open(backup, "rb").read() != open(ACCURACY_LOG, "rb").read():
                shutil.copy2(backup, ACCURACY_LOG)
                print(f"복원: {ACCURACY_LOG} (로컬 실행이 항목을 추가했다)")

    missing = [t for t in TABS if not os.path.exists(f"{OUT_DIR}/{t}.png")]
    if missing:
        print("생성되지 않은 파일:", ", ".join(missing))
        return 1
    for t in TABS:
        p = f"{OUT_DIR}/{t}.png"
        os.chmod(p, 0o644)
        print(f"  {p}  {os.path.getsize(p):,} bytes")
    print("\n완료 — `git status` 로 네 장이 바뀌었는지 확인하고 커밋할 것.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
