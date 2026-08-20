# refresh-data 트리거 워커

`refresh-data.yml`(15분 주기 데이터 갱신)을 정시에 깨우기 위한 Cloudflare Worker다.
GitHub Actions 의 `schedule:` 이벤트가 15분 같은 고빈도 주기를 지키지 못한다는
것이 실측으로 확인돼(`.github/workflows/refresh-data.yml` 상단 주석 참고) 대신
이 워커가 GitHub REST API 의 `workflow_dispatch` 를 15분마다 호출한다.

이 워커는 관측 데이터나 KMA API 키를 다루지 않는다 — 워크플로 실행을
예약하는 것이 유일한 역할이다.

## ⚠️ wrangler 버전 — 반드시 3.x, `npx --yes wrangler`로 받은 4.x 쓰지 말 것

**`npx --yes wrangler ...`(버전 미고정, 매번 최신 4.x를 내려받음)로 등록한
시크릿은 이 계정에서 실제 서빙되는 런타임에 연결되지 않는다(2026-08-20
실측).** `wrangler secret put`이든 `wrangler versions secret put`이든
CLI는 매번 "Success"를 출력하고 `wrangler versions view`에도 시크릿이
목록에 뜨지만, 실제 Worker 코드에서 `env.GITHUB_PAT`을 읽으면 빈 문자열이다
— GitHub API가 401 Bad credentials로 거부하고, 원인 추적을 위해 값 대신
길이·SHA256 지문만 로그로 남기는 디버그 코드로 확인했다(길이 0, 빈
문자열의 해시값과 일치). `wrangler deploy`·`wrangler versions deploy`
순서를 바꿔봐도, 몇 분을 기다려봐도, 프로덕션 트래픽에 100%로 명시
승격해도 재현됐다.

**해결책은 wrangler 3.x(클래식 배포 모델)를 이 디렉터리에 로컬로 고정
설치해 쓰는 것이다** — `package.json`에 이미 `"wrangler": "^3.90.0"`로
박아뒀다. `npm install`을 한 번 실행해두면, 이후 `npx wrangler <명령>`은
(네트워크에서 최신을 받아오는 대신) 로컬 `node_modules/.bin/wrangler`를
우선 사용해 자동으로 3.x로 동작한다 — 별도 옵션이나 경로 지정이 필요
없다. 3.x로 등록한 시크릿은 재등록 없이 재배포해도 그대로 유지되는 것도
확인했다(2026-08-20). `package.json`의 `wrangler` 버전 범위를 실수로
`^4`나 버전 미지정으로 바꾸지 말 것 — 이 문제가 재발한다.

## 최초 설정 (사람이 직접 해야 하는 부분)

1. **Cloudflare 무료 계정 생성** — https://dash.cloudflare.com/sign-up
2. **GitHub fine-grained PAT 발급** — GitHub → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token
   - Repository access: **Only select repositories** → `yssong01/climate-forecast` 하나만
   - Permissions → Actions: **Read and write** (다른 권한은 전부 No access로 둔다 —
     특히 Contents·Secrets 는 절대 켜지 않는다)
   - 만료 기한을 짧게 잡고(예: 90일) 만료 전 재발급하는 쪽을 권장한다.
3. **패키지 설치(wrangler 3.x 로컬 고정) + 로그인 + 시크릿 등록** (이 디렉터리에서):
   ```bash
   npm install                 # package.json 에 고정된 wrangler 3.x 설치
   npx wrangler login          # 브라우저로 Cloudflare 계정 인증
   npx wrangler secret put GITHUB_PAT   # 위에서 발급한 PAT 붙여넣기 (평문 파일에 저장 금지)
   ```
   `npm install`을 먼저 해야 `npx wrangler`가 4.x 대신 로컬 3.x를 쓴다(위
   경고 참고). 토큰을 파일로 준비해서 파이프로 넣고 싶다면(대화창에
   직접 붙여넣지 않기 위해) `.dev.vars`에 `GITHUB_PAT=...` 한 줄로 저장한
   뒤 `grep '^GITHUB_PAT=' .dev.vars | cut -d= -f2- | tr -d '\r\n' | npx wrangler secret put GITHUB_PAT`
   로 등록하고, 등록 후 그 파일은 지운다(`.gitignore`가 이미 막고 있지만
   평문으로 남겨둘 이유가 없다).
4. **배포**:
   ```bash
   npx wrangler deploy
   ```
5. **확인** — 15분 안에 GitHub Actions의 `refresh-data.yml` 실행 목록에
   새 `workflow_dispatch` 실행이 자동으로(사람 개입 없이) 뜨는지 대조한다:
   ```bash
   curl -s "https://api.github.com/repos/yssong01/climate-forecast/actions/workflows/refresh-data.yml/runs?per_page=5"
   ```
   Cloudflare 대시보드의 Worker → 설정 → 트리거 섹션에 Cron 목록이 "없음"
   으로 표시될 수 있는데, 이건 대시보드 UI 표시 지연/버그로 확인됐다
   (2026-08-20) — 실제 동작 여부는 위 GitHub Actions 실행 목록으로만
   판단할 것.

## 유지보수

- PAT 만료가 다가오면 3번을 반복해 재등록한다(재배포는 필요 없다 —
  시크릿만 갱신하면 되고, wrangler 3.x에서는 재배포해도 재등록 없이
  시크릿이 유지되는 것도 확인됐다).
- 이 워커가 죽어도 `refresh-data.yml` 의 `schedule:`(55분 안전망)이 대신
  돌아가므로 데이터 갱신이 완전히 멈추지는 않는다 — 다만 최신성이
  15분에서 최대 55분+지연으로 떨어진다.
