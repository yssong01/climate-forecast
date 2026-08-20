# refresh-data 트리거 워커

`refresh-data.yml`(15분 주기 데이터 갱신)을 정시에 깨우기 위한 Cloudflare Worker다.
GitHub Actions 의 `schedule:` 이벤트가 15분 같은 고빈도 주기를 지키지 못한다는
것이 실측으로 확인돼(`.github/workflows/refresh-data.yml` 상단 주석 참고) 대신
이 워커가 GitHub REST API 의 `workflow_dispatch` 를 15분마다 호출한다.

이 워커는 관측 데이터나 KMA API 키를 다루지 않는다 — 워크플로 실행을
예약하는 것이 유일한 역할이다.

## ⚠️ 시크릿은 `npx` 를 거치지 말고 바이너리를 직접 호출해 등록할 것

**`... | npx wrangler secret put KEY` 는 값이 전달되지 않은 채 빈 문자열을
등록한다(2026-08-20 대조 실험).** `npx` 가 파이프로 들어온 표준입력을
가로채기 때문이다. CLI 는 그래도 `✨ Success! Uploaded secret KEY` 를
출력하므로 겉으로는 성공과 구분되지 않는다. 그 결과 Worker 코드에서
`env.GITHUB_PAT` 이 빈 문자열이 되고 GitHub 이 `401 Bad credentials` 로
거부한다.

동일한 값을 두 경로로 등록해 비교한 결과가 근거다:

| 등록 방식 | 런타임에 바인딩된 길이 |
|---|---|
| `echo -n "abcdefghij" \| npx wrangler secret put TEST_NPX` | **0** |
| `echo -n "abcdefghij" \| ./node_modules/.bin/wrangler secret put TEST_DIRECT` | **10** |

따라서 시크릿 등록은 반드시 이렇게 한다:

```bash
... | ./node_modules/.bin/wrangler secret put GITHUB_PAT
```

**등록 후에는 "Success" 출력을 믿지 말고 런타임에서 길이를 직접 확인한다.**
Worker 에 임시로 아래 같은 경로를 두고 `curl` 로 확인한 뒤 지우는 것이
확실하다(값은 절대 반환하지 말고 길이만):

```js
if (new URL(request.url).pathname === "/__probe") {
  return new Response(`len=${(env.GITHUB_PAT || "").length}\n`);
}
```

> **정정 이력(2026-08-20).** 이 문제를 처음 만났을 때는 원인을 "wrangler 4.x
> 의 결함"으로 적었으나 **틀렸다.** 이후 같은 값을 `npx` 경유와 바이너리
> 직접 호출로 나눠 등록해보니, 실패한 시도는 전부 `npx` 경유였고 성공한
> 시도는 전부 직접 호출이었다 — wrangler 버전은 교란 변수였다. 실패 사례가
> 우연히 모두 `npx --yes`(최신 4.x)와 겹쳤던 탓에 버전 차이로 오인했다.
> 교훈: 두 조건이 함께 바뀐 관찰에서 한쪽만 원인으로 지목하지 말고, 조건을
> 하나씩만 바꿔 대조할 것.

`package.json` 에 wrangler 3.x 를 고정해 둔 것은 유지한다(재현성을 위해
버전을 못 박아두는 것 자체는 유효하다). 다만 그것이 이 문제의 해결책은
아니다.

## 최초 설정 (사람이 직접 해야 하는 부분)

1. **Cloudflare 무료 계정 생성** — https://dash.cloudflare.com/sign-up
2. **GitHub fine-grained PAT 발급** — GitHub → Settings → Developer settings →
   Personal access tokens → Fine-grained tokens → Generate new token
   - Repository access: **Only select repositories** → `yssong01/climate-forecast` 하나만
   - Permissions → Actions: **Read and write** (다른 권한은 전부 No access로 둔다 —
     특히 Contents·Secrets 는 절대 켜지 않는다)
   - 만료 기한은 1년을 권장한다(현행 토큰은 2027-08-20 만료). 무기한도
     가능하나, 만료가 남아 있으면 "이 토큰이 아직 필요한가"를 되돌아보는
     점검 지점이 생긴다.
3. **패키지 설치 + 로그인** (이 디렉터리에서):
   ```bash
   npm install                            # package.json 에 고정된 wrangler 설치
   ./node_modules/.bin/wrangler login     # 브라우저로 Cloudflare 계정 인증
   ```
4. **시크릿 등록** — 대화창이나 셸 히스토리에 토큰이 남지 않도록 파일로
   준비해 파이프로 넣는다. `.dev.vars` 에 `GITHUB_PAT=...` 한 줄로 저장한 뒤:
   ```bash
   grep '^GITHUB_PAT=' .dev.vars | cut -d= -f2- | tr -d '\r\n' \
     | ./node_modules/.bin/wrangler secret put GITHUB_PAT
   ```
   **`npx` 를 끼워 넣지 말 것** — 값이 빈 문자열로 등록된다(위 경고 참고).
   등록 후 `.dev.vars` 는 지운다(`.gitignore` 가 막고 있지만 평문으로
   남겨둘 이유가 없다).
5. **배포**:
   ```bash
   ./node_modules/.bin/wrangler deploy
   ```
6. **확인** — 15분 안에 GitHub Actions의 `refresh-data.yml` 실행 목록에
   새 `workflow_dispatch` 실행이 자동으로(사람 개입 없이) 뜨는지 대조한다:
   ```bash
   curl -s "https://api.github.com/repos/yssong01/climate-forecast/actions/workflows/refresh-data.yml/runs?per_page=5"
   ```
   Cloudflare 대시보드의 Worker → 설정 → 트리거 섹션에 Cron 목록이 "없음"
   으로 표시될 수 있는데, 이건 대시보드 UI 표시 지연/버그로 확인됐다
   (2026-08-20) — 실제 동작 여부는 위 GitHub Actions 실행 목록으로만
   판단할 것.

## 유지보수

- PAT 만료(2027-08-20)가 다가오면 4번을 반복해 재등록한다. 재배포는 필요
  없고(시크릿만 갱신하면 되며, 재배포해도 재등록 없이 유지되는 것을
  확인했다), **등록 후 런타임 길이 확인은 반드시 할 것** — "Success"
  출력만 믿었다가 빈 값이 올라간 사례가 실제로 있었다.
- 새 토큰이 동작하는 것을 확인하기 전에는 이전 토큰을 폐기하지 않는다.
  순서를 지켜야 갱신이 끊기지 않는다.
- 이 워커가 죽어도 `refresh-data.yml` 의 `schedule:`(55분 안전망)이 대신
  돌아가므로 데이터 갱신이 완전히 멈추지는 않는다 — 다만 최신성이
  15분에서 최대 55분+지연으로 떨어진다.
