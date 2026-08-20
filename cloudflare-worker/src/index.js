/**
 * refresh-data.yml 를 15분마다 워크플로 디스패치로 깨우는 트리거.
 *
 * 왜 필요한가: GitHub Actions 의 `schedule:` 이벤트는 15분 같은 고빈도
 * 주기를 지키지 못한다(GitHub 공식 문서에도 "고부하 시간대에는 지연될 수
 * 있다"고 명시돼 있고, 실측(2026-08-20)으로도 15분 주기 설정 후 실제
 * 실행 간격이 20~81분이었다). Cloudflare Cron Trigger 는 GitHub Actions
 * 러너 큐와 무관하게 동작해 훨씬 정시성이 높다.
 *
 * 이 워커는 관측 데이터를 직접 다루지 않는다 — GitHub REST API 의
 * workflow_dispatch 엔드포인트를 호출해 refresh-data.yml 실행을 예약할
 * 뿐이다. 필요한 토큰(GITHUB_PAT)도 이 저장소 하나에 "Actions: write"
 * 권한만 준 fine-grained PAT 이라, 유출돼도 워크플로 실행 남발 이상의
 * 피해가 없다(코드·시크릿 읽기 권한 없음).
 */

const OWNER = "yssong01";
const REPO = "climate-forecast";
const WORKFLOW_FILE = "refresh-data.yml";
const REF = "main";

async function dispatchWorkflow(env) {
  const url = `https://api.github.com/repos/${OWNER}/${REPO}/actions/workflows/${WORKFLOW_FILE}/dispatches`;
  const res = await fetch(url, {
    method: "POST",
    headers: {
      "Authorization": `Bearer ${env.GITHUB_PAT}`,
      "Accept": "application/vnd.github+json",
      "X-GitHub-Api-Version": "2022-11-28",
      "User-Agent": "climate-forecast-refresh-trigger",
      "Content-Type": "application/json",
    },
    body: JSON.stringify({ ref: REF }),
  });

  if (!res.ok) {
    const body = await res.text();
    throw new Error(`workflow dispatch failed: ${res.status} ${body}`);
  }
  return res.status;
}

export default {
  async scheduled(event, env, ctx) {
    ctx.waitUntil(
      dispatchWorkflow(env)
        .then((status) => console.log(`dispatch ok: ${status}`))
        .catch((err) => console.error(err.message))
    );
  },

  // 브라우저로 접속했을 때 상태만 보여준다 — 이 경로는 트리거를 발동하지
  // 않는다(외부에서 아무나 호출해 워크플로를 남발시키지 못하게).
  async fetch(request) {
    return new Response(
      "climate-forecast refresh trigger worker — Cron Trigger 로만 동작한다.\n",
      { status: 200 }
    );
  },
};
