"""
Pull Q&A pairs from GitHub Discussions using the GitHub GraphQL API.

Targets discussions that have an accepted answer (answerChosenAt is set),
which gives clean (question, ground_truth_answer) pairs to seed the eval set.

Usage:
    uv run python -m rag_eval.ingestion.github_discussions
"""

import httpx
from tqdm import tqdm

from rag_eval.common.config import settings
from rag_eval.common.schemas import DiscussionQA

GRAPHQL_URL = "https://api.github.com/graphql"

DISCUSSIONS_QUERY = """
query($owner: String!, $name: String!, $cursor: String) {
  repository(owner: $owner, name: $name) {
    discussions(first: 50, after: $cursor, orderBy: {field: CREATED_AT, direction: DESC}) {
      pageInfo { hasNextPage endCursor }
      nodes {
        id
        title
        bodyText
        url
        category { name }
        answer { bodyText }
      }
    }
  }
}
"""


def _headers() -> dict:
    if not settings.github_token:
        raise RuntimeError("GITHUB_TOKEN is not set (see .env.example).")
    return {"Authorization": f"Bearer {settings.github_token}"}


def fetch_discussion_qas(max_pages: int | None = None) -> list[DiscussionQA]:
    """Page through Discussions GraphQL results and keep only answered ones."""
    results: list[DiscussionQA] = []
    cursor = None
    page = 0

    with httpx.Client(timeout=30.0) as client:
        while True:
            variables = {
                "owner": settings.github_repo_owner,
                "name": settings.github_repo_name,
                "cursor": cursor,
            }
            resp = client.post(
                GRAPHQL_URL,
                json={"query": DISCUSSIONS_QUERY, "variables": variables},
                headers=_headers(),
            )
            resp.raise_for_status()
            payload = resp.json()
            if "errors" in payload:
                raise RuntimeError(payload["errors"])

            discussions = payload["data"]["repository"]["discussions"]
            for node in tqdm(discussions["nodes"], desc=f"page {page}"):
                answer = node.get("answer")
                if not answer or not answer.get("bodyText"):
                    continue
                results.append(
                    DiscussionQA(
                        discussion_id=node["id"],
                        title=node["title"],
                        question_body=node["bodyText"],
                        answer_body=answer["bodyText"],
                        url=node["url"],
                        category=(node.get("category") or {}).get("name"),
                    )
                )

            page += 1
            page_info = discussions["pageInfo"]
            if not page_info["hasNextPage"] or (max_pages and page >= max_pages):
                break
            cursor = page_info["endCursor"]

    return results


if __name__ == "__main__":
    qas = fetch_discussion_qas(max_pages=2)
    print(f"Fetched {len(qas)} answered discussions")
