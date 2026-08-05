"""
Pull Q&A pairs from GitHub Discussions using the GitHub GraphQL API.

Targets discussions that have an accepted answer (answerChosenAt is set),
which gives clean (question, ground_truth_answer) pairs to seed the eval set.

Usage:
    uv run python -m rag_eval.ingestion.github_discussions
"""

import time

import httpx
from tqdm import tqdm

from rag_eval.common.config import settings
from rag_eval.common.schemas import DiscussionQA

GRAPHQL_URL = "https://api.github.com/graphql"

# Retry/backoff tuning: GitHub's GraphQL API returns 5xx and secondary-rate-
# limit 403s often enough under sustained paging that a bare
# `raise_for_status()` turns a transient blip into a failed fetch.
MAX_RETRIES = 5
BACKOFF_BASE_SECONDS = 1.0

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


def _post_with_retry(client: httpx.Client, **kwargs) -> httpx.Response:
    """POST with exponential backoff on transient failures: 5xx, secondary
    rate limits (403), and network-level errors."""
    for attempt in range(MAX_RETRIES):
        try:
            resp = client.post(**kwargs)
            if resp.status_code >= 500 or resp.status_code == 403:
                raise httpx.HTTPStatusError(
                    f"transient status {resp.status_code}", request=resp.request, response=resp
                )
            resp.raise_for_status()
            return resp
        except (httpx.HTTPStatusError, httpx.TransportError):
            if attempt == MAX_RETRIES - 1:
                raise
            time.sleep(BACKOFF_BASE_SECONDS * (2**attempt))
    raise AssertionError("unreachable")  # pragma: no cover -- loop always returns or raises above


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
            resp = _post_with_retry(
                client,
                url=GRAPHQL_URL,
                json={"query": DISCUSSIONS_QUERY, "variables": variables},
                headers=_headers(),
            )
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
