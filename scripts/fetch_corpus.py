"""
Fetch a pinned, byte-reproducible snapshot of the FastAPI docs corpus.

Downloads the fastapi/fastapi repo at a fixed commit SHA -- one tarball
request, no local git clone needed -- and copies `docs/en/docs/**/*.md` +
`docs_src/**/*.py` into data/corpus/, then writes SNAPSHOT.json recording
the commit and a hash of the resulting chunk id set. Without a pinned
snapshot, the index and the eval set can silently be built from different
days' docs (docs/plan.md problem 3); SNAPSHOT.json makes that detectable
in CI instead (test_corpus_snapshot.py, Phase 5's eval gate).

Usage:
    uv run python scripts/fetch_corpus.py                     # pin current master
    uv run python scripts/fetch_corpus.py --sha <commit-sha>  # pin an exact commit
    uv run python scripts/fetch_corpus.py --out data/corpus   # (default) output dir
"""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import shutil
import tarfile
from datetime import UTC, datetime
from pathlib import Path

import httpx

from rag_eval.common.config import settings

REPO = "fastapi/fastapi"
API_BASE = f"https://api.github.com/repos/{REPO}"
DOCS_SUBPATH = "docs/en/docs"
DOCS_SRC_SUBPATH = "docs_src"
CORPUS_DIR = Path("data/corpus")


def _headers() -> dict[str, str]:
    return {"Authorization": f"Bearer {settings.github_token}"} if settings.github_token else {}


def resolve_sha(ref: str = "master") -> str:
    """Resolve a branch/tag name to its current commit SHA."""
    resp = httpx.get(f"{API_BASE}/commits/{ref}", headers=_headers(), timeout=30.0)
    resp.raise_for_status()
    return resp.json()["sha"]


def download_tarball(sha: str) -> bytes:
    url = f"https://codeload.github.com/{REPO}/tar.gz/{sha}"
    resp = httpx.get(url, timeout=120.0, follow_redirects=True)
    resp.raise_for_status()
    return resp.content


def extract_corpus(tar_bytes: bytes, sha: str, out_dir: Path = CORPUS_DIR) -> tuple[int, int]:
    """Extract only docs/en/docs/**/*.md and docs_src/**/*.py from the
    tarball into out_dir/docs and out_dir/docs_src, replacing whatever was
    there before. Returns (n_pages, n_snippets)."""
    docs_out = out_dir / "docs"
    docs_src_out = out_dir / "docs_src"
    shutil.rmtree(docs_out, ignore_errors=True)
    shutil.rmtree(docs_src_out, ignore_errors=True)
    docs_out.mkdir(parents=True)
    docs_src_out.mkdir(parents=True)

    root_prefix = f"fastapi-{sha}/"
    n_pages = 0
    n_snippets = 0
    with tarfile.open(fileobj=io.BytesIO(tar_bytes), mode="r:gz") as tar:
        for member in tar:
            if not member.isfile():
                continue
            rel = member.name.removeprefix(root_prefix)
            if rel.startswith(f"{DOCS_SUBPATH}/") and rel.endswith(".md"):
                dest = docs_out / Path(rel).relative_to(DOCS_SUBPATH)
                n_pages += 1
            elif rel.startswith(f"{DOCS_SRC_SUBPATH}/") and rel.endswith(".py"):
                dest = docs_src_out / Path(rel).relative_to(DOCS_SRC_SUBPATH)
                n_snippets += 1
            else:
                continue
            dest.parent.mkdir(parents=True, exist_ok=True)
            extracted = tar.extractfile(member)
            assert extracted is not None
            dest.write_bytes(extracted.read())

    return n_pages, n_snippets


def _chunk_stats(docs_dir: Path, docs_src_dir: Path) -> tuple[int, str]:
    """Chunk the freshly-written corpus and hash the resulting sorted chunk
    id set, so a later re-run (or a chunker change) that silently shifts
    the corpus is detectable by comparing against SNAPSHOT.json."""
    from rag_eval.ingestion.docs_chunker import load_doc_chunks

    chunk_ids = sorted(c["id"] for c in load_doc_chunks(docs_dir, docs_src_dir))
    chunk_count_hash = hashlib.sha256("\n".join(chunk_ids).encode()).hexdigest()[:16]
    return len(chunk_ids), chunk_count_hash


def fetch_corpus(ref: str = "master", sha: str | None = None, out_dir: Path = CORPUS_DIR) -> dict:
    resolved_sha = sha or resolve_sha(ref)
    tar_bytes = download_tarball(resolved_sha)
    n_pages, n_snippets = extract_corpus(tar_bytes, resolved_sha, out_dir)
    n_chunks_expected, chunk_count_hash = _chunk_stats(out_dir / "docs", out_dir / "docs_src")

    snapshot = {
        "fastapi_sha": resolved_sha,
        "fetched_at": datetime.now(UTC).isoformat(),
        "n_pages": n_pages,
        "n_snippets": n_snippets,
        "n_chunks_expected": n_chunks_expected,
        "chunk_count_hash": chunk_count_hash,
    }
    (out_dir / "SNAPSHOT.json").write_text(json.dumps(snapshot, indent=2) + "\n", encoding="utf-8")
    return snapshot


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ref", default="master", help="Branch/tag to resolve if --sha is not given")
    parser.add_argument("--sha", default=None, help="Exact commit SHA to pin (skips ref resolution)")
    parser.add_argument("--out", type=Path, default=CORPUS_DIR)
    args = parser.parse_args()

    result = fetch_corpus(ref=args.ref, sha=args.sha, out_dir=args.out)
    print(json.dumps(result, indent=2))
