# CLAUDE.md

Instructions for Claude Code (or any agent) working in this repo.

## What this project is

A RAG pipeline over the FastAPI docs + GitHub Discussions, with a rigorous,
judge-free retrieval eval as the primary metric and RAGAS-based generation
scoring on top. Currently mid-way through a phased rebuild driven by
`docs/plan.md` (see below) after auditing the original scaffold and finding
the published RAGAS numbers reflected three measurement bugs, not retrieval
quality: a 30-page ingestion cap that silently excluded `tutorial/`, an eval
set with no gold document for 2/3 of its rows, and a non-reproducible corpus
snapshot.

## Where the plan lives

The full phased plan (11 phases, cross-cutting design decisions for config
splitting, gold-label handling, run artifacts, metrics) is the source of
truth for ongoing work. If `docs/plan.md` doesn't exist yet, ask before
assuming a phase is done — check `runs/`, `configs/`, and git log instead of
trusting a stale summary.

## Commands

```bash
make setup   # uv sync --extra dev
make lint    # ruff check
make type    # mypy src
make test    # pytest -m "not slow and not llm"
make index   # build the Chroma index
make eval    # run the RAGAS eval
make serve   # uvicorn --reload
make demo    # index + serve
```

Pytest markers: `slow`, `integration`, `llm`. CI runs `-m "not slow and not llm"`.

## Ground rules

- **Config discipline**: once `RunConfig` (yaml, `configs/*.yaml`) exists,
  anything that can change a metric belongs there, not in `Settings` (env).
  `RunConfig` uses `extra="forbid"` — never silently swallow a typo'd key.
- **Gold labels are keyed by URL, not chunk id.** Chunk ids are derived from
  content hashes and shift whenever the chunker changes.
- **Self-retrieval leakage matters.** When an eval item's gold answer came
  from the same document being indexed, exclude that chunk from retrieval
  (`deny_ids`) before trusting a recall number — see `eval.self_retrieval`
  in the run config once it lands.
- **Report the honest number next to the flattering one.** This project's
  differentiation is measured self-criticism (label error rates with CIs,
  synthetic-vs-real eval-set gaps, rejected ideas with why) — don't quietly
  drop caveats to make a metric look cleaner.
- **Index the corpus exactly once per embedding model change.** Re-indexing
  is expensive and namespaced collections (`<source>__<embedder-slug>`)
  exist so old and new embeddings can coexist for an A/B, not to be
  reindexed casually.
- **Bootstrap CIs on every reported metric**, not point estimates — sample
  sizes here (n=27, n=127) are small enough that a 3-point delta can be
  noise.
- Prefer constructor injection over `unittest.mock.patch` for new retrieval
  code (`RetrievalPipeline(dense=FakeDense(), ...)`) so tests can assert a
  disabled stage was never called.
- Don't reach for `sentence-transformers` for reranking — it pulls torch and
  roughly triples the Docker image size, breaking the single-container
  deploy target. Use `fastembed`'s ONNX cross-encoder.

## Known follow-ups (see plan for detail)

- `tests/unit/test_retriever.py`'s 5 tests are written against the naive
  dense-only merge in `rag/retriever.py` and are expected to break once
  `retrieval/pipeline.py` replaces it — port them to use injected fakes,
  keep the url→source_id and empty-url assertions.
- `data/raw/` (gitignored) needs to move to `data/corpus/` (committed) before
  a container build has anything to index from.
