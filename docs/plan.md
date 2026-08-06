# RAG-Eval → deployed, resume-grade project

## Context

`D:\RAG-Eval` currently holds a working but under-measured RAG scaffold: FastAPI docs + GitHub Discussions ingested into Chroma, dense retrieval, Ollama generation, RAGAS scoring with three selectable judges, ~40 unit tests. Its published numbers are bad — faithfulness 0.295, answer_relevancy 0.237, context_recall 0.378.

I verified against the sqlite and re-ran the chunker over the full corpus. **The scores are not a retrieval-algorithm result. They are three bugs:**

1. **The index holds only 30 alphabetically-first docs pages, all under `advanced/`.** `INGEST_DOCS_LIMIT=30` truncates `sorted(rglob("*.md"))` before it ever reaches `tutorial/` — the pages that answer most FastAPI questions. Every `tutorial/` page is absent.
2. **Two thirds of the eval set has no gold document in the corpus.** Chroma holds 9 discussion chunks; the eval set has 27 rows; 8 overlap. `context_recall = 0.378` ≈ the fraction of questions whose answer is physically present (9/27), plus partial credit. No reranker fixes a missing document.
3. **The corpus is non-reproducible.** `orderBy: CREATED_AT DESC` + `max_pages=1` makes "page 1" a sliding window over time. The index and eval set were built on different days from different discussions, so cross-day comparison is meaningless.

Plus a weak CPU generator (`fdm-llama` emitted a 13-character "answer" on one row), self-retrieval leakage (`ragas_results.csv` row 0: a retrieved context is verbatim the reference), and chunks averaging 210 tokens against a documented 400–600 target (only 9.6% land in band; 134 chunks are under 50 tokens).

**Goal:** fix the measurement first, then earn improvements against a trustworthy baseline, then deploy it publicly with a dashboard and a CI gate. The finished artifact should demonstrate RAG architecture, evaluation rigour, and a real deploy pipeline.

### Locked decisions

| | Decision |
|---|---|
| Serving LLM | Groq `llama-3.3-70b-versatile` default; Gemini fallback; Ollama for local dev. Plain `httpx`, not LangChain, in the serving path |
| Embeddings | `fastembed` `BAAI/bge-small-en-v1.5` (ONNX, in-process, 384-dim). Ollama retained as a local backend |
| Deploy | Hugging Face Spaces, Docker SDK, single container, port 7860. **Thin slice ships early (Phase 3), then every phase redeploys via CD** |
| Frontend | Next.js App Router + TS + Tailwind, `output: 'export'`, served by FastAPI `StaticFiles` |
| Retrieval metrics | Judge-free recall@k / MRR / nDCG over a synthetic docs set, gating CI |
| Label verification | Automated filters **plus a human pass**: review ~50 synthetic pairs (y/n/edit) and hand-label gold docs sections for ~16 real discussion questions. Yields a measured label error rate |
| Tracks | Retrieval lab · eval depth + CI gates · answer quality & safety. Telemetry stays minimal (no OpenTelemetry) |
| Dropped | Anthropic provider branch (unused branches read as scaffolding). `configs/experiments/judge-anthropic.yaml` would be the only reason to keep it |

---

## Cross-cutting design

### C1 — Config split: `Settings` (env) vs `RunConfig` (yaml)

**Rule: `Settings` holds only what cannot change a metric. `RunConfig` holds only what can.** Today a metric silently depends on a `.env` file that isn't in git — that is the root of problem 3 above.

`Settings` keeps API keys, `github_token`, `chroma_persist_dir`, `ollama_base_url`, `artifacts_root`, `port`, `log_level`, `default_run_config`. It **loses** `ollama_llm_model`, `ollama_embed_model`, `ingest_docs_limit`, `ingest_discussion_pages`, `ragas_judge`, `groq_model`, `gemini_model`, `ragas_max_workers`, `ragas_timeout`, `eval_sample_limit`, `eval_set_path` — all move to yaml.

`configs/_base.yaml` + one file per experiment:

```yaml
name: baseline-dense
extends: _base.yaml
corpus:    { sources: [docs, discussions], snapshot: data/corpus/SNAPSHOT.json }
embedding: { provider: fastembed, model: BAAI/bge-small-en-v1.5 }
retrieval:
  top_k: 5
  candidates_k: 30
  dense:            { enabled: true }
  bm25:             { enabled: false, k1: 1.2, b: 0.75 }
  fusion:           { method: rrf, rrf_k: 60 }
  rerank:           { enabled: false, model: Xenova/ms-marco-MiniLM-L-6-v2, top_n: 5 }
  query_rewrite:    { enabled: false, mode: hyde, n: 1 }
  parent_expansion: { enabled: false, mode: section, max_tokens: 1200 }
  per_source_caps:  { docs: 4, discussions: 2 }
generation:
  enabled: false
  llm: { provider: groq, model: llama-3.3-70b-versatile, temperature: 0.0, max_tokens: 900 }
  prompt_version: v2-cited
  require_citations: true
  groundedness: { enabled: true, mode: embedding_support, abstain_below: 0.45 }
eval:
  datasets: [docs_synth_v1, discussions_v2]
  k_values: [1, 3, 5, 10]
  self_retrieval: holdout          # none | holdout | separate_index
  judge: { enabled: false, provider: gemini, model: gemini-2.5-flash, max_workers: 1 }
  thresholds:
    recall_at_5: { min: 0.70, regression_tolerance: 0.02 }
    mrr:         { min: 0.55, regression_tolerance: 0.02 }
    ndcg_at_10:  { min: 0.60, regression_tolerance: 0.02 }
```

`src/rag_eval/config/run_config.py`:
```python
class RunConfig(BaseModel):
    model_config = ConfigDict(extra="forbid")   # a typo'd key must fail loudly
def load_run_config(path: Path, overrides: Sequence[str] = ()) -> RunConfig
def config_hash(cfg: RunConfig) -> str                  # sha256 of canonical JSON, [:12]
def split_hashes(cfg: RunConfig) -> tuple[str, str]     # (retrieval_hash, generation_hash)
```
`extra="forbid"` is load-bearing — silently-ignored config keys are how eval harnesses lie. CLI `--set retrieval.top_k=10` values are parsed as YAML scalars (don't fight typer coercion) and recorded in the manifest, so a run is reproducible from its manifest alone.

### C2 — Gold labels, and isolating retrieval from generation

**Gold is keyed on URL, not chunk id.** Chunk id is `sha256(path::breadcrumb::idx)[:16]`; any chunker change invalidates baked ids. `resolve_gold_chunks(item, index)` maps `url → {chunk_ids}` at load and warns when a gold URL stops resolving. A section spanning several chunks makes all of them gold. Anchor-less pages fall back to page-level gold, are flagged `gold_granularity: page`, and are **reported in a separate column** — page-level gold inflates recall and hiding that would be dishonest.

```python
class EvalItem(BaseModel):
    id: str; dataset: str; question: str
    ground_truth: str | None; ground_truth_raw: str | None = None
    gold_urls: list[str]                       # PRIMARY, survives re-chunking
    gold_chunk_ids: list[str] = []             # cached hint, re-resolved at load
    gold_granularity: Literal["anchor", "page"] = "anchor"
    exclude_chunk_ids: list[str] = []          # leakage guard
    provenance: str
```

**Self-retrieval leakage.** `eval.self_retrieval`: `none` (naive, kept *only* so the inflated number can be published beside the honest one), `holdout` (default — `exclude_chunk_ids` threaded to `retrieve(..., deny_ids=...)` and filtered before fusion), `separate_index`. Automatic relabeling of discussion gold to cited docs pages was tested and **rejected: 1 of 27 answers links to `fastapi.tiangolo.com`.** Mining the docs' own cross-links was also tested and rejected: 146 links, 99 distinct anchor texts, degenerate (`"Deployment"`, `"dependencies"`). Both rejections go in the README.

**Three datasets, three jobs:**

| dataset | n | built how | gates CI | role |
|---|---|---|---|---|
| `docs_synth_v1` | ~127 | generated from known gold chunks + auto-filters, **~50 human-reviewed** | **yes** | primary retrieval metric |
| `discussions_v2` | ~16 | the 27 rows minus 11 unanswerable ones, **hand-labeled with gold docs sections** | no | human-labeled counterweight; the bridge between real questions and retrieval targets |
| `discussions_gen_v1` | ~16 | same items, own chunk excluded | no | answer quality / faithfulness only |

Note 11 of 27 current ground truths are under 200 chars and mostly social closure text (`"Thanks for your intrest! Let's track it in..."`) — unanswerable by construction, dragging every aggregate down. They get filtered.

**Synthetic construction** (`eval/synth_eval_set.py`) with filters applied in order — these are what make it defensible rather than circular:
1. **Lexical-overlap reject** — question shares a rare 3-gram with its source chunk ⇒ drop (else BM25 wins for free).
2. **Closed-book reject** — a *different* model answers correctly with no context ⇒ tests parametric knowledge, not retrieval ⇒ drop.
3. **Near-duplicate reject** — question-embedding cosine > 0.95 against an accepted item.
4. **Human spot-check** — `rag-eval eval review --dataset docs_synth_v1 --n 50` (backed by `eval/review.py`) shows question + gold chunk + which filters it passed, and takes `y` / `e`dit / `n`o / `s`kip. Sample is **stratified by top-level docs section** so the error rate generalizes. Decisions are written back as `verified: yes|no|edited` + `verified_at`, and the tool is resumable.

Emit a `BuildReport` and put its numbers in the README verbatim — e.g. *"150 generated, 23 auto-rejected (9 lexical, 11 closed-book, 3 dup), 127 retained, 50 human-reviewed, 4 edited, 2 rejected → 4.0% label error [95% CI 1.1–13.5%]."* **That sentence answers the question every sharp reviewer asks — "how do you know your labels are right?" — and is worth more than any metric on the page.** Report the CI too: at n=50 the error-rate estimate is itself uncertain, and saying so is the point.

Commit all three JSONL files; they are the project's most valuable artifact.

**Four mechanisms keep retrieval gains separable from generation gains:**
1. Retrieval metrics never invoke the generator (`generation.enabled: false`). This is the CI default and the README's primary number — zero confound by construction.
2. **Frozen-generator protocol**: answer-quality runs pin llm + `temperature: 0` + `prompt_version`. The manifest stores retrieval and generation hashes separately, and `compare_runs()` **returns `CONFOUNDED` rather than a delta** when the generation hash differs. The tool enforces the discipline.
3. Ablation matrix where each `configs/experiments/*.yaml` differs from its neighbour in exactly one field.
4. The generator upgrade is its own leaderboard row, so "+0.45 faithfulness from the generator" and "+0.19 recall@5 from hybrid+rerank" read as two separately-earned claims.

### C3 — Metrics

`src/rag_eval/eval/retrieval_metrics.py` — pure functions, no I/O:
```python
def recall_at_k / precision_at_k / hit_rate_at_k(retrieved, gold, k) -> float
def reciprocal_rank(retrieved, gold) -> float
def ndcg_at_k(retrieved, gold, k) -> float          # binary rel, ideal = min(k, |gold|)
def aggregate(per_item, k_values) -> AggregateMetrics
def bootstrap_ci(values, n=1000, seed=0) -> tuple[float, float]
def paired_bootstrap(base, cand, n=1000, seed=0) -> float
```
**Bootstrap CIs are non-optional.** At n=127 a 3-point delta is borderline; at n=27 the CI is roughly ±0.15. Reporting point estimates from a 27-row set as solid is the most common failure in RAG portfolio repos — saying so in the README is free differentiation. **Do not gate CI on the discussion set** for this reason.

### C4 — Run artifacts

```
runs/
  2026-08-05T14-22-31Z__baseline-dense__a1b2c3/
    manifest.json          # git sha+dirty, resolved config, split hashes, dataset sha256s,
                           # corpus sha, collection name, metrics, timings, cost
    config.resolved.yaml
    retrieval.jsonl        # per item: query, rewrites, candidates[{chunk_id,url,scores,stages,rank}], gold, metrics
    generation.jsonl       # per item: prompt_version, answer, citations, usage, latency, abstained, groundedness
    judge.jsonl            # written later by `rag-eval eval judge <run_id>`
    metrics.json           # aggregates + CIs + threshold results
  _pinned/<run_id>/        # promoted milestones (committed)
  leaderboard.json  baseline.json
```
`runs/manifest.py`: `new_run`, `write_manifest`, `load_run`, `list_runs`, `pin_run`.
`runs/leaderboard.py`: `build_leaderboard`, `compare_runs` (→ `CONFOUNDED`), `regressed_items(base, cand, metric)`.

**Git policy.** Committed: `configs/**`, `data/eval_sets/*.jsonl`, `data/corpus/**` (md + py + `SNAPSHOT.json`, ~2 MB — excludes img/js/css, which is why `data/raw` is 19 MB but the useful part is 2), `runs/_pinned/**`, `runs/leaderboard.json`, `runs/baseline.json`, `frontend/public/artifacts/*.json`. Gitignored: everything else under `runs/`, `data/processed/**`, `data/raw/**`. Promotion is explicit (`rag-eval runs pin <id>`), so **moving the baseline is a reviewable commit — the bar cannot be silently lowered.** Also fix the false README claim that `data/eval_sets` contents are gitignored; they are tracked, and should be.

---

## Phases

Phases 0–7 stand alone as a coherent, defensible project. 8–10 are shine. Do not start the frontend before the numbers are real.

### Phase 0 — Hygiene, worktree resolution, CI skeleton

No behaviour change. A repo a stranger can clone, and a PR-based workflow.

- **Create** `CLAUDE.md`, `LICENSE` (MIT), `Makefile` (`setup/lint/type/test/index/eval/serve/demo`), `.github/workflows/ci.yml` (lint + typecheck + test only), `tests/conftest.py`, `.github/pull_request_template.md`, `docs/adr/0001-record-architecture-decisions.md`.
- **Fix the known lies**: README's truncated `uv run python -m rag_eval.ingestion.` → `.embed_and_store`; README says `llama3.1` but config default is `fdm-llama`; the false eval_sets-gitignored claim; `chunker.py` docstring ("docs_chunker.py, once it exists" — it exists); `api/main.py` docstring promising eval triggers that don't exist; `generator.py` `SYSTEM_PROMPT` claiming context is only from Discussions.
- **Move** `ragas_results.csv` (minus its 3 garbage trailing rows, one of which is the aggregate dict written into `user_input`) to `runs/_pinned/0000-legacy-baseline/` as the honest "before" artifact.
- **Worktree** `D:\RAG-Eval\.claude\worktrees\kaggle-eval-offload`: first harvest its uncommitted `fastapi_discussions_eval.jsonl` (28 rows vs master's 27 — free labels, keep the union). Cherry-pick onto `feat/eval-export-split`, keeping **only** `notebooks/kaggle_judge_eval.py` verbatim (genuinely differentiating — Kaggle T4×2 running `qwen2.5:32b-instruct` as judge) and the `--export-only`/`--score-only` split with `dataset_to_jsonl`/`jsonl_to_dataset`. Drop `pipeline_output_path` (becomes `runs/<id>/generation.jsonl`) and the `ingest_*_limit = 0` diff (superseded by the corpus snapshot). Then `git worktree remove` + `git branch -D worktree-kaggle-eval-offload`. An unmerged branch in a portfolio repo reads as abandoned work.
- Rename `master` → `main`; branch protection requiring CI. Every later phase is a PR — the PR history is part of the portfolio.

**Verify:** `make lint type test` green; `git branch -a` shows only `main` + feature branches; CI green on the Phase 0 PR.

### Phase 1 — Deterministic corpus, coverage, chunk quality ← biggest score movement

Kills causes 1, 2, 3, and the chunk-size problem.

- **Create** `scripts/fetch_corpus.py` — pins the FastAPI repo at a commit SHA, copies `docs/en/docs/**/*.md` + `docs_src/**/*.py` into `data/corpus/`, writes `SNAPSHOT.json` `{fastapi_sha, fetched_at, n_pages, n_snippets, n_chunks_expected, chunk_count_hash}`.
- **Create** `ingestion/discussions_snapshot.py` — `fetch_snapshot(max_pages, out)` / `load_snapshot(path)` writing explicit discussion ids + `fetched_at` to `data/corpus/discussions.json`. Refresh is explicit, never implicit. Add retry/backoff to `github_discussions.py` while in there.
- **Create** `ingestion/packing.py` — extract `_split_into_blocks`, `_split_oversized_block`, `_atomic_blocks`, `_pack_blocks`, `_estimate_tokens` out of `docs_chunker.py` **verbatim** so both chunkers share them. `docs_chunker.py` re-imports; its 24 tests keep passing.
- **Modify** `docs_loader.py`: `EXCLUDED_DOCS |= {"_llm-test.md"}` (a test fixture currently being embedded and retrieved); `DOCS_DIR = Path("data/corpus/docs")`; make it parameter-overridable everywhere (it's relative-path-fragile today).
- **Modify** `docs_chunker.py`: add a post-pass `_merge_undersized(chunks, min_tokens=150)` inside `doc_to_chunks` merging consecutive sub-target chunks under a shared `##` parent, dropping chunks under 25 tokens. Add `parent_id = sha256(path::h2_breadcrumb)` for parent expansion. Root cause: `_pack_blocks` packs *within* a section and never across, so every short `###` becomes a micro-chunk.
- **Modify** `chunker.py`: pack long discussion answers through `packing.py`, emit `parent_id`/`chunk_index`, keep the single-chunk fast path for short answers. A 7 k-char chunk has a blurry averaged embedding *and* floods the context when retrieved. Update `scripts/inspect_chroma.py::EXPECTED_METADATA_KEYS_BY_SOURCE`.
- **Modify** `embed_and_store.py`: read snapshots, drop the `ingest_*_limit` knobs.
- **Modify** `build_eval_set.py`: query becomes `title + question_body` head, not the bare title (bug-report headlines, median 79 chars, body discarded).

**Verify:** `SNAPSHOT.json` chunk count matches a fresh run; ≥60% of chunks land in 150–600 tokens and none under 25 (from: avg 210, 134 under 50, 9.6% in band); `discussions.json` covers ⊇ the eval set's `source_url`s; **re-run the legacy eval with the generator unchanged and confirm `context_recall` moves 0.378 → ≥0.65.** Record that number — it is the headline of the debugging narrative.

**Result (see `runs/_pinned/0001-phase1-corpus-fix/`):** ✅ `SNAPSHOT.json`
reproducible (987 chunks); 62.6% of chunks in the 150–600 token band, none
under 25 (avg 210 → 244); `discussions.json` covers all 27 eval-set
`source_url`s. ⚠️ `context_recall` moved **0.378 → 0.4284**, short of the
≥0.65 target — the honest reasons (unregenerated eval set still using bare
titles, one RAGAS job dropped to a judge context-window overflow, no
self-retrieval guard yet, n=8) are in the pinned README. `faithfulness`
nearly doubled (0.2952 → 0.5917), the more trustworthy signal at this
sample size. Closing the recall gap needs the query fix applied to a
regenerated eval set (Phase 4) and hybrid retrieval (Phase 6) — not
something to force here.

**Do not re-index yet.** Phase 2 changes the embedding space.

### Phase 2 — Providers, namespaced collections, the one re-index

```python
# providers/base.py
class LLMProvider(Protocol):
    def complete(self, messages, *, temperature=0.0, max_tokens=1024) -> LLMResponse: ...
    async def astream(self, messages, ...) -> AsyncIterator[StreamChunk]: ...
class EmbeddingProvider(Protocol):
    name: str; model: str; dim: int; slug: str          # "bge-small-en-v15"
    def embed_documents(self, texts, batch_size=64) -> list[list[float]]: ...
    def embed_query(self, text) -> list[float]: ...
# providers/__init__.py
@lru_cache def get_llm(provider: str, model: str) -> LLMProvider
@lru_cache def get_embedder(provider: str, model: str) -> EmbeddingProvider
```
`llm/{groq,gemini,ollama}.py` use plain `httpx` — light, streamable, no LangChain churn in the hot path. `langchain_adapters.py` maps a judge spec to LangChain objects, used **only** by the eval judge because RAGAS requires them. Worth an ADR.

**Rewrite `vector_store.py`:**
```python
@lru_cache def get_client(persist_dir: str) -> chromadb.PersistentClient   # fixes per-call rebuild
def collection_name(source: str, embedder) -> str        # "fastapi_docs__bge-small-en-v15"
def get_collection(source, embedder, create=False)
def upsert_chunks(chunks, source, embedder, batch_size=64) -> int
def query(embedding, source, embedder, k, where=None) -> list[dict]
```
Stop using Chroma's built-in embedding functions — compute vectors ourselves and pass `embeddings=` / `query_embeddings=`. Makes provider swap trivial, makes it testable without a server, removes the one-at-a-time Ollama upsert. Store `{embedding_model, embedding_dim, corpus_sha, created_at}` in collection metadata and **assert on read** — querying a collection embedded with a different model must fail loudly, not return garbage.

**Re-index once.** Build `fastapi_docs__bge-small-en-v15` + `fastapi_discussions__bge-small-en-v15`. **Leave the old `fastapi_docs`/`fastapi_discussions` (nomic, 264 chunks) in place** — that's the point of namespacing, and it makes nomic-vs-bge a leaderboard row. Prune only after that A/B is recorded.

**Verify:** ~1200 chunks in the new namespace; `inspect_chroma.py --collection <full name>` shows correct metadata; both namespaces coexist in one sqlite; a mismatched-embedder query raises.

### Phase 3 — Thin-slice deploy + CD ← first public URL

Ship the container now, while it's cheap to debug. Four-stage Dockerfile, with the web stage a placeholder page for now:

`web` (node:20-slim → `out/`) → `deps` (`uv sync --frozen --no-dev`) → `index` (runs `rag-eval index build --config configs/deploy.yaml` against the committed corpus, warms fastembed + reranker ONNX caches into the layer) → `runtime` (venv + `out/` → `/app/static` + chroma + caches, `USER 1000`, `CMD uvicorn ... --port ${PORT:-7860}`).

**Bake the index at build time.** ~1220 chunks × 384 dims ≈ 1.8 MB of vectors, ~15 MB sqlite; fastembed encodes in 30–60 s on a build CPU. Building at container start would add that to every cold start, and HF free tier cold-starts constantly — a 60 s startup makes the demo look broken. Baking also means no `GITHUB_TOKEN` at runtime and a byte-identical index across restarts. **This is only viable because the corpus snapshot is committed — the second reason Phase 1 matters.** Runtime guard: empty collection or mismatched `embedding_model` metadata fails `/api/health/ready` loudly. Image estimate ~1.1–1.4 GB.

- `deploy/space/README.md` holds the HF YAML front-matter (`sdk: docker`, `app_port: 7860`) — assembled by the workflow, kept out of the portfolio README.
- `.github/workflows/deploy.yml` — on `main` after CI green, build and push to the HF Space remote with `HF_TOKEN`. Secrets via Space secrets; no `.env` in the image.
- `.github/workflows/keepalive.yml` — `cron: '0 */6 * * *'` curling `/api/health`. HF free tier sleeps after 48 h; a recruiter's first click otherwise takes 20–40 s.
- Mount `StaticFiles(html=True)` at `/` with the API under `/api`; keep the existing `/health` so its integration test passes unchanged.

**Verify:** `docker build` locally; `docker run -p 7860:7860 -e GROQ_API_KEY=...`; `/api/health/ready` green; push to `main` and confirm the Space redeploys and answers a question end to end.

### Phase 4 — Eval sets, gold labels, metrics, manifests, CLI

A judge-free, generator-free, reproducible number you can trust.

**Create** `config/run_config.py`, `eval/{datasets,gold,retrieval_metrics,runner,synth_eval_set,review}.py`, `runs/{manifest,leaderboard}.py`, `cli.py` (typer; `[project.scripts] rag-eval = "rag_eval.cli:app"` — the declared-but-unused dep finally earns its place), `configs/{_base,baseline,ci,deploy}.yaml`.

```python
# eval/runner.py
def run_experiment(cfg: RunConfig, config_path: Path, overrides: Sequence[str]) -> RunRecord
```

Build `docs_synth_v1` (~150 generated → ~127 retained after filters), derive `discussions_v2` and `discussions_gen_v1`. Commit all three. **Freeze the chunker before generating** — Phase 1's merge pass shifts chunk ids, which is why gold is URL-keyed, but generating earlier still means labeling chunks that no longer exist.

**Two human checkpoints — this phase pauses for you** (everything else runs unattended; both are resumable, so they can be done across sittings):

```
$ uv run rag-eval eval review --dataset docs_synth_v1 --n 50        # ~1h

[12/50]  section: tutorial/  (stratified sample)
  Q: How do I make a query parameter required without giving it a default value?

  GOLD  tutorial/query-params.md#required-query-parameters
  "When you want to make a query parameter required, you can just not
   declare any default value..."

  filters passed: lexical-overlap OK | closed-book OK | dedup OK

  [y] correct   [e] edit question   [n] reject   [s] skip   >
```

```
$ uv run rag-eval eval label --dataset discussions_v2               # ~1h, ~16 items

[3/16]  "Feat: HTMLResponse should support __html__ conversion of content"
  answer excerpt: "...you'd want a custom response class, see..."

  candidate docs sections (top-10 by current retrieval, for convenience only):
   1. advanced/custom-response.md#html-response
   2. advanced/custom-response.md#return-a-response
   ...
  enter gold URLs (comma-separated, or `?` to search, `n` for none) >
```

The second one is what makes `discussions_v2` the bridge: **real questions with real retrieval targets**, so you can check whether the synthetic set's absolute numbers are lying. Expect them to diverge — synthetic questions are LLM-shaped and reuse their source chunk's vocabulary — and report the gap rather than hiding it.

**Verify:** `rag-eval eval run --config configs/baseline.yaml` writes a complete run dir; `metrics.json` carries recall@{1,3,5,10}, MRR, nDCG@10 with bootstrap CIs; re-running the same config twice yields identical metrics (a `slow` determinism test); `rag-eval leaderboard` renders two runs; the `BuildReport` label error rate is computed from your review decisions; **the `none` vs `holdout` self-retrieval delta on `discussions_v2` is measured and recorded — that delta is a README headline**; and `docs_synth_v1` vs `discussions_v2` recall@5 are compared on the same config to quantify the synthetic-set optimism gap.

### Phase 5 — CI eval gate + leaderboard

**Create** `eval/gate.py`, `.github/workflows/eval-llm.yml`; extend `ci.yml` with an `eval-gate` job; pin the first baseline.

```
FAIL  metric < thresholds[m].min                                (absolute floor)
FAIL  metric < baseline[m] - thresholds[m].regression_tolerance (relative)
WARN  metric > baseline[m] + 0.02  -> "consider `rag-eval runs pin`"
PASS  otherwise; missing baseline -> PASS with a note
```
`GateResult.to_markdown()` → `$GITHUB_STEP_SUMMARY` + sticky PR comment; exit 1 on FAIL.

**The gate uses zero API quota and no Ollama.** `configs/ci.yaml` sets `generation.enabled: false`, `judge.enabled: false`, `query_rewrite.enabled: false`. Embeddings are fastembed ONNX in-process — precisely why that decision was right. Cache the ~130 MB model with `actions/cache`. **Leave the reranker enabled** (also ONNX/CPU) — reranking regressions are exactly what you want caught; ~127 items × 30 candidates ≈ 1–2 min. Rebuild the index in CI from the committed snapshot (~60 s) rather than caching a Chroma dir: deterministic, and it regression-tests ingestion for free. Budget < 6 min.

`eval-llm.yml` — nightly + `workflow_dispatch` + PRs labelled `run-llm-eval`, using `secrets.GEMINI_API_KEY`, `sample_limit: 40`. **Does not gate**; opens an issue if faithfulness drops > 0.05 week-over-week.

**Verify:** a throwaway PR setting `retrieval.top_k=1` FAILs the gate with a markdown table naming the metric and delta; one that improves a metric WARNs and suggests pinning.

### Phase 6 — Retrieval quality lab

**Create** `src/rag_eval/retrieval/{base,dense,bm25,fusion,rerank,rewrite,expand,pipeline}.py`.
```python
@dataclass class Candidate:
    chunk_id: str; content: str; url: str; title: str; source_type: str
    scores: dict[str, float]; ranks: dict[str, int]; stages: list[str]; final_score: float
@dataclass class RetrievalResult:
    query: str; rewritten_queries: list[str]; candidates: list[Candidate]
    stage_timings: dict[str, float]; stage_counts: dict[str, int]
class RetrievalPipeline:
    @classmethod def from_config(cls, cfg: RunConfig) -> RetrievalPipeline
    def retrieve(self, query, *, k=None, deny_ids: AbstractSet[str] = frozenset()) -> RetrievalResult
def reciprocal_rank_fusion(rankings: list[list[Candidate]], k: int = 60) -> list[Candidate]
```
`RetrievalResult` is simultaneously the frontend trace payload, the `retrieval.jsonl` row, and the metric input — one shape, three consumers. Call that out in the README.

**Why RRF and not score normalization:** today `retriever.py:12-15` sorts `1 - cosine` across two collections with different length and topic distributions, so long discussion answers systematically beat terse docs chunks. Rank-based fusion is distribution-free. Add per-source caps.

BM25 builds from `collection.get(...)` at startup (1158 docs, < 1 s, ~5 MB), module-cached by collection name. **Code-aware tokenizer**: lowercase, split on non-alphanumeric but keep `_` and `.` inside identifiers so `response_model` and `jsonable_encoder` stay single terms. On a code-docs corpus that's the difference between BM25 helping and BM25 being noise — and a good README paragraph.

Reranker: `fastembed.rerank.cross_encoder.TextCrossEncoder("Xenova/ms-marco-MiniLM-L-6-v2")` — ONNX, ~90 MB, no torch. **Do not reach for `sentence-transformers`**: it pulls torch, takes the image from ~1.2 GB to ~4 GB, and breaks the single-container deploy. Lock this dependency before writing code against a torch API.

Parent expansion needs no new storage — `path`, `section`, `parent_id`, `chunk_index` are already in metadata, so siblings are a Chroma `where` query. Keep `rag/retriever.py::retrieve` as a thin back-compat wrapper.

**Verify:** run the ablation grid; each variant differs in exactly one field; `hybrid+rrf+rerank` beats `dense` on recall@5 with non-overlapping bootstrap CIs on `docs_synth_v1`. **Record HyDE's latency cost beside its recall gain** — reporting the cost of a win is a senior habit.

### Phase 7 — Generation, citations, groundedness, streaming

**Create** `rag/prompts/{v1_legacy,v2_cited}.py` (`PROMPTS: dict[str, PromptTemplate]`, version in the manifest), `rag/citations.py`, `rag/groundedness.py`, `common/telemetry.py`, `api/routes/{ask,eval,health}.py`, `api/deps.py`; rewrite `api/main.py` as an app factory whose lifespan warms embedder, reranker, and BM25 index.

v2 prompt: numbered context blocks `[n] (source: <url>)`, cite `[n]` after every factual sentence, and an explicit `INSUFFICIENT_CONTEXT: <what is missing>` sentinel. That sentinel fixes the row-4 pathology where an "I don't know" scored 0/0/0/0 — abstention then reports as `abstention_rate` + `abstention_precision` rather than as total failure.

```python
# citations.py
def extract_citations(answer, candidates) -> list[Citation]
def validate_citations(answer, candidates) -> CitationReport   # unknown_indices, uncited_sentences, coverage
class StreamingCitationScanner:  def feed(self, delta: str) -> list[Citation]
# groundedness.py — no extra LLM call
def sentence_support(answer, candidates, embedder) -> list[SentenceSupport]
def groundedness_score(supports) -> float
def should_abstain(score, threshold) -> bool
```

**SSE** on `POST /api/ask/stream`, `async def` + `StreamingResponse(media_type="text/event-stream")`:
```
event: meta       {request_id, config_hash, k, stages}
event: retrieval  {candidates:[...], timings:{embed,dense,bm25,rrf,rerank}}
event: token      {"t": "FastAPI "}
event: citation   {index, chunk_id, url, char_start, char_end}
event: done       {answer, citations, groundedness, abstained, usage:{...,cost_usd}, latency_ms}
event: error      {detail}
```
**Ordering is load-bearing** — `retrieval` fires before the first token, so the trace panel renders while the answer streams. Two gotchas baked in now: set `X-Accel-Buffering: no` (HF sits behind buffering proxies and will otherwise deliver the whole stream at once), and never use a sync `def` endpoint for streaming. Provider streams are normalized inside each provider (Groq SSE, Gemini `alt=sse`, Ollama **NDJSON not SSE**).

Sentence-level groundedness needs the whole answer, so support badges appear on `done` only — an honest constraint to state, not a gap.

`common/telemetry.py` (~60 lines, no OpenTelemetry): `Usage`, `PRICES`, `timed()`, `estimate_cost()`, `log_request()` appending `runs/serve/requests-YYYY-MM-DD.jsonl`. `GET /api/stats` → p50/p95 latency, total cost, request count.

**Rate limiting + daily budget cap** on `/api/ask*` — a public demo on your Groq key will get scraped. Per-IP token bucket (10 req / 5 min), 500-char question cap, global daily cap flipping to a canned "demo budget exhausted, here's a recorded transcript" response. ~50 lines, reads as production thinking.

**Verify:** `curl -N` shows `meta → retrieval → token* → done`; every `[n]` resolves to a real URL; a no-context question abstains with `abstained: true`; `/api/stats` non-empty.

### Phase 8 — Judge depth

`eval/run_ragas.py` → `eval/judge.py`: `rag-eval eval judge <run_id>` reads `runs/<id>/generation.jsonl`, writes `judge.jsonl` — the generalized form of the worktree's `--export-only`/`--score-only`, anchored to run artifacts. Keep the Kaggle notebook working against the same file.

`eval/rubric.py` — one structured judge call per item scoring `{correctness, completeness, citation_accuracy, hallucination, tone}` 1–5 with required justification, **plus `citation_accuracy` computed deterministically** via the embedding-support scorer as a cross-check on the judge. A deterministic metric that audits your LLM judge is a strong signal. Set `answer_relevancy.strictness = 3` now that the judge is hosted (the current `1` is a single-sample estimate — hence four exact 0.0s). The judge is always hosted and never the generator's model; recorded in the manifest.

**Verify:** judge two runs differing only in generator; rubric and RAGAS agree directionally; report judge-vs-deterministic citation-accuracy correlation (if it's low, that's a finding worth writing up).

### Phase 9 — Frontend

`frontend/` — Next 15 App Router, TS, Tailwind, `output: 'export'`, `trailingSlash: true`, `images: {unoptimized: true}`. Two export constraints designed around now rather than discovered later: dynamic routes would need `generateStaticParams`, coupling `next build` to run artifacts — so **make the drilldown a client-side query route `/eval?run=<id>` fetching `/api/runs/{id}`**, pre-copying `runs/_pinned/**` into `frontend/public/artifacts/` only for the leaderboard's first paint. And `trailingSlash` must match `StaticFiles(html=True)` or you get 404s only in the container, never in `next dev`.

**Ask view:** question box, streamed answer with inline `CitationChip`s (hover → snippet, click → scroll to the chunk), per-sentence support badges after `done`, and a trace panel showing rank, a stacked dense/bm25/rerank score bar, stage pills (`dense`, `bm25`, `rrf`, `reranked ↑12`), and a `stage_timings` timeline. **This panel is the single best screenshot in the project — build it first and build it well.**

**Eval view:** leaderboard (run, config, embedder, stage pills, recall@5 with CI error bars, MRR, nDCG@10, faithfulness, cost, p50 latency, git-sha link), baseline pinned; metric bars with error bars; diff view with CI-overlap indicator **and a per-item regression list** (gold rank 2 → 14). Per-item regression lists are what make an eval dashboard useful and what most portfolio dashboards omit.

`lib/sse.ts` exports `async function* streamAsk(...)` on `fetch` + `ReadableStream` (not `EventSource` — it can't POST). Events reduced with `useReducer`; that reducer is the vitest-tested unit. **Load the `dataviz` skill before writing any chart code.**

**Verify:** `npm run build` produces `out/`; reducer test passes on a canned event sequence; Playwright smoke test drives Ask against a mocked API; the deployed Space serves `/` and `/eval`.

### Phase 10 — Portfolio presentation

Restructure `README.md` to lead with results:
1. One paragraph + live link + a 30 s GIF of Ask streaming with citations and the trace panel.
2. **Results table first** (with CIs), then the "what moved the needle" ablation table.
3. **Architecture diagram** (`docs/architecture.svg`): corpus snapshot → chunker → embedder → namespaced Chroma → [rewrite → dense ‖ bm25 → RRF → rerank → parent-expand] → prompt → LLM → citations/groundedness → SSE → UI; with a parallel lower track: eval sets → runner → run artifacts → leaderboard → CI gate.
4. **Evaluation methodology** — the strongest section. Gold-label derivation; self-retrieval leakage stated plainly with the measured inflation delta; the three datasets and their roles; the `BuildReport` line with the measured label error rate and its CI; why retrieval metrics are LLM-free; bootstrap CIs; the frozen-generator protocol; the honest caveat that synthetic questions are LLM-shaped and overestimate absolute quality, **quantified against the hand-labeled `discussions_v2` set**; and the two eval-set ideas measured and rejected (1/27 doc links; 99 degenerate anchor texts). **A self-caught, quantified methodological limitation is the most senior thing in a portfolio repo.**
5. **What I found when I measured** — the corpus-coverage bug narrative with before/after numbers. Debugging stories beat feature lists.
6. **Cost & latency** — p50/p95, $ per 1k questions, why fastembed over a hosted embedding API.
7. **Limitations & what's next** — 151-page corpus is small; only ~40% of the synthetic set was human-reviewed; synthetic questions remain LLM-shaped; no multi-hop; binary relevance underuses graded judgments; n=16 on the hand-labeled set means its CIs are wide (~±0.15) and it can gauge but not gate.
8. **Reproduce it** — four commands, `make demo`.

Plus 6 short ADRs: RRF over score normalization · fastembed over hosted embeddings · bake the index at build time · retrieval metrics are judge-free · embedding-namespaced collections · plain-httpx providers with LangChain confined to the judge.

---

## Testing

Follow the existing seam style, but prefer **constructor injection over patching** for new code — `RetrievalPipeline(dense=FakeDense(), bm25=None, ...)` with only `from_config()` touching the real world lets you assert *"the disabled stage was never called"*.

`tests/conftest.py` (missing today): `sample_chunks` (~12 hand-written), `fake_embedder` (deterministic token-hash vectors, dim 8, "planted" mode), `fake_llm` (scripted, records calls, supports `astream`), `ephemeral_collection` (`chromadb.EphemeralClient()` — a real Chroma, in memory), `run_config`, `tmp_runs_dir`. **The deterministic fake embedder is the keystone** — it makes the whole retrieval stack testable with no network, no ONNX, no server.

| file | proves |
|---|---|
| `test_retrieval_metrics.py` | hand-computed nDCG at ranks 1 and 3; MRR=0 when nothing gold retrieved; recall when \|gold\|>k; bootstrap determinism. **If this is wrong, every README number is wrong** |
| `test_run_config.py` | `extra="forbid"` rejects a typo; `--set` coercion; `extends` merge; hash stable under key reordering |
| `test_fusion.py` / `test_bm25.py` | RRF ties + dedup + stage accumulation; tokenizer keeps `response_model` whole; planted-identifier query beats dense |
| `test_retrieval_pipeline.py` | **disabled stages record zero calls** (the classic "my config did nothing" bug); `deny_ids`; per-source caps |
| `test_citations.py` | offsets, `[1][2]` adjacency, unknown index, scanner fed in arbitrary splits |
| `test_gold.py` / `test_manifest.py` / `test_gate.py` | url→chunk incl. page fallback; holdout deny set; `compare_runs` → `CONFOUNDED`; all four gate outcomes |
| `test_synth_filters.py` / `test_review.py` | each reject filter fires on a planted positive and stays quiet on a negative; stratified sampling covers every top-level section; review decisions round-trip and resume mid-session without losing prior verdicts |
| `test_ask_stream.py` (integration) | **highest value**: `TestClient` + fake provider via `dependency_overrides`; asserts event order `meta → retrieval → token+ → done` and that every `done.citations` entry validates |
| `test_static_mount.py` | `/` serves `index.html` **and** `/api/health` still works — catches the mount-order bug |
| `test_corpus_snapshot.py` | pinned chunk count from `SNAPSHOT.json` — a chunker change that silently halves the corpus fails CI |

`pyproject.toml`: `markers = ["slow", "integration", "llm"]`; CI runs `-m "not slow and not llm"`, `--cov-fail-under=80`.

**Known breakage:** the 5 `test_retriever.py` tests die in Phase 6 (the naive merge is deleted) — port them to `RetrievalPipeline` with injected fakes, keeping the url→source_id and empty-url tests against the compat wrapper. 2–3 of the 24 `test_docs_chunker.py` tests need updating in Phase 1; the other 21 and all 18 `test_docs_loader.py` tests survive untouched — that's the payoff for extracting `packing.py` verbatim instead of rewriting.

## Sequencing rules

- **Index exactly once.** Phases 0–1 touch data and config only; nothing queries the new namespace until Phase 2. Re-indexing at the end of Phase 1 with nomic and again in Phase 2 with bge is wasted work.
- **Freeze the chunker before generating any eval set** (Phase 1 lands before Phase 4).
- **Old collections stay** until the nomic-vs-bge A/B is on the leaderboard. `chroma.sqlite3` is gitignored; this costs only disk.
- **Corpus move** `data/raw/` (gitignored, 19 MB) → `data/corpus/` (committed, ~2 MB, md + py + `SNAPSHOT.json`). **Without this the Docker build has nothing to index** — it is a hard blocker for Phase 3, not a detail.
- **Harvest the worktree's uncommitted eval JSONL before removing it** — those are free labels.

## Critical files

| path | change |
|---|---|
| `src/rag_eval/rag/vector_store.py` | cached client, embedding-namespaced collections, self-computed embeddings, metadata assertions — everything else depends on this landing first |
| `src/rag_eval/ingestion/docs_chunker.py` | extract `packing.py`, add the undersized-merge post-pass and `parent_id`; the 210-token average originates here |
| `src/rag_eval/ingestion/docs_loader.py` | repoint `DOCS_DIR` to `data/corpus/docs`, exclude `_llm-test.md`; the 30-page alphabetical truncation flows through `iter_raw_docs` |
| `src/rag_eval/common/config.py` | shrink to secrets/paths; metric-affecting settings move to `configs/*.yaml` |
| `src/rag_eval/rag/retriever.py` | replaced by `retrieval/pipeline.py`; the naive cross-collection merge at lines 12–15 is the retrieval defect |
| `src/rag_eval/eval/run_ragas.py` | becomes `eval/judge.py`, artifact-driven, absorbing the worktree's export/score split |
