# STATUS

## Done
- Design plan agreed for Phase 9 (frontend-design skill applied), then
  superseded by an external design brief per your call.
- `docs/plan.md` Phase 9 updated with an addendum: console design
  supersedes the original Ask-view spec, Eval-view (leaderboard) unchanged,
  gold-metrics scoping resolved, backend contract additions scoped.
- PR1 (chat-only console + backend contract additions) built on
  `feat/phase9-rag-console`: gold-aware citations/`path` field, `POST
  /api/feedback`, `GET /api/suggestions` (`src/rag_eval/api/routes/ask.py`,
  `deps.py`, `common/telemetry.py`); new `frontend/` (Next 15 static
  export) with the "Answer only" chat screen per the design brief.
- Reviewer subagent verdict: **ship after fixes**, risk **low**. No
  gold-leak path, no RunConfig/Settings drift, no secrets or build output
  staged under `frontend/`. Both pre-ship findings fixed: negative `n` on
  `/api/suggestions` now clamps to 0 (was an unhandled 500), and
  `FeedbackRequest.request_id` now has `max_length=64`.
- `make lint type test` green — **344 passed**. Frontend `lint`/
  `tsc --noEmit`/`test`/`build` all green.
- Manually verified the full ask flow in a browser: eval-set question ->
  gold tile + "Gate passed"; free-typed question -> ordinary tiles only,
  never gold; feedback POST confirmed landing in
  `runs/serve/feedback-*.jsonl`. Added dev-only CORS middleware to
  `src/rag_eval/api/main.py` (`localhost:3000`/`127.0.0.1:3000` only) --
  without it `npm run dev` can't reach the API at all.
- **Bug found and fixed**: Groq occasionally emits fullwidth citation
  brackets (`【n】`) instead of the prompted `[n]`, so
  `rag/citations.py`'s marker regex silently dropped the citation --
  no tile rendered client-side because the backend never told it one
  existed. Broadened `_CITATION_RE` and `StreamingCitationScanner`'s
  buffer boundary to accept both bracket styles; 2 regression tests added
  (`tests/unit/test_citations.py`). **346 passed** after the fix.

- Follow-up reviewer pass on the CORS middleware + citation-bracket fix:
  **verdict pass, risk low, no fixes required**. `ruff`/`mypy`/pytest
  reverified (346 passed).
- **2026-09-18 — PR2 reviewer finding fixed**: `dense_candidates` in
  `retrieval/pipeline.py` now builds independent `Candidate` snapshots
  (`dataclasses.replace` with copied `scores`/`ranks`/`stages`) instead of
  `list(candidates)`, which shared object references with the list the
  reranker mutates in place. Added a regression test
  (`test_dense_candidates_are_not_mutated_by_in_place_reranking`, using a
  `MutatingReranker` fake that mirrors `FastEmbedReranker.rerank()`'s
  real in-place-mutation behaviour) so a future reranker change can't
  reintroduce this silently. `ruff`/`mypy`/pytest reverified — **355
  passed**.

- **2026-09-18 — Stage pills collapsed from 4 to 3, per your call**: you
  found the panel showing identical data across all four pills twice --
  once because `configs/deploy.yaml` (the local server's default) has
  `retrieval.rerank.enabled: false`, so `dense_candidates` and `candidates`
  were legitimately identical (confirmed genuinely correct behavior by
  diffing a live `retrieval` SSE payload against
  `configs/experiments/hybrid_rerank.yaml`, which does show them diverge --
  not a bug); and once because `Embed`/`Dense 50` are spec'd by the design
  brief to always render the same pre-rerank pool regardless of config
  (also not a bug, just two controls for one dataset). You chose to merge
  `Embed`+`Dense 50` into a single `Dense` pill rather than keep the
  spec's 4-pill layout -- a deliberate deviation from the design brief.
  Reviewer subagent verdict on this delta: **ship**, risk **low**, no
  fixes needed. Reducer/evalPanel tests updated to match (23 passed).
- **2026-09-18 — PR2 committed, pushed, PR opened**: `feat/phase9-pr2-eval-
  panel` (22 files, +1046/-57), commit `35c36d5`. Backend:
  `dense_candidates` on `RetrievalResult`/`RetrievalPipeline`
  (`retrieval/base.py`, `retrieval/pipeline.py`) so the pre-rerank pool is
  real data, not a re-derivation, captured as an independent snapshot so
  the reranker's in-place mutation can't contaminate it; `_candidate_out()`
  now sends `gold` and `content`; the `retrieval` SSE event sends
  `dense_candidates` alongside `candidates`; new `POST /api/verdict` +
  `log_verdict()` (`common/telemetry.py`) mirroring `POST /api/feedback`
  exactly, logged to a separate `verdict-*.jsonl`. Frontend: header view
  switch, `stage`/`openRow`/`view`/`reviewerVerdicts` on `AskState`,
  `lib/evalPanel.ts` (pure chunk-row/gold-metric derivation, unit-tested),
  new `components/EvalPanel.tsx` (3 stage pills: Dense/Rerank 8/Answer),
  `SourceTile` now opens the panel on the matching chunk row instead of a
  new tab (PR1 stopgap), header gate indicator fixed to track
  `state.activeId` instead of `.at(-1)`. `HANDOFF.md` deleted per its own
  last line. PR: https://github.com/RiverRover-stack/RAG-Eval-Harness/pull/54
  -- open, awaiting CI + your merge.
- **2026-09-18 — Citation fullwidth-bracket render fix, committed and
  pushed** (separate branch `fix/inline-citation-fullwidth-brackets`,
  commit `e783009`, kept apart from PR2 per your one-PR-per-logical-change
  call): widened `frontend/lib/inline.tsx`'s render regex to accept `【n】`
  as well as `[n]`, mirroring the backend's existing `_CITATION_RE`. Added
  `lib/inline.test.tsx` (6 cases) plus the `@/` alias and JSX-transform
  config `vitest.config.ts` needed to test component code for the first
  time. Reviewer subagent verdict: **ship**, risk **low**, one comment nit
  (fixed). Pushed, **no PR opened yet** -- only explicitly asked to push
  this one; say the word and I'll open it too.

## In progress
- **2026-09-22 — Real Next.js frontend wired into the Docker deploy image**,
  branch `infra/deploy-real-frontend` off `origin/main` (not committed yet).
  `Dockerfile`'s `web` stage now actually builds `frontend/` (`npm ci` +
  `npm run build`) instead of copying the static placeholder;
  `.dockerignore`'s blanket `/frontend` line narrowed to just
  `node_modules`/`.next`/`out` so the build context includes frontend
  source. Runtime stage copies the Next export to a new path
  (`/app/deploy/web`, `STATIC_DIR` updated to match) rather than
  overwriting `deploy/web-placeholder/`, which stays as-is for the bare
  `uvicorn`-with-no-`STATIC_DIR` local fallback (`main.py`'s default,
  docstring updated to say so, no logic changed). Removed
  `RAG_LLM_PROVIDER`/`RAG_LLM_MODEL` from the runtime `ENV` block after
  confirming by repo-wide grep they're read nowhere -- model choice is
  entirely `RunConfig`-driven per `configs/deploy.yaml`, matching this
  file's config-discipline rule.
  Verified: `docker build` succeeds end-to-end (frontend compiles, 987+112
  chunks indexed into Chroma); ran the built image and confirmed `/` serves
  the real app (`<title>FastAPI Docs Assistant</title>`, `_next/static`
  chunks, not the placeholder stub), `/api/health` -> `{"status":"ok"}`,
  `/api/health/ready` -> `{"ready":true}`; test container/image removed
  after. `make lint type test` -- 355 passed, ruff and mypy clean.
  First reviewer subagent pass: **ship**, risk low, no findings (independently
  rebuilt/ran the image and confirmed the same things).
- **2026-09-22 — You actually tested it and found two more real bugs**, both
  pre-existing (not introduced by the frontend-wiring change above, just
  never hit until a real browser test happened):
  1. Asking a question in the browser failed with `Failed to fetch` (a
     network-level failure, not an HTTP error -- curl to the same URL got a
     clean response). Root cause: `.dockerignore`'s `.env` / `.env.*` rule
     is **not recursive** (Docker's dockerignore patterns, unlike
     gitignore's, only match at the context root unless prefixed `**/`) --
     so your local `frontend/.env.local` (a `next dev` convenience file
     pointing at `localhost:8000`) slipped into the Docker build context
     and got baked into the static export as an absolute
     `NEXT_PUBLIC_API_BASE_URL`, instead of the intended same-origin `""`.
     The browser then tried to fetch a cross-origin `localhost:8000` that
     had nothing listening. Fixed: added explicit
     `/frontend/.env.local` + `/frontend/.env.production.local` lines to
     `.dockerignore`. Confirmed via `docker exec ... grep -rl
     localhost:8000 /app/deploy/web` -- no match after the fix.
  2. Next error was a clean `503` (`RAG backend unavailable`) -- worse, a
     genuinely broken RAG backend the whole time, unrelated to the
     frontend work. Container logs showed `build_app_state()`
     (`api/deps.py`) throwing at startup on two missing files:
     `data/corpus/discussions.json` (the `runtime` stage only ever copied
     `SNAPSHOT.json` out of the `index` stage's full `data/corpus/`, never
     the rest) and `data/eval_sets/docs_synth_v1.jsonl` /
     `discussions_v2.jsonl` (the `docs_synth_v1`/`discussions_v2` eval
     sets `deps.py` loads for gold-aware citations and the suggestion
     chips -- these are committed to git but `.dockerignore` explicitly
     excluded `/data/eval_sets` from the build context entirely, and
     nothing ever `COPY`'d them in). Both are **pre-existing bugs, not
     caused by this branch** -- the Dockerfile's original placeholder-only
     version never exercised `build_app_state()`'s eval-item loading path
     in a way anyone tested against a real container before now, so this
     was latent since whenever that `deps.py` logic first shipped (Phase
     7/9). If the live Render service has ever actually served a real
     `/api/ask` request, it's plausible it's been 503ing there too.
     Fixed: `runtime` stage now does
     `COPY --from=index /app/data/corpus/ data/corpus/` (whole directory,
     not just the snapshot) and a new `COPY data/eval_sets/
     data/eval_sets/`; `.dockerignore`'s `/data/eval_sets` exclusion
     removed.
  Reverified end-to-end after both fixes: `/api/health/ready` ->
  `{"ready":true}` with **no startup traceback** in the logs (first time);
  a real `curl POST /api/ask/stream` returned a genuine `meta` -> `retrieval`
  SSE sequence with real retrieved chunks. `make lint type test` reverified
  -- 355 passed. Not committed, pushed, or PR'd yet -- a second reviewer
  pass on the updated diff and your go-ahead are next.

- **2026-09-22 — PR #53 (citation fullwidth-bracket render fix) and PR #54
  (Phase 9 PR2, eval panel) are both merged.** (The two "needs your call"
  items about opening/merging them were stale as of this line -- corrected
  here rather than left to confuse the next read.)
- **2026-09-22 — Second reviewer pass on `infra/deploy-real-frontend`:
  verdict needs-changes, risk low-medium**, one real finding: the
  `.dockerignore` fix for the `frontend/.env.local` leak (see "In progress"
  above) stopped only that one exact filename, not the whole Next.js
  env-file convention (`.env`, `.env.development[.local]`, `.env.test*`,
  etc.) -- confirmed empirically both ways with scratch
  `docker buildx build --output type=local` context exports. **Fixed**:
  swapped the two explicit lines for `/frontend/.env*` +
  `!/frontend/.env.local.example`; re-verified with the same export
  technique -- only `.env.local.example` reaches the build context now,
  `.env.local` does not. Everything else in that reviewer pass had already
  checked out clean (config-discipline on the removed env vars, no
  secrets/oversized files in `data/corpus/`, layer-caching order). `make
  lint type test` reverified again after this fix -- 355 passed.
- **2026-09-22 — Found and fixed a local-only `.env` bug** while
  diagnosing a Groq `401 Unauthorized` you hit running the Docker image:
  `GROQ_API_KEY` (and `GITHUB_TOKEN`/`GEMINI_API_KEY`/`LITEROUTER_API_KEY`)
  were quoted (`KEY="value"`) in your gitignored local `.env`. Python's
  dotenv loader strips quotes (works fine outside Docker), but `docker run
  --env-file .env` does not -- it passed the literal quote characters as
  part of the bearer token, which Groq correctly rejected. Stripped the
  quotes from all four values locally (verified identical key
  prefix/length before and after); `.env.example` was already unquoted, so
  no repo change needed. Not a code bug, nothing to commit.

## Needs your call
- **Docs system update** (reviewer finding from PR1, not a code defect):
  your standing instruction is to update
  `docs/{PROJECT,ARCHITECTURE,DECISIONS,EXPERIMENTS}.md` per milestone
  without being asked; still outstanding. Want that done now, or after
  PR1/PR2/the citation fix all land?
- **Left untouched, not part of this PR**: untracked
  `src/rag_eval/providers/llm/literouter.py` +
  `configs/experiments/gen_hybrid_literouter.yaml` — pre-existing,
  unrelated scratch work in the working tree (unregistered provider, half
  finished). Excluded from `git add` again.
