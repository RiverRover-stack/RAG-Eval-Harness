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

## In progress
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
  Changed: `lib/evalPanel.ts` (`STAGE_LABELS`, `activeCandidates`,
  `scoreFor` threshold `stage < 2` -> `stage < 1`), `components/
  EvalPanel.tsx` (`showingDense` threshold), `lib/reducer.ts`/`lib/
  types.ts`/`app/page.tsx` (default/reset stage `2` -> `1`, still "Rerank
  8", now at index 1 of 3 instead of 2 of 4). `RetrievingState.tsx`'s own
  4-phase loading-text labels (`Turn.stageIndex`, a separate mechanism)
  were deliberately left untouched -- unrelated to the eval panel's stage
  pills. Reducer/evalPanel tests updated to match (23 passed). `lint`/
  `tsc --noEmit`/`test`/`build` all green.
- PR2 (evaluation panel) built on `feat/phase9-pr2-eval-panel`, off
  current `main` (which already has PR1's chat-only console merged).
  Backend: `dense_candidates` on `RetrievalResult`/`RetrievalPipeline`
  (`retrieval/base.py`, `retrieval/pipeline.py`) so the pre-rerank pool is
  real data, not a re-derivation; `_candidate_out()` now sends `gold` and
  `content`; the `retrieval` SSE event sends `dense_candidates` alongside
  `candidates`; new `POST /api/verdict` + `log_verdict()`
  (`common/telemetry.py`) mirroring `POST /api/feedback` exactly, logged
  to a separate `verdict-*.jsonl`. Frontend: header view switch, `stage`/
  `openRow`/`view`/`reviewerVerdicts` on `AskState`, `lib/evalPanel.ts`
  (pure chunk-row/gold-metric derivation, unit-tested), new
  `components/EvalPanel.tsx`, `SourceTile` now opens the panel on the
  matching chunk row instead of a new tab (PR1 stopgap, per its own
  "Assumptions" note), header gate indicator fixed to track
  `state.activeId` instead of `.at(-1)`. Not yet committed, pushed, or
  opened as a PR -- waiting on your go-ahead per the merge policy.
  `HANDOFF.md` deleted per its own last line, once PR2 was built.

## Needs your call
- **2026-09-18 — Diff review for PR2, go-ahead to commit/push/open PR**:
  reviewer subagent verdict **ship after fixes**, risk **low-medium**, one
  real finding (the `dense_candidates`/reranker-mutation aliasing bug,
  detailed above under Done — now fixed and reverified, 355 passed).
  Everything else the reviewer checked came back clean: gold-flagging
  unchanged and still leak-free, `POST /api/verdict`/`log_verdict()` exact
  mirrors of the existing feedback pattern, reducer/`activeId` correctness
  verified including the null-`activeId`-on-first-load case, no secrets/
  design-brief-folder/build output staged. Not yet committed, pushed, or
  opened as a PR; waiting on your go-ahead per the merge policy.
- **Docs system update** (reviewer finding from PR1, not a code defect):
  your standing instruction is to update
  `docs/{PROJECT,ARCHITECTURE,DECISIONS,EXPERIMENTS}.md` per milestone
  without being asked; still outstanding. Want that done now, alongside
  PR2, or after PR1 and PR2 both land?
- **Pre-existing bug observed while manually testing PR2, not caused by
  this change**: Groq sometimes emits fullwidth citation brackets
  (`【1】`) in the *final* answer text even though `rag/citations.py`'s
  `_CITATION_RE` already recognizes both bracket styles for citation
  *extraction* (a PR1 fix, see the dated entry above) -- the answer text
  itself is never rewritten to the ASCII form, and the frontend's
  `renderAnswerInline` marker regex (`lib/inline.tsx`) only matches
  `[\d+]`, so those citations render as literal bracket text instead of a
  clickable `SourceTile`. Confirmed the citation is still tracked
  correctly server-side (shows up gold/cited in the eval panel's chunk
  list) -- this is purely a missed-render case for the inline pill,
  observed on roughly half the eval questions tried during manual
  verification. Left out of PR2's scope (not part of the brief, and
  fixing it means touching either the prompt/generation path or widening
  the frontend regex, both outside "build the eval panel") -- flagging so
  it doesn't get mistaken for a PR2 regression later.
- **Left untouched, not part of this PR**: untracked
  `src/rag_eval/providers/llm/literouter.py` +
  `configs/experiments/gen_hybrid_literouter.yaml` — pre-existing,
  unrelated scratch work in the working tree (unregistered provider, half
  finished). Excluded from `git add` again.
