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

## In progress
- **2026-09-18 — Citation fullwidth-bracket render fix**, on its own
  branch `fix/inline-citation-fullwidth-brackets` (off `main`, separate
  from PR2's `feat/phase9-pr2-eval-panel` per your call — one PR per
  logical change). You reported it live via screenshots: `【2】【3】`
  rendering as inert text instead of a clickable source tile. Root cause:
  two separate regexes -- `rag/citations.py`'s `_CITATION_RE` (backend
  extraction/scoring) already matched both `[n]`/`【n】` brackets from the
  PR1 fix, but `frontend/lib/inline.tsx`'s `renderAnswerInline()` (the
  inline-pill *renderer*) was never updated and only matched ASCII `[n]`
  -- so the backend always tracked the citation correctly (gold/cited data
  was right) while the frontend silently failed to render it as a tile.
  Widened `inline.tsx`'s split/marker regexes to `[\[【]\d+[\]】]`,
  mirroring the backend pattern exactly. Added `lib/inline.test.tsx` (6
  cases: ASCII, fullwidth, mixed, unresolved-index, code-span
  non-interference) -- first test file to exercise JSX/component code, so
  also added the `@/` alias (`resolve.alias`) and JSX transform
  (`esbuild.jsx: "automatic"`) to `vitest.config.ts`, both missing
  because no prior test needed them. `npm run lint`/`tsc --noEmit`/
  `test` (11 passed)/`build` all green. Manually verified live in a
  browser against the real API with your exact reported question ("What
  benefit does using asynchronous code provide for web APIs?") -- both
  citations now render as clickable pills. Not yet committed, pushed, or
  opened as a PR; waiting on your go-ahead per the merge policy.

## Needs your call
- **Diff review for PR1** — not yet committed, pushed, or opened as a PR;
  waiting on your go-ahead per the merge policy.
- **Docs system update** (reviewer finding, not a code defect): your
  standing instruction is to update
  `docs/{PROJECT,ARCHITECTURE,DECISIONS,EXPERIMENTS}.md` per milestone
  without being asked; their mtimes predate this Phase 9 work. Want that
  done now, alongside PR1, or after the eval-panel PR lands too?
- Done: the external design brief folder is excluded from this PR and now
  gitignored outright; every mention of its path was scrubbed from
  `docs/plan.md`, `STATUS.md`, and 3 frontend source comments (replaced
  with generic "design brief" wording) so nothing pointing at a
  non-existent repo path ships to GitHub.
- **FYI, not a blocker**: the builder subagent reported a prompt-injection
  attempt inside a Bash tool-output block during its run (a fake
  attribution/URL "system-reminder"). It was ignored as untrusted data; no
  commits were made either way.
- **Left untouched, not part of this PR**: untracked
  `src/rag_eval/providers/llm/literouter.py` +
  `configs/experiments/gen_hybrid_literouter.yaml` — pre-existing,
  unrelated scratch work in the working tree (unregistered provider, half
  finished). Excluded from `git add` for PR1.
