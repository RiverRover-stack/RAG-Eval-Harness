# 0001-phase1-corpus-fix

The same 8-row sample as `0000-legacy-baseline`, same generator (`fdm-llama`
via Ollama) and same judge (`RAGAS_JUDGE=ollama`, also `fdm-llama`), re-run
after Phase 1's corpus/chunking fixes (see `docs/plan.md`): the docs corpus
is pinned to a commit SHA and covers all of `tutorial/` instead of a
30-page alphabetical slice, discussions are a frozen snapshot instead of a
live re-fetch, and undersized chunks are merged instead of left as
low-signal fragments. Nothing about the generator, the judge, or the eval
set's queries changed between these two runs.

```
                    legacy (0000)   phase 1 (0001)   delta
faithfulness        0.2952          0.5917           +0.296
answer_relevancy     0.2365          0.2536           +0.017
context_precision   0.6875          0.6562           -0.031
context_recall      0.3780          0.4284           +0.050
```

`context_recall` moved in the right direction but **fell short of the
docs/plan.md Phase 1 target of >=0.65**. Recorded honestly, with the
reasons it likely fell short rather than rounding it up:

1. **The eval set itself is still the old one.** `build_eval_set.py`'s
   query bug (bare discussion title instead of title + question body) was
   fixed in Phase 1, but `data/eval_sets/fastapi_discussions_eval.jsonl`
   (the 27 committed rows used here) was deliberately *not* regenerated --
   the plan assigns real eval-set construction to Phase 4, after the
   chunker is frozen. So this run's retrieval queries never benefited from
   the query fix; that's plausibly most of the shortfall.
2. **One job failed outright.** Row 5's `context_recall` in
   `ragas_results.csv` is blank -- `fdm-llama`'s 4096-token context window
   was exceeded (4180 tokens requested) and RAGAS dropped it from the
   aggregate. The reported 0.4284 is the mean of the other 7 rows, not 8.
   This is a real side effect of Phase 1: the undersized-chunk merge pass
   raised average chunk size (210 -> 244 tokens), making judge-prompt
   overflow on a small local context window more likely.
3. **No self-retrieval guard yet.** This eval set's gold is "the
   discussion's own accepted answer," retrieved across docs+discussions
   with no `deny_ids` exclusion -- exactly what Phase 4's
   `eval.self_retrieval: holdout` and hand-labeled `discussions_v2` are
   built to fix.
4. **n=8, no CI.** Treat the delta as directional, not conclusive.

`faithfulness` nearly doubling is the more trustworthy signal here: full
`tutorial/` coverage plus larger, better-formed chunks means the generator
has real material to ground answers in, where before it was often working
from `advanced/`-only context unrelated to the question.

`ragas_results.csv` here is the untouched output of
`uv run python -m rag_eval.eval.run_ragas` against the Phase 1 index
(987 `fastapi_docs` chunks + 112 `fastapi_discussions` chunks, both from
the pinned `data/corpus/` snapshot).
