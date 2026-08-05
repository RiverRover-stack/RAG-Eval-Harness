# 0000-legacy-baseline

The RAGAS scores this project originally shipped with, before the Phase 0–1
audit and rebuild described in `docs/plan.md`. Kept as the honest "before"
artifact, not a run to compare against going forward — it does not have a
`manifest.json` (that format starts in Phase 4) and its numbers are known to
be a measurement artifact, not a retrieval-quality result:

```
faithfulness:      0.2952
answer_relevancy:  0.2365
context_precision: 0.6875
context_recall:    0.3780
```

`ragas_results.csv` here is the original `data/eval_sets/ragas_results.csv`
with its 3 trailing garbage rows removed (two blank rows, and one row where
the aggregate score dict had been accidentally written into the `user_input`
column instead of reported separately). 8 data rows, not the eval set's full
27 — RAGAS only scored the subset it could retrieve *something* for.

See `docs/plan.md`'s Context section for why these numbers don't reflect
retrieval quality: a 30-page ingestion cap that silently excluded every
`tutorial/` page, an eval set where two-thirds of questions have no gold
document in the index at all, and a non-reproducible corpus snapshot.
