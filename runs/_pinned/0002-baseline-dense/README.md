# 0002-baseline-dense

The first pinned run produced by the Phase 4 harness itself (`rag-eval eval
run --config configs/baseline.yaml`), against the current committed corpus
(987 `fastapi_docs` + 112 `fastapi_discussions` chunks) and eval sets. This
is the baseline `eval/gate.py` (Phase 5) will compare PRs against — see
`runs/baseline.json` for the dataset → run-id pointer.

Dense-only retrieval, generation disabled, `self_retrieval: holdout`. Clean
checkout (`git_sha c501fc2`, `git_dirty: false`).

```
                recall@1  recall@3  recall@5  recall@10   mrr
docs_synth_v1     0.440     0.650     0.740     0.812     0.746   (n=56)
discussions_v2    0.250     0.350     0.421     0.533     0.550   (n=7)
```

Caveats, recorded rather than smoothed over:

1. **`docs_synth_v1` is n=56, not the ~127 the plan targets.** The dataset
   is mid-build (generation + filtering + the human review pass in
   `docs/plan.md` Phase 4 aren't finished). `recall@5 = 0.740` clears the
   `configs/_base.yaml` threshold (`min: 0.70`) today, but the threshold was
   set anticipating the fuller set — re-pin once `docs_synth_v1` grows,
   since a shrinking/changing dataset under a fixed absolute floor is
   exactly the kind of drift the "pinning is a reviewable commit" rule
   exists to catch.
2. **`discussions_v2` is n=7, not ~16.** One gold URL
   (`https://fastapi.tiangolo.com/release-notes/`) doesn't resolve to any
   chunk in the current corpus and is silently excluded from every
   aggregate (items with empty gold are dropped — see
   `retrieval_metrics.aggregate`). At n=7 the recall@5 CI is
   `[0.143, 0.714]` — essentially uninformative. Per `docs/plan.md` C3,
   the CI gate does **not** gate on this dataset for exactly this reason;
   it's a directional counterweight, not a pass/fail bar.
