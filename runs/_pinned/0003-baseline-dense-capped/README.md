# 0003-baseline-dense-capped

Re-pin of `0002-baseline-dense`, made necessary by Phase 6 (`docs/plan.md`)
landing `RetrievalPipeline` and wiring it into `eval/runner.py`. Same
dense-only retrieval, same `configs/ci.yaml`, same eval sets — but
`retrieval.per_source_caps` (`docs: 4, discussions: 2`) was always present
in the config and was **never actually enforced** by the old
`rag/retriever.py`-based closure. The new pipeline applies it for real.

That means `0002-baseline-dense` and this run measure two different
retrieval behaviors under the same config file — comparing a PR's numbers
against `0002` was a confounded (apples-to-oranges) comparison, not a
signal of real quality change. This run re-establishes a baseline that
reflects caps actually being enforced, so future PRs get gated against
like-for-like behavior. See `runs/baseline.json` for the pointer update.

`git_sha ca32434` (tip of the `phase6/*` PR stack at re-pin time),
`git_dirty: false`.

```
                recall@1  recall@3  recall@5  recall@10   mrr
docs_synth_v1     0.440     0.650     0.717     0.723     0.731   (n=56)
discussions_v2    0.250     0.350     0.421     0.421     0.536   (n=7)
```

Versus `0002-baseline-dense`:

```
                recall@5 delta   recall@10 delta   mrr delta
docs_synth_v1      -0.023           -0.090           -0.015
discussions_v2     -0.000           -0.112           -0.014
```

recall@5 and mrr moved by less than the width of their own bootstrap CIs
(recall@5 CI is `[0.617, 0.826]`, a 0.21-point-wide interval against a
0.023-point delta) -- consistent with the caps mostly not binding for a
single-source-per-query answer. recall@10 moved more: with `docs: 4,
discussions: 2` as the cap, at most 6 chunks total can survive per query
regardless of `candidates_k`, so once dense's own ranking would have
returned more than 6 relevant-looking chunks from one source, recall@10
has no way to reach them anymore. That's the cap doing exactly what it's
for (stopping one source from crowding out the other) -- it's just not
free, and this run is what documents the cost.

Same caveats as `0002-baseline-dense` still apply (n=56 vs the ~127
target, n=7 discussions_v2 CI is uninformative) -- see that README.
