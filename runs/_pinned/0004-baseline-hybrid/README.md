# 0004-baseline-hybrid

`configs/ci.yaml` moved off dense-only: `retrieval.bm25.enabled: true`
(dense + BM25, fused with RRF), matching what the ablation grid in
`configs/experiments/hybrid.yaml` already measured. Reranker stays off --
its ONNX model cache isn't wired into the `eval-gate` workflow yet, so
enabling it would add a network download to every PR run.

This is the baseline `eval/gate.py` compares PRs against from this commit
on (see `runs/baseline.json`); supersedes `0003-baseline-dense-capped` for
that purpose, which stays pinned as the dense-only historical record.

`git_sha f7371e9`, `git_dirty: false`.

```
                recall@1  recall@3  recall@5  recall@10   mrr
docs_synth_v1     0.507     0.688     0.727     0.727     0.787   (n=56)
discussions_v2    0.179     0.326     0.326     0.326     0.452   (n=7)
```

Versus `0003-baseline-dense-capped` (dense-only, same config otherwise):

```
                recall@5 delta   recall@10 delta   mrr delta
docs_synth_v1      +0.010           +0.004           +0.056
discussions_v2     -0.095           -0.095           -0.084
```

`docs_synth_v1`'s recall@5/recall@10 deltas are well inside their own
bootstrap CI width -- not distinguishable from noise at n=56. MRR's climb
(+0.056) is the more consistent signal, matching the full ablation grid in
PR #26, where MRR rose monotonically across every added stage while
recall@5 stayed flat within noise.

`discussions_v2` moved down, but at n=7 its CI (`[0.107, 0.576]` for
recall@5) is wide enough that this single-digit-item set can't
meaningfully support any directional claim either way -- this dataset
does not gate CI for exactly this reason (`docs/plan.md` C3).

Same corpus-coverage/eval-set-size caveats as `0002`/`0003` apply -- see
those READMEs.
