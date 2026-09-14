# Phase 8 — Judge depth

Two judge stages sit on top of a run's `generation.jsonl` (written by
`eval/generate.py` when `generation.enabled`). Both are always hosted and
never the generator's own model — `judge_run` / `rubric_run` assert that and
record the judge in the manifest.

| command | reads | writes | what it is |
|---|---|---|---|
| `rag-eval eval judge <run_id>` | `generation.jsonl` | `judge.jsonl`, `manifest.metrics.judge` | RAGAS: faithfulness, answer_relevancy (strictness 3), context_precision, context_recall — per item + bootstrap-CI aggregates |
| `rag-eval eval rubric <run_id>` | `generation.jsonl` | `rubric.jsonl`, `manifest.metrics.rubric` | one structured judge call per item scoring 5 dimensions 1–5 with a justification, **plus a deterministic `citation_accuracy`** from the embedding-support scorer as a cross-check |

## Kaggle GPU offload

`judge` splits the same way the old `run_ragas.py` did, now anchored to run
artifacts:

```
rag-eval eval judge <run_id> --export-only          # -> runs/<id>/judge_export.jsonl
# upload judge_export.jsonl as a Kaggle Dataset, run notebooks/kaggle_judge_eval.py
rag-eval eval judge <run_id> --score-only judge_scored.jsonl   # merge back, no judge call
```

## Running the Phase 8 verification

The plan's Verify step: judge two runs that differ **only** in the
generator, check that the rubric and RAGAS move in the same direction, and
report the correlation between the judge's `citation_accuracy` and the
deterministic one.

```bash
# needs GROQ_API_KEY + GEMINI_API_KEY, and a local Ollama running fdm-llama
# (see Modelfile.fdm-llama at the repo root -- ollama create fdm-llama -f
# Modelfile.fdm-llama) + the RAGAS judge embeddings (nomic-embed-text).
rag-eval eval run --config configs/experiments/gen_hybrid_ollama.yaml
rag-eval eval run --config configs/experiments/gen_hybrid_groq.yaml

rag-eval eval judge  <ollama_run_id>
rag-eval eval judge  <groq_run_id>
rag-eval eval rubric <ollama_run_id>
rag-eval eval rubric <groq_run_id>

rag-eval leaderboard          # generator upgrade is its own row
```

The two configs share a retrieval hash and differ by one field
(`generation.llm`), so `compare_runs()` returns a clean generator-only
delta rather than `CONFOUNDED`.

### Results

_To be filled in once the runs above have been executed with live keys and
pinned to `runs/_pinned/`._

| | RAGAS faithfulness | rubric correctness (1–5) | rubric hallucination (1–5) | judge citation_accuracy (1–5) | deterministic citation_accuracy (0–1) |
|---|---|---|---|---|---|
| `exp-gen-ollama` (fdm-llama) | | | | | |
| `exp-gen-groq` (gpt-oss-120b) | | | | | |

- **Directional agreement**: does rubric `correctness` move the same way as
  RAGAS `faithfulness` between the two arms?
- **Judge audit**: Pearson r between the judge's `citation_accuracy` and the
  deterministic fraction (`manifest.metrics.rubric.judge_vs_deterministic_citation_accuracy_r`).
  A low r means the LLM judge and the embedding scorer disagree about which
  citations are sound — worth writing up either way.
