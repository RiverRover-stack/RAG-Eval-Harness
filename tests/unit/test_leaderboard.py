import json

import pytest

from rag_eval.config.run_config import RunConfig
from rag_eval.runs.leaderboard import build_leaderboard, compare_runs, regressed_items
from rag_eval.runs.manifest import new_run, write_manifest


def _run(tmp_path, name, *, mrr, gen_model="llama-3.3-70b-versatile", top_k=5):
    cfg = RunConfig(name=name)
    cfg = cfg.model_copy(
        update={
            "retrieval": cfg.retrieval.model_copy(update={"top_k": top_k}),
            "generation": cfg.generation.model_copy(
                update={"llm": cfg.generation.llm.model_copy(update={"model": gen_model})}
            ),
        }
    )
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    metrics = {"docs_synth_v1": {"n": 20, "mrr": mrr, "cis": {"mrr": [mrr - 0.05, mrr + 0.05]}}}
    manifest = write_manifest(run_dir, cfg, "c.yaml", metrics=metrics)
    return run_dir, manifest


def test_build_leaderboard_reports_requested_dataset_only(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.5)
    rows = build_leaderboard([base], dataset="docs_synth_v1", metrics=["mrr"])
    assert rows[0]["mrr"] == 0.5
    assert rows[0]["mrr_ci"] == [0.45, 0.55]
    assert rows[0]["n"] == 20


def test_build_leaderboard_missing_dataset_is_none_not_a_crash(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.5)
    rows = build_leaderboard([base], dataset="discussions_v2", metrics=["mrr"])
    assert rows[0]["mrr"] is None
    assert rows[0]["n"] == 0


def test_compare_runs_same_generator_returns_deltas(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.5)
    _, cand = _run(tmp_path, "cand", mrr=0.65)
    result = compare_runs(base, cand, dataset="docs_synth_v1", metrics=["mrr"])
    assert result != "CONFOUNDED"
    assert result["mrr"]["delta"] == pytest.approx(0.15)


def test_compare_runs_different_generator_is_confounded(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.5, gen_model="llama-3.3-70b-versatile")
    _, cand = _run(tmp_path, "cand", mrr=0.65, gen_model="a-different-model")
    result = compare_runs(base, cand, dataset="docs_synth_v1")
    assert result == "CONFOUNDED"


def test_compare_runs_confounded_even_when_generation_disabled(tmp_path):
    # generation.enabled defaults to False for both, but the model field
    # still differs underneath it -- the frozen-generator check is on the
    # hash, not on whether generation actually ran.
    _, base = _run(tmp_path, "base", mrr=0.5, gen_model="model-a")
    _, cand = _run(tmp_path, "cand", mrr=0.6, gen_model="model-b")
    assert compare_runs(base, cand, dataset="docs_synth_v1") == "CONFOUNDED"


def _write_retrieval_jsonl(run_dir, rows):
    with open(run_dir / "retrieval.jsonl", "w", encoding="utf-8") as f:
        f.writelines(json.dumps(row) + "\n" for row in rows)


def test_regressed_items_flags_only_items_that_got_worse(tmp_path):
    base_dir, _ = _run(tmp_path, "base", mrr=0.5)
    cand_dir, _ = _run(tmp_path, "cand", mrr=0.5)

    _write_retrieval_jsonl(
        base_dir,
        [
            {
                "item_id": "q1",
                "question": "How do I do X?",
                "gold_chunk_ids": ["c1"],
                "retrieved": [{"chunk_id": "c1", "rank": 1}],
                "metrics": {"recall_at_5": 1.0},
            },
            {
                "item_id": "q2",
                "question": "How do I do Y?",
                "gold_chunk_ids": ["c2"],
                "retrieved": [{"chunk_id": "c2", "rank": 1}],
                "metrics": {"recall_at_5": 1.0},
            },
        ],
    )
    _write_retrieval_jsonl(
        cand_dir,
        [
            {
                "item_id": "q1",
                "question": "How do I do X?",
                "gold_chunk_ids": ["c1"],
                "retrieved": [{"chunk_id": "c1", "rank": 14}],
                "metrics": {"recall_at_5": 0.0},
            },
            {
                "item_id": "q2",
                "question": "How do I do Y?",
                "gold_chunk_ids": ["c2"],
                "retrieved": [{"chunk_id": "c2", "rank": 1}],
                "metrics": {"recall_at_5": 1.0},
            },
        ],
    )

    regressions = regressed_items(base_dir, cand_dir, metric="recall_at_5")
    assert len(regressions) == 1
    assert regressions[0]["item_id"] == "q1"
    assert regressions[0]["base_gold_rank"] == 1
    assert regressions[0]["cand_gold_rank"] == 14


def test_regressed_items_empty_when_no_retrieval_jsonl(tmp_path):
    base_dir, _ = _run(tmp_path, "base", mrr=0.5)
    cand_dir, _ = _run(tmp_path, "cand", mrr=0.5)
    assert regressed_items(base_dir, cand_dir, metric="recall_at_5") == []
