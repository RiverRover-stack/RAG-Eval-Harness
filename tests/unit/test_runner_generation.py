import json

import pytest

from rag_eval.config.run_config import RunConfig
from rag_eval.eval.gold import EvalItem, build_gold_index
from rag_eval.eval.runner import run_experiment

CHUNKS = [
    {"id": "gold-1", "document": "...", "metadata": {"url": "https://x/#a"}},
    {"id": "noise-1", "document": "...", "metadata": {"url": "https://x/#b"}},
]


def _write_dataset(eval_sets_dir, name, items):
    eval_sets_dir.mkdir(parents=True, exist_ok=True)
    (eval_sets_dir / f"{name}.jsonl").write_text(
        "\n".join(item.model_dump_json() for item in items), encoding="utf-8"
    )


@pytest.fixture
def gold_index():
    return build_gold_index(CHUNKS)


def _retrieve_fn(question, run_cfg, deny_ids):
    return [{"chunk_id": "gold-1", "url": "https://x/#a", "score": 0.9}]


def _cfg(generation_enabled: bool) -> RunConfig:
    cfg = RunConfig(name="gen-run")
    cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"datasets": ["d"], "k_values": [1]})})
    return cfg.model_copy(
        update={"generation": cfg.generation.model_copy(update={"enabled": generation_enabled})}
    )


def test_generation_jsonl_written_when_enabled(tmp_path, gold_index):
    dataset_dir = tmp_path / "eval_sets"
    _write_dataset(
        dataset_dir,
        "d",
        [EvalItem(id="i1", dataset="d", question="q1", ground_truth="gt1", gold_urls=["https://x/#a"])],
    )

    calls: list[str] = []

    def generate_fn(item, run_cfg):
        calls.append(item.id)
        return {"item_id": f"{item.dataset}::{item.id}", "answer": "A [1].", "abstained": False}

    manifest = run_experiment(
        _cfg(generation_enabled=True),
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=_retrieve_fn,
        generate_fn=generate_fn,
        corpus_sha="test-sha",
    )

    assert calls == ["i1"]
    gen_path = tmp_path / "runs" / manifest.run_id / "generation.jsonl"
    rows = [json.loads(r) for r in gen_path.read_text().splitlines()]
    assert rows == [{"item_id": "d::i1", "answer": "A [1].", "abstained": False}]
    assert manifest.timings["generation_items"] == 1
    assert manifest.generation_hash


def test_no_generation_jsonl_and_fn_untouched_when_disabled(tmp_path, gold_index):
    dataset_dir = tmp_path / "eval_sets"
    _write_dataset(
        dataset_dir, "d", [EvalItem(id="i1", dataset="d", question="q1", gold_urls=["https://x/#a"])]
    )

    def generate_fn(item, run_cfg):
        raise AssertionError("generator must not run when generation.enabled is false")

    manifest = run_experiment(
        _cfg(generation_enabled=False),
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=_retrieve_fn,
        generate_fn=generate_fn,
        corpus_sha="test-sha",
    )

    assert not (tmp_path / "runs" / manifest.run_id / "generation.jsonl").exists()
    assert "generation_seconds" not in manifest.timings
