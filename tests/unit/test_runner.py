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
    path = eval_sets_dir / f"{name}.jsonl"
    path.write_text("\n".join(item.model_dump_json() for item in items), encoding="utf-8")


@pytest.fixture
def gold_index():
    return build_gold_index(CHUNKS)


@pytest.fixture
def dataset_dir(tmp_path):
    return tmp_path / "eval_sets"


def _cfg(**overrides) -> RunConfig:
    cfg = RunConfig(name="test-run")
    return cfg.model_copy(update={"eval": cfg.eval.model_copy(update=overrides)})


def test_run_experiment_scores_a_hit_and_a_miss(tmp_path, dataset_dir, gold_index):
    items = [
        EvalItem(id="hit", dataset="d", question="q1", gold_urls=["https://x/#a"]),
        EvalItem(id="miss", dataset="d", question="q2", gold_urls=["https://x/#a"]),
    ]
    _write_dataset(dataset_dir, "d", items)
    cfg = _cfg(datasets=["d"], k_values=[1])

    def retrieve_fn(question, run_cfg, deny_ids):
        if question == "q1":
            return [{"chunk_id": "gold-1", "url": "https://x/#a", "score": 0.9}]
        return [{"chunk_id": "noise-1", "url": "https://x/#b", "score": 0.9}]

    manifest = run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="test-sha",
    )

    assert manifest.metrics["d"]["n"] == 2
    assert manifest.metrics["d"]["recall_at_k"][1] == pytest.approx(0.5)
    assert manifest.corpus_sha == "test-sha"
    assert manifest.dataset_shas["d"]

    rows = (tmp_path / "runs" / manifest.run_id / "retrieval.jsonl").read_text().splitlines()
    parsed = [json.loads(r) for r in rows]
    assert {r["item_id"] for r in parsed} == {"d::hit", "d::miss"}
    hit_row = next(r for r in parsed if r["item_id"] == "d::hit")
    assert hit_row["metrics"]["recall_at_1"] == 1.0
    assert hit_row["retrieved"][0]["rank"] == 1


def test_run_experiment_excludes_items_rejected_by_human_review(tmp_path, dataset_dir, gold_index):
    items = [
        EvalItem(id="good", dataset="d", question="q1", gold_urls=["https://x/#a"]),
        EvalItem(
            id="bad",
            dataset="d",
            question="q2",
            gold_urls=["https://x/#a"],
            verified="no",
            verified_at="t",
        ),
    ]
    _write_dataset(dataset_dir, "d", items)
    cfg = _cfg(datasets=["d"], k_values=[1])

    def retrieve_fn(question, run_cfg, deny_ids):
        return [{"chunk_id": "gold-1", "url": "https://x/#a", "score": 0.9}]

    manifest = run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="test-sha",
    )
    assert manifest.metrics["d"]["n"] == 1

    rows = (tmp_path / "runs" / manifest.run_id / "retrieval.jsonl").read_text().splitlines()
    parsed = [json.loads(r) for r in rows]
    assert {r["item_id"] for r in parsed} == {"d::good"}


def test_self_retrieval_holdout_passes_exclude_ids_as_deny_ids(tmp_path, dataset_dir, gold_index):
    items = [
        EvalItem(
            id="leaky",
            dataset="d",
            question="q1",
            gold_urls=["https://x/#a"],
            exclude_chunk_ids=["gold-1"],
        )
    ]
    _write_dataset(dataset_dir, "d", items)
    cfg = _cfg(datasets=["d"], k_values=[1], self_retrieval="holdout")

    seen_deny_ids = []

    def retrieve_fn(question, run_cfg, deny_ids):
        seen_deny_ids.append(deny_ids)
        return []

    run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="test-sha",
    )
    assert seen_deny_ids == [{"gold-1"}]


def test_self_retrieval_none_ignores_exclude_ids(tmp_path, dataset_dir, gold_index):
    items = [
        EvalItem(
            id="leaky",
            dataset="d",
            question="q1",
            gold_urls=["https://x/#a"],
            exclude_chunk_ids=["gold-1"],
        )
    ]
    _write_dataset(dataset_dir, "d", items)
    cfg = _cfg(datasets=["d"], k_values=[1], self_retrieval="none")

    seen_deny_ids = []

    def retrieve_fn(question, run_cfg, deny_ids):
        seen_deny_ids.append(deny_ids)
        return []

    run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="test-sha",
    )
    assert seen_deny_ids == [set()]


def test_generation_enabled_raises_not_implemented(tmp_path, dataset_dir, gold_index):
    cfg = RunConfig(name="t")
    cfg = cfg.model_copy(update={"generation": cfg.generation.model_copy(update={"enabled": True})})
    with pytest.raises(NotImplementedError, match="generation"):
        run_experiment(
            cfg,
            "c.yaml",
            runs_root=tmp_path / "runs",
            eval_sets_dir=dataset_dir,
            gold_index=gold_index,
            retrieve_fn=lambda q, c, d: [],
        )


def test_separate_index_self_retrieval_raises(tmp_path, dataset_dir, gold_index):
    cfg = _cfg(datasets=["d"], self_retrieval="separate_index")
    with pytest.raises(NotImplementedError, match="separate_index"):
        run_experiment(
            cfg,
            "c.yaml",
            runs_root=tmp_path / "runs",
            eval_sets_dir=dataset_dir,
            gold_index=gold_index,
            retrieve_fn=lambda q, c, d: [],
        )


def test_missing_dataset_file_raises_file_not_found(tmp_path, dataset_dir, gold_index):
    cfg = _cfg(datasets=["nope"])
    with pytest.raises(FileNotFoundError):
        run_experiment(
            cfg,
            "c.yaml",
            runs_root=tmp_path / "runs",
            eval_sets_dir=dataset_dir,
            gold_index=gold_index,
            retrieve_fn=lambda q, c, d: [],
        )


def test_run_experiment_is_deterministic_given_fixed_inputs(tmp_path, dataset_dir, gold_index):
    items = [EvalItem(id="i", dataset="d", question="q1", gold_urls=["https://x/#a"])]
    _write_dataset(dataset_dir, "d", items)
    cfg = _cfg(datasets=["d"], k_values=[1, 5])

    def retrieve_fn(question, run_cfg, deny_ids):
        return [{"chunk_id": "gold-1", "url": "https://x/#a", "score": 0.9}]

    m1 = run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs1",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="s",
    )
    m2 = run_experiment(
        cfg,
        "c.yaml",
        runs_root=tmp_path / "runs2",
        eval_sets_dir=dataset_dir,
        gold_index=gold_index,
        retrieve_fn=retrieve_fn,
        corpus_sha="s",
    )
    assert m1.metrics == m2.metrics
