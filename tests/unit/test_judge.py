import json

import pytest

from rag_eval.eval.judge import (
    aggregate_judge,
    judge_run,
    load_scored_file,
    merge_judge_into_manifest,
    to_ragas_rows,
)

GEN_ROWS = [
    {
        "item_id": "d::a",
        "question": "how?",
        "answer": "like this [1].",
        "ground_truth": "like this",
        "contexts": ["ctx a1", "ctx a2"],
    },
    {
        "item_id": "d::b",
        "question": "why?",
        "answer": "because [1].",
        "ground_truth": None,
        "contexts": ["ctx b1"],
    },
]


def _make_run_dir(tmp_path, *, judge=None, gen_llm=None):
    run_dir = tmp_path / "runs" / "2026-01-01T00-00-00Z__t__abc123"
    run_dir.mkdir(parents=True)
    config = {
        "eval": {"judge": judge or {"provider": "gemini", "model": "gemini-2.5-flash", "max_workers": 1}},
        "generation": {"llm": gen_llm or {"provider": "groq", "model": "openai/gpt-oss-120b"}},
    }
    (run_dir / "manifest.json").write_text(
        json.dumps({"run_id": run_dir.name, "config": config, "metrics": {}}), encoding="utf-8"
    )
    (run_dir / "generation.jsonl").write_text(
        "\n".join(json.dumps(r) for r in GEN_ROWS) + "\n", encoding="utf-8"
    )
    return run_dir


def test_to_ragas_rows_keeps_no_reference_row_with_empty_string():
    rows = to_ragas_rows(GEN_ROWS)
    assert [r["item_id"] for r in rows] == ["d::a", "d::b"]
    assert rows[0]["ground_truth"] == "like this"
    assert rows[1]["ground_truth"] == ""
    assert rows[0]["contexts"] == ["ctx a1", "ctx a2"]


def test_aggregate_judge_skips_nan_and_reports_n():
    per_item = [
        {"item_id": "a", "faithfulness": 1.0, "context_recall": 0.5},
        {"item_id": "b", "faithfulness": 0.0, "context_recall": float("nan")},
    ]
    agg = aggregate_judge(per_item, bootstrap_n=200)
    assert agg["faithfulness"]["mean"] == pytest.approx(0.5)
    assert agg["faithfulness"]["n"] == 2
    assert agg["context_recall"]["n"] == 1
    assert agg["answer_relevancy"] == {"mean": None, "ci": [None, None], "n": 0}


def test_load_scored_file_jsonl_by_item_id_and_csv_by_order(tmp_path):
    ragas_rows = [{"item_id": "d::a"}, {"item_id": "d::b"}]
    metrics = {m: 0.5 for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}

    jl = tmp_path / "s.jsonl"
    jl.write_text(
        json.dumps({"item_id": "d::b", **metrics}) + "\n" + json.dumps({"item_id": "d::a", **metrics}) + "\n",
        encoding="utf-8",
    )
    assert [r["item_id"] for r in load_scored_file(jl, ragas_rows)] == ["d::b", "d::a"]

    header = "faithfulness,answer_relevancy,context_precision,context_recall"
    csvf = tmp_path / "s.csv"
    csvf.write_text(f"{header}\n0.5,0.5,0.5,0.5\n0.5,0.5,0.5,0.5\n", encoding="utf-8")
    assert [r["item_id"] for r in load_scored_file(csvf, ragas_rows)] == ["d::a", "d::b"]


def test_merge_judge_into_manifest(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    merge_judge_into_manifest(run_dir, {"provider": "gemini", "aggregates": {}})
    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["metrics"]["judge"]["provider"] == "gemini"


def test_judge_run_export_only_writes_portable_jsonl(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    path = judge_run("2026-01-01T00-00-00Z__t__abc123", runs_root=tmp_path / "runs", export_only=True)
    assert path == run_dir / "judge_export.jsonl"
    exported = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["item_id"] for r in exported] == ["d::a", "d::b"]
    assert not (run_dir / "judge.jsonl").exists()


def test_judge_run_score_only_merges_without_calling_judge(tmp_path):
    run_dir = _make_run_dir(tmp_path)
    metrics = {m: 0.5 for m in ("faithfulness", "answer_relevancy", "context_precision", "context_recall")}
    scored = tmp_path / "scored.jsonl"
    scored.write_text(
        "\n".join(json.dumps({"item_id": r["item_id"], **metrics}) for r in GEN_ROWS) + "\n",
        encoding="utf-8",
    )

    path = judge_run(
        "2026-01-01T00-00-00Z__t__abc123", runs_root=tmp_path / "runs", score_only=scored
    )
    assert path == run_dir / "judge.jsonl"
    judged = [json.loads(line) for line in path.read_text().splitlines()]
    assert [r["item_id"] for r in judged] == ["d::a", "d::b"]

    manifest = json.loads((run_dir / "manifest.json").read_text())
    block = manifest["metrics"]["judge"]
    assert block["scored_offline"] is True
    assert block["strictness"] == 3
    assert block["aggregates"]["faithfulness"]["mean"] == pytest.approx(0.5)


def test_judge_run_rejects_generator_own_model(tmp_path):
    same = {"provider": "groq", "model": "openai/gpt-oss-120b"}
    _make_run_dir(tmp_path, judge={**same, "max_workers": 1}, gen_llm=same)
    with pytest.raises(ValueError, match="must not be the generator"):
        judge_run(
            "2026-01-01T00-00-00Z__t__abc123",
            runs_root=tmp_path / "runs",
            score_only=tmp_path / "unused.jsonl",
        )
