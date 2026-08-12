import json

from typer.testing import CliRunner

from rag_eval.cli import app
from rag_eval.config.run_config import RunConfig
from rag_eval.eval.gold import EvalItem
from rag_eval.runs.manifest import new_run, write_manifest

runner = CliRunner()


def _write_dataset(eval_sets_dir, name, items):
    eval_sets_dir.mkdir(parents=True, exist_ok=True)
    path = eval_sets_dir / f"{name}.jsonl"
    path.write_text("\n".join(i.model_dump_json() for i in items), encoding="utf-8")
    return path


def test_top_level_help_lists_subcommands():
    result = runner.invoke(app, ["--help"])
    assert result.exit_code == 0
    assert "eval" in result.output
    assert "runs" in result.output
    assert "leaderboard" in result.output


def test_eval_review_updates_and_persists_decisions(tmp_path):
    eval_sets_dir = tmp_path / "eval_sets"
    items = [
        EvalItem(id="1", dataset="d", question="q1", gold_urls=["https://x/tutorial/a/#a"]),
        EvalItem(id="2", dataset="d", question="q2", gold_urls=["https://x/tutorial/b/#b"]),
    ]
    path = _write_dataset(eval_sets_dir, "d", items)

    result = runner.invoke(
        app,
        ["eval", "review", "--dataset", "d", "--n", "2", "--eval-sets-dir", str(eval_sets_dir)],
        input="y\nn\n",
    )
    assert result.exit_code == 0, result.output
    assert "2/2 items verified" in result.output

    saved = [json.loads(line) for line in path.read_text().splitlines()]
    verdicts = {row["id"]: row["verified"] for row in saved}
    assert verdicts == {"1": "yes", "2": "no"}


def test_eval_review_resumes_without_reprompting_verified_items(tmp_path):
    eval_sets_dir = tmp_path / "eval_sets"
    items = [
        EvalItem(
            id="1",
            dataset="d",
            question="q1",
            gold_urls=["https://x/tutorial/a/#a"],
            verified="yes",
            verified_at="earlier",
        ),
        EvalItem(id="2", dataset="d", question="q2", gold_urls=["https://x/tutorial/b/#b"]),
    ]
    path = _write_dataset(eval_sets_dir, "d", items)

    result = runner.invoke(
        app,
        ["eval", "review", "--dataset", "d", "--n", "2", "--eval-sets-dir", str(eval_sets_dir)],
        input="n\n",  # only item 2 should prompt
    )
    assert result.exit_code == 0, result.output
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    by_id = {row["id"]: row for row in saved}
    assert by_id["1"]["verified_at"] == "earlier"
    assert by_id["2"]["verified"] == "no"


def test_eval_label_records_gold_urls(tmp_path):
    eval_sets_dir = tmp_path / "eval_sets"
    items = [EvalItem(id="1", dataset="discussions_v2", question="how do I do X?", gold_urls=[])]
    path = _write_dataset(eval_sets_dir, "discussions_v2", items)

    result = runner.invoke(
        app,
        [
            "eval",
            "label",
            "--dataset",
            "discussions_v2",
            "--eval-sets-dir",
            str(eval_sets_dir),
            "--no-show-candidates",
        ],
        input="https://x/#a\n",
    )
    assert result.exit_code == 0, result.output
    saved = [json.loads(line) for line in path.read_text().splitlines()]
    assert saved[0]["gold_urls"] == ["https://x/#a"]
    assert saved[0]["verified"] == "yes"


def _cfg(name: str) -> RunConfig:
    return RunConfig(name=name)


def test_runs_list_prints_each_run(tmp_path):
    cfg = _cfg("base")
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    write_manifest(run_dir, cfg, "c.yaml", metrics={})

    result = runner.invoke(app, ["runs", "list", "--runs-root", str(tmp_path)])
    assert result.exit_code == 0
    assert run_dir.name in result.output


def test_runs_pin_copies_the_run(tmp_path):
    cfg = _cfg("base")
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    write_manifest(run_dir, cfg, "c.yaml", metrics={})

    result = runner.invoke(
        app, ["runs", "pin", run_dir.name, "--name", "0001-first", "--runs-root", str(tmp_path)]
    )
    assert result.exit_code == 0, result.output
    assert (tmp_path / "_pinned" / "0001-first" / "manifest.json").exists()


def test_leaderboard_prints_requested_metrics(tmp_path):
    cfg = _cfg("base")
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    write_manifest(
        run_dir, cfg, "c.yaml", metrics={"docs_synth_v1": {"n": 10, "mrr": 0.6, "cis": {}}}
    )

    result = runner.invoke(
        app,
        [
            "leaderboard",
            "--dataset",
            "docs_synth_v1",
            "--metric",
            "mrr",
            "--runs-root",
            str(tmp_path),
        ],
    )
    assert result.exit_code == 0, result.output
    assert "mrr=0.600" in result.output
