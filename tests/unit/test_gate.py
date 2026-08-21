import json

from rag_eval.config.run_config import RunConfig, ThresholdConfig
from rag_eval.eval.gate import evaluate_gate, load_baseline
from rag_eval.runs.manifest import new_run, write_manifest


def _run(tmp_path, name, *, mrr, thresholds=None, **extra_metrics):
    cfg = RunConfig(name=name)
    if thresholds is not None:
        cfg = cfg.model_copy(update={"eval": cfg.eval.model_copy(update={"thresholds": thresholds})})
    run_dir = new_run(cfg, "c.yaml", runs_root=tmp_path)
    metrics = {
        "docs_synth_v1": {"n": 20, "mrr": mrr, "cis": {"mrr": [mrr - 0.05, mrr + 0.05]}, **extra_metrics}
    }
    return cfg, write_manifest(run_dir, cfg, "c.yaml", metrics=metrics)


def test_below_floor_fails_even_with_no_baseline(tmp_path):
    cfg, cand = _run(tmp_path, "cand", mrr=0.3, thresholds={"mrr": ThresholdConfig(min=0.5)})
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=None)
    assert result.verdict == "FAIL"
    assert result.results[0].note.startswith("0.300 below floor")


def test_missing_baseline_passes_with_note(tmp_path):
    cfg, cand = _run(tmp_path, "cand", mrr=0.6, thresholds={"mrr": ThresholdConfig(min=0.5)})
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=None)
    assert result.verdict == "PASS"
    assert "no pinned baseline" in result.results[0].note


def test_regression_past_tolerance_fails(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.6)
    cfg, cand = _run(
        tmp_path, "cand", mrr=0.55, thresholds={"mrr": ThresholdConfig(min=0.3, regression_tolerance=0.02)}
    )
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=base)
    assert result.verdict == "FAIL"
    assert "below baseline" in result.results[0].note


def test_within_tolerance_passes(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.6)
    cfg, cand = _run(
        tmp_path, "cand", mrr=0.59, thresholds={"mrr": ThresholdConfig(min=0.3, regression_tolerance=0.02)}
    )
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=base)
    assert result.verdict == "PASS"


def test_improvement_past_margin_warns(tmp_path):
    _, base = _run(tmp_path, "base", mrr=0.6)
    cfg, cand = _run(
        tmp_path, "cand", mrr=0.65, thresholds={"mrr": ThresholdConfig(min=0.3, regression_tolerance=0.02)}
    )
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=base)
    assert result.verdict == "WARN"
    assert "consider `rag-eval runs pin`" in result.results[0].note


def test_missing_metric_in_candidate_fails(tmp_path):
    cfg, cand = _run(tmp_path, "cand", mrr=0.6, thresholds={"recall_at_5": ThresholdConfig(min=0.5)})
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=None)
    assert result.verdict == "FAIL"
    assert "missing from docs_synth_v1" in result.results[0].note


def test_baseline_missing_the_metric_passes_with_note(tmp_path):
    # base never recorded recall_at_5; cand did. The floor still applies to
    # cand's value, but there's nothing to regress-check against.
    _, base = _run(tmp_path, "base", mrr=0.6)
    cfg, cand = _run(
        tmp_path, "cand", mrr=0.6, recall_at_5=0.8, thresholds={"recall_at_5": ThresholdConfig(min=0.5)}
    )
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=base)
    assert result.verdict == "PASS"
    assert "baseline has no recall_at_5" in result.results[0].note


def test_overall_verdict_is_worst_of_all_metrics(tmp_path):
    # recall_at_5 WARNs (well past baseline), mrr FAILs the floor -- the
    # overall verdict must be FAIL, not the WARN from the other metric.
    _, base = _run(tmp_path, "base", mrr=0.6, recall_at_5=0.7)
    cfg, cand = _run(
        tmp_path,
        "cand",
        mrr=0.05,
        recall_at_5=0.8,
        thresholds={
            "mrr": ThresholdConfig(min=0.2, regression_tolerance=0.02),
            "recall_at_5": ThresholdConfig(min=0.1, regression_tolerance=0.02),
        },
    )
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=base)
    assert result.verdict == "FAIL"
    verdicts = {r.metric: r.verdict for r in result.results}
    assert verdicts["mrr"] == "FAIL"
    assert verdicts["recall_at_5"] == "WARN"


def test_to_markdown_includes_verdict_and_metric_rows(tmp_path):
    cfg, cand = _run(tmp_path, "cand", mrr=0.6, thresholds={"mrr": ThresholdConfig(min=0.5)})
    result = evaluate_gate(cand, cfg, dataset="docs_synth_v1", baseline=None)
    md = result.to_markdown()
    assert md.splitlines()[0] == "## Eval gate: PASS (docs_synth_v1)"
    assert "| mrr | 0.600 | 0.500 | - | PASS |" in md


def test_load_baseline_missing_file_returns_none(tmp_path):
    assert load_baseline("docs_synth_v1", baseline_path=tmp_path / "baseline.json", runs_root=tmp_path) is None


def test_load_baseline_missing_dataset_key_returns_none(tmp_path):
    pointer = tmp_path / "baseline.json"
    pointer.write_text(json.dumps({"discussions_v2": "some-run"}), encoding="utf-8")
    assert load_baseline("docs_synth_v1", baseline_path=pointer, runs_root=tmp_path) is None


def test_load_baseline_resolves_pinned_run(tmp_path):
    cfg, _ = _run(tmp_path, "base", mrr=0.6)
    run_dir = tmp_path / "0001-base"
    run_dir.mkdir()
    write_manifest(run_dir, cfg, "c.yaml", metrics={"docs_synth_v1": {"n": 1, "mrr": 0.6, "cis": {}}})

    pointer = tmp_path / "baseline.json"
    pointer.write_text(json.dumps({"docs_synth_v1": "0001-base"}), encoding="utf-8")

    baseline = load_baseline("docs_synth_v1", baseline_path=pointer, runs_root=tmp_path)
    assert baseline is not None
    assert baseline.get_metric("docs_synth_v1", "mrr") == 0.6
