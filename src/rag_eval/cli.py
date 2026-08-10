"""`rag-eval` -- the command-line surface over config/eval/runs. typer was a
declared-but-unused dependency until this file (docs/plan.md Phase 4); it
finally earns its place.
"""

from __future__ import annotations

from pathlib import Path

import typer

from rag_eval.config.run_config import load_run_config
from rag_eval.eval.datasets import DEFAULT_EVAL_SETS_DIR, dataset_path, load_dataset
from rag_eval.eval.gold import EvalItem
from rag_eval.eval.review import CandidateFn, run_label_session, run_review_session
from rag_eval.eval.runner import run_experiment
from rag_eval.runs.leaderboard import DEFAULT_METRICS, build_leaderboard
from rag_eval.runs.manifest import DEFAULT_RUNS_ROOT, list_runs, pin_run

app = typer.Typer(help="rag-eval: config-driven retrieval eval harness")
eval_app = typer.Typer(help="build, run, and review eval sets")
runs_app = typer.Typer(help="inspect and pin run artifacts")
app.add_typer(eval_app, name="eval")
app.add_typer(runs_app, name="runs")


def _write_dataset_jsonl(items: list[EvalItem], path: Path) -> None:
    path.write_text(
        "\n".join(item.model_dump_json() for item in items) + "\n", encoding="utf-8"
    )


def _docs_candidate_fn(top_k: int = 10) -> CandidateFn:
    """Top-k docs sections for the current question, shown in `eval label`
    purely as a shortlist to speed up hand-labeling -- never taken as the
    answer, the human still has to confirm or override it."""
    from rag_eval.providers import get_embedder
    from rag_eval.rag.vector_store import DOCS_SOURCE
    from rag_eval.rag.vector_store import query as vector_query

    embedder = get_embedder()

    def candidate_fn(question: str) -> list[str]:
        vec = embedder.embed_query(question)
        hits = vector_query(vec, DOCS_SOURCE, embedder, k=top_k)
        return [h["metadata"].get("url", "") for h in hits]

    return candidate_fn


@eval_app.command("run")
def eval_run(
    config: Path = typer.Option(..., "--config", help="path to a RunConfig yaml"),
    set_: list[str] = typer.Option([], "--set", help="override, e.g. retrieval.top_k=10"),
    runs_root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--runs-root"),
) -> None:
    cfg = load_run_config(config, overrides=set_)
    manifest = run_experiment(cfg, config, runs_root=runs_root)

    typer.echo(f"run_id: {manifest.run_id}")
    for dataset, block in manifest.metrics.items():
        typer.echo(f"  {dataset}  (n={block['n']})")
        cis = block.get("cis", {})
        for k, value in block.get("recall_at_k", {}).items():
            lo, hi = cis.get(f"recall_at_{k}", (value, value))
            typer.echo(f"    recall@{k}: {value:.3f}  [{lo:.3f}, {hi:.3f}]")
        mrr = block.get("mrr", 0.0)
        mrr_lo, mrr_hi = cis.get("mrr", (mrr, mrr))
        typer.echo(f"    mrr: {mrr:.3f}  [{mrr_lo:.3f}, {mrr_hi:.3f}]")


@eval_app.command("review")
def eval_review(
    dataset: str = typer.Option(..., "--dataset"),
    n: int = typer.Option(50, "--n"),
    seed: int = typer.Option(0, "--seed"),
    eval_sets_dir: Path = typer.Option(DEFAULT_EVAL_SETS_DIR, "--eval-sets-dir"),
) -> None:
    items = load_dataset(dataset, eval_sets_dir)
    path = dataset_path(dataset, eval_sets_dir)

    def save(all_items: list[EvalItem]) -> None:
        _write_dataset_jsonl(all_items, path)

    updated = run_review_session(items, n, seed=seed, print_fn=typer.echo, save_fn=save)
    reviewed = sum(1 for item in updated if item.verified is not None)
    typer.echo(f"{reviewed}/{len(updated)} items verified in {dataset} so far")


@eval_app.command("label")
def eval_label(
    dataset: str = typer.Option(..., "--dataset"),
    eval_sets_dir: Path = typer.Option(DEFAULT_EVAL_SETS_DIR, "--eval-sets-dir"),
    show_candidates: bool = typer.Option(True, help="show retrieval's own top-10 as a shortlist"),
) -> None:
    items = load_dataset(dataset, eval_sets_dir)
    path = dataset_path(dataset, eval_sets_dir)

    def save(all_items: list[EvalItem]) -> None:
        _write_dataset_jsonl(all_items, path)

    candidate_fn = _docs_candidate_fn() if show_candidates else None
    updated = run_label_session(items, candidate_fn=candidate_fn, print_fn=typer.echo, save_fn=save)
    labeled = sum(1 for item in updated if item.verified is not None)
    typer.echo(f"{labeled}/{len(updated)} items labeled in {dataset} so far")


@runs_app.command("list")
def runs_list_cmd(runs_root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--runs-root")) -> None:
    for run in list_runs(runs_root=runs_root, include_pinned=True):
        dirty = "*" if run.git_dirty else ""
        typer.echo(f"{run.run_id}  {run.name}  cfg={run.config_hash}  git={run.git_sha[:8]}{dirty}")


@runs_app.command("pin")
def runs_pin_cmd(
    run_id: str,
    name: str = typer.Option(None, "--name"),
    runs_root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--runs-root"),
) -> None:
    dest = pin_run(run_id, runs_root=runs_root, name=name)
    typer.echo(f"pinned -> {dest}")


@app.command("leaderboard")
def leaderboard_cmd(
    dataset: str = typer.Option(..., "--dataset"),
    metric: list[str] = typer.Option(list(DEFAULT_METRICS), "--metric"),
    runs_root: Path = typer.Option(DEFAULT_RUNS_ROOT, "--runs-root"),
) -> None:
    runs = list_runs(runs_root=runs_root, include_pinned=True)
    rows = build_leaderboard(runs, dataset=dataset, metrics=metric)
    for row in rows:
        parts = [row["run_id"], f"n={row['n']}"]
        for m in metric:
            value = row.get(m)
            parts.append(f"{m}={value:.3f}" if isinstance(value, int | float) else f"{m}=-")
        typer.echo("  ".join(parts))


if __name__ == "__main__":
    app()
