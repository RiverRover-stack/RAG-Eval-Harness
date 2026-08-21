"""CI eval gate: turns one run's metrics into a FAIL / WARN / PASS verdict
against `RunConfig.eval.thresholds` and the pinned baseline in
`runs/baseline.json` (docs/plan.md Phase 5). Zero API quota, zero Ollama --
see configs/ci.yaml.

`evaluate_gate` takes an already-loaded baseline `RunManifest | None` rather
than reading `runs/baseline.json` itself, so tests can exercise every
verdict with plain manifests (CLAUDE.md: prefer constructor injection over
patching). `load_baseline` is the thin I/O wrapper the CLI/CI job uses.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Literal

from rag_eval.config.run_config import RunConfig
from rag_eval.runs.manifest import DEFAULT_RUNS_ROOT, RunManifest, load_run

Verdict = Literal["FAIL", "WARN", "PASS"]

DEFAULT_BASELINE_POINTER = Path("runs/baseline.json")
WARN_MARGIN = 0.02


@dataclass
class MetricResult:
    metric: str
    verdict: Verdict
    value: float | None
    floor: float
    baseline_value: float | None
    note: str


@dataclass
class GateResult:
    dataset: str
    results: list[MetricResult]

    @property
    def verdict(self) -> Verdict:
        if any(r.verdict == "FAIL" for r in self.results):
            return "FAIL"
        if any(r.verdict == "WARN" for r in self.results):
            return "WARN"
        return "PASS"

    def to_markdown(self) -> str:
        lines = [
            f"## Eval gate: {self.verdict} ({self.dataset})",
            "",
            "| metric | value | floor | baseline | verdict | note |",
            "|---|---|---|---|---|---|",
        ]
        for r in self.results:
            value = f"{r.value:.3f}" if r.value is not None else "-"
            baseline = f"{r.baseline_value:.3f}" if r.baseline_value is not None else "-"
            lines.append(f"| {r.metric} | {value} | {r.floor:.3f} | {baseline} | {r.verdict} | {r.note} |")
        return "\n".join(lines)


def evaluate_gate(
    manifest: RunManifest,
    cfg: RunConfig,
    *,
    dataset: str,
    baseline: RunManifest | None,
) -> GateResult:
    results = []
    for metric, threshold in cfg.eval.thresholds.items():
        value = manifest.get_metric(dataset, metric)
        if value is None:
            results.append(
                MetricResult(metric, "FAIL", None, threshold.min, None, f"{metric} missing from {dataset}")
            )
            continue
        if value < threshold.min:
            results.append(
                MetricResult(
                    metric, "FAIL", value, threshold.min, None, f"{value:.3f} below floor {threshold.min:.3f}"
                )
            )
            continue

        baseline_value = baseline.get_metric(dataset, metric) if baseline is not None else None
        if baseline_value is None:
            note = "no pinned baseline yet" if baseline is None else f"baseline has no {metric} for {dataset}"
            results.append(MetricResult(metric, "PASS", value, threshold.min, None, note))
            continue

        if value < baseline_value - threshold.regression_tolerance:
            results.append(
                MetricResult(
                    metric,
                    "FAIL",
                    value,
                    threshold.min,
                    baseline_value,
                    f"{value:.3f} below baseline {baseline_value:.3f} - tolerance {threshold.regression_tolerance:.3f}",
                )
            )
        elif value > baseline_value + WARN_MARGIN:
            results.append(
                MetricResult(
                    metric,
                    "WARN",
                    value,
                    threshold.min,
                    baseline_value,
                    f"{value:.3f} beats baseline {baseline_value:.3f} by > {WARN_MARGIN} -- consider `rag-eval runs pin`",
                )
            )
        else:
            results.append(
                MetricResult(metric, "PASS", value, threshold.min, baseline_value, "within tolerance of baseline")
            )
    return GateResult(dataset=dataset, results=results)


def load_baseline(
    dataset: str,
    *,
    baseline_path: Path = DEFAULT_BASELINE_POINTER,
    runs_root: Path = DEFAULT_RUNS_ROOT,
) -> RunManifest | None:
    if not baseline_path.exists():
        return None
    pointer = json.loads(baseline_path.read_text(encoding="utf-8"))
    run_id = pointer.get(dataset)
    if run_id is None:
        return None
    return load_run(run_id, runs_root=runs_root)
