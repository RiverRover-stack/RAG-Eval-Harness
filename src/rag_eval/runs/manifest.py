"""Run artifacts: every `rag-eval eval run` writes a self-contained
directory under `runs/` that a later run can be reproduced from without
touching this session's memory -- git sha, resolved config, split hashes,
dataset checksums, and the metrics themselves all live in one manifest.json
(docs/plan.md C4).
"""

from __future__ import annotations

import json
import shutil
import subprocess
from dataclasses import asdict, dataclass, field
from datetime import UTC, datetime
from pathlib import Path

import yaml

from rag_eval.config.run_config import RunConfig, config_hash, split_hashes

DEFAULT_RUNS_ROOT = Path("runs")
PINNED_DIRNAME = "_pinned"


def _git_info(cwd: Path | None = None) -> tuple[str, bool]:
    """(sha, is_dirty). Falls back to ("unknown", True) outside a git
    checkout -- a run manifest should say so rather than fail the run."""
    try:
        sha = (
            subprocess.run(
                ["git", "rev-parse", "HEAD"],
                cwd=cwd,
                capture_output=True,
                text=True,
                check=True,
            )
            .stdout.strip()
        )
        status = subprocess.run(
            ["git", "status", "--porcelain"],
            cwd=cwd,
            capture_output=True,
            text=True,
            check=True,
        ).stdout
        return sha, bool(status.strip())
    except (subprocess.CalledProcessError, FileNotFoundError, OSError):
        return "unknown", True


def _slugify(name: str) -> str:
    return "".join(c if c.isalnum() or c in "-_." else "-" for c in name)


@dataclass
class RunManifest:
    run_id: str
    name: str
    created_at: str
    git_sha: str
    git_dirty: bool
    config_path: str
    config: dict
    config_hash: str
    retrieval_hash: str
    generation_hash: str
    corpus_sha: str = ""
    collection_names: dict[str, str] = field(default_factory=dict)
    dataset_shas: dict[str, str] = field(default_factory=dict)
    metrics: dict = field(default_factory=dict)
    timings: dict = field(default_factory=dict)
    cost: dict | None = None

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, data: dict) -> RunManifest:
        return cls(**data)

    def get_metric(self, dataset: str, key: str) -> float | None:
        """metrics[dataset] is an AggregateMetrics.to_dict() blob; `key` can
        be a flat field ("mrr") or "recall_at_5"-style, which looks inside
        the per-k dicts."""
        block = self.metrics.get(dataset)
        if not block:
            return None
        if key in block and isinstance(block[key], int | float):
            return block[key]
        for per_k_field in ("recall_at_k", "precision_at_k", "hit_rate_at_k", "ndcg_at_k"):
            prefix = per_k_field.replace("_k", "_")
            if key.startswith(prefix):
                k = key[len(prefix) :]
                try:
                    return block[per_k_field][k]
                except KeyError:
                    try:
                        return block[per_k_field][int(k)]
                    except (KeyError, ValueError):
                        return None
        return None


def _run_dir_name(cfg: RunConfig, timestamp: datetime) -> str:
    ts = timestamp.strftime("%Y-%m-%dT%H-%M-%SZ")
    return f"{ts}__{_slugify(cfg.name)}__{config_hash(cfg)[:6]}"


def new_run(
    cfg: RunConfig,
    config_path: str | Path,
    *,
    runs_root: Path = DEFAULT_RUNS_ROOT,
    timestamp: datetime | None = None,
) -> Path:
    """Create (and return) an empty run directory. Metrics get filled in
    later by write_manifest once the run has actually executed."""
    timestamp = timestamp or datetime.now(UTC)
    run_dir = runs_root / _run_dir_name(cfg, timestamp)
    run_dir.mkdir(parents=True, exist_ok=False)
    return run_dir


def write_manifest(
    run_dir: Path,
    cfg: RunConfig,
    config_path: str | Path,
    *,
    metrics: dict,
    timings: dict | None = None,
    corpus_sha: str = "",
    collection_names: dict[str, str] | None = None,
    dataset_shas: dict[str, str] | None = None,
    cost: dict | None = None,
    git_cwd: Path | None = None,
    created_at: datetime | None = None,
) -> RunManifest:
    git_sha, git_dirty = _git_info(git_cwd)
    retrieval_hash, generation_hash = split_hashes(cfg)
    manifest = RunManifest(
        run_id=run_dir.name,
        name=cfg.name,
        created_at=(created_at or datetime.now(UTC)).isoformat(),
        git_sha=git_sha,
        git_dirty=git_dirty,
        config_path=str(config_path),
        config=cfg.model_dump(mode="json"),
        config_hash=config_hash(cfg),
        retrieval_hash=retrieval_hash,
        generation_hash=generation_hash,
        corpus_sha=corpus_sha,
        collection_names=collection_names or {},
        dataset_shas=dataset_shas or {},
        metrics=metrics,
        timings=timings or {},
        cost=cost,
    )
    (run_dir / "manifest.json").write_text(
        json.dumps(manifest.to_dict(), indent=2, sort_keys=True), encoding="utf-8"
    )
    (run_dir / "config.resolved.yaml").write_text(
        yaml.dump(manifest.config, sort_keys=False), encoding="utf-8"
    )
    return manifest


def load_run(run_id_or_dir: str | Path, *, runs_root: Path = DEFAULT_RUNS_ROOT) -> RunManifest:
    run_dir = Path(run_id_or_dir)
    if not run_dir.exists():
        run_dir = runs_root / run_id_or_dir
    if not run_dir.exists():
        run_dir = runs_root / PINNED_DIRNAME / run_id_or_dir
    manifest_path = run_dir / "manifest.json"
    if not manifest_path.exists():
        raise FileNotFoundError(f"no manifest.json under {run_dir}")
    return RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))


def list_runs(
    *, runs_root: Path = DEFAULT_RUNS_ROOT, include_pinned: bool = False
) -> list[RunManifest]:
    if not runs_root.exists():
        return []
    manifests = []
    for child in sorted(runs_root.iterdir()):
        if child.name == PINNED_DIRNAME:
            continue
        manifest_path = child / "manifest.json"
        if manifest_path.exists():
            manifests.append(RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8"))))
    if include_pinned:
        pinned_root = runs_root / PINNED_DIRNAME
        if pinned_root.exists():
            for child in sorted(pinned_root.iterdir()):
                manifest_path = child / "manifest.json"
                if manifest_path.exists():
                    manifests.append(
                        RunManifest.from_dict(json.loads(manifest_path.read_text(encoding="utf-8")))
                    )
    manifests.sort(key=lambda m: m.created_at)
    return manifests


def pin_run(
    run_id: str, *, runs_root: Path = DEFAULT_RUNS_ROOT, name: str | None = None
) -> Path:
    """Copy a run directory into runs/_pinned/<name or run_id>/ -- promotion
    is an explicit, reviewable commit rather than a silent pointer update,
    so the baseline the project compares against can't quietly move."""
    src = runs_root / run_id
    if not src.exists():
        raise FileNotFoundError(f"no run directory {src}")
    dest_name = name or run_id
    dest = runs_root / PINNED_DIRNAME / dest_name
    if dest.exists():
        raise FileExistsError(f"{dest} already exists -- pick a different name")
    shutil.copytree(src, dest)
    return dest
