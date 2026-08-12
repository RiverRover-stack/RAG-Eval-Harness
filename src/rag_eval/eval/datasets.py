"""Load eval sets from data/eval_sets/*.jsonl into EvalItem lists. Loading
is deliberately dumb -- one JSON object per line, one EvalItem per object --
because the interesting logic (gold resolution) lives in gold.py and runs
separately at run time, not baked into what's on disk.
"""

from __future__ import annotations

import hashlib
from pathlib import Path

from rag_eval.eval.gold import EvalItem

DEFAULT_EVAL_SETS_DIR = Path("data/eval_sets")


def dataset_path(name: str, eval_sets_dir: Path = DEFAULT_EVAL_SETS_DIR) -> Path:
    return eval_sets_dir / f"{name}.jsonl"


def load_dataset(name: str, eval_sets_dir: Path = DEFAULT_EVAL_SETS_DIR) -> list[EvalItem]:
    path = dataset_path(name, eval_sets_dir)
    if not path.exists():
        raise FileNotFoundError(f"no eval set at {path} -- has dataset {name!r} been built yet?")
    items = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if not line:
            continue
        items.append(EvalItem.model_validate_json(line))
    return items


def dataset_sha256(name: str, eval_sets_dir: Path = DEFAULT_EVAL_SETS_DIR) -> str:
    path = dataset_path(name, eval_sets_dir)
    return hashlib.sha256(path.read_bytes()).hexdigest()
