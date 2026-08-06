"""The yaml half of the config split: anything that can move a metric lives
here, not in Settings (env). See CLAUDE.md's config-discipline rule.

A run config is just a yaml file, optionally `extends`-ing a base one, with
`--set path.to.field=value` overrides layered on top before validation. The
whole point of `extra="forbid"` everywhere is that a typo'd key should blow
up loudly instead of silently being ignored -- that's how eval harnesses end
up lying about what they measured.
"""

from __future__ import annotations

import copy
import hashlib
import json
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, ConfigDict


class _Strict(BaseModel):
    model_config = ConfigDict(extra="forbid")


class CorpusConfig(_Strict):
    sources: list[Literal["docs", "discussions"]] = ["docs", "discussions"]
    snapshot: str = "data/corpus/SNAPSHOT.json"


class EmbeddingConfig(_Strict):
    provider: str = "fastembed"
    model: str = "BAAI/bge-small-en-v1.5"


class DenseStageConfig(_Strict):
    enabled: bool = True


class Bm25StageConfig(_Strict):
    enabled: bool = False
    k1: float = 1.2
    b: float = 0.75


class FusionConfig(_Strict):
    method: str = "rrf"
    rrf_k: int = 60


class RerankStageConfig(_Strict):
    enabled: bool = False
    model: str = "Xenova/ms-marco-MiniLM-L-6-v2"
    top_n: int = 5


class QueryRewriteConfig(_Strict):
    enabled: bool = False
    mode: str = "hyde"
    n: int = 1


class ParentExpansionConfig(_Strict):
    enabled: bool = False
    mode: str = "section"
    max_tokens: int = 1200


class PerSourceCaps(_Strict):
    docs: int = 4
    discussions: int = 2


class RetrievalConfig(_Strict):
    top_k: int = 5
    candidates_k: int = 30
    dense: DenseStageConfig = DenseStageConfig()
    bm25: Bm25StageConfig = Bm25StageConfig()
    fusion: FusionConfig = FusionConfig()
    rerank: RerankStageConfig = RerankStageConfig()
    query_rewrite: QueryRewriteConfig = QueryRewriteConfig()
    parent_expansion: ParentExpansionConfig = ParentExpansionConfig()
    per_source_caps: PerSourceCaps = PerSourceCaps()


class LLMConfig(_Strict):
    provider: str = "groq"
    model: str = "llama-3.3-70b-versatile"
    temperature: float = 0.0
    max_tokens: int = 900


class GroundednessConfig(_Strict):
    enabled: bool = True
    mode: str = "embedding_support"
    abstain_below: float = 0.45


class GenerationConfig(_Strict):
    enabled: bool = False
    llm: LLMConfig = LLMConfig()
    prompt_version: str = "v2-cited"
    require_citations: bool = True
    groundedness: GroundednessConfig = GroundednessConfig()


class JudgeConfig(_Strict):
    enabled: bool = False
    provider: str = "gemini"
    model: str = "gemini-2.5-flash"
    max_workers: int = 1


class ThresholdConfig(_Strict):
    min: float
    regression_tolerance: float = 0.02


class EvalConfig(_Strict):
    datasets: list[str] = []
    k_values: list[int] = [1, 3, 5, 10]
    # none: naive, kept only so the inflated number can be published beside
    # the honest one. holdout: exclude_chunk_ids threaded through as
    # deny_ids. separate_index: not built yet.
    self_retrieval: Literal["none", "holdout", "separate_index"] = "holdout"
    judge: JudgeConfig = JudgeConfig()
    thresholds: dict[str, ThresholdConfig] = {}


class RunConfig(_Strict):
    name: str
    corpus: CorpusConfig = CorpusConfig()
    embedding: EmbeddingConfig = EmbeddingConfig()
    retrieval: RetrievalConfig = RetrievalConfig()
    generation: GenerationConfig = GenerationConfig()
    eval: EvalConfig = EvalConfig()


def _deep_merge(base: dict, override: dict) -> dict:
    merged = dict(base)
    for key, value in override.items():
        if key in merged and isinstance(merged[key], dict) and isinstance(value, dict):
            merged[key] = _deep_merge(merged[key], value)
        else:
            merged[key] = value
    return merged


def _load_yaml_with_extends(path: Path, _chain: tuple[Path, ...] = ()) -> dict:
    resolved = path.resolve()
    if resolved in _chain:
        chain = " -> ".join(str(p) for p in (*_chain, resolved))
        raise ValueError(f"circular 'extends' chain: {chain}")

    data = yaml.safe_load(resolved.read_text(encoding="utf-8")) or {}
    extends = data.pop("extends", None)
    if extends is None:
        return data

    base_path = resolved.parent / extends
    base_data = _load_yaml_with_extends(base_path, (*_chain, resolved))
    return _deep_merge(base_data, data)


def _parse_override_value(raw: str) -> object:
    # yaml.safe_load turns "10" -> int, "true" -> bool, "0.5" -> float,
    # "foo" -> str -- exactly the scalar coercion `--set` needs, and it's
    # already a dependency, so no reason to write a second parser.
    return yaml.safe_load(raw)


def _apply_overrides(data: dict, overrides: list[str]) -> dict:
    data = copy.deepcopy(data)
    for item in overrides:
        key_path, sep, raw_value = item.partition("=")
        if not sep:
            raise ValueError(f"--set override {item!r} must look like path.to.field=value")
        value = _parse_override_value(raw_value)
        keys = key_path.strip().split(".")
        cursor = data
        for key in keys[:-1]:
            nxt = cursor.get(key)
            if not isinstance(nxt, dict):
                nxt = {}
                cursor[key] = nxt
            cursor = nxt
        cursor[keys[-1]] = value
    return data


def load_run_config(path: str | Path, overrides: list[str] | None = None) -> RunConfig:
    data = _load_yaml_with_extends(Path(path))
    data = _apply_overrides(data, overrides or [])
    return RunConfig.model_validate(data)


def _canonical_hash(payload: dict) -> str:
    blob = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()[:12]


def config_hash(cfg: RunConfig) -> str:
    return _canonical_hash(cfg.model_dump(mode="json"))


def split_hashes(cfg: RunConfig) -> tuple[str, str]:
    """(retrieval_hash, generation_hash) -- the pair that lets
    `compare_runs` tell a real retrieval delta from a confounded one where
    the generator also changed underneath it."""
    retrieval_payload = {
        "corpus": cfg.corpus.model_dump(mode="json"),
        "embedding": cfg.embedding.model_dump(mode="json"),
        "retrieval": cfg.retrieval.model_dump(mode="json"),
    }
    generation_payload = {"generation": cfg.generation.model_dump(mode="json")}
    return _canonical_hash(retrieval_payload), _canonical_hash(generation_payload)
