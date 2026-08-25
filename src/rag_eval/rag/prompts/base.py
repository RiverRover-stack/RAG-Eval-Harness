"""Shared shape for versioned prompt templates.

`version` is the same string `RunConfig.generation.prompt_version` selects
(docs/plan.md C1) and is recorded in the run manifest, so a generation run
is reproducible from its config alone -- a prompt change is a version bump,
not a silent edit.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

from rag_eval.retrieval.base import Candidate


@dataclass(frozen=True)
class PromptTemplate:
    version: str
    system_prompt: str
    build_user_prompt: Callable[[str, list[Candidate]], str]
