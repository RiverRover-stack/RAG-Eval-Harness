"""Versioned prompt templates, keyed by the string
`RunConfig.generation.prompt_version` uses (docs/plan.md C1) so the active
prompt is a config value, not a code change.
"""

from rag_eval.rag.prompts.base import PromptTemplate
from rag_eval.rag.prompts.v1_legacy import PROMPT as V1_LEGACY
from rag_eval.rag.prompts.v2_cited import PROMPT as V2_CITED

PROMPTS: dict[str, PromptTemplate] = {
    V1_LEGACY.version: V1_LEGACY,
    V2_CITED.version: V2_CITED,
}

__all__ = ["PROMPTS", "PromptTemplate"]
