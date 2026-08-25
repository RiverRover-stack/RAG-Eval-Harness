from rag_eval.rag.prompts import PROMPTS
from rag_eval.retrieval.base import Candidate


def _cand(chunk_id: str, content: str, url: str) -> Candidate:
    return Candidate(chunk_id=chunk_id, content=content, url=url, title="t", source_type="docs")


def test_prompts_keyed_by_their_own_version_string():
    assert set(PROMPTS) == {"v1-legacy", "v2-cited"}
    for version, template in PROMPTS.items():
        assert template.version == version


def test_v1_legacy_prompt_has_no_numbered_blocks():
    candidates = [_cand("a", "first chunk", "https://x/a"), _cand("b", "second chunk", "https://x/b")]
    prompt = PROMPTS["v1-legacy"].build_user_prompt("What is FastAPI?", candidates)

    assert "[1]" not in prompt
    assert "first chunk" in prompt
    assert "second chunk" in prompt
    assert "What is FastAPI?" in prompt


def test_v2_cited_prompt_numbers_blocks_and_tags_source_url():
    candidates = [_cand("a", "first chunk", "https://x/a"), _cand("b", "second chunk", "https://x/b")]
    prompt = PROMPTS["v2-cited"].build_user_prompt("What is FastAPI?", candidates)

    assert "[1] (source: https://x/a)" in prompt
    assert "[2] (source: https://x/b)" in prompt
    assert prompt.index("[1]") < prompt.index("[2]")


def test_v2_cited_system_prompt_mentions_the_abstention_sentinel():
    assert "INSUFFICIENT_CONTEXT" in PROMPTS["v2-cited"].system_prompt
