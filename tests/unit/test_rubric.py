import json
from dataclasses import dataclass

import pytest

from rag_eval.eval.rubric import (
    RubricScore,
    deterministic_citation_accuracy,
    judge_vs_deterministic_correlation,
    parse_rubric_response,
    rubric_run,
    score_item,
)

VALID_JSON = json.dumps(
    {
        "correctness": 5,
        "completeness": 4,
        "citation_accuracy": 3,
        "hallucination": 5,
        "tone": 4,
        "justification": "solid answer",
    }
)

_KEYWORDS = ("response", "class", "sky", "blue", "green", "weather")


class _KeywordEmbedder:
    def embed_query(self, text: str) -> list[float]:
        t = text.lower()
        return [1.0 if kw in t else 0.0 for kw in _KEYWORDS]

    def embed_documents(self, texts):
        return [self.embed_query(t) for t in texts]


@dataclass
class _Resp:
    content: str
    prompt_tokens: int = 10
    completion_tokens: int = 20


class _ScriptedLLM:
    def __init__(self, content: str) -> None:
        self.content = content
        self.calls: list = []

    def complete(self, messages, temperature: float = 0.0, max_tokens: int = 1024):
        self.calls.append(messages)
        return _Resp(self.content)


def _gen_row():
    answer = "Use a custom response class [1]. The sky is green [2]."
    return {
        "item_id": "d::x",
        "question": "how do I return HTML?",
        "ground_truth": "",
        "answer": answer,
        "contexts": ["custom response class docs", "weather report: sky is blue"],
        "citations": {
            "citations": [
                {"index": 1, "chunk_id": "c1", "url": "u1", "char_start": answer.index("[1]"), "char_end": answer.index("[1]") + 3},
                {"index": 2, "chunk_id": "c2", "url": "u2", "char_start": answer.index("[2]"), "char_end": answer.index("[2]") + 3},
            ]
        },
    }


def test_parse_rubric_response_valid():
    scores, justification = parse_rubric_response(VALID_JSON)
    assert scores == {"correctness": 5, "completeness": 4, "citation_accuracy": 3, "hallucination": 5, "tone": 4}
    assert justification == "solid answer"


def test_parse_rubric_response_prose_wrapped():
    wrapped = f"Here is my assessment:\n```json\n{VALID_JSON}\n```\nThanks."
    scores, _ = parse_rubric_response(wrapped)
    assert scores["correctness"] == 5


def test_parse_rubric_response_missing_dimension_raises():
    payload = json.loads(VALID_JSON)
    del payload["tone"]
    with pytest.raises(ValueError, match="missing 'tone'"):
        parse_rubric_response(json.dumps(payload))


def test_parse_rubric_response_out_of_range_raises():
    payload = json.loads(VALID_JSON)
    payload["correctness"] = 9
    with pytest.raises(ValueError, match="out of range"):
        parse_rubric_response(json.dumps(payload))


def test_deterministic_citation_accuracy_mixes_supported_and_unsupported():
    det, n = deterministic_citation_accuracy(_gen_row(), _KeywordEmbedder())
    assert n == 2
    assert det == pytest.approx(0.5)


def test_deterministic_citation_accuracy_none_without_citations():
    row = _gen_row()
    row["citations"] = {"citations": []}
    assert deterministic_citation_accuracy(row, _KeywordEmbedder()) == (None, 0)


def test_judge_vs_deterministic_correlation_perfect():
    scores = [
        RubricScore("a", {"citation_accuracy": 5}, "", 1.0, 1),
        RubricScore("b", {"citation_accuracy": 4}, "", 0.5, 2),
        RubricScore("c", {"citation_accuracy": 3}, "", 0.0, 1),
    ]
    assert judge_vs_deterministic_correlation(scores) == pytest.approx(1.0)


def test_judge_vs_deterministic_correlation_none_when_too_few():
    scores = [RubricScore("a", {"citation_accuracy": 5}, "", None, 0)]
    assert judge_vs_deterministic_correlation(scores) is None


def test_score_item_combines_judge_and_deterministic():
    llm = _ScriptedLLM(VALID_JSON)
    result = score_item(_gen_row(), llm=llm, embedder=_KeywordEmbedder())
    assert result.item_id == "d::x"
    assert result.scores["completeness"] == 4
    assert result.deterministic_citation_accuracy == pytest.approx(0.5)
    assert result.n_citations == 2
    assert len(llm.calls) == 1


def test_rubric_run_writes_artifact_and_manifest_block(tmp_path):
    run_dir = tmp_path / "runs" / "2026-01-01T00-00-00Z__t__abc123"
    run_dir.mkdir(parents=True)
    config = {
        "eval": {"judge": {"provider": "gemini", "model": "gemini-2.5-flash"}},
        "generation": {"llm": {"provider": "groq", "model": "openai/gpt-oss-120b"}},
        "embedding": {"provider": "fastembed", "model": "BAAI/bge-small-en-v1.5"},
    }
    (run_dir / "manifest.json").write_text(json.dumps({"config": config, "metrics": {}}), encoding="utf-8")
    (run_dir / "generation.jsonl").write_text(json.dumps(_gen_row()) + "\n", encoding="utf-8")

    path = rubric_run(
        "2026-01-01T00-00-00Z__t__abc123",
        runs_root=tmp_path / "runs",
        llm=_ScriptedLLM(VALID_JSON),
        embedder=_KeywordEmbedder(),
    )
    assert path == run_dir / "rubric.jsonl"
    rows = [json.loads(line) for line in path.read_text().splitlines()]
    assert rows[0]["scores"]["correctness"] == 5

    block = json.loads((run_dir / "manifest.json").read_text())["metrics"]["rubric"]
    assert block["n_items"] == 1
    assert block["dimension_means"]["tone"] == pytest.approx(4.0)
