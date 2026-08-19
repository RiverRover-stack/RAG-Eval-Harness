"""Generate docs_synth_v1: one question per sampled docs chunk, run through
three auto-filters before a human ever sees it (docs/plan.md C2).

The filters exist to keep the set from being circular or trivial:

1. lexical-overlap reject -- the question just parrots a rare, distinctive
   phrase straight out of its source chunk, which would let BM25 (or any
   keyword match) win for free. That's not a retrieval test, it's a
   string-match test wearing a retrieval eval's clothes.
2. closed-book reject -- a *different* model answers the question
   correctly with no context at all, which means the question tests the
   model's parametric knowledge of FastAPI, not whether retrieval found
   the right chunk.
3. near-duplicate reject -- the question is a near-paraphrase of one
   already accepted, which would silently inflate n without adding signal.

Everything that survives still needs the human pass in eval/review.py --
these filters catch the mechanical failure modes, not "is this a good
question."
"""

from __future__ import annotations

import logging
import random
import re
from collections import Counter
from collections.abc import Callable, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from rag_eval.eval.gold import EvalItem

if TYPE_CHECKING:
    from rag_eval.providers.base import EmbeddingProvider, LLMProvider

logger = logging.getLogger(__name__)

RARE_TRIGRAM_DOC_FREQ_MAX = 2
CLOSED_BOOK_SUPPORT_THRESHOLD = 0.85
NEAR_DUPLICATE_THRESHOLD = 0.95

GENERATION_SYSTEM_PROMPT = (
    "You write a single short, natural question a developer would type into "
    "a search box, fully answerable from the given documentation passage. "
    "Ask about one concrete thing it explains. Don't mention 'the passage' "
    "or 'the documentation'. Reply with only the question -- no preamble, "
    "no quotes, no markdown."
)


# ---------------------------------------------------------------------------
# Trigram lexical-overlap filter
# ---------------------------------------------------------------------------

_TOKEN_RE = re.compile(r"[a-z0-9_]+")

# On a corpus this small (~1000 chunks), almost *any* three-word English
# phrase has a low document frequency just by combinatorial chance -- "to
# the project", "support for reading", "the encryption of" all came back as
# "rare" on a real run, and none of them would make BM25 win for free.
# Requiring every token in the shared trigram to be a content word (no
# stopwords) is what actually separates "distinctive technical phrase" from
# "ordinary sentence glue that happened not to repeat elsewhere."
_STOPWORDS = frozenset(
    """
    a an the of to for in on at is are was were be been being do does did
    doesn don t and or but with from by as that this these those it its i
    you your my we our they their he she his her if not no so than then
    there here when where which who whom what how why can could should
    would will shall may might must have has had having also into out up
    down over under again further once
    """.split()  # noqa: SIM905 -- a literal word list is more legible than one giant list literal
)


def _tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


def _trigrams(tokens: Sequence[str]) -> set[tuple[str, ...]]:
    return {tuple(tokens[i : i + 3]) for i in range(len(tokens) - 2)}


def _is_content_trigram(trigram: tuple[str, ...]) -> bool:
    return all(token not in _STOPWORDS for token in trigram)


def build_trigram_doc_freq(chunk_texts: Sequence[str]) -> Counter[tuple[str, ...]]:
    """Document frequency of each word-trigram across the corpus -- how many
    chunks it appears in, not how many times total. A trigram appearing in
    only one or two chunks is distinctive enough that reusing it verbatim in
    a question would make retrieval trivial."""
    df: Counter[tuple[str, ...]] = Counter()
    for text in chunk_texts:
        df.update(_trigrams(_tokenize(text)))
    return df


def lexical_overlap_reject(
    question: str, chunk_text: str, trigram_df: Counter[tuple[str, ...]]
) -> bool:
    shared = _trigrams(_tokenize(question)) & _trigrams(_tokenize(chunk_text))
    content_shared = {t for t in shared if _is_content_trigram(t)}
    return any(trigram_df.get(t, 0) <= RARE_TRIGRAM_DOC_FREQ_MAX for t in content_shared)


# ---------------------------------------------------------------------------
# Closed-book filter
# ---------------------------------------------------------------------------


def _cosine(a: Sequence[float], b: Sequence[float]) -> float:
    dot = sum(x * y for x, y in zip(a, b))
    norm_a = sum(x * x for x in a) ** 0.5
    norm_b = sum(y * y for y in b) ** 0.5
    if norm_a == 0 or norm_b == 0:
        return 0.0
    return dot / (norm_a * norm_b)


def closed_book_reject(
    question: str,
    chunk_text: str,
    *,
    closed_book_llm: LLMProvider,
    embedder: EmbeddingProvider,
    threshold: float = CLOSED_BOOK_SUPPORT_THRESHOLD,
) -> bool:
    """True if a model with *no* access to the passage still lands close
    enough to the passage's own content that the question isn't really
    testing retrieval. "Close enough" is a crude embedding-similarity proxy,
    not a full judge rubric -- good enough to catch "what is dependency
    injection" (the model already knows), not subtle enough to catch every
    partial-credit case, and that's an honest tradeoff for Phase 4."""
    response = closed_book_llm.complete(
        [{"role": "user", "content": question}], temperature=0.0, max_tokens=300
    )
    answer_vec = embedder.embed_query(response.content)
    chunk_vec = embedder.embed_query(chunk_text)
    return _cosine(answer_vec, chunk_vec) >= threshold


# ---------------------------------------------------------------------------
# Near-duplicate filter
# ---------------------------------------------------------------------------


def near_duplicate_reject(
    question_embedding: Sequence[float],
    accepted_embeddings: Sequence[Sequence[float]],
    threshold: float = NEAR_DUPLICATE_THRESHOLD,
) -> bool:
    return any(_cosine(question_embedding, e) >= threshold for e in accepted_embeddings)


# ---------------------------------------------------------------------------
# Stratified sampling
# ---------------------------------------------------------------------------


def _top_level_section(chunk: dict) -> str:
    path = chunk["metadata"].get("path", "")
    return path.split("/")[0] if path else "_root"


def stratified_sample(chunks: Sequence[dict], n: int, *, seed: int = 0) -> list[dict]:
    """Round-robins across top-level docs sections (tutorial/, advanced/,
    reference/, ...) so a large section can't crowd out small ones -- the
    label-error-rate estimate in the human review only generalizes if the
    sample does."""
    by_section: dict[str, list[dict]] = {}
    for chunk in chunks:
        by_section.setdefault(_top_level_section(chunk), []).append(chunk)

    rng = random.Random(seed)
    for bucket in by_section.values():
        rng.shuffle(bucket)

    sections = sorted(by_section)
    sampled: list[dict] = []
    cursors = dict.fromkeys(sections, 0)
    while len(sampled) < n and any(cursors[s] < len(by_section[s]) for s in sections):
        for section in sections:
            if len(sampled) >= n:
                break
            cursor = cursors[section]
            if cursor < len(by_section[section]):
                sampled.append(by_section[section][cursor])
                cursors[section] = cursor + 1
    return sampled


# ---------------------------------------------------------------------------
# Build report
# ---------------------------------------------------------------------------


@dataclass
class BuildReport:
    generated: int = 0
    rejected_lexical: int = 0
    rejected_closed_book: int = 0
    rejected_duplicate: int = 0
    rejected_generation_error: int = 0
    retained: int = 0
    sections_covered: set[str] = field(default_factory=set)

    def to_dict(self) -> dict:
        return {
            "generated": self.generated,
            "rejected_lexical": self.rejected_lexical,
            "rejected_closed_book": self.rejected_closed_book,
            "rejected_duplicate": self.rejected_duplicate,
            "rejected_generation_error": self.rejected_generation_error,
            "retained": self.retained,
            "sections_covered": sorted(self.sections_covered),
        }

    def summary(self) -> str:
        rejected = (
            self.rejected_lexical
            + self.rejected_closed_book
            + self.rejected_duplicate
            + self.rejected_generation_error
        )
        return (
            f"{self.generated} generated, {rejected} auto-rejected "
            f"({self.rejected_lexical} lexical, {self.rejected_closed_book} closed-book, "
            f"{self.rejected_duplicate} dup, {self.rejected_generation_error} generation error), "
            f"{self.retained} retained"
        )


# ---------------------------------------------------------------------------
# Orchestration
# ---------------------------------------------------------------------------


def build_docs_synth_v1(
    chunks: Sequence[dict],
    *,
    llm: LLMProvider,
    closed_book_llm: LLMProvider,
    embedder: EmbeddingProvider,
    n_target: int = 150,
    seed: int = 0,
    dataset_name: str = "docs_synth_v1",
    checkpoint_fn: Callable[[list[EvalItem], BuildReport], None] | None = None,
) -> tuple[list[EvalItem], BuildReport]:
    """checkpoint_fn, if given, is called after every chunk (accepted or
    not) with the accepted items and report so far -- a real run is ~300
    sequential API calls, and without incremental persistence a single
    failure near the end would throw away everything before it."""
    trigram_df = build_trigram_doc_freq([c["document"] for c in chunks])
    sampled = stratified_sample(chunks, n_target, seed=seed)

    report = BuildReport()
    accepted_items: list[EvalItem] = []
    accepted_embeddings: list[list[float]] = []

    for chunk in sampled:
        chunk_text = chunk["document"]
        try:
            try:
                response = llm.complete(
                    [
                        {"role": "system", "content": GENERATION_SYSTEM_PROMPT},
                        {
                            "role": "user",
                            "content": f"Page: {chunk['metadata'].get('title', '')}\n\n"
                            f"Passage:\n{chunk_text}\n\nQuestion:",
                        },
                    ],
                    temperature=0.7,
                    max_tokens=100,
                )
                question = response.content.strip().strip('"')
            except Exception:
                logger.exception("generation failed for chunk %s", chunk["id"])
                report.rejected_generation_error += 1
                continue

            report.generated += 1
            try:
                if not question:
                    report.rejected_generation_error += 1
                    continue

                if lexical_overlap_reject(question, chunk_text, trigram_df):
                    report.rejected_lexical += 1
                    continue

                if closed_book_reject(
                    question, chunk_text, closed_book_llm=closed_book_llm, embedder=embedder
                ):
                    report.rejected_closed_book += 1
                    continue

                question_vec = embedder.embed_query(question)
                if near_duplicate_reject(question_vec, accepted_embeddings):
                    report.rejected_duplicate += 1
                    continue
            except Exception:
                logger.exception("filtering failed for chunk %s", chunk["id"])
                report.rejected_generation_error += 1
                continue

            item = EvalItem(
                id=chunk["id"],
                dataset=dataset_name,
                question=question,
                ground_truth=chunk_text,
                gold_urls=[chunk["metadata"].get("url", "")],
                provenance=f"synth:{chunk['id']}",
            )
            accepted_items.append(item)
            accepted_embeddings.append(question_vec)
            report.retained += 1
            report.sections_covered.add(_top_level_section(chunk))
        finally:
            if checkpoint_fn is not None:
                checkpoint_fn(accepted_items, report)

    return accepted_items, report


if __name__ == "__main__":
    # uv run python -m rag_eval.eval.synth_eval_set
    #
    # Generation uses Groq's llama-3.3-70b-versatile (docs/plan.md's locked
    # serving model); the closed-book check deliberately uses a *different*
    # Groq model (llama-3.1-8b-instant) rather than Gemini -- the project's
    # GEMINI_API_KEY was invalid when this ran, and the filter only needs
    # "a model that isn't the generator," not Gemini specifically.
    import logging
    from pathlib import Path

    from rag_eval.ingestion.docs_chunker import load_doc_chunks
    from rag_eval.providers import get_embedder, get_llm

    logging.basicConfig(level=logging.INFO)

    out_path = Path("data/eval_sets/docs_synth_v1.jsonl")

    def _checkpoint(items: list[EvalItem], report: BuildReport) -> None:
        out_path.write_text(
            "\n".join(item.model_dump_json() for item in items) + ("\n" if items else ""),
            encoding="utf-8",
        )
        logger.info("checkpoint: %s", report.summary())

    chunks = list(load_doc_chunks())
    generated_items, build_report = build_docs_synth_v1(
        chunks,
        llm=get_llm("groq", "llama-3.3-70b-versatile"),
        closed_book_llm=get_llm("groq", "llama-3.1-8b-instant"),
        embedder=get_embedder(),
        n_target=150,
        seed=0,
        checkpoint_fn=_checkpoint,
    )
    _checkpoint(generated_items, build_report)
    print(build_report.summary())
