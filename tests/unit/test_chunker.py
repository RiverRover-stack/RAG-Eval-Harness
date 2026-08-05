from rag_eval.common.schemas import DiscussionQA
from rag_eval.ingestion.chunker import qa_to_chunks
from rag_eval.ingestion.packing import CHARS_PER_TOKEN, TARGET_MAX_TOKENS


def _base_qa(**overrides) -> DiscussionQA:
    base = {
        "discussion_id": "D_1",
        "title": "How do I use Depends?",
        "question_body": "I'm confused about Depends()",
        "answer_body": "Depends() lets you declare a dependency...",
        "url": "https://github.com/fastapi/fastapi/discussions/1",
        "category": "Q&A",
    }
    base.update(overrides)
    return DiscussionQA(**base)


def test_qa_to_chunks_produces_one_chunk_with_metadata():
    qa = _base_qa()

    chunks = qa_to_chunks(qa)

    assert len(chunks) == 1
    assert chunks[0]["id"] == "D_1"
    assert chunks[0]["document"] == qa.answer_body
    assert chunks[0]["metadata"]["title"] == qa.title
    assert chunks[0]["metadata"]["url"] == qa.url
    assert chunks[0]["metadata"]["source_type"] == "discussion"
    assert chunks[0]["metadata"]["content_hash"]
    assert chunks[0]["metadata"]["chunk_index"] == 0
    assert chunks[0]["metadata"]["parent_id"]


def test_qa_to_chunks_packs_a_long_answer_into_multiple_chunks():
    # Well past TARGET_MAX_TOKENS, with blank-line-separated paragraphs so
    # the packer has more than one atomic block to work with.
    long_answer = "\n\n".join(
        "a" * (100 * CHARS_PER_TOKEN) for _ in range(2 * TARGET_MAX_TOKENS // 100)
    )
    qa = _base_qa(answer_body=long_answer)

    chunks = qa_to_chunks(qa)

    assert len(chunks) > 1
    assert [c["id"] for c in chunks] == [f"D_1::{i}" for i in range(len(chunks))]
    assert [c["metadata"]["chunk_index"] for c in chunks] == list(range(len(chunks)))
    # every chunk from the same answer shares one parent_id
    assert len({c["metadata"]["parent_id"] for c in chunks}) == 1


def test_qa_to_chunks_content_hash_changes_with_answer_text():
    base = {
        "discussion_id": "D_1",
        "title": "How do I use Depends?",
        "question_body": "I'm confused about Depends()",
        "url": "https://github.com/fastapi/fastapi/discussions/1",
        "category": "Q&A",
    }
    qa_a = DiscussionQA(**base, answer_body="Depends() lets you declare a dependency...")
    qa_b = DiscussionQA(**base, answer_body="Different answer text entirely.")

    hash_a = qa_to_chunks(qa_a)[0]["metadata"]["content_hash"]
    hash_b = qa_to_chunks(qa_b)[0]["metadata"]["content_hash"]

    assert hash_a != hash_b
