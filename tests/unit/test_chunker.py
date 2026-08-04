from rag_eval.common.schemas import DiscussionQA
from rag_eval.ingestion.chunker import qa_to_chunks


def test_qa_to_chunks_produces_one_chunk_with_metadata():
    qa = DiscussionQA(
        discussion_id="D_1",
        title="How do I use Depends?",
        question_body="I'm confused about Depends()",
        answer_body="Depends() lets you declare a dependency...",
        url="https://github.com/fastapi/fastapi/discussions/1",
        category="Q&A",
    )

    chunks = qa_to_chunks(qa)

    assert len(chunks) == 1
    assert chunks[0]["id"] == "D_1"
    assert chunks[0]["document"] == qa.answer_body
    assert chunks[0]["metadata"]["title"] == qa.title
    assert chunks[0]["metadata"]["url"] == qa.url
    assert chunks[0]["metadata"]["source_type"] == "discussion"
    assert chunks[0]["metadata"]["content_hash"]


def test_qa_to_chunks_content_hash_changes_with_answer_text():
    base = dict(
        discussion_id="D_1",
        title="How do I use Depends?",
        question_body="I'm confused about Depends()",
        url="https://github.com/fastapi/fastapi/discussions/1",
        category="Q&A",
    )
    qa_a = DiscussionQA(**base, answer_body="Depends() lets you declare a dependency...")
    qa_b = DiscussionQA(**base, answer_body="Different answer text entirely.")

    hash_a = qa_to_chunks(qa_a)[0]["metadata"]["content_hash"]
    hash_b = qa_to_chunks(qa_b)[0]["metadata"]["content_hash"]

    assert hash_a != hash_b
