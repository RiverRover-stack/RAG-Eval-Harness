from rag_eval.common.schemas import DiscussionQA
from rag_eval.eval.build_eval_set import qa_to_eval_example


def test_qa_to_eval_example_query_is_title_plus_body_head():
    qa = DiscussionQA(
        discussion_id="D_1",
        title="jsonable_encoder drops exclude_defaults",
        question_body="Given a nested model with exclude_defaults=True set...",
        answer_body="Thanks for the report, tracked in #123.",
        url="https://github.com/fastapi/fastapi/discussions/1",
        category="Q&A",
    )

    example = qa_to_eval_example(qa)

    assert qa.title in example.question
    assert qa.question_body in example.question
    assert example.ground_truth == qa.answer_body
    assert example.source_url == qa.url


def test_qa_to_eval_example_truncates_long_body_to_500_chars():
    qa = DiscussionQA(
        discussion_id="D_1",
        title="Title",
        question_body="x" * 1000,
        answer_body="answer",
        url="https://github.com/fastapi/fastapi/discussions/1",
    )

    example = qa_to_eval_example(qa)

    assert example.question.count("x") == 500


def test_qa_to_eval_example_falls_back_to_bare_title_when_body_empty():
    qa = DiscussionQA(
        discussion_id="D_1",
        title="Just a title",
        question_body="",
        answer_body="answer",
        url="https://github.com/fastapi/fastapi/discussions/1",
    )

    example = qa_to_eval_example(qa)

    assert example.question == "Just a title"
