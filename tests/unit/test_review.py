import pytest

from rag_eval.eval.gold import EvalItem
from rag_eval.eval.review import (
    apply_label_decision,
    apply_review_decision,
    run_label_session,
    run_review_session,
    select_review_sample,
)


def _item(item_id: str, section: str, **overrides) -> EvalItem:
    defaults = {
        "id": item_id,
        "dataset": "docs_synth_v1",
        "question": f"question for {item_id}",
        "gold_urls": [f"https://fastapi.tiangolo.com/{section}/page/#anchor"],
    }
    defaults.update(overrides)
    return EvalItem(**defaults)


def _scripted_input(responses: list[str]):
    it = iter(responses)

    def _input(_prompt: str) -> str:
        return next(it)

    return _input


# ---------------------------------------------------------------------------
# select_review_sample
# ---------------------------------------------------------------------------


def test_select_review_sample_covers_every_section():
    items = (
        [_item(f"t{i}", "tutorial") for i in range(10)]
        + [_item(f"a{i}", "advanced") for i in range(2)]
        + [_item(f"r{i}", "reference") for i in range(1)]
    )
    sample = select_review_sample(items, n=3, seed=0)
    sections = {i.gold_urls[0].split("/")[3] for i in sample}
    assert sections == {"tutorial", "advanced", "reference"}


def test_select_review_sample_deterministic():
    items = [_item(f"t{i}", "tutorial") for i in range(10)]
    a = select_review_sample(items, n=5, seed=7)
    b = select_review_sample(items, n=5, seed=7)
    assert [i.id for i in a] == [i.id for i in b]


# ---------------------------------------------------------------------------
# apply_review_decision
# ---------------------------------------------------------------------------


def test_apply_review_decision_yes():
    item = _item("1", "tutorial")
    updated = apply_review_decision(item, "y", now="2026-01-01T00:00:00Z")
    assert updated.verified == "yes"
    assert updated.verified_at == "2026-01-01T00:00:00Z"


def test_apply_review_decision_no():
    updated = apply_review_decision(_item("1", "tutorial"), "n", now="t")
    assert updated.verified == "no"


def test_apply_review_decision_edit_replaces_question():
    updated = apply_review_decision(
        _item("1", "tutorial"), "e", edited_question="a better question", now="t"
    )
    assert updated.verified == "edited"
    assert updated.question == "a better question"


def test_apply_review_decision_edit_without_text_raises():
    with pytest.raises(ValueError, match="edited_question"):
        apply_review_decision(_item("1", "tutorial"), "e")


def test_apply_review_decision_skip_is_a_no_op():
    item = _item("1", "tutorial")
    updated = apply_review_decision(item, "s")
    assert updated == item
    assert updated.verified is None


def test_apply_review_decision_unknown_raises():
    with pytest.raises(ValueError, match="unknown review decision"):
        apply_review_decision(_item("1", "tutorial"), "x")


# ---------------------------------------------------------------------------
# run_review_session
# ---------------------------------------------------------------------------


def test_run_review_session_applies_decisions_in_order():
    items = [_item("1", "tutorial"), _item("2", "tutorial")]
    input_fn = _scripted_input(["y", "n"])
    result = run_review_session(items, n=2, input_fn=input_fn, print_fn=lambda _: None)
    by_id = {i.id: i for i in result}
    assert by_id["1"].verified == "yes"
    assert by_id["2"].verified == "no"


def test_run_review_session_skips_already_verified_items():
    items = [
        _item("1", "tutorial", verified="yes", verified_at="earlier"),
        _item("2", "tutorial"),
    ]
    input_fn = _scripted_input(["n"])  # only item 2 should prompt
    result = run_review_session(items, n=2, input_fn=input_fn, print_fn=lambda _: None)
    by_id = {i.id: i for i in result}
    assert by_id["1"].verified_at == "earlier"  # untouched
    assert by_id["2"].verified == "no"


def test_run_review_session_quit_leaves_remaining_unverified():
    items = [_item("1", "tutorial"), _item("2", "tutorial"), _item("3", "tutorial")]
    input_fn = _scripted_input(["y", "q"])
    result = run_review_session(items, n=3, input_fn=input_fn, print_fn=lambda _: None)
    by_id = {i.id: i for i in result}
    assert by_id["1"].verified == "yes"
    # whichever of 2/3 was asked next got no answer beyond "q" -- still unverified
    assert sum(1 for i in result if i.verified is None) == 2


def test_run_review_session_calls_save_fn_after_every_decision():
    items = [_item("1", "tutorial"), _item("2", "tutorial")]
    saved: list[list] = []
    input_fn = _scripted_input(["y", "n"])
    run_review_session(items, n=2, input_fn=input_fn, print_fn=lambda _: None, save_fn=saved.append)
    assert len(saved) == 2


def test_run_review_session_unrecognized_input_treated_as_skip():
    items = [_item("1", "tutorial")]
    input_fn = _scripted_input(["banana"])
    messages = []
    result = run_review_session(items, n=1, input_fn=input_fn, print_fn=messages.append)
    assert result[0].verified is None
    assert any("unrecognized" in m for m in messages)


def test_run_review_session_edit_prompts_for_new_text():
    items = [_item("1", "tutorial")]
    input_fn = _scripted_input(["e", "much better question"])
    result = run_review_session(items, n=1, input_fn=input_fn, print_fn=lambda _: None)
    assert result[0].verified == "edited"
    assert result[0].question == "much better question"


# ---------------------------------------------------------------------------
# apply_label_decision / run_label_session
# ---------------------------------------------------------------------------


def test_apply_label_decision_parses_comma_separated_urls():
    item = _item("1", "tutorial", gold_urls=[])
    updated = apply_label_decision(item, "https://x/#a, https://x/#b", now="t")
    assert updated.gold_urls == ["https://x/#a", "https://x/#b"]
    assert updated.verified == "yes"


def test_apply_label_decision_n_means_no_gold():
    item = _item("1", "tutorial")
    updated = apply_label_decision(item, "n", now="t")
    assert updated.gold_urls == []
    assert updated.verified == "no"


def test_apply_label_decision_empty_input_raises():
    with pytest.raises(ValueError):
        apply_label_decision(_item("1", "tutorial"), "   ")


def test_run_label_session_shows_candidates_and_applies_decisions():
    items = [_item("1", "tutorial", gold_urls=[])]
    seen_questions = []

    def candidate_fn(question):
        seen_questions.append(question)
        return ["https://x/#top1", "https://x/#top2"]

    input_fn = _scripted_input(["https://x/#top1"])
    result = run_label_session(
        items, candidate_fn=candidate_fn, input_fn=input_fn, print_fn=lambda _: None
    )
    assert result[0].gold_urls == ["https://x/#top1"]
    assert seen_questions == [items[0].question]


def test_run_label_session_skips_already_labeled_items():
    items = [_item("1", "tutorial", verified="yes", gold_urls=["https://x/#a"])]
    input_fn = _scripted_input([])  # should never be called
    result = run_label_session(items, input_fn=input_fn, print_fn=lambda _: None)
    assert result[0].gold_urls == ["https://x/#a"]


def test_run_label_session_bad_input_is_skipped_not_crashed():
    items = [_item("1", "tutorial", gold_urls=[])]
    input_fn = _scripted_input(["   "])
    messages = []
    result = run_label_session(items, input_fn=input_fn, print_fn=messages.append)
    assert result[0].verified is None
    assert any("expected comma-separated" in m for m in messages)
