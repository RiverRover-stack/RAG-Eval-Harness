"""The two human checkpoints (docs/plan.md Phase 4):

`run_review_session` -- a human looks at a stratified sample of
docs_synth_v1 (question + gold chunk) and calls it y(es)/e(dit)/n(o)/s(kip).
That's what turns "127 retained after auto-filters" into a measured label
error rate with a confidence interval, instead of a number nobody checked.

`run_label_session` -- a human hand-labels discussions_v2 with the actual
docs section(s) that answer each real question, using retrieval's own
top-10 as a convenience shortlist, never as the answer.

Both are built around injectable input/print/save functions rather than
real stdin, so the whole interactive loop -- including resumability -- is
unit-testable without a terminal (CLAUDE.md: constructor injection over
patching). The CLI wires real input()/print()/disk-writes on top.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from datetime import UTC, datetime
from urllib.parse import urlparse

from rag_eval.eval.gold import EvalItem

InputFn = Callable[[str], str]
PrintFn = Callable[[str], None]
SaveFn = Callable[[list[EvalItem]], None]


def _now_iso() -> str:
    return datetime.now(UTC).isoformat()


def _url_section(url: str) -> str:
    path = urlparse(url).path.strip("/")
    return path.split("/")[0] if path else "_root"


def _item_section(item: EvalItem) -> str:
    return _url_section(item.gold_urls[0]) if item.gold_urls else "_root"


# ---------------------------------------------------------------------------
# review: docs_synth_v1, y/e/n/s against the generated question + gold chunk
# ---------------------------------------------------------------------------


def select_review_sample(items: Sequence[EvalItem], n: int, *, seed: int = 0) -> list[EvalItem]:
    """Same round-robin-by-section approach as synth_eval_set.stratified_sample,
    over already-built EvalItems instead of raw chunks, keyed by the docs
    section their gold URL falls under -- deterministic given (items, n,
    seed), so a resumed session samples the identical subset."""
    import random

    by_section: dict[str, list[EvalItem]] = {}
    for item in items:
        by_section.setdefault(_item_section(item), []).append(item)

    rng = random.Random(seed)
    for bucket in by_section.values():
        rng.shuffle(bucket)

    sections = sorted(by_section)
    sampled: list[EvalItem] = []
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


def apply_review_decision(
    item: EvalItem, decision: str, *, edited_question: str | None = None, now: str | None = None
) -> EvalItem:
    """decision: 'y' correct, 'e' edit, 'n' reject, 's' skip (leave
    unverified, ask again next session)."""
    if decision == "s":
        return item
    if decision == "y":
        return item.model_copy(update={"verified": "yes", "verified_at": now or _now_iso()})
    if decision == "n":
        return item.model_copy(update={"verified": "no", "verified_at": now or _now_iso()})
    if decision == "e":
        if not edited_question:
            raise ValueError("decision 'e' requires edited_question")
        return item.model_copy(
            update={
                "question": edited_question,
                "verified": "edited",
                "verified_at": now or _now_iso(),
            }
        )
    raise ValueError(f"unknown review decision {decision!r} -- expected y/e/n/s")


def run_review_session(
    all_items: list[EvalItem],
    n: int,
    *,
    seed: int = 0,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    save_fn: SaveFn | None = None,
) -> list[EvalItem]:
    """Walks the stratified sample, prompting for each not-yet-verified item,
    saving the full item list after every single decision so a session that
    dies mid-review loses at most the item in progress. Typing 'q' stops the
    loop early; everything already decided stays decided."""
    by_id = {item.id: item for item in all_items}
    sample = select_review_sample(all_items, n, seed=seed)
    to_review = [item for item in sample if item.verified is None]

    for idx, item in enumerate(to_review, start=1):
        print_fn(f"[{idx}/{len(to_review)}] section: {_item_section(item)}")
        print_fn(f"  Q: {item.question}")
        if item.gold_urls:
            print_fn(f"  GOLD  {item.gold_urls[0]}")
        raw = input_fn("  [y] correct  [e] edit  [n] reject  [s] skip  [q] quit  > ").strip().lower()

        if raw == "q":
            break
        if raw == "e":
            edited = input_fn("  new question text > ").strip()
            updated = apply_review_decision(item, "e", edited_question=edited)
        elif raw in ("y", "n", "s"):
            updated = apply_review_decision(item, raw)
        else:
            print_fn(f"  unrecognized input {raw!r}, treating as skip")
            updated = apply_review_decision(item, "s")

        by_id[item.id] = updated
        if save_fn is not None:
            save_fn(list(by_id.values()))

    return list(by_id.values())


# ---------------------------------------------------------------------------
# label: discussions_v2, free-text gold URL entry per real question
# ---------------------------------------------------------------------------

CandidateFn = Callable[[str], list[str]]


def apply_label_decision(item: EvalItem, raw_input: str, *, now: str | None = None) -> EvalItem:
    """raw_input: comma-separated gold URLs, or 'n' for "no matching docs
    section" (this question gets excluded from the retrieval-eval slice of
    discussions_v2, kept only for generation quality via
    discussions_gen_v1)."""
    stripped = raw_input.strip()
    if stripped.lower() == "n":
        return item.model_copy(update={"gold_urls": [], "verified": "no", "verified_at": now or _now_iso()})
    urls = [u.strip() for u in stripped.split(",") if u.strip()]
    if not urls:
        raise ValueError("expected comma-separated URLs or 'n'")
    return item.model_copy(update={"gold_urls": urls, "verified": "yes", "verified_at": now or _now_iso()})


def run_label_session(
    all_items: list[EvalItem],
    *,
    candidate_fn: CandidateFn | None = None,
    input_fn: InputFn = input,
    print_fn: PrintFn = print,
    save_fn: SaveFn | None = None,
) -> list[EvalItem]:
    by_id = {item.id: item for item in all_items}
    to_label = [item for item in all_items if item.verified is None]

    for idx, item in enumerate(to_label, start=1):
        print_fn(f"[{idx}/{len(to_label)}]  {item.question}")
        if item.ground_truth:
            print_fn(f"  answer excerpt: {item.ground_truth[:200]}")
        if candidate_fn is not None:
            for rank, url in enumerate(candidate_fn(item.question), start=1):
                print_fn(f"   {rank}. {url}")
        raw = input_fn("  enter gold URLs (comma-separated) or 'n' for none  > ")

        try:
            updated = apply_label_decision(item, raw)
        except ValueError as exc:
            print_fn(f"  {exc}, skipping for now")
            continue

        by_id[item.id] = updated
        if save_fn is not None:
            save_fn(list(by_id.values()))

    return list(by_id.values())
