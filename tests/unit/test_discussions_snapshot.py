from pathlib import Path

import pytest

from rag_eval.common.schemas import DiscussionQA
from rag_eval.ingestion import discussions_snapshot as ds


def _qa(n: int) -> DiscussionQA:
    return DiscussionQA(
        discussion_id=f"D_{n}",
        title=f"Question {n}",
        question_body=f"Body of question {n}",
        answer_body=f"Answer to question {n}",
        url=f"https://github.com/fastapi/fastapi/discussions/{n}",
        category="Q&A",
    )


def test_fetch_snapshot_writes_frozen_json(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    qas = [_qa(1), _qa(2)]
    monkeypatch.setattr(ds, "fetch_discussion_qas", lambda max_pages: qas)
    out = tmp_path / "discussions.json"

    snapshot = ds.fetch_snapshot(max_pages=3, out=out)

    assert out.exists()
    assert snapshot["n_discussions"] == 2
    assert snapshot["max_pages"] == 3
    assert "fetched_at" in snapshot


def test_load_snapshot_round_trips_discussion_qas(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    qas = [_qa(1), _qa(2)]
    monkeypatch.setattr(ds, "fetch_discussion_qas", lambda max_pages: qas)
    out = tmp_path / "discussions.json"
    ds.fetch_snapshot(max_pages=None, out=out)

    loaded = ds.load_snapshot(out)

    assert loaded == qas


def test_fetch_snapshot_creates_parent_dirs(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(ds, "fetch_discussion_qas", lambda max_pages: [])
    out = tmp_path / "nested" / "dir" / "discussions.json"

    ds.fetch_snapshot(max_pages=1, out=out)

    assert out.exists()
