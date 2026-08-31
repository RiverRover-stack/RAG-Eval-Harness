from types import SimpleNamespace

import pytest
from fastapi import HTTPException

from rag_eval.api.deps import get_app_state


def test_get_app_state_raises_503_when_backend_missing():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace()))
    with pytest.raises(HTTPException) as exc_info:
        get_app_state(request)
    assert exc_info.value.status_code == 503


def test_get_app_state_raises_503_when_backend_is_none():
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag=None)))
    with pytest.raises(HTTPException) as exc_info:
        get_app_state(request)
    assert exc_info.value.status_code == 503


def test_get_app_state_returns_state_when_present():
    sentinel = object()
    request = SimpleNamespace(app=SimpleNamespace(state=SimpleNamespace(rag=sentinel)))
    assert get_app_state(request) is sentinel
