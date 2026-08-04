"""Shared pydantic models used across ingestion, rag, and eval modules."""

from pydantic import BaseModel


class DiscussionQA(BaseModel):
    """One accepted-answer Q&A pair pulled from GitHub Discussions via GraphQL."""

    discussion_id: str
    title: str
    question_body: str
    answer_body: str
    url: str
    category: str | None = None


class RetrievedChunk(BaseModel):
    content: str
    source_id: str
    score: float


class EvalExample(BaseModel):
    """One row of the eval set, in RAGAS-compatible shape."""

    question: str
    ground_truth: str
    contexts: list[str] = []
    answer: str | None = None
    source_url: str | None = None
