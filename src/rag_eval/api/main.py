"""FastAPI app exposing the RAG pipeline for ad-hoc querying."""

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from rag_eval.rag.pipeline import answer_question

app = FastAPI(title="RAG Eval Harness API")


class AskRequest(BaseModel):
    question: str
    k: int = 5


class AskResponse(BaseModel):
    question: str
    answer: str
    contexts: list[str]
    sources: list[str]


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/ask", response_model=AskResponse)
def ask(req: AskRequest) -> AskResponse:
    try:
        result = answer_question(req.question, k=req.k)
    except ConnectionError as e:
        raise HTTPException(status_code=503, detail=f"RAG backend unavailable: {e}") from e
    return AskResponse(**result)
