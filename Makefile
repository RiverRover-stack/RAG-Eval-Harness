.PHONY: setup lint type test index eval serve demo

setup:
	uv sync --extra dev

lint:
	uv run ruff check .

type:
	uv run mypy src

test:
	uv run pytest -m "not slow and not llm"

index:
	uv run python -m rag_eval.ingestion.embed_and_store

eval:
	uv run rag-eval eval run --config configs/baseline.yaml

serve:
	uv run uvicorn rag_eval.api.main:app --reload

demo: index serve
