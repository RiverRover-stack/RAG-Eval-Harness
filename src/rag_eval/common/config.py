"""Central settings, loaded from environment / .env via pydantic-settings."""

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    # GitHub GraphQL
    github_token: str = ""
    github_repo_owner: str = "fastapi"
    github_repo_name: str = "fastapi"

    # Chroma
    chroma_persist_dir: str = "./data/processed/chroma"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_llm_model: str = "fdm-llama"
    ollama_embed_model: str = "nomic-embed-text"

    # Ingestion scope: caps on how much of each source gets indexed, so a
    # full ingest + eval pass stays fast and the retrieval corpus stays
    # small enough for a clean local run. 0 / None = no cap (full corpus).
    ingest_docs_limit: int = 30
    ingest_discussion_pages: int = 1

    # Eval
    eval_set_path: str = "./data/eval_sets/fastapi_discussions_eval.jsonl"

    # RAGAS judge: "ollama" (local), "groq" (hosted, free tier), or "gemini"
    # (hosted, free tier, native JSON mode -- best structured-output reliability)
    ragas_judge: str = "ollama"
    groq_api_key: str = ""
    groq_model: str = "llama-3.1-8b-instant"
    gemini_api_key: str = ""
    gemini_model: str = "gemini-2.5-flash"

    # RAGAS execution: kept conservative for CPU-only local inference, where
    # concurrency just queues jobs behind a single-threaded model instead of
    # speeding anything up. Bump max_workers only if the judge is hosted or
    # OLLAMA_NUM_PARALLEL is raised. Note: Gemini's free tier is only 10
    # requests/minute -- more workers won't raise that ceiling, they just
    # mean more simultaneous 429s for ragas's own retry/backoff to absorb.
    ragas_max_workers: int = 1
    ragas_timeout: int = 600

    # Eval set sampling: 0 = use the full eval set. Set to a small number
    # (e.g. 10-15) to validate a judge/config change quickly and cheaply
    # before spending a full day's free-tier quota on all 27 questions.
    eval_sample_limit: int = 0

    # Docs (for building chunk source URLs, path + anchor)
    docs_base_url: str = "https://fastapi.tiangolo.com"


settings = Settings()
