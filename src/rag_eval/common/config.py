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

    # fastembed's ONNX model cache -- a path, not metric-affecting, so it
    # stays in Settings rather than RunConfig. Phase 3 bakes this into a
    # Docker layer.
    fastembed_cache_dir: str = "./data/processed/fastembed"

    # Ollama
    ollama_base_url: str = "http://localhost:11434"
    ollama_embed_model: str = "nomic-embed-text"

    # Eval (legacy single-file discussions set; superseded by the per-dataset
    # files under data/eval_sets/ built via eval/synth_eval_set.py, but
    # eval/build_eval_set.py still targets this path)
    eval_set_path: str = "./data/eval_sets/fastapi_discussions_eval.jsonl"

    # API keys only -- *which* provider/model to use for generation or
    # judging is decided entirely by RunConfig (configs/*.yaml:
    # generation.llm, eval.judge -- see the role table in configs/_base.yaml).
    # A key belongs here because it can't change a metric; a model choice
    # can, so it never does.
    groq_api_key: str = ""
    gemini_api_key: str = ""
    literouter_api_key: str = ""

    # HTTP request timeout for the Ollama provider -- the serving path, and
    # (when eval.judge.provider is "ollama") the judge.
    ollama_timeout: int = 600

    # Eval set sampling: 0 = use the full eval set. Set to a small number
    # (e.g. 10-15) to validate a judge/config change quickly and cheaply
    # before spending a full day's free-tier quota on all 27 questions.
    eval_sample_limit: int = 0

    # Docs (for building chunk source URLs, path + anchor)
    docs_base_url: str = "https://fastapi.tiangolo.com"

    # Which RunConfig the served API builds its RetrievalPipeline/LLM/prompt
    # from (api/deps.py) -- a path, not a metric-affecting value itself, so
    # it lives here rather than inside the RunConfig it points at.
    default_run_config: str = "configs/deploy.yaml"

    # Daily spend cap on /api/ask* (api/rate_limit.py) -- a public demo on a
    # real API key gets scraped; this is an operational safety valve, not
    # anything that changes a retrieval/generation metric.
    daily_budget_usd: float = 5.0


settings = Settings()
