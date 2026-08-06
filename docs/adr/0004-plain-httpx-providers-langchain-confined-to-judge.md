# 4. Plain-httpx providers, LangChain confined to the judge

## Status

Accepted

## Context

Pre-Phase-2, `rag/generator.py` called the `ollama` Python package directly
and `eval/run_ragas.py` built `ChatGroq` / `ChatGoogleGenerativeAI` /
`ChatOllama` LangChain objects inline for the RAGAS judge. Extending that
pattern to three serving-path providers (Groq, Gemini, Ollama) would mean
three more LangChain chat-model dependencies in the hot request path, plus
whatever abstraction LangChain imposes on streaming (Phase 7 needs Groq
SSE, Gemini `alt=sse`, and Ollama's NDJSON-not-SSE format normalized to one
shape) and on error handling.

RAGAS, however, is not optional here: `ragas.evaluate()`'s public API
takes LangChain LLM/embeddings objects as its `llm=`/`embeddings=`
arguments. There is no judge-side way to avoid that dependency without
forking RAGAS itself.

## Decision

The serving path (`providers/llm/{groq,gemini,ollama}.py`) is plain
`httpx` calling each provider's REST API directly -- no LangChain in the
request path a live user waits on. `providers/langchain_adapters.py` is
the **one** module in the codebase permitted to import LangChain, and it
exists solely to hand `eval/run_ragas.py` the `(llm, embeddings)` pair
RAGAS's `evaluate()` requires. `langchain-ollama`, `langchain-groq`, and
`langchain-google-genai` remain dependencies for exactly this reason;
`langchain-core` / `langchain-community` are transitive from those.

Phase 2 ships `complete()` only on every serving provider --
`astream()`/SSE-normalization is Phase 7's problem, once there's an actual
streaming consumer to write it against.

## Consequences

A provider swap or a new provider (e.g. adding Anthropic later) touches
one small `httpx`-based file and never touches `langchain_adapters.py`.
Streaming, when it lands in Phase 7, is normalized once per provider
inside `providers/llm/*.py` rather than inheriting whatever streaming
abstraction LangChain's chat models expose. The cost: response parsing
(pulling `choices[0].message.content` out of Groq's JSON, `candidates[0]
.content.parts[0].text` out of Gemini's) is written and maintained by
hand instead of borrowed from a client library -- acceptable because each
provider's REST shape is small and stable, and it's exactly the code
`test_providers.py` exists to pin down.
