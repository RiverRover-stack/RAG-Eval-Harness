---
title: RAG Eval Harness
emoji: 🔍
colorFrom: blue
colorTo: indigo
sdk: docker
app_port: 7860
pinned: false
---

# RAG Eval Harness

A RAG pipeline over the FastAPI docs + GitHub Discussions, with a
judge-free retrieval eval as the primary metric and RAGAS-based generation
scoring on top.

Source: https://github.com/RiverRover-stack/RAG-Eval-Harness

- `GET /api/health` — liveness
- `GET /api/health/ready` — readiness (asserts the baked index is present and non-empty)
- `POST /api/ask` — `{"question": "...", "k": 5}`
