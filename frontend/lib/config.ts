// Same-origin ('') in the exported build, since FastAPI serves the built
// frontend and /api/* from one process (StaticFiles + the API router in
// src/rag_eval/api/main.py). In `next dev`, set NEXT_PUBLIC_API_BASE_URL to
// the FastAPI dev server's origin (see frontend/.env.local.example) --
// `output: 'export'` drops next.config rewrites from the static build, so
// an env var is simpler than a rewrite that would only work in one of the
// two run modes.
export const API_BASE = process.env.NEXT_PUBLIC_API_BASE_URL ?? "";
