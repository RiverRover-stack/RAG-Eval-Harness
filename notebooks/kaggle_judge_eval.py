"""
Score a RAG eval run's answers with RAGAS, judged by Qwen2.5-32B-Instruct
served locally (inside this notebook) via Ollama on Kaggle's free GPU.

This is the GPU-offload half of `rag-eval eval judge` (docs/plan.md Phase
8). It is intentionally self-contained (no dependency on installing the
rag_eval package, no git remote needed) -- paste it into a Kaggle Notebook,
split on the `# %%` markers into cells (or run top to bottom as a Script
kernel), and run it as a batch job ("Save & Run All").

--- One-time setup, before running ---

1. Notebook Settings -> Accelerator -> GPU T4 x2.
   NOT P100 -- a single 16GB P100 cannot fit Qwen2.5-32B-Instruct even at
   4-bit quantization (~20GB on disk, ~22-25GB at runtime with KV cache).
   T4 x2 pools 2x16GB=32GB and Ollama (v0.4+) auto-splits model layers
   across both GPUs with no extra config.
2. Notebook Settings -> Internet -> on (needed to install Ollama and pull
   the model; requires phone verification on your Kaggle account).
3. Locally, export the run's answers and upload them as a Kaggle Dataset:
     uv run rag-eval eval judge <run_id> --export-only
   This writes runs/<run_id>/judge_export.jsonl (item_id, question,
   answer, contexts, ground_truth -- no judge scores yet). Upload that file
   as a Kaggle Dataset and attach it to this notebook, then set INPUT_PATH
   below to match its mounted path under /kaggle/input/.
4. Run as "Save & Run All" (batch/commit mode), not just interactively --
   Kaggle's interactive sessions idle-kill after ~1h. Batch sessions run
   uninterrupted and are capped at ~9h.
5. After the run, download judge_scored.jsonl from the notebook's Output
   tab and merge it back into the run:
     uv run rag-eval eval judge <run_id> --score-only path/to/judge_scored.jsonl

Notes on quotas (verified against Kaggle's own product-feedback threads and
the Ollama library page as of 2026-08): free tier gives ~30 GPU-hours/week
(resets Saturday UTC); a q4_K_M pull of qwen2.5:32b-instruct is ~20GB, so
budget a few minutes of the session for the download itself.

This script deliberately does NOT expose Ollama to the outside world (no
ngrok/cloudflared tunnel) -- Kaggle's docs and community reports flag SSH
tunneling into kernels as against the Acceptable Use Policy. Keeping
everything self-contained inside the notebook (upload input -> run ->
download output) sidesteps that risk and fits how this is actually used:
one-shot dev-time validation and ad-hoc bulk runs, not an always-on judge.
"""

# %% Cell 1 -- install Ollama (its installer shells out to zstd for
# extraction; Kaggle's base image doesn't have it -- confirmed live 2026-09)
import subprocess

subprocess.run("apt-get update -qq && apt-get install -y -qq zstd", shell=True, check=True)
subprocess.run("curl -fsSL https://ollama.com/install.sh | sh", shell=True, check=True)

# %% Cell 2 -- sanity-check the GPU setup before committing to a 20GB pull
subprocess.run(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv"], check=True)

# %% Cell 3 -- start `ollama serve` in the background, wait until it's ready
import time
import urllib.error
import urllib.request

OLLAMA_URL = "http://localhost:11434"

ollama_proc = subprocess.Popen(
    ["ollama", "serve"], stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
)

for _ in range(30):
    try:
        urllib.request.urlopen(OLLAMA_URL, timeout=2)
        break
    except (urllib.error.URLError, ConnectionError, TimeoutError):
        time.sleep(1)
else:
    raise RuntimeError("ollama serve did not become ready within 30s")

print("ollama serve is up")

# %% Cell 4 -- pull the judge model + embedding model
JUDGE_MODEL = "qwen2.5:32b-instruct"  # default tag = q4_K_M, ~20GB
EMBED_MODEL = "nomic-embed-text"      # matches ollama_embed_model in local .env

subprocess.run(["ollama", "pull", JUDGE_MODEL], check=True)
subprocess.run(["ollama", "pull", EMBED_MODEL], check=True)

# %% Cell 5 -- install RAGAS + friends (versions match pyproject.toml)
subprocess.run(
    [
        "pip", "install", "-q",
        "ragas>=0.2,<0.4",
        "datasets>=2.20",
        "langchain-ollama>=0.2",
        "langchain-core>=0.3",
    ],
    check=True,
)

# %% Cell 6 -- load the exported run answers (uploaded as a Kaggle Dataset)
import json

# Update this to match your attached dataset's mounted path, e.g.
# "/kaggle/input/rag-eval-judge-export/judge_export.jsonl"
INPUT_PATH = "/kaggle/input/<your-dataset-slug>/judge_export.jsonl"

with open(INPUT_PATH, encoding="utf-8") as f:
    rows = [json.loads(line) for line in f if line.strip()]

item_ids = [r["item_id"] for r in rows]
score_cols = ("question", "answer", "contexts", "ground_truth")

from datasets import Dataset

dataset = Dataset.from_list([{k: r[k] for k in score_cols} for r in rows])
print(f"Loaded {len(dataset)} rows")

# %% Cell 7 -- score with RAGAS, judged by Qwen 32B Instruct on the GPU
from langchain_ollama import ChatOllama, OllamaEmbeddings
from ragas import evaluate
from ragas.metrics import answer_relevancy, context_precision, context_recall, faithfulness
from ragas.run_config import RunConfig

judge_llm = ChatOllama(model=JUDGE_MODEL, base_url=OLLAMA_URL)
judge_embeddings = OllamaEmbeddings(model=EMBED_MODEL, base_url=OLLAMA_URL)
# 3 matches `rag-eval eval judge` (docs/plan.md Phase 8); the old 1 was a
# single-sample estimate and produced runs of exact 0.0s.
answer_relevancy.strictness = 3

# Real GPU judge behind ollama's own request queue -- 2 workers is a safe
# starting point (vs. max_workers=1 forced locally on CPU-only Ollama).
run_config = RunConfig(max_workers=2, timeout=600)

results = evaluate(
    dataset,
    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
    llm=judge_llm,
    embeddings=judge_embeddings,
    run_config=run_config,
)
print(results)

# %% Cell 8 -- save results; download from the notebook's Output tab after the run
METRIC_NAMES = ("faithfulness", "answer_relevancy", "context_precision", "context_recall")
frame = results.to_pandas()

# judge_scored.jsonl: item_id + the four metric columns, in input order --
# `rag-eval eval judge <run_id> --score-only judge_scored.jsonl` merges it back.
with open("/kaggle/working/judge_scored.jsonl", "w", encoding="utf-8") as f:
    for item_id, (_, record) in zip(item_ids, frame.iterrows()):
        f.write(
            json.dumps(
                {"item_id": item_id, **{m: float(record[m]) for m in METRIC_NAMES}}
            )
            + "\n"
        )

frame.to_csv("/kaggle/working/ragas_results.csv", index=False)
