# GroundTruth

An **agentic answer engine** over an AI / GenAI concepts knowledge base (LLMs, embeddings,
RAG, vector search, prompting, agents). It retrieves, re-ranks, and
answers **with inline citations** — it tries harder on weak queries (rewrite + retry) and
**abstains** ("not in the docs") instead of hallucinating. The headline isn't the chatbot;
it's the **evaluation + reliability harness** that proves it works.

Built on **Cohere** (embeddings + Command + Rerank) and **Postgres / pgvector**.

```
                          ┌─────────── agent loop (LangGraph or plain) ───────────┐
question ─► embed query ─►│  retrieve (vector + keyword → RRF → Cohere rerank)     │
                          │        │                                              │
                          │     grade top score                                   │
                          │   ├ good  ──────────────► generate cited answer       │
                          │   ├ weak + retries ─────► rewrite query → retrieve     │
                          │   └ weak, no retries ───► ABSTAIN                      │
                          └───────────────────────────────────────────────────────┘
                                   │ every call logs latency + token cost
                                   ▼
                    answer + sources + trace + latency_ms + est_cost_usd
```

## What makes it stand out (not a tutorial clone)

- **Abstention, measured.** Out-of-scope questions are *supposed* to be refused. The gold
  set includes "should-refuse" cases and we score **abstention accuracy** explicitly.
- **An agent, not a one-shot.** It grades retrieval and rewrites the query to retry before
  giving up — implemented as a real **LangGraph** state graph (`app/graph.py`).
- **Evaluation as a CI gate.** `eval/run_eval.py` scores retrieval recall, faithfulness
  (LLM judge), and abstention accuracy, writes `report.json`, compares to the previous run,
  and **exits non-zero on regressions** — wired into GitHub Actions.
- **Observability + cost.** Every answer carries `latency_ms` and `est_cost_usd`; `/metrics`
  reports rolling p50/p95 latency and average cost.
- **Hybrid retrieval.** Vector + keyword search fused with Reciprocal Rank Fusion, then a
  Cohere reranker picks the final passages.

## Quickstart

```bash
# 0) Postgres with pgvector (Docker)
docker compose up -d db

# 1) Environment
python -m venv .venv && .venv\Scripts\activate      # mac/linux: source .venv/bin/activate
pip install -r requirements.txt
copy .env.example .env                               # then paste your Cohere key

# 2) Build the index from the sample docs
python -m app.ingest

# 3) Ask (terminal-free: use the API)
uvicorn app.main:app --reload
#   open http://localhost:8000/docs  and POST /query  {"question": "..."}

# 4) Measure quality (the important part)
python -m eval.run_eval
```

Try an in-scope and an out-of-scope question to see grounding vs. abstention:

```json
POST /query  {"question": "What is retrieval-augmented generation?"}
POST /query  {"question": "What is the capital of Australia?"}   // -> abstains
```

## Run the whole stack in Docker

```bash
docker compose up --build       # db + api; then run `python -m app.ingest` once
```

## Use your own documents (scale up)

The repo ships with a small AI/GenAI knowledge base so it runs end-to-end out of the box. To
point it at any other corpus, just drop `.md` files into `data/sample_docs/` and re-ingest:

```bash
# add your own .md files to data/sample_docs/, then:
python -m app.ingest
```

Then grow `eval/gold.jsonl` toward ~150 questions (including more out-of-scope cases) so the
benchmark stays meaningful as the corpus grows.

## Project layout

```
app/
  config.py        typed settings from .env
  chunking.py      sentence-aware chunker with overlap
  db.py            Postgres + pgvector schema
  embed.py         Cohere document/query embeddings
  retrieve.py      vector + keyword + RRF + Cohere rerank
  agent.py         agentic loop (retrieve→grade→rewrite/generate/abstain)
  graph.py         the same loop as a LangGraph StateGraph
  observability.py latency + token-cost tracking, rolling metrics
  rag.py           answer() entry point (timing + cost + metrics)
  main.py          FastAPI: /health /query /metrics
  ingest.py        offline: docs → chunks → embeddings → Postgres
eval/
  gold.jsonl       in-scope + out-of-scope questions
  run_eval.py      recall + faithfulness + abstention; report + CI gate
tests/             unit tests (chunking, RRF)
.github/workflows/eval.yml   lint + tests always; eval gate when COHERE_API_KEY is set
```

## Resume numbers to fill in (after you run the eval)

Run `python -m eval.run_eval` and copy the real figures into your resume:
retrieval recall, faithfulness, abstention accuracy, p95 latency, and cost/query.

> Status: working core + agent + eval + CI. Next polish: deploy a public demo (Render / Fly /
> HF Spaces) and grow the gold set. See `WALKTHROUGH.md` for how to talk about it in interviews.
