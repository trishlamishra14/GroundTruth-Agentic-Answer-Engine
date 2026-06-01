"""
app/main.py — FastAPI web service.

  GET  /health   -> liveness
  POST /query    -> {question} -> grounded answer + sources + trace + latency/cost
  GET  /metrics  -> rolling p50/p95 latency, avg cost, abstention rate

Run:  uvicorn app.main:app --reload
Docs: http://localhost:8000/docs
"""

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse

from app.rag import answer
from app.observability import METRICS
from app.schemas import QueryRequest, QueryResponse

app = FastAPI(
    title="GroundTruth",
    description="Agentic answer engine over docs — cited answers, abstention, and live metrics.",
    version="1.0.0",
)

STATIC_DIR = Path(__file__).parent / "static"


@app.get("/", include_in_schema=False)
def home() -> FileResponse:
    """Serve the web UI."""
    return FileResponse(STATIC_DIR / "index.html")


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/query", response_model=QueryResponse)
def query(req: QueryRequest) -> dict:
    return answer(req.question)


@app.get("/metrics")
def metrics() -> dict:
    return METRICS.summary()


@app.get("/report")
def report() -> dict:
    """Latest eval scorecard (written by `python -m eval.run_eval`)."""
    path = Path("eval/report.json")
    return json.loads(path.read_text()) if path.exists() else {}
