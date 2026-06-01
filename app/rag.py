"""
app/rag.py — public entry point used by the API and the eval harness.

Wraps the agent with timing + cost so every answer carries latency_ms and
est_cost_usd, and records them into the rolling METRICS window.
"""

from app.config import settings
from app.cohere_utils import take_slept_seconds
from app.observability import METRICS, Timer, estimate_cost
from app.agent import run_agent


def answer(question: str) -> dict:
    with Timer() as t:
        if settings.use_langgraph:
            from app.graph import run_agent_graph
            result = run_agent_graph(question)
        else:
            result = run_agent(question)

    # Subtract any rate-limit sleeping so latency reflects real inference time.
    latency_ms = max(0.0, t.ms - take_slept_seconds() * 1000)
    cost = estimate_cost(result["in_tokens"], result["out_tokens"])
    result["latency_ms"] = round(latency_ms, 1)
    result["est_cost_usd"] = round(cost, 6)
    METRICS.record(latency_ms, cost, result["abstained"])
    return result
