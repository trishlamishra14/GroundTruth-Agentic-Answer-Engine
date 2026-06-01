"""
app/observability.py — the boring-but-senior part: measure latency and cost.

- `token_usage(resp)` pulls input/output token counts out of a Cohere response
  (the field has moved around across SDK versions, so we look in a few places).
- `estimate_cost(...)` turns tokens into an approximate USD figure.
- `METRICS` keeps the last N requests in memory so /metrics can report p50/p95
  latency and average cost — a tiny observability dashboard with no extra infra.
"""

import time
from collections import deque

from app.config import settings


def token_usage(resp) -> tuple[int, int]:
    """Best-effort (input_tokens, output_tokens) from a Cohere chat response."""
    usage = getattr(resp, "usage", None)
    if usage is not None:
        for attr in ("billed_units", "tokens"):
            units = getattr(usage, attr, None)
            if units is not None:
                inp = getattr(units, "input_tokens", None)
                out = getattr(units, "output_tokens", None)
                if inp is not None or out is not None:
                    return int(inp or 0), int(out or 0)
    return 0, 0


def estimate_cost(input_tokens: int, output_tokens: int) -> float:
    return (
        input_tokens / 1_000_000 * settings.price_in_per_mtok
        + output_tokens / 1_000_000 * settings.price_out_per_mtok
    )


class Metrics:
    """Rolling window of recent requests for a lightweight /metrics endpoint."""

    def __init__(self, maxlen: int = 500):
        self.latencies: deque[float] = deque(maxlen=maxlen)
        self.costs: deque[float] = deque(maxlen=maxlen)
        self.abstentions = 0
        self.total = 0

    def record(self, latency_ms: float, cost_usd: float, abstained: bool) -> None:
        self.latencies.append(latency_ms)
        self.costs.append(cost_usd)
        self.total += 1
        if abstained:
            self.abstentions += 1

    @staticmethod
    def _pct(values: list[float], p: float) -> float:
        if not values:
            return 0.0
        ordered = sorted(values)
        idx = min(len(ordered) - 1, int(round((p / 100) * (len(ordered) - 1))))
        return ordered[idx]

    def summary(self) -> dict:
        lat = list(self.latencies)
        return {
            "requests": self.total,
            "p50_latency_ms": round(self._pct(lat, 50), 1),
            "p95_latency_ms": round(self._pct(lat, 95), 1),
            "avg_cost_usd": round(sum(self.costs) / len(self.costs), 6) if self.costs else 0.0,
            "abstention_rate": round(self.abstentions / self.total, 3) if self.total else 0.0,
        }


# Single shared instance.
METRICS = Metrics()


class Timer:
    """`with Timer() as t: ...`  ->  t.ms gives elapsed milliseconds."""

    def __enter__(self):
        self._start = time.perf_counter()
        self.ms = 0.0
        return self

    def __exit__(self, *exc):
        self.ms = (time.perf_counter() - self._start) * 1000
        return False
