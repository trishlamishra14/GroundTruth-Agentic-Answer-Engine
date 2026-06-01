"""
app/cohere_utils.py — make Cohere calls survive the trial key's 10 calls/minute
limit. On a 429 we wait and retry with increasing backoff; other errors raise
immediately. Wrap every Cohere call with call_with_retry(...).
"""

import time

# Backoff schedule (seconds). The trial limit is a rolling 1-minute window, so
# waits build up to ~60s to let it clear.
WAITS = [20, 40, 60, 60, 60]


def _is_rate_limit(exc: Exception) -> bool:
    return (
        exc.__class__.__name__ == "TooManyRequestsError"
        or getattr(exc, "status_code", None) == 429
    )


# Time spent sleeping for rate limits — so latency metrics can subtract it and
# report *real* inference latency instead of trial-key throttling.
_slept_seconds = 0.0


def take_slept_seconds() -> float:
    """Return seconds slept on rate limits since the last call, then reset."""
    global _slept_seconds
    value = _slept_seconds
    _slept_seconds = 0.0
    return value


def call_with_retry(fn, *args, **kwargs):
    global _slept_seconds
    last: Exception | None = None
    for wait in [0, *WAITS]:
        if wait:
            print(f"  [cohere rate limit] waiting {wait}s then retrying...")
            _slept_seconds += wait
            time.sleep(wait)
        try:
            return fn(*args, **kwargs)
        except Exception as exc:  # noqa: BLE001 - re-raised below unless rate limit
            if not _is_rate_limit(exc):
                raise
            last = exc
    raise last
