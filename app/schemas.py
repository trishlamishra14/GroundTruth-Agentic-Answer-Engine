"""
app/schemas.py — typed request/response shapes for the API.
"""

from pydantic import BaseModel, Field


class QueryRequest(BaseModel):
    question: str = Field(..., min_length=3,
                          examples=["How do I declare a query parameter in FastAPI?"])


class Source(BaseModel):
    rank: int
    doc: str
    score: float
    snippet: str


class QueryResponse(BaseModel):
    answer: str
    sources: list[Source]
    grounded: bool
    abstained: bool
    loops: int
    latency_ms: float
    est_cost_usd: float
    trace: list[str] = []
