"""
app/retrieve.py — find the best passages for a question.

Hybrid retrieval:
  vector search (meaning)  +  keyword search (exact words)
      -> Reciprocal Rank Fusion (combine the two ranked lists)
      -> Cohere Rerank (read query + passage together, score true relevance)

`retrieve()` returns the top reranked passages, each with a `score`
(the rerank relevance score, 0..1) used later for the abstain decision.
"""

import cohere

from app.config import settings
from app.cohere_utils import call_with_retry
from app.db import connect
from app.embed import embed_query

co = cohere.ClientV2(api_key=settings.cohere_api_key)


def _vec_literal(values: list[float]) -> str:
    """pgvector text form: [0.1,0.2,...]. Cast with ::vector so the <=> operator
    sees a real vector instead of a generic float array."""
    return "[" + ",".join(map(str, values)) + "]"


def vector_search(conn, query_vec: list[float], k: int) -> list[dict]:
    vec = _vec_literal(query_vec)
    rows = conn.execute(
        """
        SELECT id, doc, content, 1 - (embedding <=> %s::vector) AS score
        FROM chunks
        ORDER BY embedding <=> %s::vector
        LIMIT %s
        """,
        (vec, vec, k),
    ).fetchall()
    return [{"id": r[0], "doc": r[1], "content": r[2], "score": float(r[3])} for r in rows]


def keyword_search(conn, question: str, k: int) -> list[dict]:
    rows = conn.execute(
        """
        SELECT id, doc, content,
               ts_rank(to_tsvector('english', content),
                       plainto_tsquery('english', %s)) AS score
        FROM chunks
        WHERE to_tsvector('english', content) @@ plainto_tsquery('english', %s)
        ORDER BY score DESC
        LIMIT %s
        """,
        (question, question, k),
    ).fetchall()
    return [{"id": r[0], "doc": r[1], "content": r[2], "score": float(r[3])} for r in rows]


def reciprocal_rank_fusion(result_lists: list[list[dict]], k: int = 60) -> list[dict]:
    """A chunk ranking high in BOTH lists bubbles to the top: score = sum 1/(k+rank)."""
    scores: dict[int, float] = {}
    meta: dict[int, dict] = {}
    for results in result_lists:
        for rank, item in enumerate(results):
            cid = item["id"]
            scores[cid] = scores.get(cid, 0.0) + 1.0 / (k + rank + 1)
            meta[cid] = item
    ordered = sorted(scores, key=lambda cid: scores[cid], reverse=True)
    return [meta[cid] for cid in ordered]


def rerank(question: str, candidates: list[dict], top_n: int) -> list[dict]:
    if not candidates:
        return []
    resp = call_with_retry(
        co.rerank,
        model=settings.rerank_model,
        query=question,
        documents=[c["content"] for c in candidates],
        top_n=min(top_n, len(candidates)),
    )
    out: list[dict] = []
    for result in resp.results:
        item = dict(candidates[result.index])
        item["score"] = float(result.relevance_score)
        out.append(item)
    return out


def retrieve(question: str, top_k: int | None = None, top_n: int | None = None) -> list[dict]:
    """Full hybrid retrieve -> fuse -> rerank. Returns top passages with scores."""
    top_k = top_k or settings.top_k
    top_n = top_n or settings.rerank_top_n

    query_vec = embed_query(question)
    with connect() as conn:
        vec_hits = vector_search(conn, query_vec, k=top_k)
        kw_hits = keyword_search(conn, question, k=top_k)

    fused = reciprocal_rank_fusion([vec_hits, kw_hits])
    return rerank(question, fused, top_n=top_n)
