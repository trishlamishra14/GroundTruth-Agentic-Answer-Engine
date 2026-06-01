"""
app/embed.py — Cohere embedding helpers (documents vs. queries).

We use input_type='search_document' when storing chunks and
input_type='search_query' when embedding a question. Matching these two
modes is what makes Cohere retrieval accurate.
"""

import cohere

from app.config import settings
from app.cohere_utils import call_with_retry

co = cohere.ClientV2(api_key=settings.cohere_api_key)


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed many chunks. Cohere accepts up to 96 texts per call, so batch."""
    out: list[list[float]] = []
    for i in range(0, len(texts), 96):
        resp = call_with_retry(
            co.embed,
            model=settings.embedding_model,
            texts=texts[i:i + 96],
            input_type="search_document",
            embedding_types=["float"],
        )
        out.extend(resp.embeddings.float)
    return out


def embed_query(question: str) -> list[float]:
    resp = call_with_retry(
        co.embed,
        model=settings.embedding_model,
        texts=[question],
        input_type="search_query",
        embedding_types=["float"],
    )
    return resp.embeddings.float[0]
