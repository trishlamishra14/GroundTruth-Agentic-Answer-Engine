"""
app/chunking.py — split a long document into smaller, overlapping chunks.

Small, focused chunks -> sharp embeddings -> precise retrieval. A little overlap
between neighbours keeps a fact that sits on a boundary from being cut in half.
"""

import re


def split_sentences(text: str) -> list[str]:
    """Cut after a '.', '!' or '?' that is followed by whitespace."""
    parts = re.split(r"(?<=[.!?])\s+", text.strip())
    return [p.strip() for p in parts if p.strip()]


def chunk_text(text: str, max_chars: int = 700, overlap_sentences: int = 1) -> list[str]:
    """Pack sentences into chunks no larger than `max_chars`, carrying a tail of
    `overlap_sentences` sentence(s) into the next chunk for context continuity."""
    sentences = split_sentences(text)
    chunks: list[str] = []
    current: list[str] = []
    current_len = 0

    for sentence in sentences:
        if current and current_len + len(sentence) > max_chars:
            chunks.append(" ".join(current))
            current = current[-overlap_sentences:] if overlap_sentences else []
            current_len = sum(len(s) for s in current)
        current.append(sentence)
        current_len += len(sentence)

    if current:
        chunks.append(" ".join(current))
    return chunks
