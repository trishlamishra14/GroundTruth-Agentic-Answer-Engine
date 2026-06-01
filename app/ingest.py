"""
app/ingest.py — OFFLINE phase: read docs -> chunk -> embed -> store in Postgres.

Run:  python -m app.ingest
"""

from pathlib import Path

from app.config import settings
from app.chunking import chunk_text
from app.db import connect, init_schema
from app.embed import embed_documents


def load_chunks(docs_dir: str) -> list[tuple[str, str]]:
    items: list[tuple[str, str]] = []
    for path in sorted(Path(docs_dir).glob("**/*.md")):
        text = path.read_text(encoding="utf-8", errors="ignore")
        for chunk in chunk_text(text):
            items.append((path.name, chunk))
    return items


def ingest(docs_dir: str = "data/sample_docs") -> None:
    init_schema()

    items = load_chunks(docs_dir)
    if not items:
        raise SystemExit(f"No .md files found under {docs_dir!r}.")

    texts = [content for _doc, content in items]
    print(f"Embedding {len(texts)} chunks with Cohere...")
    vectors = embed_documents(texts)

    with connect() as conn:
        conn.execute("TRUNCATE chunks RESTART IDENTITY;")
        for (doc, content), vector in zip(items, vectors):
            conn.execute(
                "INSERT INTO chunks (doc, content, embedding) VALUES (%s, %s, %s)",
                (doc, content, vector),
            )
        conn.commit()

    print(f"Stored {len(items)} chunks from {docs_dir}. Done.")


if __name__ == "__main__":
    ingest()
