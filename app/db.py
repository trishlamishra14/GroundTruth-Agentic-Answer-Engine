"""
app/db.py — connect to Postgres and create the pgvector-backed `chunks` table.
"""

import psycopg
from pgvector.psycopg import register_vector

from app.config import settings


def connect():
    """Open a connection and teach it the pgvector 'vector' type."""
    conn = psycopg.connect(settings.database_url)
    register_vector(conn)
    return conn


def init_schema():
    """Create the extension and table if missing. Safe to run repeatedly."""
    with psycopg.connect(settings.database_url) as conn:
        conn.execute("CREATE EXTENSION IF NOT EXISTS vector;")
        conn.commit()
        register_vector(conn)
        conn.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id        BIGSERIAL PRIMARY KEY,
                doc       TEXT NOT NULL,
                content   TEXT NOT NULL,
                embedding VECTOR({settings.embedding_dim})
            );
            """
        )
        # Full-text index speeds up the keyword retriever.
        conn.execute(
            "CREATE INDEX IF NOT EXISTS chunks_fts "
            "ON chunks USING GIN (to_tsvector('english', content));"
        )
        conn.commit()
    print("Schema ready: 'chunks' table exists.")
