"""
00_init_db.py — (Re)create the pgvector schema for the e2e RAG pipeline.

Connects to a Postgres database with the pgvector extension (e.g. a Neon
project with pgvector enabled) and creates a single `chunks` table.

This script is safe to re-run: by default it drops the existing `chunks`
table first, so you can run it again after changing chunking strategy
(different chunk sizes/overlap change what should be stored) without
leaving behind stale rows from a previous approach.

Usage:
    python 00_init_db.py
    python 00_init_db.py --no-drop   # keep existing data, just ensure table exists
"""

from __future__ import annotations

import argparse
import os

from dotenv import load_dotenv
from psycopg import connect

load_dotenv()

# Embedding dimension must match the SentenceTransformer model used in 01_ingest_documents.py.
# BAAI/bge-m3 produces 1024-dimensional embeddings (same model used in db/ingest_kb.py).
EMBEDDING_DIM = 1024  # EDITABLE: must match the embedding model's output dimension


def get_connection_string() -> str:
    """Build the Postgres connection string from env vars.

    Neon (and most hosted Postgres providers) give you a single connection
    string — DATABASE_URL — instead of separate host/port/user/password vars.
    """
    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in your "
            "Neon connection string."
        )
    return database_url


def init_schema(conn, drop_existing: bool) -> None:
    with conn.cursor() as cur:
        # pgvector must be enabled before the VECTOR type can be used.
        cur.execute("CREATE EXTENSION IF NOT EXISTS vector;")

        if drop_existing:
            cur.execute("DROP TABLE IF EXISTS chunks;")

        cur.execute(
            f"""
            CREATE TABLE IF NOT EXISTS chunks (
                id BIGSERIAL PRIMARY KEY,
                file_path TEXT NOT NULL,
                file_name TEXT NOT NULL,
                chunk_order INT NOT NULL,
                chunk_text TEXT NOT NULL,
                chunk_char_count INT NOT NULL,
                chunking_strategy TEXT NOT NULL,
                embedding VECTOR({EMBEDDING_DIM}),
                metadata JSONB DEFAULT '{{}}'::jsonb,
                created_at TIMESTAMPTZ DEFAULT NOW(),
                CONSTRAINT uq_file_chunk_order UNIQUE (file_path, chunk_order)
            );
            """
        )

        # HNSW index for fast approximate nearest-neighbour search by cosine distance.
        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine
                ON chunks USING hnsw (embedding vector_cosine_ops);
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS idx_chunks_metadata
                ON chunks USING GIN(metadata);
            """
        )

    conn.commit()


def main() -> None:
    parser = argparse.ArgumentParser(description="(Re)create the chunks table for e2e_rag.")
    parser.add_argument(
        "--no-drop",
        action="store_true",
        help="Keep the existing table/data instead of dropping it first.",
    )
    args = parser.parse_args()

    conninfo = get_connection_string()

    with connect(conninfo) as conn:
        init_schema(conn, drop_existing=not args.no_drop)

    if args.no_drop:
        print("Schema ensured (existing data kept).")
    else:
        print("Schema (re)created. The chunks table is now empty — run 01_ingest_documents.py next.")


if __name__ == "__main__":
    main()
