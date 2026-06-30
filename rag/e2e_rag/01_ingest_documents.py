"""
01_ingest_documents.py — chunk, embed, and store markdown documents in pgvector.

Reads every .md file from documents/, splits it into chunks using the
selected chunking strategy, embeds each chunk, and stores the result in the
`chunks` table created by 00_init_db.py.

Chunking strategies (--strategy):
  paragraph   Paragraph-aware chunking with char-based overlap. Reused as-is
              from db/ingest_kb.py: prefers splitting on blank lines, and
              only hard-splits very long paragraphs.
  fixed       Naive fixed-size sliding window: cuts every --max-chars
              characters with --overlap characters shared between
              consecutive chunks, ignoring paragraph boundaries.
  recursive   Tries a hierarchy of separators (markdown headings, blank
              lines, single newlines, sentences, words) and only falls back
              to a finer-grained separator when a piece is still too big —
              similar to LangChain's RecursiveCharacterTextSplitter.

Usage:
    python 01_ingest_documents.py
    python 01_ingest_documents.py --strategy fixed --max-chars 500 --overlap 50
"""

from __future__ import annotations

import argparse
import os
import re
import sys
from pathlib import Path

from dotenv import load_dotenv
from psycopg import connect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

load_dotenv()

HERE = Path(__file__).parent
DOCUMENTS_DIR = HERE / "documents"

# Must match EMBEDDING_DIM in 00_init_db.py.
EMBEDDING_MODEL = "BAAI/bge-m3"  # EDITABLE: also update EMBEDDING_MODEL in 02_chatbot.py and EMBEDDING_DIM in 00_init_db.py if you change this


# --- Chunking strategies -----------------------------------------------
#
# `chunk_paragraph` below is copied from db/ingest_kb.py's `chunk_markdown` /
# `normalize_text` so this folder stays self-contained (per instructions to
# keep everything in a single directory).

def normalize_text(text: str) -> str:
    text = text.replace(" ", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_paragraph(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:  # EDITABLE: default max_chars/overlap
    """Paragraph-aware chunking with char-based overlap (from db/ingest_kb.py)."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    # Prefer paragraph-aware chunks first
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]
    chunks: list[str] = []
    current = ""

    for paragraph in paragraphs:
        candidate = f"{current}\n\n{paragraph}".strip() if current else paragraph
        if len(candidate) <= max_chars:
            current = candidate
            continue

        if current:
            chunks.append(current)
            current = paragraph
        else:
            # Very long paragraph: hard-split with overlap
            start = 0
            while start < len(paragraph):
                end = min(start + max_chars, len(paragraph))
                piece = paragraph[start:end].strip()
                if piece:
                    chunks.append(piece)
                if end == len(paragraph):
                    break
                start = max(0, end - overlap)
            current = ""

    if current:
        chunks.append(current)

    # Light overlap between adjacent chunks for context continuity
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev_tail = chunks[idx - 1][-overlap:].strip()
            combined = f"{prev_tail}\n{chunk}".strip()
            overlapped.append(combined if len(combined) <= max_chars + overlap else chunk)
        chunks = overlapped

    return chunks


def chunk_fixed(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:  # EDITABLE: default max_chars/overlap
    """Naive fixed-size sliding window, ignoring paragraph/sentence boundaries."""
    text = normalize_text(text)
    if len(text) <= max_chars:
        return [text]

    chunks: list[str] = []
    start = 0
    while start < len(text):
        end = min(start + max_chars, len(text))
        piece = text[start:end].strip()
        if piece:
            chunks.append(piece)
        if end == len(text):
            break
        start = max(0, end - overlap)
    return chunks


# Separator hierarchy for recursive chunking, from coarsest to finest.
# Markdown headings are tried first so sections stay intact where possible,
# then paragraph breaks, then single newlines, then sentences, then words.
_RECURSIVE_SEPARATORS = ["\n## ", "\n# ", "\n\n", "\n", ". ", " "]  # EDITABLE: separator hierarchy


def _split_with_separator(text: str, separator: str) -> list[str]:
    if separator == "":
        return list(text)
    parts = text.split(separator)
    # Re-attach the separator to each piece (except the first) so headings/
    # sentence punctuation aren't silently dropped from the chunk text.
    return [parts[0]] + [separator + p for p in parts[1:]]


def _recursive_split(text: str, max_chars: int, separators: list[str]) -> list[str]:
    if len(text) <= max_chars:
        return [text]

    if not separators:
        # No separators left: hard-split at the character limit.
        return [text[i : i + max_chars] for i in range(0, len(text), max_chars)]

    separator, *rest = separators
    pieces = _split_with_separator(text, separator)

    # If this separator didn't actually split anything, try the next one.
    if len(pieces) == 1:
        return _recursive_split(text, max_chars, rest)

    chunks: list[str] = []
    for piece in pieces:
        if not piece.strip():
            continue
        if len(piece) <= max_chars:
            chunks.append(piece)
        else:
            # Piece is still too big: recurse with the remaining, finer separators.
            chunks.extend(_recursive_split(piece, max_chars, rest))
    return chunks


def chunk_recursive(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:  # EDITABLE: default max_chars/overlap
    """Recursively split on a separator hierarchy, falling back to finer
    separators only where needed (LangChain-style recursive splitting)."""
    text = normalize_text(text)
    pieces = _recursive_split(text, max_chars, _RECURSIVE_SEPARATORS)
    chunks = [p.strip() for p in pieces if p.strip()]

    # Merge tiny trailing pieces (e.g. a lone leftover sentence) back into
    # the previous chunk when they'd still fit, to avoid near-empty chunks.
    # Joined with a newline (not a space) so headings stay on their own line.
    merged: list[str] = []
    for chunk in chunks:
        if merged and len(merged[-1]) + len(chunk) + 1 <= max_chars:
            merged[-1] = f"{merged[-1]}\n{chunk}".strip()
        else:
            merged.append(chunk)
    chunks = merged

    # Same light overlap pass used by chunk_paragraph, for consistency
    # across strategies and context continuity at chunk boundaries.
    if overlap > 0 and len(chunks) > 1:
        overlapped: list[str] = []
        for idx, chunk in enumerate(chunks):
            if idx == 0:
                overlapped.append(chunk)
                continue
            prev_tail = chunks[idx - 1][-overlap:].strip()
            combined = f"{prev_tail}\n{chunk}".strip()
            overlapped.append(combined if len(combined) <= max_chars + overlap else chunk)
        chunks = overlapped

    return chunks


CHUNKING_STRATEGIES = {
    "paragraph": chunk_paragraph,
    "fixed": chunk_fixed,
    "recursive": chunk_recursive,
}


# --- Ingestion ------------------------------------------------------------

def read_markdown_files(root: Path) -> list[Path]:
    return sorted(p for p in root.rglob("*.md") if p.name.lower() != "readme.md")


def store_chunks(
    conn,
    file_path: Path,
    chunks: list[str],
    embeddings: list[list[float]],
    strategy: str,
) -> None:
    rel_path = str(file_path.relative_to(DOCUMENTS_DIR))
    with conn.cursor() as cur:
        # Replace any previous chunks for this file so re-ingesting is safe.
        cur.execute("DELETE FROM chunks WHERE file_path = %s;", (rel_path,))
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (
                    file_path, file_name, chunk_order, chunk_text,
                    chunk_char_count, chunking_strategy, embedding
                )
                VALUES (%s, %s, %s, %s, %s, %s, %s);
                """,
                (rel_path, file_path.name, idx, chunk, len(chunk), strategy, emb),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Chunk, embed, and store markdown documents.")
    parser.add_argument(
        "--strategy",
        choices=sorted(CHUNKING_STRATEGIES),
        default="paragraph",  # EDITABLE: default chunking strategy
        help="Chunking strategy to use (default: paragraph).",
    )
    parser.add_argument("--max-chars", type=int, default=900, help="Maximum characters per chunk.")  # EDITABLE: default max_chars
    parser.add_argument("--overlap", type=int, default=150, help="Character overlap between chunks.")  # EDITABLE: default overlap
    parser.add_argument("--batch-size", type=int, default=32, help="Embedding batch size.")  # EDITABLE: default batch size
    args = parser.parse_args()

    database_url = os.getenv("DATABASE_URL")
    if not database_url:
        raise EnvironmentError(
            "DATABASE_URL is not set. Copy .env.example to .env and fill in your Neon connection string."
        )

    files = read_markdown_files(DOCUMENTS_DIR)
    if not files:
        print(f"No .md files found in {DOCUMENTS_DIR}. Add some documents and re-run.", file=sys.stderr)
        return

    print(f"Found {len(files)} document(s) to ingest.")

    chunk_fn = CHUNKING_STRATEGIES[args.strategy]
    print(f"Chunking strategy: {args.strategy} (max_chars={args.max_chars}, overlap={args.overlap})")

    print("Loading embedding model… (this may take a minute on first run while the model downloads)")
    model = SentenceTransformer(EMBEDDING_MODEL)
    print("Embedding model loaded.")

    print("Connecting to database…")
    total_chunks = 0
    with connect(database_url) as conn:
        for file_path in tqdm(files, desc="Ingesting documents"):
            text = file_path.read_text(encoding="utf-8").strip()
            if not text:
                tqdm.write(f"  Skipping {file_path.name} (empty file)")
                continue

            chunks = chunk_fn(text, max_chars=args.max_chars, overlap=args.overlap)
            tqdm.write(f"  {file_path.name}: {len(chunks)} chunk(s) — embedding…")
            embeddings = model.encode(
                chunks,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            tqdm.write(f"  {file_path.name}: storing in database…")
            store_chunks(conn, file_path, chunks, embeddings, args.strategy)
            total_chunks += len(chunks)
        conn.commit()

    print(f"\nDone. Ingested {len(files)} document(s), {total_chunks} chunk(s) total, using the '{args.strategy}' strategy.")


if __name__ == "__main__":
    main()
