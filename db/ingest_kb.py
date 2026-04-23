from __future__ import annotations

import argparse
import os
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable

from dotenv import load_dotenv
from psycopg import connect
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
from psycopg.types.json import Jsonb

VALID_FOLDERS = {"contract_concept", "crm_case", "faq", "procedure", "product_doc"}


@dataclass
class DocumentRecord:
    title: str
    doc_type: str
    audience: str | None
    sensitivity: str | None
    source: str | None
    tags: list[str]
    file_path: str
    relative_path: str
    folder_name: str
    file_name: str
    file_stem: str
    raw_text: str


def infer_doc_type(folder_name: str) -> str:
    return folder_name


def infer_audience(folder_name: str) -> str:
    mapping = {
        "faq": "customers",
        "crm_case": "agents",
        "procedure": "agents",
        "product_doc": "customers",
        "contract_concept": "internal",
    }
    return mapping.get(folder_name, "internal")


def infer_sensitivity(folder_name: str) -> str:
    mapping = {
        "faq": "internal",
        "crm_case": "confidential",
        "procedure": "internal",
        "product_doc": "internal",
        "contract_concept": "internal",
    }
    return mapping.get(folder_name, "internal")


def make_tags(folder_name: str, file_stem: str) -> list[str]:
    parts = re.split(r"[_\-\s]+", file_stem.lower())
    cleaned = [p for p in parts if p]
    # folder_name is included so lexical filters can be used later
    return [folder_name] + cleaned[:10]


def derive_title(file_stem: str, text: str) -> str:
    lines = [line.strip() for line in text.splitlines() if line.strip()]
    if lines and len(lines[0]) <= 120:
        return lines[0].lstrip("# ").strip()
    return file_stem.replace("_", " ").replace("-", " ").strip().title()


def read_markdown_files(root: Path) -> Iterable[DocumentRecord]:
    for path in sorted(root.rglob("*.md")):
        rel = path.relative_to(root)
        if len(rel.parts) < 2:
            continue

        folder_name = rel.parts[0]
        if folder_name not in VALID_FOLDERS:
            continue

        text = path.read_text(encoding="utf-8").strip()
        if not text:
            continue

        file_name = path.name
        file_stem = path.stem

        yield DocumentRecord(
            title=derive_title(file_stem, text),
            doc_type=infer_doc_type(folder_name),
            audience=infer_audience(folder_name),
            sensitivity=infer_sensitivity(folder_name),
            source="knowledge_base",
            tags=make_tags(folder_name, file_stem),
            file_path=str(path.resolve()),
            relative_path=str(rel),
            folder_name=folder_name,
            file_name=file_name,
            file_stem=file_stem,
            raw_text=text,
        )


def normalize_text(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"\r\n?", "\n", text)
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def chunk_markdown(text: str, max_chars: int = 900, overlap: int = 150) -> list[str]:
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


def upsert_document(conn, doc: DocumentRecord) -> int:
    sql = """
        INSERT INTO documents (
            title,
            doc_type,
            audience,
            language,
            sensitivity,
            source,
            tags,
            file_path,
            relative_path,
            folder_name,
            file_name,
            file_stem,
            raw_text,
            metadata
        )
        VALUES (
            %(title)s,
            %(doc_type)s,
            %(audience)s,
            'hr',
            %(sensitivity)s,
            %(source)s,
            %(tags)s,
            %(file_path)s,
            %(relative_path)s,
            %(folder_name)s,
            %(file_name)s,
            %(file_stem)s,
            %(raw_text)s,
            %(metadata)s
        )
        ON CONFLICT (file_path)
        DO UPDATE SET
            title = EXCLUDED.title,
            doc_type = EXCLUDED.doc_type,
            audience = EXCLUDED.audience,
            sensitivity = EXCLUDED.sensitivity,
            source = EXCLUDED.source,
            tags = EXCLUDED.tags,
            relative_path = EXCLUDED.relative_path,
            folder_name = EXCLUDED.folder_name,
            file_name = EXCLUDED.file_name,
            file_stem = EXCLUDED.file_stem,
            raw_text = EXCLUDED.raw_text,
            metadata = EXCLUDED.metadata
        RETURNING id;
    """

    payload = {
        **doc.__dict__,
        "metadata": Jsonb(
            {
                "folder_name": doc.folder_name,
                "file_name": doc.file_name,
                "relative_path": doc.relative_path,
            }
        ),
    }

    with conn.cursor() as cur:
        cur.execute(sql, payload)
        return cur.fetchone()[0]


def replace_chunks(conn, document_id: int, chunks: list[str], embeddings: list[list[float]]) -> None:
    if len(chunks) != len(embeddings):
        raise ValueError("Chunks and embeddings length mismatch.")

    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s;", (document_id,))
        for idx, (chunk, emb) in enumerate(zip(chunks, embeddings)):
            cur.execute(
                """
                INSERT INTO chunks (
                    document_id,
                    chunk_order,
                    chunk_text,
                    chunk_char_count,
                    embedding,
                    metadata
                )
                VALUES (%s, %s, %s, %s, %s, %s);
                """,
                (
                    document_id,
                    idx,
                    chunk,
                    len(chunk),
                    emb,
                    Jsonb({"document_id": document_id, "chunk_order": idx}),
                ),
            )


def main() -> None:
    parser = argparse.ArgumentParser(description="Ingest Markdown knowledge base into PostgreSQL + pgvector.")
    parser.add_argument("--kb-root", default="knowledge_base", help="Path to knowledge base root folder.")
    parser.add_argument("--model", default="BAAI/bge-m3", help="SentenceTransformer model name or local path.")
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--max-chars", type=int, default=900)
    parser.add_argument("--overlap", type=int, default=150)
    parser.add_argument(
        "--device",
        default=None,
        help="Optional device override for SentenceTransformer, e.g. cpu or cuda.",
    )
    args = parser.parse_args()

    load_dotenv()

    db_host = os.getenv("POSTGRES_HOST", "localhost")
    db_port = os.getenv("POSTGRES_PORT", "5432")
    db_name = os.getenv("POSTGRES_DB", "ai_workshop")
    db_user = os.getenv("POSTGRES_USER", "postgres")
    db_password = os.getenv("POSTGRES_PASSWORD", "postgres")

    kb_root = Path(args.kb_root)
    if not kb_root.exists():
        raise FileNotFoundError(f"Knowledge base folder not found: {kb_root}")

    documents = list(read_markdown_files(kb_root))
    if not documents:
        raise RuntimeError("No markdown documents found in expected subfolders.")

    model = SentenceTransformer(args.model, device=args.device)

    conninfo = f"host={db_host} port={db_port} dbname={db_name} user={db_user} password={db_password}"

    with connect(conninfo) as conn:
        for doc in tqdm(documents, desc="Ingesting documents"):
            document_id = upsert_document(conn, doc)
            chunks = chunk_markdown(doc.raw_text, max_chars=args.max_chars, overlap=args.overlap)
            embeddings = model.encode(
                chunks,
                batch_size=args.batch_size,
                normalize_embeddings=True,
                show_progress_bar=False,
                convert_to_numpy=True,
            ).tolist()
            replace_chunks(conn, document_id, chunks, embeddings)
        conn.commit()

    print(f"Done. Ingested {len(documents)} documents from {kb_root}.")


if __name__ == "__main__":
    main()
