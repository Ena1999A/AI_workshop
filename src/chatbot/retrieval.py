"""
Vector retrieval from PostgreSQL / pgvector.

Intent-to-doc_type mapping drives which partition of the knowledge base is searched.
When a specific CRM case is identified (via PII lookup), it is fetched directly
by file_stem before the semantic search runs.
"""

from __future__ import annotations

from typing import Optional

from sentence_transformers import SentenceTransformer

INTENT_TO_DOC_TYPES: dict[str, list[str]] = {
    "faq":              ["faq"],
    "crm_case":         ["crm_case"],
    "contract_concept": ["contract_concept"],
    "procedure":        ["procedure"],
    "product_doc":      ["product_doc"],
    "unsupported":      [],
}

_TOP_K = 4


def retrieve_chunks(
    conn,
    model: SentenceTransformer,
    query: str,
    intent: str,
    case_id: Optional[str] = None,
    top_k: int = _TOP_K,
) -> list[dict]:
    """
    Return relevant chunks for the given query and intent.

    If case_id is provided (CRM intent), the specific case document is fetched
    directly first, followed by a broader semantic search.
    """
    doc_types = INTENT_TO_DOC_TYPES.get(intent, [])
    if not doc_types:
        return []

    embedding = model.encode(query, normalize_embeddings=True).tolist()
    results: list[dict] = []
    seen_stems: set[str] = set()

    # Direct lookup for identified CRM case
    if case_id and intent == "crm_case":
        with conn.cursor() as cur:
            cur.execute(
                """
                SELECT c.chunk_text, d.title, d.file_stem, d.doc_type
                FROM chunks c
                JOIN documents d ON c.document_id = d.id
                WHERE d.file_stem = %s
                ORDER BY c.chunk_order
                LIMIT 3;
                """,
                (case_id,),
            )
            for row in cur.fetchall():
                results.append({
                    "chunk_text": row[0],
                    "title": row[1],
                    "file_stem": row[2],
                    "doc_type": row[3],
                    "source": "direct_lookup",
                })
                seen_stems.add(row[2])

    # Semantic search filtered by doc_type
    placeholders = ", ".join(["%s"] * len(doc_types))
    with conn.cursor() as cur:
        cur.execute(
            f"""
            SELECT c.chunk_text, d.title, d.file_stem, d.doc_type,
                   1 - (c.embedding <=> %s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON c.document_id = d.id
            WHERE d.doc_type IN ({placeholders})
            ORDER BY c.embedding <=> %s::vector
            LIMIT %s;
            """,
            [embedding] + doc_types + [embedding, top_k],
        )
        for row in cur.fetchall():
            results.append({
                "chunk_text": row[0],
                "title": row[1],
                "file_stem": row[2],
                "doc_type": row[3],
                "similarity": float(row[4]),
                "source": "vector_search",
            })

    return results


def format_chunks_for_prompt(chunks: list[dict]) -> str:
    if not chunks:
        return "No relevant documents found."
    parts = []
    for i, chunk in enumerate(chunks, 1):
        parts.append(f"[{i}] {chunk['title']}\n{chunk['chunk_text']}")
    return "\n\n---\n\n".join(parts)
