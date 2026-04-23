# Database Setup and Knowledge Base Ingestion

This folder contains everything needed to initialize the PostgreSQL database and fill it with knowledge base documents that the chatbot searches over.

---

## Overview

The chatbot stores knowledge base documents in PostgreSQL with the **pgvector** extension. pgvector adds a `VECTOR` column type and nearest-neighbour search operators to standard PostgreSQL, which is how semantic search works — each text chunk is stored alongside its embedding (a list of 1024 numbers), and queries find the most similar chunks by cosine distance.

```
knowledge_base/          ← Markdown files organized by category
      │
      ▼
ingest_kb.py             ← Reads, chunks, embeds, and stores them
      │
      ▼
PostgreSQL + pgvector    ← documents table + chunks table with VECTOR(1024) column
      │
      ▼
chatbot retrieval.py     ← Queries the DB at runtime using semantic search
```

---

## Folder Structure

```
db/
├── init/
│   ├── 01_init_extension.sql   ← Activates the pgvector extension
│   └── 02_schema.sql           ← Creates the documents and chunks tables
└── ingest_kb.py                ← Script to read, chunk, embed, and store documents
```

---

## Step 1 — Start the Database

The project uses Docker to run PostgreSQL with pgvector pre-installed. From the project root:

```bash
docker compose up -d
```

This does several things automatically on first run:
1. Pulls the `pgvector/pgvector:pg16` image — a version of PostgreSQL 16 that has the pgvector extension files already installed on its filesystem.
2. Creates the database and user defined in your `.env` file.
3. Runs the SQL files from `db/init/` in alphabetical order (`01_` before `02_`).

The database data is persisted in `data/postgres/` on your machine, so it survives container restarts.

### Why two SQL init files?

**`01_init_extension.sql`**

```sql
CREATE EXTENSION IF NOT EXISTS vector;
```

This activates pgvector for the database. PostgreSQL extensions must be explicitly enabled per database before their types and operators become available. The extension name is `vector` — that is how pgvector registers itself (not `pgvector`). This file must run before the schema because `02_schema.sql` uses the `VECTOR(1024)` type, which only exists after the extension is loaded.

**`02_schema.sql`**

Creates two tables and their indexes:

- `documents` — one row per knowledge base file. Stores the title, doc type, audience, raw text, and metadata.
- `chunks` — one row per text chunk. Each chunk holds a portion of a document's text and its `VECTOR(1024)` embedding. Linked to `documents` via `document_id`.

The most important index:

```sql
CREATE INDEX idx_chunks_embedding_cosine
    ON chunks USING hnsw (embedding vector_cosine_ops);
```

`HNSW` (Hierarchical Navigable Small World) is an approximate nearest-neighbour algorithm built into pgvector. It makes similarity search fast even with hundreds of thousands of chunks — without it, every query would scan every row.

### Verifying pgvector is active

Connect to the database and run either of these:

```sql
-- Option 1: check the extensions table
SELECT * FROM pg_extension WHERE extname = 'vector';

-- Option 2: check the type is registered
SELECT typname FROM pg_type WHERE typname = 'vector';
```

If either returns a row, pgvector is active. If not, `01_init_extension.sql` did not run — re-create the container or run the SQL manually.

---

## Step 2 — Ingest the Knowledge Base

`ingest_kb.py` reads all Markdown files from the knowledge base folder, splits them into chunks, generates an embedding for each chunk, and stores everything in the database.

```bash
python -m db.ingest_kb
```

By default it looks for a folder named `knowledge_base/` in the current directory. You can override this and other settings:

```bash
python -m db.ingest_kb \
  --kb-root knowledge_base \
  --model BAAI/bge-m3 \
  --batch-size 32 \
  --max-chars 900 \
  --overlap 150 \
  --device cpu
```

| Flag | Default | Description |
|------|---------|-------------|
| `--kb-root` | `knowledge_base` | Path to the folder containing Markdown documents |
| `--model` | `BAAI/bge-m3` | SentenceTransformer model to generate embeddings |
| `--batch-size` | `32` | How many chunks to embed in one GPU/CPU batch |
| `--max-chars` | `900` | Maximum characters per chunk |
| `--overlap` | `150` | Characters of overlap between adjacent chunks |
| `--device` | auto | `cpu` or `cuda` |

> **First run:** The script downloads the `BAAI/bge-m3` embedding model (~570 MB). This only happens once — it is cached by the sentence-transformers library.

Re-running the script is safe. It uses `ON CONFLICT (file_path) DO UPDATE`, so existing documents are updated rather than duplicated.

---

## Knowledge Base Folder Structure

Documents must be placed in subfolders named after their category. The folder name determines how the chatbot searches for them (it maps directly to the `doc_type` column):

```
knowledge_base/
├── faq/                ← General customer questions about leasing
├── crm_case/           ← Individual customer case files
├── contract_concept/   ← Explanations of financial/leasing terms
├── procedure/          ← Step-by-step internal process documents
└── product_doc/        ← Software module and portal documentation
```

Any `.md` file placed outside these five folders is ignored by the ingestion script.

---

## Function Reference — `ingest_kb.py`

| Function | What it does |
|----------|-------------|
| `read_markdown_files(root)` | Walks the knowledge base folder and yields one `DocumentRecord` per valid `.md` file |
| `derive_title(file_stem, text)` | Uses the first non-empty line of the file as the document title; falls back to the filename |
| `infer_doc_type(folder_name)` | Returns the folder name as the `doc_type` (e.g. `faq`, `procedure`) |
| `infer_audience(folder_name)` | Maps folder to audience: `customers`, `agents`, or `internal` |
| `infer_sensitivity(folder_name)` | Maps folder to sensitivity level: `internal` or `confidential` (CRM cases) |
| `make_tags(folder_name, file_stem)` | Builds a list of tags from the folder name and words in the filename |
| `normalize_text(text)` | Cleans whitespace: removes non-breaking spaces, normalises line endings, collapses multiple blank lines |
| `chunk_markdown(text, max_chars, overlap)` | Splits text into chunks of at most `max_chars` characters. Prefers splitting on paragraph boundaries (`\n\n`). For very long paragraphs, hard-splits with `overlap` characters of overlap between adjacent chunks. |
| `upsert_document(conn, doc)` | Inserts a document row or updates it if the file path already exists. Returns the database `id`. |
| `replace_chunks(conn, document_id, chunks, embeddings)` | Deletes all existing chunks for a document and inserts fresh ones with new embeddings. |
| `main()` | Entry point — parses args, connects to DB, iterates over documents, calls `upsert_document` then `replace_chunks` for each. |

---

## Database Schema

### `documents` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL | Primary key |
| `title` | TEXT | Document title (derived from first line or filename) |
| `doc_type` | TEXT | Category: `faq`, `crm_case`, `contract_concept`, `procedure`, `product_doc` |
| `audience` | TEXT | Who the document is for: `customers`, `agents`, `internal` |
| `language` | TEXT | Always `hr` (Croatian) |
| `sensitivity` | TEXT | `internal` or `confidential` |
| `file_stem` | TEXT | Filename without extension — used to look up CRM cases by customer ID |
| `raw_text` | TEXT | Full original document text |
| `tags` | TEXT[] | Searchable tags derived from folder and filename |
| `metadata` | JSONB | Additional metadata (folder, relative path) |

### `chunks` table

| Column | Type | Description |
|--------|------|-------------|
| `id` | BIGSERIAL | Primary key |
| `document_id` | BIGINT | Foreign key to `documents.id` |
| `chunk_order` | INT | Position of this chunk within the document (0-indexed) |
| `chunk_text` | TEXT | The actual text content of this chunk |
| `chunk_char_count` | INT | Character length of the chunk |
| `embedding` | VECTOR(1024) | The semantic embedding — used for cosine similarity search |

The `embedding <=> query_vector` expression in the retrieval query computes cosine distance between two vectors. The `HNSW` index makes this fast without scanning every row.
