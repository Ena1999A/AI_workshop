CREATE TABLE IF NOT EXISTS documents (
    id BIGSERIAL PRIMARY KEY,
    title TEXT NOT NULL,
    doc_type TEXT NOT NULL,
    audience TEXT,
    language TEXT DEFAULT 'hr',
    sensitivity TEXT,
    source TEXT,
    tags TEXT[],
    file_path TEXT UNIQUE NOT NULL,
    relative_path TEXT,
    folder_name TEXT,
    file_name TEXT,
    file_stem TEXT,
    raw_text TEXT NOT NULL,
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW()
);

CREATE TABLE IF NOT EXISTS chunks (
    id BIGSERIAL PRIMARY KEY,
    document_id BIGINT NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    chunk_order INT NOT NULL,
    chunk_text TEXT NOT NULL,
    chunk_char_count INT NOT NULL,
    embedding VECTOR(1024),
    metadata JSONB DEFAULT '{}'::jsonb,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    CONSTRAINT uq_document_chunk_order UNIQUE (document_id, chunk_order)
);

CREATE INDEX IF NOT EXISTS idx_documents_doc_type
    ON documents(doc_type);

CREATE INDEX IF NOT EXISTS idx_documents_tags
    ON documents USING GIN(tags);

CREATE INDEX IF NOT EXISTS idx_documents_metadata
    ON documents USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_documents_file_name
    ON documents(file_name);

CREATE INDEX IF NOT EXISTS idx_documents_file_stem
    ON documents(file_stem);

CREATE INDEX IF NOT EXISTS idx_documents_folder_name
    ON documents(folder_name);

CREATE INDEX IF NOT EXISTS idx_chunks_metadata
    ON chunks USING GIN(metadata);

CREATE INDEX IF NOT EXISTS idx_chunks_document_id
    ON chunks(document_id);

CREATE INDEX IF NOT EXISTS idx_chunks_embedding_cosine
    ON chunks USING hnsw (embedding vector_cosine_ops);