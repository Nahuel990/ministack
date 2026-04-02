-- MiniStack Bedrock Knowledge Base — pgvector schema initialization
-- This script runs automatically when the pgvector container starts for the first time.

CREATE EXTENSION IF NOT EXISTS vector;

-- Main document store for Knowledge Base
CREATE TABLE IF NOT EXISTS kb_documents (
    id UUID PRIMARY KEY DEFAULT gen_random_uuid(),
    knowledge_base_id VARCHAR(255) NOT NULL,
    data_source_id VARCHAR(255) NOT NULL,
    s3_uri TEXT,
    content TEXT,
    metadata JSONB DEFAULT '{}',
    embedding vector(768),  -- nomic-embed-text outputs 768 dimensions
    status VARCHAR(50) DEFAULT 'INDEXED',
    created_at TIMESTAMP DEFAULT NOW(),
    updated_at TIMESTAMP DEFAULT NOW()
);

CREATE INDEX IF NOT EXISTS idx_kb_docs_kb_id ON kb_documents(knowledge_base_id);
CREATE INDEX IF NOT EXISTS idx_kb_docs_ds_id ON kb_documents(data_source_id);
CREATE INDEX IF NOT EXISTS idx_kb_docs_status ON kb_documents(status);

-- IVFFlat index for approximate nearest neighbor search
-- Note: This index requires at least some rows to be effective.
-- For small datasets, pgvector falls back to exact search automatically.
CREATE INDEX IF NOT EXISTS idx_kb_docs_embedding ON kb_documents
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
