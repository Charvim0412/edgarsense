CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS chunks (
    id SERIAL PRIMARY KEY,
    ticker TEXT NOT NULL,
    accession_number TEXT NOT NULL,
    filing_date TEXT NOT NULL,
    item_number TEXT,
    heading TEXT,
    chunk_index INTEGER NOT NULL,
    text TEXT NOT NULL,
    embedding vector(384)
);

CREATE INDEX IF NOT EXISTS chunks_ticker_idx ON chunks (ticker);
CREATE INDEX IF NOT EXISTS chunks_embedding_idx ON chunks
    USING ivfflat (embedding vector_cosine_ops) WITH (lists = 100);
