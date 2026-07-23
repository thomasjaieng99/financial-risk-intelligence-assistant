-- Runs ONCE, on first creation of the database volume (see docker-compose.yml).
-- If you change this file or EMBED_DIM, you must `docker compose down -v` to
-- drop the volume and let it re-run — Postgres does NOT re-run it otherwise.

-- pgvector: adds the VECTOR column type and the distance operators (<=> etc.).
CREATE EXTENSION IF NOT EXISTS vector;

-- One row per filing. accession_number is EDGAR's globally-unique id for a
-- filing, so it's the natural key that makes re-ingesting the same 10-K
-- idempotent: we upsert on it instead of inserting a duplicate document.
CREATE TABLE IF NOT EXISTS documents (
    id               BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    ticker           TEXT        NOT NULL,
    cik              TEXT        NOT NULL,
    form_type        TEXT        NOT NULL,
    filing_date      DATE,
    accession_number TEXT        NOT NULL UNIQUE,
    primary_document TEXT,
    created_at       TIMESTAMPTZ NOT NULL DEFAULT now()
);

-- Many chunks per document. ON DELETE CASCADE means the re-ingest path (delete
-- a document's old chunks, insert the new ones) can't leave orphaned rows.
CREATE TABLE IF NOT EXISTS chunks (
    id          BIGINT GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
    document_id BIGINT      NOT NULL REFERENCES documents(id) ON DELETE CASCADE,
    section     TEXT        NOT NULL,
    chunk_index INT         NOT NULL,
    text        TEXT        NOT NULL,
    embedding   VECTOR(384) NOT NULL
);

-- HNSW index for fast approximate nearest-neighbour search under COSINE
-- distance. vector_cosine_ops MUST match the operator db.py's search query uses
-- (<=>). The 384 here MUST match EMBED_DIM in .env — a mismatch fails at insert
-- time, not startup, so keep them in lockstep.
CREATE INDEX IF NOT EXISTS chunks_embedding_hnsw
    ON chunks USING hnsw (embedding vector_cosine_ops);

-- Speeds up the "delete this document's chunks before re-insert" step.
CREATE INDEX IF NOT EXISTS chunks_document_id_idx ON chunks (document_id);