"""
db.py — all Postgres access for the app: raw SQL through psycopg 3, no ORM.

CLAUDE.md draws a line through this file:
  - Plumbing: open a connection, upsert a document, insert its chunks.
  - The cosine-similarity SEARCH query is the retrieval core of the project — written to
    the design the author approved: embed the question, rank by pgvector cosine distance,
    report (1 - distance) as a similarity score, with an optional ticker filter.
"""

import psycopg
from psycopg.rows import dict_row
from pgvector.psycopg import register_vector

from app.config import DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD
from app.embed import embed_query


def connect() -> psycopg.Connection:
    """Open a connection and teach psycopg the pgvector `vector` type, so we can
    hand it a plain Python list of floats as an embedding (and read one back)
    without hand-formatting vector literals anywhere in the app.

    The caller owns the connection's lifetime: `with connect() as conn: ...`
    commits on a clean exit and rolls back on an exception, so one filing
    ingests as a single atomic transaction."""
    conn = psycopg.connect(
        host=DB_HOST,
        port=DB_PORT,
        dbname=DB_NAME,
        user=DB_USER,
        password=DB_PASSWORD,
    )
    register_vector(conn)
    return conn


def upsert_document(
    conn: psycopg.Connection,
    *,
    ticker: str,
    cik: str,
    form_type: str,
    filing_date: str | None,
    accession_number: str,
    primary_document: str | None,
) -> int:
    """Insert the filing, or update it if this accession_number already exists,
    and return its documents.id.

    ON CONFLICT is what makes re-ingesting the same 10-K idempotent — one row
    per filing instead of piling up duplicates (the design the author chose).
    Args are keyword-only (the `*`) so a call site can't silently transpose two
    same-typed fields like ticker and cik."""
    with conn.cursor() as cur:
        cur.execute(
            """
            INSERT INTO documents
                (ticker, cik, form_type, filing_date, accession_number, primary_document)
            VALUES (%s, %s, %s, %s, %s, %s)
            ON CONFLICT (accession_number) DO UPDATE SET
                ticker           = EXCLUDED.ticker,
                cik              = EXCLUDED.cik,
                form_type        = EXCLUDED.form_type,
                filing_date      = EXCLUDED.filing_date,
                primary_document = EXCLUDED.primary_document
            RETURNING id
            """,
            (ticker, cik, form_type, filing_date, accession_number, primary_document),
        )
        return cur.fetchone()[0]


def insert_chunks(conn: psycopg.Connection, document_id: int, chunks: list[dict]) -> int:
    """Replace a document's chunks with a fresh set; return how many were inserted.

    We DELETE the document's existing chunks first so re-ingesting a filing can't
    leave stale rows behind — the flip side of upsert_document's idempotency (and
    why chunks has ON DELETE CASCADE on document_id). Each chunk is
    {"section", "text", "embedding"}; chunk_index is its 0-based order within the
    document, assigned here."""
    rows = [
        (document_id, c["section"], i, c["text"], c["embedding"])
        for i, c in enumerate(chunks)
    ]
    with conn.cursor() as cur:
        cur.execute("DELETE FROM chunks WHERE document_id = %s", (document_id,))
        # executemany batches the inserts; register_vector adapts each embedding
        # list into a pgvector `vector` on the way in.
        cur.executemany(
            """
            INSERT INTO chunks (document_id, section, chunk_index, text, embedding)
            VALUES (%s, %s, %s, %s, %s)
            """,
            rows,
        )
    return len(rows)


def search(
    conn: psycopg.Connection,
    query: str,
    *,
    ticker: str | None = None,
    k: int = 5,
) -> list[dict]:
    """
    Return the k chunks most semantically similar to `query`, most similar
    first, each with its section label and a similarity score. This is the
    retrieval step the whole project is built around.

    How it works:
      1. Embed the question with embed_query() so it carries the bge search
         instruction and lands in the same 384-dim space as the stored chunks.
      2. Rank chunks by pgvector's cosine DISTANCE operator `<=>` (0 = identical
         direction, 2 = opposite). ORDER BY it ascending = most similar first,
         and the HNSW cosine index makes that fast.
      3. Report a similarity SCORE, not a distance: for normalized vectors
         cosine_distance = 1 - cosine_similarity, so similarity = 1 - (a <=> b).
         A score near 1.0 is a strong match; CLAUDE.md's ~0.6 bar lives here.
      4. Optional ticker filter: ticker=None searches everything; a ticker
         restricts results to one company's filings.
    """
    query_vec = embed_query(query)
    with conn.cursor(row_factory=dict_row) as cur:
        # psycopg sends the Python list as a Postgres float8[]; `::vector` casts
        # it so pgvector's cosine operator (vector <=> vector) has a match. On
        # INSERT the same cast happens implicitly against the vector column, but
        # an operator expression doesn't get that for free.
        cur.execute(
            """
            SELECT
                d.ticker,
                d.accession_number,
                c.section,
                c.chunk_index,
                c.text,
                1 - (c.embedding <=> %(qvec)s::vector) AS similarity
            FROM chunks c
            JOIN documents d ON d.id = c.document_id
            WHERE (%(ticker)s::text IS NULL OR d.ticker = %(ticker)s::text)
            ORDER BY c.embedding <=> %(qvec)s::vector
            LIMIT %(k)s
            """,
            {"qvec": query_vec, "ticker": ticker, "k": k},
        )
        return cur.fetchall()