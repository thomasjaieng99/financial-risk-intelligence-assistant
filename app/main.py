"""
main.py — the FastAPI app. Three endpoints, matching CLAUDE.md's file map:

  GET  /health  — liveness + a quick DB reachability check
  POST /search  — semantic search over the stored chunks (the retrieval demo)
  POST /ask     — retrieval-augmented answer: search, then have Claude synthesize
                  a cited answer from the hits (needs ANTHROPIC_API_KEY)

This file is thin on purpose. All the real work lives in db.py (retrieval) and
embed.py (vectors); here we just expose it over HTTP.
"""

from anthropic import Anthropic
from fastapi import FastAPI
from pydantic import BaseModel

from app import db
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL

app = FastAPI(title="Financial Risk Intelligence Assistant")


# One request shape covers both /search and /ask — they take the same inputs.
class Query(BaseModel):
    query: str
    ticker: str | None = None   # None searches every filing; a ticker restricts it
    k: int = 5                  # how many chunks to return / feed the model


@app.get("/health")
def health():
    """Liveness plus a real DB check: if Postgres is unreachable, this reports
    'degraded' with the error instead of the app looking healthy while search
    would fail."""
    try:
        with db.connect() as conn, conn.cursor() as cur:
            cur.execute("SELECT count(*) FROM documents")
            docs = cur.fetchone()[0]
            cur.execute("SELECT count(*) FROM chunks")
            chunks = cur.fetchone()[0]
        return {"status": "ok", "documents": docs, "chunks": chunks}
    except Exception as exc:  # surface the cause rather than a bare 500
        return {"status": "degraded", "error": str(exc)}


@app.post("/search")
def search(req: Query):
    """Return the k most relevant chunks with their section labels and scores —
    the raw retrieval, no LLM involved."""
    with db.connect() as conn:
        hits = db.search(conn, req.query, ticker=req.ticker, k=req.k)
    return {"query": req.query, "results": hits}


@app.post("/ask")
def ask(req: Query):
    """Retrieve the top-k chunks, then have Claude answer the question using ONLY
    those chunks and cite the section each claim came from. Retrieval always
    runs; if no ANTHROPIC_API_KEY is set we return the sources with a note
    instead of a synthesized answer, so the endpoint still works offline."""
    with db.connect() as conn:
        hits = db.search(conn, req.query, ticker=req.ticker, k=req.k)

    if not ANTHROPIC_API_KEY:
        return {
            "query": req.query,
            "answer": None,
            "note": "Set ANTHROPIC_API_KEY in .env to enable answer synthesis.",
            "sources": hits,
        }

    # Number each retrieved chunk so the model can cite it, and keep its section
    # label visible so citations name the actual 10-K section.
    context = "\n\n".join(
        f"[Source {i}] (section: {h['section']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    system = (
        "You are a financial risk analyst assistant. Answer the user's question "
        "using ONLY the provided sources from SEC filings. Cite the section you "
        "drew each claim from, e.g. (Item 1A. Risk Factors). If the sources do "
        "not contain the answer, say so plainly rather than guessing."
    )

    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=1024,  # a cited answer is short by design
        system=system,
        messages=[{"role": "user",
                   "content": f"Question: {req.query}\n\nSources:\n{context}"}],
    )
    answer = "".join(block.text for block in message.content if block.type == "text")

    return {"query": req.query, "answer": answer, "sources": hits}
