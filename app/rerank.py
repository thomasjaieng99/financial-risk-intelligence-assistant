"""
rerank.py — second-stage reranking with a cross-encoder (Phase 2).

Why a reranker at all — the bi-encoder vs cross-encoder distinction (the concept
to know cold):

  * First-stage search (embed.py + db.py) is a BI-ENCODER. It embeds the query
    and each chunk SEPARATELY into one vector apiece, then compares vectors with
    cosine distance. That's fast — every chunk's vector is precomputed once at
    ingest — but lossy: the query and the chunk never actually "meet," so the
    model can't reason about how they relate. Similarity is only an approximation.

  * A CROSS-ENCODER instead feeds the (query, chunk) PAIR through the model
    TOGETHER and outputs one relevance score, so it can attend across both texts
    at once. Much more accurate — but far too slow to run over all 1,371 chunks
    on every query (there's nothing to precompute; every pair is fresh work).

The standard resolution is TWO-STAGE retrieval: let the cheap bi-encoder narrow
1,371 chunks down to ~20 candidates (high recall, rough order), then let the
accurate cross-encoder re-score just those 20 and pick the best. You get the
bi-encoder's speed and the cross-encoder's precision. Measured against the eval
set, this is what lifts recall@1 / MRR without touching recall@5.
"""

from sentence_transformers import CrossEncoder

from app.config import RERANK_MODEL

# Loaded once and reused, like the embedding model — the first call downloads the
# cross-encoder weights; every call after reuses this instance.
_model: CrossEncoder | None = None


def _get_model() -> CrossEncoder:
    global _model
    if _model is None:
        _model = CrossEncoder(RERANK_MODEL)
    return _model


def rerank_hits(query: str, hits: list[dict], top_k: int) -> list[dict]:
    """
    Re-score first-stage `hits` with the cross-encoder and return the top_k by
    that score (highest first), attaching a 'rerank_score' to each.

    Each hit is a search row (has a "text" field). We build one (query, text)
    pair per hit, score them all in a single batched forward pass, then sort.
    Empty input returns an empty list so callers don't special-case it.
    """
    if not hits:
        return []

    model = _get_model()
    pairs = [(query, hit["text"]) for hit in hits]
    scores = model.predict(pairs)  # one relevance score per (query, chunk) pair

    for hit, score in zip(hits, scores):
        hit["rerank_score"] = float(score)  # numpy float -> plain float for JSON

    ranked = sorted(hits, key=lambda h: h["rerank_score"], reverse=True)
    return ranked[:top_k]
