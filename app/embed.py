"""
embed.py — turns text into vectors with a local sentence-transformers model.

Why this is its own file: everything about the embedding model lives in one
place — which model, which device, whether we normalize, and the query-vs-
document asymmetry that bge models need. The rest of the app just calls
embed_documents() / embed_query() and never has to think about torch or devices.
"""

from sentence_transformers import SentenceTransformer

from app.config import EMBED_MODEL, EMBED_DIM, EMBED_DEVICE

# bge-*-v1.5 retrieval models were trained so that a SEARCH QUERY gets a short
# instruction prepended, while the passages being searched do NOT. Skipping this
# on the query measurably hurts recall, so we bake the official instruction in
# here rather than trusting every caller to remember it.
_QUERY_INSTRUCTION = "Represent this sentence for searching relevant passages: "

# Loading the model reads ~130MB of weights and is slow, so we load it once on
# first use and reuse the same instance for every call after (a module cache).
_model: SentenceTransformer | None = None


def _resolve_device() -> str:
    """Map EMBED_DEVICE='auto' to the M1 GPU (mps) when it's available, else
    cpu. Any explicit value ('cpu', 'mps', ...) is used as-is, so the device
    stays a config knob rather than a hardcoded assumption."""
    if EMBED_DEVICE != "auto":
        return EMBED_DEVICE
    # torch is only touched when we actually need to auto-detect the device.
    import torch
    return "mps" if torch.backends.mps.is_available() else "cpu"


def _get_model() -> SentenceTransformer:
    """Load the embedding model once and cache it. Also fail loud immediately if
    the model's output dimension doesn't match EMBED_DIM — otherwise the
    mismatch wouldn't surface until a confusing insert error against the
    VECTOR(384) column in Postgres, which is a much worse place to discover it."""
    global _model
    if _model is None:
        model = SentenceTransformer(EMBED_MODEL, device=_resolve_device())
        actual_dim = model.get_embedding_dimension()
        if actual_dim != EMBED_DIM:
            raise ValueError(
                f"Model {EMBED_MODEL} outputs {actual_dim}-dim vectors, but "
                f"EMBED_DIM={EMBED_DIM}. Update EMBED_DIM in .env AND VECTOR(n) "
                f"in db/init.sql to match, then rebuild the DB volume."
            )
        _model = model
    return _model


def embed_documents(texts: list[str]) -> list[list[float]]:
    """Embed passages (chunk text) for storage. No query instruction. Vectors
    are L2-normalized so cosine distance in pgvector behaves cleanly and the
    similarity scores stay directly comparable across queries."""
    model = _get_model()
    vectors = model.encode(
        texts,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.tolist()


def embed_query(text: str) -> list[float]:
    """Embed a single search query. Prepends the bge instruction (the query/
    document asymmetry these models were trained with) and normalizes, so the
    query vector lands in the same space as the stored document vectors."""
    model = _get_model()
    vector = model.encode(
        _QUERY_INSTRUCTION + text,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vector.tolist()