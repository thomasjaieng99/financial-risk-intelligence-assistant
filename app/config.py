"""
config.py — the one place every tunable value gets read from .env.

Why this file exists: CLAUDE.md's working agreement says nothing gets
hardcoded outside of here. That's a defensibility choice — in an interview,
"where does this value come from" should always have the same one-line
answer: config.py, which reads .env. Nowhere else in the app calls
os.environ directly.
"""

import os
import sys
from pathlib import Path

from dotenv import load_dotenv

# Load .env from the repo root, not the current working directory — so this
# works whether a script is run as `python -m app.ingest` from the repo root
# or invoked some other way later.
_ENV_PATH = Path(__file__).resolve().parent.parent / ".env"
load_dotenv(dotenv_path=_ENV_PATH)


def _require(key: str) -> str:
    """Fail immediately at import time with a clear message, rather than
    failing later inside edgar.py or db.py with a confusing symptom (e.g. a
    403 with no obvious cause) the first time the value is actually used."""
    value = os.environ.get(key)
    if not value:
        sys.exit(
            f"Missing required setting '{key}' in .env "
            f"(see .env.example, edit {_ENV_PATH})"
        )
    return value


# --- SEC EDGAR ---
# SEC returns 403 without a real name + email here. Their stated policy, not a bug.
SEC_USER_AGENT = _require("SEC_USER_AGENT")

# --- Postgres / pgvector ---
DB_HOST = os.environ.get("DB_HOST", "localhost")
DB_PORT = int(os.environ.get("DB_PORT", "5433"))
DB_NAME = os.environ.get("DB_NAME", "finrisk")
DB_USER = os.environ.get("DB_USER", "finrisk")
DB_PASSWORD = os.environ.get("DB_PASSWORD", "")

# --- Embeddings ---
# EMBED_DIM must match the VECTOR(n) column in db/init.sql — they're separate
# files by necessity (Python config vs SQL schema) but have to change together.
EMBED_MODEL = os.environ.get("EMBED_MODEL", "BAAI/bge-small-en-v1.5")
EMBED_DIM = int(os.environ.get("EMBED_DIM", "384"))
EMBED_DEVICE = os.environ.get("EMBED_DEVICE", "auto")

# --- Chunking ---
# Tunable knobs for chunk.py — kept here so nothing about chunk size is
# hardcoded in the splitting logic (CLAUDE.md: every tunable lives in config).
CHUNK_SIZE = int(os.environ.get("CHUNK_SIZE", "1000"))
CHUNK_OVERLAP = int(os.environ.get("CHUNK_OVERLAP", "150"))

# --- /ask endpoint (optional) ---
ANTHROPIC_API_KEY = os.environ.get("ANTHROPIC_API_KEY", "")
# The model that synthesizes answers in /ask. Opus is the strong default;
# override in .env with a cheaper/faster model (e.g. claude-haiku-4-5) if wanted.
ANTHROPIC_MODEL = os.environ.get("ANTHROPIC_MODEL", "claude-opus-4-8")
