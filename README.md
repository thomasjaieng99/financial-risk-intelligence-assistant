# Financial Risk Intelligence Assistant

A retrieval-augmented question-answering system over real SEC EDGAR filings. Ask
*"what credit risks did Citi disclose around commercial real estate,"* and it
retrieves the relevant passages from Citigroup's actual 10-K and answers with
citations back to the source section.

Built to be **explained and defended**, not just run. Every design choice below
has a rationale and, where relevant, the interview question it answers.

---

## What it does (end to end)

```
SEC EDGAR  ──►  clean HTML  ──►  detect sections  ──►  chunk  ──►  embed  ──►  Postgres+pgvector
 (edgar.py)     (chunk.py)       (chunk.py)          (chunk.py)   (embed.py)   (db.py + init.sql)
                                                                                     │
                        "commercial real estate credit exposure"  ──► embed query ──┘
                                                                          │
                                                            cosine search (db.py) ──► cited answer (main.py)
```

On Citi's most recent 10-K this produces **1,371 section-labeled chunks** across
**12 sections** (including `Item 1A. Risk Factors` and Citi's large
`Managing Global Risk` section). The Phase-1 sample query returns, as its top
hit, the filing's *Exposure to Commercial Real Estate* passage at a cosine
similarity of **0.85** — real disclosure text, correctly cited, not boilerplate.

---

## Stack

| Layer | Choice |
|---|---|
| API | FastAPI + uvicorn |
| Text splitting | LangChain `RecursiveCharacterTextSplitter` |
| Embeddings | `BAAI/bge-small-en-v1.5` via sentence-transformers, 384-dim, **local** (MPS/M1 GPU) |
| Vector store | PostgreSQL 16 + pgvector (HNSW index, cosine distance) |
| DB driver | psycopg 3, **raw SQL, no ORM** |
| HTML parsing | BeautifulSoup (stdlib `html.parser` backend) |
| Source data | SEC EDGAR REST API directly — no third-party SEC wrapper |
| Answer synthesis | Anthropic Claude (`claude-opus-4-8`), optional |

---

## Setup / runbook

**Prerequisites** (macOS, Apple Silicon):
- Python 3.11 (`brew install python@3.11`)
- Docker Desktop (running)
- A real name + email for `SEC_USER_AGENT` (SEC returns **403** without it)

```bash
# 1. Environment
python3.11 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt          # ~2GB torch on first install; slow once

# 2. Config — copy the template and put a REAL name+email in SEC_USER_AGENT
cp .env.example .env
#   edit .env: SEC_USER_AGENT="Your Name your.email@example.com"
#   (optional) ANTHROPIC_API_KEY=... to enable /ask synthesis

# 3. Database (Postgres 16 + pgvector on host port 5433)
docker compose up -d --wait

# 4. Ingest a filing (Citi's most recent 10-K)
python -m app.ingest --tickers C --forms 10-K --limit 1
#   → prints the detected sections (incl. "Item 1A. Risk Factors") and chunk count

# 5. Serve the API
uvicorn app.main:app --port 8000

# 6. Search
curl -X POST localhost:8000/search -H 'content-type: application/json' \
  -d '{"query":"commercial real estate credit exposure","ticker":"C","k":5}'
```

**Endpoints:** `GET /health`, `POST /search`, `POST /ask`.

**Gotchas** (in the order you'll hit them):
- SEC **403** → `SEC_USER_AGENT` isn't a real name+email. Their policy, not a bug.
- Changed the schema or `EMBED_DIM`? `docker compose up` will **not** re-run
  `db/init.sql` on an existing volume — run `docker compose down -v` first.
- macOS has no `timeout` command — don't wrap `docker`/`curl` in it.

---

## How it works — and why (the defensible decisions)

### 1. Fetch straight from EDGAR, no wrapper library (`edgar.py`)
EDGAR's REST API is small and well documented: `company_tickers.json` maps
ticker→CIK, `submissions/CIK{cik}.json` lists filings, and the archive serves
the document. We call it with stdlib `urllib` and a single rate-limited helper
(~4 req/sec) that enforces the required `User-Agent` in one place.

- **Why no `requests` / no `sec-edgar` wrapper?** A wrapper trades ~100 lines of
  readable code for an opaque dependency an interviewer can't see inside. The
  whole point is defensibility. stdlib `urllib` covers a handful of GETs fine.
- **One subtlety worth knowing:** the accession number is formatted
  `0000831001-26-000011` in the JSON but the archive URL uses it **without
  dashes**. That's an EDGAR inconsistency we normalize at the one place it matters.

### 2. HTML → clean text (`chunk.py`, stage 1)
Modern 10-Ks are **inline XBRL**: ~16 MB of XHTML with machine-readable
accounting tags interleaved with the prose. We parse with BeautifulSoup,
`decompose()` the `<ix:header>` block (hidden facts + `fasb.org` context URLs)
plus `<script>`/`<style>`, then `get_text(separator="\n")` and drop blank lines.
Result: **16 MB → ~1.16 MB** of clean prose.

- **Why BeautifulSoup and not stdlib `html.parser`?** stdlib *can* strip tags,
  but cleanly removing the XBRL metadata block and navigating nested tables in
  callback code is far less readable. Readability is the priority; we use BS4's
  built-in `html.parser` backend so we don't also pull in `lxml`.

### 3. Section-aware splitting — the intellectual core (`chunk.py`, stage 2)
This is what lets a search hit cite *"Item 1A. Risk Factors"* instead of quoting
text from nowhere. It's also where the real work was, because **Citi's 10-K does
not use the textbook `Item 1A` headings.** We discovered its actual structure
empirically:

- The body uses **all-caps section titles** (`RISK FACTORS`, `MANAGING GLOBAL
  RISK`, `CAPITAL RESOURCES`, …), read from the filing's own table of contents.
- The **same title appears in two tables of contents** near the top; there it's
  followed by a page number, while in the body it's followed by a real sentence.

So the detector: (a) matches a line against Citi's known section titles
(normalizing curly quotes / dashes / case), and (b) accepts it as a real heading
only if a **real paragraph** — `≥ 8 words, ≥ 50 chars, contains lowercase —
starts within the next few lines. Both tables of contents fail that test; only
the body passes. We then slice the text between consecutive headings and tag each
slice with a canonical label (`RISK FACTORS → "Item 1A. Risk Factors"`; Citi's
own names, like `Managing Global Risk`, are kept as-is because forcing an Item
number onto them would be *less* accurate).

- **Hard-won lesson (a great thing to tell in an interview):** `OVERVIEW` is
  *not* a usable anchor. Citi reuses it as a subsection header inside almost
  every major section, so anchoring on it fired **9 times** and stole other
  sections' content. **A section anchor must uniquely mark one section.**
- **Why not just regex `Item\s+1A`?** Proven empirically not to work — that
  literal string appears *nowhere* in Citi's filing.

### 4. Chunking (`chunk.py`, stage 3)
Each section is split with `RecursiveCharacterTextSplitter` (paragraph → line →
word boundaries, so chunks rarely cut mid-sentence). **Every chunk inherits its
section label** — the payoff of stage 2.

- **`CHUNK_SIZE=1000` chars (~250 tokens), `CHUNK_OVERLAP=150`** — comfortably
  under bge-small's 512-token limit, big enough to hold one idea, with overlap so
  a sentence straddling a boundary stays findable from both chunks. Both live in
  `.env`/`config.py`, not hardcoded (every tunable in config).

### 5. Embeddings — local, normalized, query/document asymmetry (`embed.py`)
`BAAI/bge-small-en-v1.5`, 384-dim, run on the M1 GPU (`mps`). The model loads
once and is cached. Two deliberate details:

- **Local, not an API.** No per-call cost, no data leaving the machine, and it's
  fast enough on-device. The only optional external call in the whole system is
  `/ask`.
- **Query/document asymmetry.** bge-v1.5 was trained so the *search query* gets a
  short instruction prepended (`"Represent this sentence for searching relevant
  passages:"`) while the stored passages do **not**. `embed_query()` adds it;
  `embed_documents()` doesn't. Skipping it measurably hurts recall.
- **L2-normalized** so cosine distance is clean and similarity scores are
  directly interpretable (relevant ≈ 0.7–0.85; the ~0.6 relevance bar lives here).
- **Fail-loud dimension check:** on load we assert the model outputs `EMBED_DIM`
  (384); otherwise the mismatch wouldn't surface until a confusing insert error
  against the `VECTOR(384)` column.

### 6. Storage & retrieval — raw SQL over pgvector (`db.py`, `db/init.sql`)
Two tables: `documents` (one row per filing, keyed by the globally-unique
`accession_number`) and `chunks` (`section`, `chunk_index`, `text`, `embedding
VECTOR(384)`, FK to the document with `ON DELETE CASCADE`), plus an **HNSW index
with `vector_cosine_ops`**.

- **Why Postgres + pgvector, not Pinecone/Weaviate?** One system to run, no extra
  service or cost, transactional, and the same DB could hold relational metadata.
  For a single-machine portfolio project, a managed vector DB is complexity with
  no payoff.
- **Why raw SQL, no ORM?** The retrieval query *is* the intellectual content —
  an ORM would hide the one thing worth understanding. The search is one readable
  statement: order by `embedding <=> query`, return `1 - distance` as similarity.
- **Idempotent ingest** (chosen deliberately): `upsert_document` does
  `ON CONFLICT (accession_number) DO UPDATE`, and `insert_chunks` deletes a
  document's old chunks before inserting new ones. Running ingest twice leaves
  `documents=1, chunks=1371` — not doubled.
- **Two casts the live database forced on us** (good "I debugged this" stories):
  - `%(qvec)s::vector` — psycopg sends a Python list as a Postgres `float8[]`, and
    there's no `vector <=> float8[]` operator. Inserts work via an implicit
    array→vector cast in assignment context; an operator expression needs the
    explicit `::vector`.
  - `%(ticker)s::text` — with `ticker` NULL, Postgres can't infer the parameter's
    type in `$1 IS NULL`; the cast tells it.

### 7. The pipeline & the API (`ingest.py`, `main.py`)
`ingest.py` is a thin argparse CLI that glues fetch → chunk → embed → store.
`main.py` is a thin FastAPI app:
- `GET /health` — liveness + a real DB reachability check.
- `POST /search` — the raw retrieval (embed query → cosine search → top-k with
  section + score). No LLM.
- `POST /ask` — retrieve, then Claude (`claude-opus-4-8`) synthesizes an answer
  **using only the retrieved chunks** and cites the section. If no
  `ANTHROPIC_API_KEY` is set it returns the sources with a note, so the app runs
  offline and `/search` never depends on a key.

---

## Project layout

```
finrisk/
├─ README.md            <- this file
├─ docker-compose.yml   <- Postgres 16 + pgvector, host port 5433
├─ requirements.txt
├─ .env.example         <- copy to .env; SEC_USER_AGENT must be real
├─ db/init.sql          <- schema; runs once on first volume creation
└─ app/
   ├─ config.py         <- every tunable, read from .env
   ├─ edgar.py          <- ticker→CIK, filing index, download, rate limiting
   ├─ chunk.py          <- HTML→text, section detection, chunking
   ├─ embed.py          <- sentence-transformers wrapper, MPS, query/doc asymmetry
   ├─ db.py             <- raw SQL: connect, upsert, insert, cosine search
   ├─ ingest.py         <- CLI pipeline
   └─ main.py           <- FastAPI: /health, /search, /ask
```

---

## Known limitations (be honest about these)

- **Section detection is tuned to Citi's filing structure.** The *approach*
  (anchor on the filing's own section titles; reject table-of-contents matches by
  requiring following prose) is general, but the title list is Citi's. A
  different filer would need its titles added — a one-line change to
  `SECTION_HEADINGS`, by design.
- **The Business overview + MD&A (pages ~4–36) are currently dropped** — Citi
  prints that section's heading split across two lines, which the single-line
  matcher misses. Not needed for the credit-risk demo (Risk Factors + Managing
  Global Risk carry it), but a real coverage gap.
- **One filing, one company** in the demo. The pipeline takes multiple tickers
  and forms; only Citi's 10-K has been ingested and verified.

## Not built yet (Phase 2, intentionally out of scope)

Query decomposition + parallel retrieval, a reranker over the top-K, and a
recall@5 eval set. Deferred until Phase 1 was proven end-to-end — which it now is.