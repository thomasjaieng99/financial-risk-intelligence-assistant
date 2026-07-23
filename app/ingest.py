"""
ingest.py — the CLI that runs the whole pipeline end to end and loads Postgres.

It's pure glue: EDGAR (edgar.py) -> clean/section/chunk (chunk.py) -> vectors
(embed.py) -> storage (db.py). No new logic lives here; the point is one command
that turns "a ticker" into "searchable rows."

    python -m app.ingest --tickers C --forms 10-K --limit 1
"""

import argparse

from app import db
from app.chunk import chunk_sections, html_to_text, split_into_sections
from app.edgar import (
    download_filing_document,
    get_cik_for_ticker,
    get_recent_filings,
)
from app.embed import embed_documents


def ingest_filing(conn, ticker: str, cik: str, form_type: str, filing: dict) -> None:
    """Run one filing all the way through and store it. Kept as its own function
    so the loop in main() reads as 'for each filing, ingest it' and the pipeline
    order (download -> section -> chunk -> embed -> store) is visible in one place."""
    html = download_filing_document(
        cik, filing["accession_number"], filing["primary_document"]
    )
    sections = split_into_sections(html_to_text(html))
    chunks = chunk_sections(sections)

    # Attach an embedding to each chunk. embed_documents runs once over all chunk
    # texts (batched on the GPU) rather than one call per chunk.
    for chunk, vector in zip(chunks, embed_documents([c["text"] for c in chunks])):
        chunk["embedding"] = vector

    # upsert + replace-chunks is idempotent: re-running never duplicates a filing.
    doc_id = db.upsert_document(
        conn,
        ticker=ticker,
        cik=cik,
        form_type=form_type,
        filing_date=filing["filing_date"],
        accession_number=filing["accession_number"],
        primary_document=filing["primary_document"],
    )
    n = db.insert_chunks(conn, doc_id, chunks)
    conn.commit()  # make this filing durable before moving to the next one

    # Print what the Phase-1 definition of done asks for: the detected sections
    # (Item 1A. Risk Factors should be among them) and a non-zero chunk count.
    section_labels = [s["section"] for s in sections]
    print(f"  {form_type} {filing['accession_number']} ({filing['filing_date']}) "
          f"-> document id {doc_id}")
    print(f"    sections detected ({len(section_labels)}): {section_labels}")
    print(f"    chunks embedded + stored: {n}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Ingest SEC filings into the finrisk vector store."
    )
    parser.add_argument("--tickers", nargs="+", required=True,
                        help="One or more tickers, e.g. C AAPL")
    parser.add_argument("--forms", nargs="+", default=["10-K"],
                        help="Filing form types to fetch (default: 10-K)")
    parser.add_argument("--limit", type=int, default=1,
                        help="Most recent N filings per form (default: 1)")
    args = parser.parse_args()

    # One connection for the whole run; each filing commits itself inside.
    with db.connect() as conn:
        for ticker in args.tickers:
            cik = get_cik_for_ticker(ticker)
            print(f"\n=== {ticker} (CIK {cik}) ===")
            for form_type in args.forms:
                filings = get_recent_filings(cik, form_type=form_type, limit=args.limit)
                if not filings:
                    print(f"  no {form_type} filings found for {ticker}")
                    continue
                for filing in filings:
                    ingest_filing(conn, ticker, cik, form_type, filing)


if __name__ == "__main__":
    main()