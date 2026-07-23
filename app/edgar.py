"""
edgar.py — talks to SEC EDGAR directly: ticker -> CIK, filing index, document
download, rate limiting.

Why this file exists: CLAUDE.md's stack deliberately rules out a third-party
SEC wrapper library. EDGAR's REST API is small and well-documented, so
depending on a wrapper here would trade a couple hundred lines of readable
code for an opaque dependency an interviewer can't see inside.
"""

import json
import time
import urllib.error
import urllib.request
from pathlib import Path

from app.config import SEC_USER_AGENT

# SEC's stated policy allows up to 10 requests/second and requires a real
# User-Agent. We stay well under that (4/sec) — a portfolio project pulling a
# handful of filings has no reason to push a free public API's limit, and
# "we're a good citizen of someone else's server" is the more defensible
# design choice than "we're fast."
_MIN_SECONDS_BETWEEN_REQUESTS = 0.25
_last_request_time = 0.0

# The ticker->CIK map is ~800KB and covers every public company. Ticker
# assignments change rarely, so caching it to disk avoids re-downloading the
# same static file on every ingest run during a multi-day project.
_CACHE_DIR = Path(__file__).resolve().parent.parent / ".cache"
_TICKER_MAP_CACHE = _CACHE_DIR / "company_tickers.json"


def _rate_limited_get(url: str) -> bytes:
    """Every EDGAR request goes through here so the User-Agent header and the
    rate limit are enforced in exactly one place, instead of being
    re-implemented (or forgotten) at each call site."""
    global _last_request_time
    elapsed = time.monotonic() - _last_request_time
    if elapsed < _MIN_SECONDS_BETWEEN_REQUESTS:
        time.sleep(_MIN_SECONDS_BETWEEN_REQUESTS - elapsed)

    request = urllib.request.Request(url, headers={"User-Agent": SEC_USER_AGENT})
    try:
        with urllib.request.urlopen(request) as response:
            body = response.read()
    finally:
        _last_request_time = time.monotonic()

    return body


def get_cik_for_ticker(ticker: str) -> str:
    """
    Resolve a ticker (e.g. 'C' for Citigroup) to a 10-digit, zero-padded CIK
    string (e.g. '0000831001'). Every other EDGAR endpoint we call needs this form.
    """
    if _TICKER_MAP_CACHE.exists():
        raw = _TICKER_MAP_CACHE.read_bytes()
    else:
        raw = _rate_limited_get("https://www.sec.gov/files/company_tickers.json")
        _CACHE_DIR.mkdir(exist_ok=True)
        _TICKER_MAP_CACHE.write_bytes(raw)

    data = json.loads(raw)
    ticker_upper = ticker.upper()
    for entry in data.values():
        if entry["ticker"].upper() == ticker_upper:
            return str(entry["cik_str"]).zfill(10)

    raise ValueError(f"No CIK found for ticker '{ticker}'")


def get_recent_filings(cik: str, form_type: str = "10-K", limit: int = 1) -> list[dict]:
    """
    Return up to `limit` most recent filings of `form_type` for a CIK, each as
    {"accession_number", "filing_date", "primary_document"}.

    Why we zip parallel arrays together: EDGAR's submissions JSON stores
    filing metadata as several same-length arrays (one of accession numbers,
    one of dates, one of form types, ...) rather than a list of filing
    objects. That's their schema, not a choice we made — we reassemble it
    into one dict per filing so the rest of the code doesn't have to think
    about index alignment.
    """
    url = f"https://data.sec.gov/submissions/CIK{cik}.json"
    data = json.loads(_rate_limited_get(url))

    recent = data["filings"]["recent"]
    matches = []
    for i, form in enumerate(recent["form"]):
        if form == form_type:
            matches.append({
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
            })
        if len(matches) == limit:
            break

    return matches


def download_filing_document(cik: str, accession_number: str, primary_document: str) -> str:
    """
    Fetch the actual filing document (HTML, typically) as text.

    Why the accession number's dashes get stripped here but not elsewhere:
    EDGAR's submissions JSON formats it like '0000831001-26-000011', but the
    document's path on their archive server uses the same number with the
    dashes removed. That's an EDGAR URL-scheme inconsistency, not something
    we introduced — so we only normalize it at the one place it matters.
    """
    accession_no_dashes = accession_number.replace("-", "")
    cik_no_leading_zeros = str(int(cik))  # archive URLs drop the zero-padding
    url = (
        f"https://www.sec.gov/Archives/edgar/data/"
        f"{cik_no_leading_zeros}/{accession_no_dashes}/{primary_document}"
    )
    return _rate_limited_get(url).decode("utf-8", errors="replace")
