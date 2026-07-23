"""
chunk.py — turns a filing's raw HTML into clean, section-labeled chunks ready
to embed.

This is the heart of the project's retrieval quality: HOW we split the text
decides what the search can and can't find. Everything here is aimed at one
goal — handing back a passage AND knowing which 10-K section it came from, so
answers can cite their source instead of quoting text from nowhere.

Built in stages: (1) HTML -> clean text, (2) split into labeled sections,
(3) split each section into embeddable chunks.
"""

import re

from bs4 import BeautifulSoup
from langchain_text_splitters import RecursiveCharacterTextSplitter

from app.config import CHUNK_SIZE, CHUNK_OVERLAP


def html_to_text(html: str) -> str:
    """
    Turn a filing's raw HTML into clean, readable plain text.

    Why this is its own step: the raw document is ~16MB of nested tables and
    inline-XBRL metadata (hidden machine-readable accounting facts, context
    and unit definitions) that are NOT prose a human reads. If we embedded
    that noise, search would match XBRL boilerplate instead of real
    disclosure. So we strip it here, before anything downstream sees the text.
    """
    # html.parser is Python's built-in backend — no lxml dependency. It's
    # lenient about the messy, not-quite-valid HTML that real filers produce.
    soup = BeautifulSoup(html, "html.parser")

    # Remove the inline-XBRL metadata block. <ix:header> holds the hidden facts
    # and context/unit definitions (the fasb.org URL soup we saw polluting the
    # top of the extracted text) — numbers for machines, not sentences for
    # readers. We also drop <script>/<style> as standard non-content noise.
    # decompose() deletes the tag AND its contents from the tree entirely.
    for tag in soup.find_all(["ix:header", "script", "style"]):
        tag.decompose()

    # separator="\n" puts a newline between each piece of text, so words from
    # different table cells (like "1A." and "Risk Factors") stay on separate
    # lines instead of being glued together — that structure helps the
    # section-detection step we'll add next. We chose newline over space
    # deliberately; it's a lever we may revisit once we see the output.
    text = soup.get_text(separator="\n")

    # get_text leaves lots of blank/whitespace-only lines behind. Strip each
    # line and drop the empty ones so the result is compact and scannable.
    lines = [line.strip() for line in text.splitlines()]
    return "\n".join(line for line in lines if line)


# Citi's ACTUAL body section titles, read straight from the filing's own table
# of contents — not the textbook Item list, because Citi organizes its 10-K
# under its own ALL-CAPS names and only some line up with standard items.
# KEY  = the heading text as Citi prints it (normalized to uppercase).
# VALUE = the label we tag chunks with. Where a section maps cleanly to a
#         standard 10-K item we use the canonical "Item N" label (so citations
#         read the way an analyst expects); Citi-specific sections keep their
#         own name, since forcing an Item number onto them would be LESS
#         accurate. This list is derived from Citi's TOC; the run confirms
#         which titles actually appear as body headings.
# NOTE: "OVERVIEW" is deliberately NOT an anchor. Citi reuses it as a subsection
# header inside almost every major section (Capital Resources, Managing Global
# Risk, etc.), so anchoring on it fires everywhere and steals other sections'
# content. An anchor must uniquely mark ONE section.
SECTION_HEADINGS = {
    "MANAGEMENT'S DISCUSSION AND ANALYSIS OF FINANCIAL CONDITION AND RESULTS OF OPERATIONS": "Item 7. Management's Discussion and Analysis",
    "CAPITAL RESOURCES": "Capital Resources",
    "RISK FACTORS": "Item 1A. Risk Factors",
    "SUSTAINABILITY": "Sustainability",
    "HUMAN CAPITAL RESOURCES AND MANAGEMENT": "Human Capital Resources and Management",
    "MANAGING GLOBAL RISK": "Managing Global Risk",
    "SIGNIFICANT ACCOUNTING POLICIES AND SIGNIFICANT ESTIMATES": "Significant Accounting Policies and Estimates",
    "DISCLOSURE CONTROLS AND PROCEDURES": "Item 9A. Controls and Procedures",
    "FORWARD-LOOKING STATEMENTS": "Forward-Looking Statements",
    "CONSOLIDATED FINANCIAL STATEMENTS": "Item 8. Consolidated Financial Statements",
    "NOTES TO CONSOLIDATED FINANCIAL STATEMENTS": "Item 8. Notes to Consolidated Financial Statements",
    "SUPERVISION, REGULATION AND OTHER": "Supervision, Regulation and Other",
    "OTHER INFORMATION": "Item 9B. Other Information",
}


def _normalize_heading(line: str) -> str:
    """Uppercase, collapse whitespace, and flatten the curly quotes and en-/em-
    dashes that real filings use — so a heading match doesn't fail purely on
    typography. The cleaned text contains "MANAGEMENT'S" (curly apostrophe),
    which would never == "MANAGEMENT'S" (straight) without this."""
    line = line.replace("’", "'").replace("‘", "'")
    line = line.replace("–", "-").replace("—", "-")
    line = re.sub(r"\s+", " ", line)
    return line.strip().upper()


def _looks_like_body_prose(line: str) -> bool:
    """True if a line reads like the start of a real paragraph — long, many
    words, with lowercase letters in it. This is what separates a real body
    heading (which is followed by a sentence) from a table-of-contents copy of
    the same title (followed by a page number, 'Not Applicable', or a short
    subsection name). Both of Citi's two tables of contents fail this test;
    only the actual body text passes it."""
    words = line.split()
    has_lowercase = any(c.islower() for c in line)
    return len(words) >= 8 and len(line) >= 50 and has_lowercase


def split_into_sections(text: str) -> list[dict]:
    """
    Split cleaned filing text into labeled sections. This is the intellectual
    core of the project.

    Strategy: Citi prints each section heading as an ALL-CAPS title on its own
    line. The SAME title also appears in the two tables of contents — but there
    it's followed by page numbers or short fragments, while in the body it's
    followed by an actual sentence. So a line is a real section boundary only
    when (a) it matches one of Citi's known section titles and (b) a real
    paragraph starts within the next few lines. We then slice the text between
    consecutive real headings; each slice is one section.
    """
    lines = text.split("\n")

    # Pass 1: find the real heading lines as (line_index, canonical_label).
    boundaries: list[tuple[int, str]] = []
    for i, line in enumerate(lines):
        label = SECTION_HEADINGS.get(_normalize_heading(line))
        if label is None:
            continue
        # A body heading is followed by prose within a few lines; a TOC copy is
        # not. Look ahead a little to allow a short sub-title or mini-TOC
        # between the heading and its first paragraph (Citi has those).
        lookahead = lines[i + 1:i + 6]
        if not any(_looks_like_body_prose(la) for la in lookahead):
            continue  # a table-of-contents entry, not the real heading
        boundaries.append((i, label))

    # Pass 2: slice the text between consecutive headings into sections. Text
    # before the first heading (cover page, TOC) is intentionally dropped.
    sections: list[dict] = []
    for j, (start_line, label) in enumerate(boundaries):
        end_line = boundaries[j + 1][0] if j + 1 < len(boundaries) else len(lines)
        body = "\n".join(lines[start_line + 1:end_line]).strip()
        sections.append({"section": label, "text": body})

    return sections


def chunk_sections(sections: list[dict]) -> list[dict]:
    """
    Split each section's text into embedding-sized chunks, carrying the section
    label onto every chunk.

    Why chunk at all: an embedding model turns text into a single vector, and a
    vector can only faithfully represent a smallish passage — embed a whole
    84k-character section and the meaning is mush. So we cut each section into
    pieces small enough to embed sharply but large enough to hold one idea.

    Why RecursiveCharacterTextSplitter: it tries a hierarchy of separators
    (paragraph break, then line break, then space, then character) and only
    falls to a coarser one when a piece is still too big. That means it breaks
    at natural boundaries first, so a chunk rarely ends mid-sentence.

    Why the label rides along: this is the whole payoff of split_into_sections.
    Because every chunk remembers it came from "Item 1A. Risk Factors", a later
    search hit can cite its source section instead of quoting text from nowhere.
    """
    # One splitter, configured from .env via config.py — nothing about chunk
    # size is hardcoded here (CLAUDE.md: every tunable lives in config).
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=CHUNK_SIZE,
        chunk_overlap=CHUNK_OVERLAP,
    )

    chunks: list[dict] = []
    for section in sections:
        for piece in splitter.split_text(section["text"]):
            chunks.append({"section": section["section"], "text": piece})

    return chunks
