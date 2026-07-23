"""
agent.py — the Phase-2 agentic layer: a LangGraph state machine that turns a
single-shot lookup into decompose -> parallel retrieve -> synthesize.

Why LangGraph (the concept to know): a plain function could call these three
steps in a row, so why a graph? A LangGraph `StateGraph` models the workflow as
NODES (plain functions that each read a shared, typed STATE object and return
the fields they want to update) wired by EDGES (control flow). The payoff is
that the pipeline becomes an explicit, inspectable object rather than nested
calls: you can add conditional branches (e.g. "if the question is simple, skip
decomposition"), loops (retry/refine), fan-out for true parallelism, streaming,
and checkpointing — without rewriting the steps. It's the standard scaffold for
multi-step LLM agents, which is exactly what an interviewer means by "agentic."

The flow here:
  decompose   — split a broad question into 2-4 focused sub-questions (LLM)
  retrieve    — search the vector store for EACH sub-question and merge results
  synthesize  — write one cited answer over all retrieved passages, flagging
                any conflicts between sources (LLM)

No ANTHROPIC_API_KEY? The two LLM nodes degrade gracefully (decompose -> the
original question; synthesize -> return the sources with a note), so the graph
still runs end-to-end and the retrieval is fully exercisable. Real decomposition
and a synthesized answer light up once a key is set.
"""

from typing import TypedDict

from anthropic import Anthropic
from langgraph.graph import StateGraph, END

from app import db
from app.config import ANTHROPIC_API_KEY, ANTHROPIC_MODEL


class AgentState(TypedDict):
    """The shared state every node reads from and writes to. LangGraph merges
    each node's returned dict into this — nodes return only the keys they change."""
    question: str
    ticker: str | None
    k: int
    sub_questions: list[str]
    hits: list[dict]
    answer: str | None
    used_llm: bool


def _claude(system: str, user: str, max_tokens: int) -> str:
    """One place that calls Claude, mirroring how /ask does it in main.py."""
    client = Anthropic(api_key=ANTHROPIC_API_KEY)
    message = client.messages.create(
        model=ANTHROPIC_MODEL,
        max_tokens=max_tokens,
        system=system,
        messages=[{"role": "user", "content": user}],
    )
    return "".join(block.text for block in message.content if block.type == "text")


def decompose_node(state: AgentState) -> dict:
    """Break the question into a few focused sub-questions. This is what lets the
    agent answer a broad prompt ('what are Citi's main risks?') by pulling from
    several different parts of the filing instead of one lucky match."""
    question = state["question"]
    if not ANTHROPIC_API_KEY:
        # Fallback: no LLM -> treat the whole question as the single sub-question.
        return {"sub_questions": [question], "used_llm": False}

    system = (
        "You decompose a financial-analysis question into 2 to 4 focused "
        "sub-questions, each targeting a distinct aspect that would live in a "
        "different part of a bank's 10-K (e.g. credit risk vs. market risk vs. "
        "capital). Return ONLY the sub-questions, one per line, no numbering."
    )
    text = _claude(system, f"Question: {question}", max_tokens=300)
    subs = [line.lstrip("-•* ").strip() for line in text.splitlines() if line.strip()]
    return {"sub_questions": (subs or [question])[:4], "used_llm": True}


def retrieve_node(state: AgentState) -> dict:
    """Retrieve for EACH sub-question and merge. Because the sub-questions target
    different sections, this fans retrieval across the filing; results are
    de-duplicated by (filing, chunk), keeping the best similarity when a chunk is
    found by more than one sub-question. (The searches are independent, so this
    is the natural place to parallelize with LangGraph's Send fan-out later.)"""
    sub_questions = state["sub_questions"]
    ticker = state.get("ticker")
    k = state.get("k", 5)

    merged: dict[tuple, dict] = {}
    with db.connect() as conn:
        for sub in sub_questions:
            for hit in db.search(conn, sub, ticker=ticker, k=k):
                key = (hit["accession_number"], hit["chunk_index"])
                if key not in merged or hit["similarity"] > merged[key]["similarity"]:
                    merged[key] = hit

    hits = sorted(merged.values(), key=lambda h: h["similarity"], reverse=True)
    return {"hits": hits}


def synthesize_node(state: AgentState) -> dict:
    """Write one answer grounded ONLY in the retrieved passages, citing the
    section per claim and explicitly flagging conflicts between sources. Without
    a key we skip synthesis and let the caller return the raw sources."""
    hits = state["hits"]
    if not ANTHROPIC_API_KEY:
        return {"answer": None}

    context = "\n\n".join(
        f"[Source {i}] (section: {h['section']})\n{h['text']}"
        for i, h in enumerate(hits, 1)
    )
    system = (
        "You are a financial risk analyst. Answer the question using ONLY the "
        "provided sources. Cite the section for each claim, e.g. "
        "(Item 1A. Risk Factors). If sources disagree or are ambiguous, flag the "
        "conflict explicitly. If the sources don't contain the answer, say so."
    )
    answer = _claude(
        system, f"Question: {state['question']}\n\nSources:\n{context}", max_tokens=1024
    )
    return {"answer": answer}


def _build_graph():
    """Wire the three nodes into a linear graph: decompose -> retrieve ->
    synthesize -> END. Compiled once at import and reused."""
    builder = StateGraph(AgentState)
    builder.add_node("decompose", decompose_node)
    builder.add_node("retrieve", retrieve_node)
    builder.add_node("synthesize", synthesize_node)
    builder.set_entry_point("decompose")
    builder.add_edge("decompose", "retrieve")
    builder.add_edge("retrieve", "synthesize")
    builder.add_edge("synthesize", END)
    return builder.compile()


_GRAPH = _build_graph()


def run_agent(question: str, *, ticker: str | None = None, k: int = 5) -> dict:
    """Run the full agent graph on a question and return a JSON-friendly result:
    the sub-questions it asked, the synthesized answer (or None without a key),
    and the source passages it grounded on."""
    result = _GRAPH.invoke({
        "question": question,
        "ticker": ticker,
        "k": k,
        "sub_questions": [],
        "hits": [],
        "answer": None,
        "used_llm": False,
    })
    return {
        "question": question,
        "sub_questions": result["sub_questions"],
        "answer": result["answer"],
        "used_llm": result["used_llm"],
        "sources": result["hits"],
        "note": None if ANTHROPIC_API_KEY else (
            "Set ANTHROPIC_API_KEY in .env for real query decomposition and a "
            "synthesized answer; this run shows retrieval only."
        ),
    }
