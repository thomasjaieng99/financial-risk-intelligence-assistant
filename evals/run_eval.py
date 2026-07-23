"""
run_eval.py — measure retrieval quality against a fixed, hand-verified question
set, and compare the first-stage bi-encoder search against two-stage
search + cross-encoder reranker (Phase 2). The reranker earns its place by a
number, not by vibes.

Metrics per configuration:
  recall@1 — target ranked #1     (the number the reranker most visibly moves)
  recall@5 — target in the top 5  (CLAUDE.md's bar; already saturated at baseline)
  MRR      — mean of 1/rank        (sensitive to promotions like rank 3 -> 1)

Ground truth is one passage per question, (accession_number, chunk_index),
verified by reading it (see eval_questions.json). Tied to the current ingest.

    python -m evals.run_eval
"""

import json
from pathlib import Path

from app import db
from app.config import RERANK_CANDIDATES
from app.rerank import rerank_hits

_QUESTIONS = Path(__file__).parent / "eval_questions.json"


def _rank_of_target(hits: list[dict], accession: str, chunk_index: int) -> int | None:
    """1-based rank of the ground-truth chunk, or None if it's absent."""
    for rank, hit in enumerate(hits, start=1):
        if hit["accession_number"] == accession and hit["chunk_index"] == chunk_index:
            return rank
    return None


def _metrics(ranks: list[int | None]) -> tuple[float, float, float]:
    """recall@1, recall@5, MRR over a list of ranks (int, or None if not found)."""
    n = len(ranks)
    recall_1 = sum(1 for r in ranks if r == 1) / n
    recall_5 = sum(1 for r in ranks if r is not None and r <= 5) / n
    mrr = sum((1.0 / r if r else 0.0) for r in ranks) / n
    return recall_1, recall_5, mrr


def main() -> None:
    questions = json.loads(_QUESTIONS.read_text())

    baseline_ranks: list[int | None] = []
    reranked_ranks: list[int | None] = []
    rows = []

    with db.connect() as conn:
        for q in questions:
            acc, idx = q["target_accession"], q["target_chunk_index"]
            # One retrieval per question; BOTH configs score the SAME candidate
            # set, so any difference is purely the reranker's doing.
            candidates = db.search(conn, q["question"], ticker="C", k=RERANK_CANDIDATES)
            b_rank = _rank_of_target(candidates, acc, idx)
            reranked = rerank_hits(q["question"], list(candidates), top_k=len(candidates))
            r_rank = _rank_of_target(reranked, acc, idx)

            baseline_ranks.append(b_rank)
            reranked_ranks.append(r_rank)
            rows.append((q["id"], b_rank, r_rank, q["question"]))

    # Per-question: baseline rank -> reranked rank, with a direction marker.
    print(f"{'id':<3} {'base':<5} {'rerank':<7} {'':<4} question")
    print("-" * 74)
    for qid, b, r, question in rows:
        if b == r:
            mark = "  ="
        elif r is not None and (b is None or r < b):
            mark = " up"
        else:
            mark = " dn"
        print(f"{qid:<3} {str(b):<5} {str(r):<7} {mark:<4} {question[:46]}")
    print("-" * 74)

    b1, b5, bmrr = _metrics(baseline_ranks)
    r1, r5, rmrr = _metrics(reranked_ranks)
    n = len(questions)
    print(f"{'':<24}{'recall@1':>9}{'recall@5':>10}{'MRR':>8}")
    print(f"{'baseline (bi-encoder)':<24}{b1:>9.3f}{b5:>10.3f}{bmrr:>8.3f}")
    print(f"{'+ cross-encoder rerank':<24}{r1:>9.3f}{r5:>10.3f}{rmrr:>8.3f}")
    print(f"{'delta':<24}{r1 - b1:>+9.3f}{r5 - b5:>+10.3f}{rmrr - bmrr:>+8.3f}")
    print(f"\n(n={n} questions, {RERANK_CANDIDATES} candidates reranked per question)")


if __name__ == "__main__":
    main()
