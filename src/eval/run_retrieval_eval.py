"""
Retrieval-only eval - tests hybrid_search directly, without calling the LLM.
No API key required, zero cost.

Two things are checked per case:
  - "hit": is the expected item present anywhere in the top-k results?
  - "top1_match": is the expected item the #1 ranked result specifically?

top1_match is the stricter, more honest signal - a chunk buried at rank 5
still gets fed to the LLM, but ranking quality matters for answer grounding.
"""

import json
from pathlib import Path
from src.retrieval.hybrid_search import hybrid_search

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def run_retrieval_eval():
    cases = json.loads(GOLDEN_PATH.read_text())
    results = []

    for case in cases:
        ticker = case.get("expected_ticker")
        retrieved = hybrid_search(case["question"], top_k=5, ticker=ticker)

        retrieved_items = [r["item_number"] for r in retrieved]

        if case.get("expected_item") is None:
            hit = True
            top1_match = True
        else:
            hit = case["expected_item"] in retrieved_items
            top1_match = (
                len(retrieved_items) > 0 and retrieved_items[0] == case["expected_item"]
            )

        results.append({
            "question": case["question"],
            "hit": hit,
            "top1_match": top1_match,
            "num_retrieved": len(retrieved),
            "top_result_heading": retrieved[0]["heading"] if retrieved else None,
            "top_result_item": retrieved[0]["item_number"] if retrieved else None,
        })

    hit_rate = sum(r["hit"] for r in results) / len(results)
    top1_rate = sum(r["top1_match"] for r in results) / len(results)
    print(f"\nHit rate (in top 5): {hit_rate:.1%} ({sum(r['hit'] for r in results)}/{len(results)})")
    print(f"Top-1 match rate:    {top1_rate:.1%} ({sum(r['top1_match'] for r in results)}/{len(results)})\n")

    for r in results:
        status = "checkmark" if r["top1_match"] else ("partial" if r["hit"] else "miss")
        print(f"[{status}] {r['question']}")
        print(f"   top result: Item {r['top_result_item']} - {r['top_result_heading']}")

    return results


if __name__ == "__main__":
    run_retrieval_eval()
