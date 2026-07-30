import json
import time
from pathlib import Path
from src.agents.graph import run_query

GOLDEN_PATH = Path(__file__).parent / "golden_set.json"


def run_eval():
    cases = json.loads(GOLDEN_PATH.read_text())
    results = []

    for case in cases:
        start = time.time()
        state = run_query(case["question"])
        latency_ms = int((time.time() - start) * 1000)

        retrieved_tickers = {e["ticker"] for e in state["excerpts"]}
        retrieved_items = {e["item_number"] for e in state["excerpts"]}

        retrieval_hit = (
            case["expected_ticker"] is None
            or (case["expected_ticker"] in retrieved_tickers
                and case["expected_item"] in retrieved_items)
        )

        results.append({
            "question": case["question"],
            "retrieval_hit": retrieval_hit,
            "num_excerpts": len(state["excerpts"]),
            "route": state["route"],
            "latency_ms": latency_ms,
            "answer": state["answer"],
        })

    hit_rate = sum(r["retrieval_hit"] for r in results) / len(results)
    print(f"\nRetrieval hit rate: {hit_rate:.1%} ({sum(r['retrieval_hit'] for r in results)}/{len(results)})")
    avg_latency = sum(r["latency_ms"] for r in results) / len(results)
    print(f"Avg latency: {avg_latency:.0f}ms\n")

    out_path = Path(__file__).parent / f"eval_run_{int(time.time())}.json"
    out_path.write_text(json.dumps(results, indent=2))
    print(f"Full results saved to {out_path}")
    return results


if __name__ == "__main__":
    run_eval()
