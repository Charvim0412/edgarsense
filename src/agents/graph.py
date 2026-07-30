import re
from typing import TypedDict, List, Optional
from langgraph.graph import StateGraph, END

from src.retrieval.hybrid_search import hybrid_search
from src.generation.llm_client import generate_answer

KNOWN_TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "AMZN", "GOOGL", "META", "NVDA"]
MAX_RETRIES = 2

# Matches numbers like $123.4, 12.3%, 1,234, 45 — anything with 2+ digits so we
# skip noise like citation brackets [1] or single-digit item numbers.
NUMERIC_CLAIM_RE = re.compile(r"\$?\d[\d,]*\.?\d*%?")


class GraphState(TypedDict):
    question: str
    tickers: List[str]
    route: str
    excerpts: List[dict]
    answer: Optional[str]
    usage: Optional[dict]
    retry_count: int
    verified: bool
    critic_notes: Optional[str]


def route_node(state: GraphState) -> GraphState:
    q = state["question"].upper()
    found = [t for t in KNOWN_TICKERS if re.search(rf"\b{t}\b", q)]
    route = "comparison" if len(found) > 1 else ("single" if found else "general")
    return {**state, "tickers": found, "route": route}


def retrieve_node(state: GraphState) -> GraphState:
    # On a retry (critic flagged something), widen the search so we have a
    # better chance of finding support for the flagged claims.
    retry_boost = 3 if state.get("retry_count", 0) > 0 else 0
    excerpts = []

    if state["route"] == "general":
        results = hybrid_search(state["question"], top_k=6 + retry_boost)
        for r in results:
            excerpts.append(r)
    else:
        per_ticker_k = (5 if state["route"] == "single" else 3) + retry_boost
        for ticker in state["tickers"]:
            results = hybrid_search(state["question"], top_k=per_ticker_k, ticker=ticker)
            for r in results:
                excerpts.append(r)

    return {**state, "excerpts": excerpts}


def synthesize_node(state: GraphState) -> GraphState:
    if not state["excerpts"]:
        return {**state, "answer": "No relevant filing content found for this question.", "usage": None}
    answer, usage = generate_answer(state["question"], state["excerpts"])
    return {**state, "answer": answer, "usage": usage}


def extract_numeric_claims(text: str) -> List[str]:
    """Pull out number-like tokens (dollars, percentages, plain figures)
    with at least 2 digits, so single-digit noise doesn't trigger false flags."""
    matches = NUMERIC_CLAIM_RE.findall(text)
    return [m for m in matches if len(re.sub(r"[^\d]", "", m)) >= 2]


def critic_node(state: GraphState) -> GraphState:
    """
    Checks every numeric claim in the drafted answer against the raw text of
    the retrieved excerpts. This is a heuristic substring check, not perfect
    (it won't catch a number that's correct but derived via math the model
    did itself, e.g. a computed YoY delta) — but it catches the most common
    and most damaging failure: a number that simply doesn't appear anywhere
    in the source material at all.
    """
    answer = state.get("answer") or ""
    if not answer or not state["excerpts"]:
        return {**state, "verified": True, "critic_notes": None}

    claims = extract_numeric_claims(answer)
    source_text = " ".join(e["text"] for e in state["excerpts"])
    source_normalized = source_text.replace(",", "").replace("$", "")

    unsupported = []
    for claim in claims:
        normalized = claim.replace(",", "").replace("$", "").rstrip("%")
        if normalized not in source_normalized:
            unsupported.append(claim)

    if unsupported:
        notes = f"Unsupported numeric claims (not found in retrieved excerpts): {unsupported}"
        return {**state, "verified": False, "critic_notes": notes}

    return {**state, "verified": True, "critic_notes": None}


def route_after_critic(state: GraphState) -> str:
    if state["verified"]:
        return "end"
    if state.get("retry_count", 0) >= MAX_RETRIES:
        # Give up gracefully after max retries — surface the concern rather
        # than looping forever or silently returning an unverified answer.
        return "end"
    return "retry"


def increment_retry(state: GraphState) -> GraphState:
    return {**state, "retry_count": state.get("retry_count", 0) + 1}


def build_graph():
    graph = StateGraph(GraphState)
    graph.add_node("route", route_node)
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("synthesize", synthesize_node)
    graph.add_node("critic", critic_node)
    graph.add_node("increment_retry", increment_retry)

    graph.set_entry_point("route")
    graph.add_edge("route", "retrieve")
    graph.add_edge("retrieve", "synthesize")
    graph.add_edge("synthesize", "critic")
    graph.add_conditional_edges("critic", route_after_critic, {"end": END, "retry": "increment_retry"})
    graph.add_edge("increment_retry", "retrieve")

    return graph.compile()


_app = build_graph()


def run_query(question: str) -> GraphState:
    return _app.invoke({
        "question": question,
        "tickers": [],
        "route": "",
        "excerpts": [],
        "answer": None,
        "usage": None,
        "retry_count": 0,
        "verified": False,
        "critic_notes": None,
    })
