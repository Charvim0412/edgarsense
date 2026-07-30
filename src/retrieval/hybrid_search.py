"""
Hybrid retrieval: combines dense vector search (pgvector) with keyword
search (BM25), merged via Reciprocal Rank Fusion (RRF).

Why: pure dense embeddings are fuzzy — a query like "supply chain risk" can
rank a chunk about general "risk factors" above a chunk that explicitly says
"single or limited sources for supply and manufacture" (a real example we
saw during testing). BM25 catches exact/near-exact keyword matches that
embeddings sometimes miss; RRF blends both rankings without needing to
normalize two different similarity scales against each other.
"""

import re
from rank_bm25 import BM25Okapi
from sentence_transformers import SentenceTransformer

from src.retrieval.db import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"
RRF_K = 60  # standard RRF damping constant


def _tokenize(text):
    return re.findall(r"[a-z0-9]+", text.lower())


def _fetch_chunks(ticker=None):
    """Pull all chunk rows (id, text, metadata, embedding) for BM25 + reference."""
    conn = get_connection()
    cur = conn.cursor()
    if ticker:
        cur.execute(
            "SELECT id, ticker, item_number, heading, text FROM chunks WHERE ticker = %s",
            (ticker,),
        )
    else:
        cur.execute("SELECT id, ticker, item_number, heading, text FROM chunks")
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows


def _dense_search(query, ticker=None, top_k=20):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])[0]

    conn = get_connection()
    cur = conn.cursor()
    if ticker:
        cur.execute(
            """
            SELECT id, ticker, item_number, heading, text
            FROM chunks WHERE ticker = %s
            ORDER BY embedding <=> %s::vector ASC LIMIT %s
            """,
            (ticker, list(query_embedding), top_k),
        )
    else:
        cur.execute(
            """
            SELECT id, ticker, item_number, heading, text
            FROM chunks ORDER BY embedding <=> %s::vector ASC LIMIT %s
            """,
            (list(query_embedding), top_k),
        )
    rows = cur.fetchall()
    cur.close()
    conn.close()
    return rows  # ordered best-to-worst by vector distance


def _bm25_search(query, ticker=None, top_k=20):
    rows = _fetch_chunks(ticker=ticker)
    if not rows:
        return []

    corpus = [_tokenize(r[4]) for r in rows]
    bm25 = BM25Okapi(corpus)
    scores = bm25.get_scores(_tokenize(query))

    ranked = sorted(zip(rows, scores), key=lambda x: x[1], reverse=True)
    return [row for row, score in ranked[:top_k]]


def hybrid_search(query, top_k=5, ticker=None, candidate_k=20):
    """
    Run dense + BM25 search, merge via Reciprocal Rank Fusion, return top_k.
    Each result: {"ticker", "item_number", "heading", "text"}
    """
    dense_results = _dense_search(query, ticker=ticker, top_k=candidate_k)
    bm25_results = _bm25_search(query, ticker=ticker, top_k=candidate_k)

    # RRF: score(doc) = sum over each ranked list of 1 / (RRF_K + rank)
    rrf_scores = {}
    row_by_id = {}

    for rank, row in enumerate(dense_results):
        chunk_id = row[0]
        row_by_id[chunk_id] = row
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank)

    for rank, row in enumerate(bm25_results):
        chunk_id = row[0]
        row_by_id[chunk_id] = row
        rrf_scores[chunk_id] = rrf_scores.get(chunk_id, 0) + 1.0 / (RRF_K + rank)

    merged_ids = sorted(rrf_scores.keys(), key=lambda cid: rrf_scores[cid], reverse=True)

    results = []
    for chunk_id in merged_ids[:top_k]:
        _, ticker_val, item_number, heading, text = row_by_id[chunk_id]
        results.append(
            {"ticker": ticker_val, "item_number": item_number, "heading": heading, "text": text}
        )
    return results
