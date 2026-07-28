import sys
from sentence_transformers import SentenceTransformer

from src.retrieval.db import get_connection

MODEL_NAME = "all-MiniLM-L6-v2"


def search(query, top_k=5, ticker=None):
    model = SentenceTransformer(MODEL_NAME)
    query_embedding = model.encode([query])[0]

    conn = get_connection()
    cur = conn.cursor()

    if ticker:
        cur.execute(
            """
            SELECT ticker, item_number, heading, text, embedding <=> %s::vector AS distance
            FROM chunks WHERE ticker = %s ORDER BY distance ASC LIMIT %s
            """,
            (list(query_embedding), ticker, top_k),
        )
    else:
        cur.execute(
            """
            SELECT ticker, item_number, heading, text, embedding <=> %s::vector AS distance
            FROM chunks ORDER BY distance ASC LIMIT %s
            """,
            (list(query_embedding), top_k),
        )

    results = cur.fetchall()
    cur.close()
    conn.close()
    return results


def main():
    if len(sys.argv) < 2:
        print('Usage: python -m src.retrieval.search "your question here"')
        sys.exit(1)

    query = sys.argv[1]
    results = search(query)

    print(f"\nTop {len(results)} results for: {query}\n")
    for i, (ticker, item_number, heading, text, distance) in enumerate(results, 1):
        print(f"--- Result {i} (distance={distance:.4f}) ---")
        print(f"{ticker} | Item {item_number} — {heading}")
        print(text[:300] + ("..." if len(text) > 300 else ""))
        print()


if __name__ == "__main__":
    main()
