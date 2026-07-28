import json
from sentence_transformers import SentenceTransformer

from src.config import DATA_PROCESSED
from src.retrieval.db import get_connection
from src.retrieval.chunker import chunk_text

MODEL_NAME = "all-MiniLM-L6-v2"


def load_filings():
    filings = []
    for path in DATA_PROCESSED.glob("*.json"):
        with open(path, encoding="utf-8") as f:
            filings.append(json.load(f))
    return filings


def main():
    print(f"Loading embedding model ({MODEL_NAME})...")
    model = SentenceTransformer(MODEL_NAME)

    filings = load_filings()
    print(f"Found {len(filings)} processed filing(s)")

    conn = get_connection()
    cur = conn.cursor()

    total_chunks = 0
    for filing in filings:
        ticker = filing["ticker"]
        accession = filing["accession_number"]
        filing_date = filing["filing_date"]

        cur.execute("SELECT COUNT(*) FROM chunks WHERE accession_number = %s", (accession,))
        existing = cur.fetchone()[0]
        if existing > 0:
            print(f"[{ticker}] {accession} already embedded, skipping")
            continue

        for section in filing["sections"]:
            pieces = chunk_text(section["text"])
            embeddings = model.encode(pieces, show_progress_bar=False)

            for i, (piece, emb) in enumerate(zip(pieces, embeddings)):
                cur.execute(
                    """
                    INSERT INTO chunks
                        (ticker, accession_number, filing_date, item_number,
                         heading, chunk_index, text, embedding)
                    VALUES (%s, %s, %s, %s, %s, %s, %s, %s)
                    """,
                    (ticker, accession, filing_date, section["item_number"],
                     section["heading"], i, piece, list(emb)),
                )
                total_chunks += 1

        conn.commit()
        print(f"[{ticker}] embedded and stored {accession}")

    cur.close()
    conn.close()
    print(f"Done. {total_chunks} new chunks stored.")


if __name__ == "__main__":
    main()
