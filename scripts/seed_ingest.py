from src.ingestion.run_ingest import ingest_one

TICKERS = ["AAPL", "MSFT", "TSLA", "JPM", "AMZN", "GOOGL", "META", "NVDA"]

for ticker in TICKERS:
    try:
        ingest_one(ticker, form_type="10-K", limit=1)
    except Exception as e:
        print(f"[{ticker}] FAILED: {e}")
