# EdgarSense

Multi-agent RAG system that answers analyst-grade questions over SEC filings.

## Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env   # fill in SEC_USER_AGENT
docker compose up -d
docker compose exec -T db psql -U edgarsense -d edgarsense < src/retrieval/schema.sql
python -m src.ingestion.run_ingest --ticker AAPL --form-type 10-K --limit 1
python -m src.retrieval.embed_and_store
python -m src.retrieval.search "your question here"
