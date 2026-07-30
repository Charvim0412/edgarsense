# EdgarSense

Multi-agent RAG system that answers analyst-grade questions over SEC filings -
built to explore what production-grade RAG looks like beyond a basic demo:
hybrid retrieval, a critic agent that verifies claims before returning them,
and an eval harness that measures whether any of it actually works.

## The problem

Equity analysts and researchers spend hours manually cross-referencing SEC
filings - 10-Ks routinely run 50-100+ pages, splitting narrative sections
(Risk Factors, MD&A) from dense financial tables. Generic "chat with a PDF"
tools tend to hallucinate specific numbers with total confidence, which is
exactly the failure mode you can least afford in financial research.

## What this does differently

- **Real EDGAR ingestion** - pulls actual filings from SEC's public APIs
  (not a static dataset), parses them into labeled sections (Item 1A Risk
  Factors, Item 7 MD&A, etc.)
- **Hybrid retrieval** - dense vector search (pgvector) combined with BM25
  keyword search, merged via Reciprocal Rank Fusion. Dense-only search
  reliably missed specific supply-chain risk language in testing; hybrid
  search surfaced it. See docs/eval-report.md.
- **Multi-agent pipeline (LangGraph)** - a state machine that routes the
  query, retrieves, drafts an answer, then runs a critic agent that
  checks every numeric claim in the draft against the retrieved source text.
  Unsupported claims trigger a re-retrieval loop (capped retries) instead of
  returning an unverified number.
- **A real eval harness, not a demo score** - a golden set of questions
  across 8 companies (AAPL, AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA),
  scored on both "is the right section in the top 5" and the stricter
  "is it ranked #1." The honest numbers, and why they matter, are in the
  eval report.

## Architecture

question -> route -> retrieve (hybrid search) -> synthesize (LLM draft)
                          ^                            |
                          +------- critic (verify) <---+
                            (loops back on unsupported claims,
                             capped at 2 retries)

## Known limitations (found via the eval harness, not guessed at)

- Retrieval top-1 accuracy is ~31% on a 16-question golden set spanning 8
  companies - short sections (Properties, foreign jurisdiction disclosures)
  are consistently outranked by longer, keyword-dense sections like
  "Business." This is a known limitation of chunk-level retrieval, not a bug.
- The Item-heading parser doesn't fully capture every filer's heading format -
  a few AMZN sections parsed with blank headings, still correctly bounded
  and searchable, just missing display labels.
- The critic agent does substring/numeric matching against source text - it
  catches numbers that don't appear anywhere in the retrieved excerpts, but
  won't catch a number that's technically present but misattributed, or
  verify a number the model computed itself (e.g. a derived YoY delta).

## Setup

python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
docker compose up -d
docker compose exec -T db psql -U edgarsense -d edgarsense < src/retrieval/schema.sql

### Ingest a filing
python -m src.ingestion.run_ingest --ticker AAPL --form-type 10-K --limit 1

### Build the vector store
python -m src.retrieval.embed_and_store

### Run the retrieval eval (no API key needed, zero cost)
python -m src.eval.run_retrieval_eval

### Run the full agent with the critic loop (requires ANTHROPIC_API_KEY)
python app.py

## Project layout

src/
  ingestion/      - EDGAR fetch + HTML section parsing
  retrieval/      - embeddings, pgvector, BM25, hybrid search (RRF)
  agents/         - LangGraph state machine: route, retrieve, synthesize, critic
  generation/     - LLM client
  eval/           - golden set + eval scripts
  observability/  - query logging
app.py            - Gradio demo UI

## Stack
Python, LangGraph, pgvector (Postgres via Docker), sentence-transformers
(local embeddings, no API cost), rank_bm25, Claude (Anthropic API) for
generation, Gradio for the demo UI.
