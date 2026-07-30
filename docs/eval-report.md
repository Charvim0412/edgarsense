# Eval report: does hybrid retrieval actually help?

## Setup

Two retrieval approaches were tested against a golden set of questions with
known-correct answers (expected ticker + expected filing section):

1. **Dense-only search** — pgvector cosine similarity over
   `all-MiniLM-L6-v2` embeddings, no keyword matching.
2. **Hybrid search** — dense search combined with BM25 keyword search,
   merged via Reciprocal Rank Fusion (RRF, k=60).

Both were tested on real, ingested 10-K filings from 8 companies: AAPL,
AMZN, GOOGL, JPM, META, MSFT, NVDA, TSLA.

Two metrics were measured:
- **Hit rate** — does the expected section appear anywhere in the top 5
  retrieved chunks?
- **Top-1 match rate** — is the expected section specifically the #1
  ranked result? This is the stricter, more honest signal, since ranking
  quality directly affects what gets fed to the LLM as primary context.

## Result 1: dense-only search missed obvious keyword matches

Manual testing with the query *"What did Apple say about supply chain
risk?"* showed dense-only search ranking a chunk containing the exact
phrase "single or limited sources for the supply and manufacture of many
critical components" at position 5 out of 5 — behind four less directly
relevant Risk Factors chunks about stock volatility, climate change, and
general risk language.

Re-running the same query with hybrid search (ticker-scoped to AAPL)
surfaced a chunk containing "custom components available from only one
source" — much closer to the actual supply-chain risk language — within
the top 2 results.

**Takeaway:** dense embeddings alone are fuzzy on financial/legal boilerplate
where "risk factors" chunks all sound similar semantically. Exact or
near-exact keyword matches (BM25) catch specific language that embeddings
blur together. This confirms the original hypothesis for using hybrid
retrieval instead of dense-only.

## Result 2: golden set size changed the measured accuracy substantially

| Golden set size | Hit rate (top 5) | Top-1 match rate |
|---|---|---|
| 6 questions, 3 tickers | 100.0% (6/6) | not measured (v1 script only checked top-5) |
| 16 questions, 8 tickers | 68.8% (11/16) | 31.2% (5/16) |

The small 6-question set showed a misleadingly perfect hit rate. Expanding
to 16 questions across 8 companies (and adding the stricter top-1 metric)
revealed materially lower, more realistic performance.

**Takeaway:** a small golden set can produce a falsely confident eval
number. This is a real, common eval-harness pitfall — the fix wasn't
tuning the retrieval system, it was recognizing the eval itself was too
small to trust.

## Result 3: root cause of top-1 misses — short sections lose to long ones

Manually reviewing every `[miss]` case in the 16-question run showed a
consistent pattern:

| Question | Expected section | Top-1 result instead |
|---|---|---|
| NVIDIA foreign jurisdiction disclosures | Item 9c (short, narrow topic) | Item 1 (Business — long) |
| Meta properties and facilities | Item 2 (short) | Item 1 (Business — long) |
| Amazon MD&A financial results | Item 7 | Item 9a (short, unrelated boilerplate) |

Short, narrowly-scoped sections (Properties, foreign jurisdiction
disclosures) were consistently outranked by longer, keyword-dense sections
like "Business." Longer chunks have more surface area for both BM25 term
overlap and semantic similarity, which systematically advantages them
regardless of topical relevance — a known limitation of chunk-level
retrieval, not a bug specific to this implementation.

## Result 4: a parser edge case surfaced during eval, not before

Investigating the Amazon MD&A miss led to discovering that several AMZN
sections (Items 9a, 9b, 10, 12–16) were correctly bounded and searchable,
but had **blank headings** — the Item-heading regex didn't fully match
Amazon's specific filing format. The section content itself was intact and
correct (verified by direct database inspection), only the display label
was missing.

**Takeaway:** this is exactly the kind of issue an eval harness is supposed
to surface — it wasn't visible from spot-checking a couple of filings
manually, but showed up clearly once systematic testing covered more
companies.

## What I'd do with more time

- **Chunk by fixed size across section boundaries**, rather than one chunk
  per Item section, so short and long sections compete more fairly in
  ranking rather than long sections dominating by sheer text volume.
- **Fix the Item-heading regex** to handle Amazon's filing format (and
  audit the other tickers for the same gap before trusting headings
  broadly).
- **Add a re-ranking step** — a cross-encoder re-rank of the top ~10 RRF
  results before returning the final top 5, which could meaningfully
  improve top-1 accuracy without changing the retrieval candidates
  themselves.
- **Grow the golden set further** (50+ questions) for a more statistically
  stable accuracy number — 16 questions is enough to reveal a real pattern,
  but not enough to precisely quantify it.

## Reproducing this

```bash
python -m src.eval.run_retrieval_eval
```
No API key required — this only exercises the retrieval layer (pgvector +
BM25 + RRF), not the LLM generation or critic agent.
