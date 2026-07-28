import time
import requests

from src.config import SEC_USER_AGENT

HEADERS = {"User-Agent": SEC_USER_AGENT}

TICKER_MAP_URL = "https://www.sec.gov/files/company_tickers.json"
SUBMISSIONS_URL = "https://data.sec.gov/submissions/CIK{cik:010d}.json"
ARCHIVES_BASE = "https://www.sec.gov/Archives/edgar/data"

_ticker_cache = None


def _get(url):
    resp = requests.get(url, headers=HEADERS, timeout=30)
    resp.raise_for_status()
    time.sleep(1)
    return resp.json()


def resolve_cik(ticker):
    global _ticker_cache
    if _ticker_cache is None:
        raw = _get(TICKER_MAP_URL)
        _ticker_cache = {v["ticker"].upper(): v["cik_str"] for v in raw.values()}

    ticker = ticker.upper()
    if ticker not in _ticker_cache:
        raise ValueError(f"Ticker '{ticker}' not found in SEC's ticker map.")
    return _ticker_cache[ticker]


def get_filing_list(cik, form_type="10-K", limit=5):
    data = _get(SUBMISSIONS_URL.format(cik=cik))
    recent = data["filings"]["recent"]

    results = []
    for i in range(len(recent["form"])):
        if recent["form"][i] != form_type:
            continue
        results.append(
            {
                "accession_number": recent["accessionNumber"][i],
                "filing_date": recent["filingDate"][i],
                "primary_document": recent["primaryDocument"][i],
                "form": recent["form"][i],
            }
        )
        if len(results) >= limit:
            break
    return results


def build_document_url(cik, accession_number, primary_document):
    accession_no_dashes = accession_number.replace("-", "")
    return f"{ARCHIVES_BASE}/{cik}/{accession_no_dashes}/{primary_document}"


def fetch_filing_html(cik, accession_number, primary_document):
    url = build_document_url(cik, accession_number, primary_document)
    resp = requests.get(url, headers=HEADERS, timeout=60)
    resp.raise_for_status()
    time.sleep(1)
    return resp.text
