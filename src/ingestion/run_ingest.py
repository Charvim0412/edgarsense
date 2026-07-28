import argparse
import json

from src.config import DATA_RAW, DATA_PROCESSED
from src.ingestion.edgar_client import resolve_cik, get_filing_list, fetch_filing_html
from src.ingestion.parser import split_into_sections


def ingest_one(ticker, form_type, limit):
    cik = resolve_cik(ticker)
    print(f"[{ticker}] resolved CIK: {cik}")

    filings = get_filing_list(cik, form_type=form_type, limit=limit)
    print(f"[{ticker}] found {len(filings)} filing(s) of type {form_type}")

    output_paths = []
    for filing in filings:
        accession = filing["accession_number"]
        print(f"[{ticker}] fetching {filing['form']} filed {filing['filing_date']} ({accession})")

        html = fetch_filing_html(cik, accession, filing["primary_document"])

        raw_path = DATA_RAW / f"{ticker}_{accession}.html"
        raw_path.write_text(html, encoding="utf-8")

        sections = split_into_sections(html)
        print(f"[{ticker}] parsed {len(sections)} section(s)")

        record = {
            "ticker": ticker,
            "cik": cik,
            "form": filing["form"],
            "filing_date": filing["filing_date"],
            "accession_number": accession,
            "sections": sections,
        }

        out_path = DATA_PROCESSED / f"{ticker}_{accession}.json"
        out_path.write_text(json.dumps(record, indent=2), encoding="utf-8")
        print(f"[{ticker}] saved -> {out_path}")
        output_paths.append(str(out_path))

    return output_paths


def main():
    parser = argparse.ArgumentParser(description="Ingest SEC filings for a ticker.")
    parser.add_argument("--ticker", required=True)
    parser.add_argument("--form-type", default="10-K")
    parser.add_argument("--limit", type=int, default=1)
    args = parser.parse_args()

    ingest_one(args.ticker, args.form_type, args.limit)


if __name__ == "__main__":
    main()
