import re
from bs4 import BeautifulSoup

ITEM_HEADING_RE = re.compile(
    r"^\s*item\s+(\d{1,2}[a-c]?)\.?\s*[-—:]?\s*(.*)$",
    re.IGNORECASE,
)

COMMON_ITEM_TITLES = {
    "1": "Business",
    "1a": "Risk Factors",
    "1b": "Unresolved Staff Comments",
    "2": "Properties",
    "3": "Legal Proceedings",
    "7": "Management's Discussion and Analysis (MD&A)",
    "7a": "Quantitative and Qualitative Disclosures About Market Risk",
    "8": "Financial Statements and Supplementary Data",
}


def html_to_text_lines(html):
    soup = BeautifulSoup(html, "lxml")
    for tag in soup(["script", "style"]):
        tag.decompose()

    lines = []
    for el in soup.find_all(["p", "div", "td", "th", "li", "h1", "h2", "h3", "h4"]):
        text = el.get_text(separator=" ", strip=True)
        if not text:
            continue
        if lines and lines[-1] == text:
            continue
        lines.append(text)
    return lines


def find_section_boundaries(lines):
    boundaries = []
    for i, line in enumerate(lines):
        m = ITEM_HEADING_RE.match(line)
        if not m:
            continue
        item_number = m.group(1).lower()
        heading_text = m.group(2).strip() or COMMON_ITEM_TITLES.get(item_number, "")
        boundaries.append((i, item_number, heading_text))
    return boundaries


def split_into_sections(html):
    lines = html_to_text_lines(html)
    boundaries = find_section_boundaries(lines)

    if not boundaries:
        return [
            {
                "item_number": None,
                "heading": "UNPARSED — no Item headings matched",
                "text": " ".join(lines),
            }
        ]

    sections = []
    for idx, (line_i, item_number, heading) in enumerate(boundaries):
        start = line_i + 1
        end = boundaries[idx + 1][0] if idx + 1 < len(boundaries) else len(lines)
        body = " ".join(lines[start:end]).strip()

        if len(body) < 200:
            continue

        sections.append(
            {
                "item_number": item_number,
                "heading": heading or COMMON_ITEM_TITLES.get(item_number, ""),
                "text": body,
            }
        )
    return sections
