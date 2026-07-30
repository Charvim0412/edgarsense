import anthropic
from src.config import ANTHROPIC_API_KEY

_client = anthropic.Anthropic(api_key=ANTHROPIC_API_KEY)

MODEL = "claude-sonnet-4-6"

SYSTEM_PROMPT = """You are a financial research assistant. Answer ONLY using the
provided filing excerpts. Every claim must cite its source like [1], [2] matching
the excerpt numbers given. If the excerpts don't contain enough information to
answer, say so explicitly instead of guessing. Do not use outside knowledge."""


def generate_answer(question, excerpts):
    numbered = "\n\n".join(
        f"[{i+1}] {e['ticker']} — Item {e['item_number']} ({e['heading']}):\n{e['text']}"
        for i, e in enumerate(excerpts)
    )
    prompt = f"Question: {question}\n\nExcerpts:\n{numbered}\n\nAnswer with citations:"

    resp = _client.messages.create(
        model=MODEL,
        max_tokens=800,
        system=SYSTEM_PROMPT,
        messages=[{"role": "user", "content": prompt}],
    )
    text = resp.content[0].text
    usage = {"input_tokens": resp.usage.input_tokens, "output_tokens": resp.usage.output_tokens}
    return text, usage
