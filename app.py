import gradio as gr
from src.agents.graph import run_query

EXAMPLES = [
    "What does AAPL say about supply chain risk?",
    "Compare AAPL and MSFT risk factors",
    "What are TSLA's main business segments?",
]


def ask(question):
    if not question.strip():
        return "Enter a question.", ""
    result = run_query(question)
    citations = "\n\n".join(
        f"[{i+1}] {e['ticker']} — Item {e['item_number']} ({e['heading']})\n{e['text'][:250]}..."
        for i, e in enumerate(result["excerpts"])
    )
    verified_label = "✅ verified" if result["verified"] else "⚠️ unverified — see notes"
    meta = (
        f"Route: {result['route']} | Tickers: {result['tickers']} | "
        f"Retries: {result['retry_count']} | {verified_label}"
    )
    if result.get("critic_notes"):
        meta += f"\nCritic notes: {result['critic_notes']}"
    return result["answer"], f"{meta}\n\n---\n\n{citations}"


with gr.Blocks(title="EdgarSense") as demo:
    gr.Markdown("# EdgarSense\nMulti-agent RAG over SEC filings, with a critic agent that verifies numeric claims.")
    q = gr.Textbox(label="Question")
    btn = gr.Button("Ask")
    answer_box = gr.Textbox(label="Answer", lines=6)
    citations_box = gr.Textbox(label="Retrieved excerpts + verification status", lines=14)
    gr.Examples(EXAMPLES, inputs=q)
    btn.click(ask, inputs=q, outputs=[answer_box, citations_box])

if __name__ == "__main__":
    demo.launch()
