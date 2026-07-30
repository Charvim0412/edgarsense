import time
from src.retrieval.db import get_connection


def log_query(question, route, tickers, num_excerpts, usage, latency_ms):
    conn = get_connection()
    cur = conn.cursor()
    cur.execute(
        """
        INSERT INTO query_logs (question, route, tickers, num_excerpts, input_tokens, output_tokens, latency_ms)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        """,
        (
            question,
            route,
            ",".join(tickers),
            num_excerpts,
            usage["input_tokens"] if usage else None,
            usage["output_tokens"] if usage else None,
            latency_ms,
        ),
    )
    conn.commit()
    cur.close()
    conn.close()


def timed_run(question):
    from src.agents.graph import run_query
    start = time.time()
    result = run_query(question)
    latency_ms = int((time.time() - start) * 1000)
    log_query(question, result["route"], result["tickers"], len(result["excerpts"]), result["usage"], latency_ms)
    return result, latency_ms
