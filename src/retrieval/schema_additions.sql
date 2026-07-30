CREATE TABLE IF NOT EXISTS query_logs (
    id SERIAL PRIMARY KEY,
    question TEXT NOT NULL,
    route TEXT,
    tickers TEXT,
    num_excerpts INT,
    input_tokens INT,
    output_tokens INT,
    latency_ms INT,
    created_at TIMESTAMP DEFAULT now()
);
