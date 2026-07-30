import os
from pathlib import Path
from dotenv import load_dotenv

load_dotenv()

SEC_USER_AGENT = os.getenv("SEC_USER_AGENT", "")
if not SEC_USER_AGENT:
    raise RuntimeError(
        "SEC_USER_AGENT is not set. Copy .env.example to .env and fill it in."
    )

DATABASE_URL = os.getenv("DATABASE_URL", "")

PROJECT_ROOT = Path(__file__).resolve().parent.parent
DATA_RAW = PROJECT_ROOT / "data" / "raw"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
DATA_RAW.mkdir(parents=True, exist_ok=True)
DATA_PROCESSED.mkdir(parents=True, exist_ok=True)
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
