import psycopg
from src.config import DATABASE_URL


def get_connection():
    return psycopg.connect(DATABASE_URL)
