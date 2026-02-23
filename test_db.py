import os
import psycopg2
from dotenv import load_dotenv

load_dotenv()

db_url = os.getenv("DATABASE_URL")

if not db_url:
    print("DATABASE_URL not found in .env")
    exit()

try:
    conn = psycopg2.connect(db_url)
    cur = conn.cursor()
    cur.execute("SELECT 1;")
    result = cur.fetchone()
    print("Database connected successfully:", result)
    conn.close()
except Exception as e:
    print("Database connection failed:")
    print(e)
