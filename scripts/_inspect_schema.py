"""Quick DB schema inspection to inform the e2e execution script."""
import asyncio, os, sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from dotenv import load_dotenv
load_dotenv()

import asyncpg

async def connect_with_retry(url, retries=3, timeout=30):
    for attempt in range(retries):
        try:
            return await asyncpg.connect(url, timeout=timeout)
        except (TimeoutError, OSError, asyncio.CancelledError) as e:
            print(f"  Attempt {attempt+1}/{retries} failed: {e}")
            if attempt < retries - 1:
                await asyncio.sleep(2)
    raise RuntimeError("Could not connect after retries")

async def inspect():
    conn = await connect_with_retry(os.getenv("DATABASE_URL"))
    try:
        # 1. All tables
        tables = await conn.fetch(
            "SELECT table_name FROM information_schema.tables "
            "WHERE table_schema = 'public' ORDER BY table_name"
        )
        print("=== TABLES ===")
        for t in tables:
            print(f"  {t['table_name']}")

        # 2. Column types for key tables
        for tbl in ["transactions", "organizations", "user_organizations",
                     "analysis_runs", "analysis_results",
                     "vendor_risk_scores", "department_risk_scores"]:
            cols = await conn.fetch(
                "SELECT column_name, data_type, udt_name FROM information_schema.columns "
                "WHERE table_name = $1 ORDER BY ordinal_position", tbl
            )
            if cols:
                print(f"\n=== {tbl} ===")
                for c in cols:
                    print(f"  {c['column_name']:<25} {c['data_type']:<20} {c['udt_name']}")
            else:
                print(f"\n=== {tbl} === (NOT FOUND)")

        # 3. Sample organizations
        print("\n=== SAMPLE organizations ===")
        try:
            rows = await conn.fetch("SELECT * FROM organizations LIMIT 3")
            for r in rows:
                print(f"  {dict(r)}")
        except Exception as e:
            print(f"  Error: {e}")

        # 4. Sample user_organizations
        print("\n=== SAMPLE user_organizations ===")
        try:
            rows = await conn.fetch("SELECT * FROM user_organizations LIMIT 3")
            for r in rows:
                print(f"  {dict(r)}")
        except Exception as e:
            print(f"  Error: {e}")

        # 5. Existing transaction count
        count = await conn.fetchval("SELECT COUNT(*) FROM transactions")
        print(f"\n=== Transactions count: {count} ===")

        # 6. Existing analysis_runs
        try:
            runs = await conn.fetch("SELECT * FROM analysis_runs LIMIT 3")
            print(f"\n=== analysis_runs ({len(runs)} rows) ===")
            for r in runs:
                print(f"  {dict(r)}")
        except Exception as e:
            print(f"\n=== analysis_runs: {e} ===")

    finally:
        await conn.close()

asyncio.run(inspect())
