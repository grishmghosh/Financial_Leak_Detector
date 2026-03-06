"""Quick targeted DB schema check using asyncpg."""
import asyncio, os, ssl
from dotenv import load_dotenv
load_dotenv()

async def main():
    import asyncpg
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    
    for tbl in ["transactions", "organizations", "user_organizations", "analysis_runs", "analysis_results", "vendor_risk_scores", "department_risk_scores"]:
        rows = await conn.fetch(f"SELECT column_name, data_type FROM information_schema.columns WHERE table_name = '{tbl}' ORDER BY ordinal_position")
        print(f"\n{tbl.upper()} COLUMNS:")
        for r in rows:
            print(f"  {r['column_name']}: {r['data_type']}")
    
    # Check existing data
    for tbl in ["organizations", "user_organizations"]:
        cnt = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl}")
        print(f"\n{tbl} count: {cnt}")
        if cnt > 0:
            rows = await conn.fetch(f"SELECT * FROM {tbl} LIMIT 3")
            for r in rows:
                print(f"  {dict(r)}")
    
    cnt = await conn.fetchval("SELECT COUNT(*) FROM transactions")
    print(f"\ntransactions count: {cnt}")
    
    for tbl in ["analysis_runs", "analysis_results", "vendor_risk_scores", "department_risk_scores"]:
        cnt = await conn.fetchval(f"SELECT COUNT(*) FROM {tbl}")
        print(f"  {tbl}: {cnt}")
    
    await conn.close()

asyncio.run(main())
