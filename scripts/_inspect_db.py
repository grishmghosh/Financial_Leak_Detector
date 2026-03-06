"""Quick DB schema inspector."""
import os, psycopg2
from dotenv import load_dotenv
load_dotenv()
conn = psycopg2.connect(os.getenv("DATABASE_URL"))
cur = conn.cursor()

cur.execute("SELECT column_name, data_type, is_nullable FROM information_schema.columns WHERE table_name = 'transactions' ORDER BY ordinal_position")
print("=== TRANSACTIONS TABLE ===")
for r in cur.fetchall():
    print(f"  {r[0]:<25} {r[1]:<25} nullable={r[2]}")

cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'organizations' ORDER BY ordinal_position")
print("\n=== ORGANIZATIONS TABLE ===")
for r in cur.fetchall():
    print(f"  {r[0]:<25} {r[1]}")

cur.execute("SELECT column_name, data_type FROM information_schema.columns WHERE table_name = 'user_organizations' ORDER BY ordinal_position")
print("\n=== USER_ORGANIZATIONS TABLE ===")
for r in cur.fetchall():
    print(f"  {r[0]:<25} {r[1]}")

cur.execute("SELECT id, name FROM organizations LIMIT 5")
print("\n=== EXISTING ORGS ===")
for r in cur.fetchall():
    print(f"  {r}")

cur.execute("SELECT user_id, org_id, role FROM user_organizations LIMIT 5")
print("\n=== EXISTING USER_ORGS ===")
for r in cur.fetchall():
    print(f"  {r}")

cur.execute("SELECT COUNT(*) FROM transactions")
print(f"\n=== TRANSACTION COUNT: {cur.fetchone()[0]} ===")

# Check analysis tables
for tbl in ["analysis_runs", "analysis_results", "vendor_risk_scores", "department_risk_scores"]:
    cur.execute(f"SELECT COUNT(*) FROM {tbl}")
    print(f"  {tbl}: {cur.fetchone()[0]} rows")

conn.close()
