"""
End-to-end system execution for LeakWatch.

Steps:
  1. Clean synthetic dataset
  2. Ensure DB schema compatibility
  3. Load data into transactions table
  4. Train ML model
  5. Run ML analysis pipeline
  6. Validate analysis result tables
  7. Test dashboard service layer
  8. Print system validation report
"""

import asyncio
import json
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from uuid import UUID, uuid4

import numpy as np
import pandas as pd

# Ensure project root is on path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from dotenv import load_dotenv
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_execution")

# ── Paths ────────────────────────────────────────────────────
SYNTHETIC_PATH = Path("data/synthetic/payments_6500_rows.csv")
CLEANED_PATH = Path("data/processed/payments_6500_rows_clean.csv")

# ── Results collector ────────────────────────────────────────
REPORT: dict = {
    "cleaning": {},
    "ingestion": {},
    "ml_analysis": {},
    "result_tables": {},
    "dashboard_api": {},
}


# =====================================================================
#  STEP 1 — Clean the synthetic dataset
# =====================================================================

def step1_clean_dataset() -> pd.DataFrame:
    logger.info("=" * 60)
    logger.info("STEP 1 — Cleaning synthetic dataset")
    logger.info("=" * 60)

    df = pd.read_csv(SYNTHETIC_PATH)
    initial_count = len(df)
    logger.info("Loaded %d rows from %s", initial_count, SYNTHETIC_PATH)

    # Keep only columns needed by the ML pipeline
    columns_to_keep = [
        "VOUCHER NUMBER",
        "AMOUNT",
        "CHECK DATE",
        "DEPARTMENT NAME",
        "VENDOR NAME",
    ]
    df = df[columns_to_keep]

    # Convert CHECK DATE to datetime
    df["CHECK DATE"] = pd.to_datetime(df["CHECK DATE"], errors="coerce")

    # Convert AMOUNT to numeric
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")

    # Remove zero or negative payments
    df = df[df["AMOUNT"] > 0]

    # Remove rows with missing critical values
    df = df.dropna(subset=["AMOUNT", "CHECK DATE", "VENDOR NAME"])

    # Remove exact duplicate rows
    df = df.drop_duplicates()

    # Create year and month columns
    df["YEAR"] = df["CHECK DATE"].dt.year
    df["MONTH"] = df["CHECK DATE"].dt.month

    rows_removed = initial_count - len(df)
    final_count = len(df)

    # Save cleaned dataset
    CLEANED_PATH.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(CLEANED_PATH, index=False)

    logger.info("Cleaning complete:")
    logger.info("  Rows processed:   %d", initial_count)
    logger.info("  Rows removed:     %d", rows_removed)
    logger.info("  Final row count:  %d", final_count)
    logger.info("  Columns:          %s", list(df.columns))
    logger.info("  Date range:       %s → %s", df["CHECK DATE"].min().date(), df["CHECK DATE"].max().date())
    logger.info("  Voucher unique:   %s", df["VOUCHER NUMBER"].nunique() == final_count)
    logger.info("  Saved to:         %s", CLEANED_PATH)

    REPORT["cleaning"] = {
        "input_rows": initial_count,
        "rows_removed": rows_removed,
        "final_rows": final_count,
        "vouchers_unique": bool(df["VOUCHER NUMBER"].nunique() == final_count),
    }

    return df


# =====================================================================
#  STEP 2 — Load dataset into database
# =====================================================================

async def step2_load_into_database(df: pd.DataFrame):
    logger.info("=" * 60)
    logger.info("STEP 2 — Loading dataset into database")
    logger.info("=" * 60)

    import asyncpg

    DATABASE_URL = os.getenv("DATABASE_URL")
    conn = await asyncpg.connect(DATABASE_URL)

    try:
        # ── Ensure required columns exist ────────────────────
        # The ML pipeline queries reference 'department', 'description',
        # and 'leak_probability' columns. Add them if they don't exist.
        existing_cols = await conn.fetch(
            "SELECT column_name FROM information_schema.columns "
            "WHERE table_name = 'transactions'"
        )
        existing_col_names = {r["column_name"] for r in existing_cols}
        logger.info("Existing transaction columns: %s", existing_col_names)

        columns_to_add = {
            "department": "TEXT",
            "description": "TEXT",
            "leak_probability": "DOUBLE PRECISION",
        }
        for col_name, col_type in columns_to_add.items():
            if col_name not in existing_col_names:
                await conn.execute(f"ALTER TABLE transactions ADD COLUMN IF NOT EXISTS {col_name} {col_type}")
                logger.info("Added column: %s (%s)", col_name, col_type)

        # ── Resolve org_id ───────────────────────────────────
        # Use existing org or create one for this test execution
        org_row = await conn.fetchrow("SELECT org_id FROM user_organizations LIMIT 1")
        if org_row:
            org_id = str(org_row["org_id"])
            logger.info("Using existing org_id: %s", org_id)
        else:
            # Check organizations table structure
            org_cols = await conn.fetch(
                "SELECT column_name FROM information_schema.columns "
                "WHERE table_name = 'organizations'"
            )
            org_col_names = {r["column_name"] for r in org_cols}
            logger.info("Organizations columns: %s", org_col_names)

            org_id = str(uuid4())
            if "id" in org_col_names:
                await conn.execute(
                    "INSERT INTO organizations (id, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    org_id if "uuid" in str(org_col_names) else org_id,
                    "LeakWatch Test Org",
                )
            else:
                # Try with whatever PK the table has
                pk_col = await conn.fetchval(
                    "SELECT column_name FROM information_schema.columns "
                    "WHERE table_name = 'organizations' ORDER BY ordinal_position LIMIT 1"
                )
                await conn.execute(
                    f"INSERT INTO organizations ({pk_col}, name) VALUES ($1, $2) ON CONFLICT DO NOTHING",
                    org_id, "LeakWatch Test Org",
                )
            logger.info("Created org_id: %s", org_id)

        # ── Clear previous synthetic data for this org ───────
        existing_count = await conn.fetchval(
            "SELECT COUNT(*) FROM transactions WHERE org_id = $1", org_id
        )
        if existing_count > 0:
            await conn.execute("DELETE FROM transactions WHERE org_id = $1", org_id)
            logger.info("Cleared %d existing transactions for org %s", existing_count, org_id)

        # Also clear previous analysis data
        for tbl in ["department_risk_scores", "vendor_risk_scores", "analysis_results", "analysis_runs"]:
            await conn.execute(f"DELETE FROM {tbl} WHERE org_id = $1", org_id)

        # ── Batch insert ─────────────────────────────────────
        inserted = 0
        skipped = 0
        batch_size = 500

        rows_to_insert = []
        seen_vouchers = set()

        for _, row in df.iterrows():
            vn = str(row["VOUCHER NUMBER"])
            if vn in seen_vouchers:
                skipped += 1
                continue
            seen_vouchers.add(vn)

            vendor = str(row["VENDOR NAME"]) if pd.notna(row["VENDOR NAME"]) else None
            dept = str(row["DEPARTMENT NAME"]) if pd.notna(row["DEPARTMENT NAME"]) else None
            check_date = row["CHECK DATE"].date() if pd.notna(row["CHECK DATE"]) else None

            rows_to_insert.append((
                org_id,                          # org_id
                vn,                              # voucher_number
                float(row["AMOUNT"]),            # amount
                check_date,                      # check_date
                dept,                            # department_name
                vendor,                          # vendor_name
                int(row["YEAR"]),                # year
                int(row["MONTH"]),               # month
                dept,                            # department (ML pipeline column)
                vendor,                          # description (used by ML for keyword detection)
            ))

        # Insert in batches
        for i in range(0, len(rows_to_insert), batch_size):
            batch = rows_to_insert[i:i + batch_size]
            await conn.executemany(
                """
                INSERT INTO transactions
                    (org_id, voucher_number, amount, check_date, department_name,
                     vendor_name, year, month, department, description)
                VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
                ON CONFLICT (voucher_number) DO NOTHING
                """,
                batch,
            )
            inserted += len(batch)
            logger.info("Inserted batch %d/%d (%d rows)",
                        i // batch_size + 1,
                        (len(rows_to_insert) + batch_size - 1) // batch_size,
                        len(batch))

        final_count = await conn.fetchval(
            "SELECT COUNT(*) FROM transactions WHERE org_id = $1", org_id
        )

        logger.info("Data loading complete:")
        logger.info("  Rows inserted:     %d", inserted)
        logger.info("  Rows skipped:      %d", skipped)
        logger.info("  Final DB count:    %d", final_count)

        REPORT["ingestion"] = {
            "rows_inserted": inserted,
            "rows_skipped": skipped,
            "final_db_count": final_count,
            "org_id": org_id,
        }

    finally:
        await conn.close()

    return org_id


# =====================================================================
#  STEP 3 — Train ML Model (prerequisite)
# =====================================================================

async def step3_train_model(org_id: str):
    logger.info("=" * 60)
    logger.info("STEP 3 — Training ML model")
    logger.info("=" * 60)

    model_path = Path("models/isolation_forest.pkl")
    if model_path.exists():
        logger.info("ML model already exists at %s — skipping training", model_path)
        return

    from app.ml.train_model import train_model
    await train_model(UUID(org_id))
    logger.info("ML model trained and saved to %s", model_path)


# =====================================================================
#  STEPS 4-5 — Run ML Analysis Pipeline & Monitor
# =====================================================================

async def step4_run_analysis(org_id: str) -> dict:
    logger.info("=" * 60)
    logger.info("STEP 4 — Running ML analysis pipeline")
    logger.info("=" * 60)

    import asyncpg
    from app.db.connection import init_db, close_db, get_pool
    from app.services.ml_service import start_analysis, run_analysis_pipeline, get_run_status

    # Initialize the app's DB pool (needed by run_analysis_pipeline)
    await init_db()
    pool = get_pool()

    async with pool.acquire() as conn:
        # Create analysis run
        run = await start_analysis(conn, org_id=UUID(org_id))
        run_id = run["id"]
        logger.info("Analysis run created: run_id=%s", run_id)
        logger.info("Analysis start time: %s", run.get("started_at"))

    REPORT["ml_analysis"]["run_id"] = str(run_id)
    REPORT["ml_analysis"]["started_at"] = str(run.get("started_at"))

    # Execute pipeline (runs synchronously in this context)
    start_time = time.time()
    await run_analysis_pipeline(run_id, UUID(org_id))
    duration = time.time() - start_time

    # Check final status
    async with pool.acquire() as conn:
        status = await get_run_status(conn, run_id)

    logger.info("Analysis completed:")
    logger.info("  Status:                %s", status.get("status"))
    logger.info("  Transactions scored:   %s", status.get("transactions_scored"))
    logger.info("  High risk count:       %s", status.get("high_risk_count"))
    logger.info("  Duplicates found:      %s", status.get("duplicates_found"))
    logger.info("  Duration:              %.1f seconds", duration)

    REPORT["ml_analysis"].update({
        "status": status.get("status"),
        "transactions_scored": status.get("transactions_scored"),
        "high_risk_count": status.get("high_risk_count"),
        "duplicates_found": status.get("duplicates_found"),
        "completed_at": str(status.get("completed_at")),
        "duration_seconds": round(duration, 1),
    })

    await close_db()
    return status


# =====================================================================
#  STEP 5 — Validate Analysis Tables
# =====================================================================

async def step5_validate_tables(org_id: str):
    logger.info("=" * 60)
    logger.info("STEP 5 — Validating analysis tables")
    logger.info("=" * 60)

    import asyncpg
    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))

    try:
        tables = {
            "analysis_runs": await conn.fetchval(
                "SELECT COUNT(*) FROM analysis_runs WHERE org_id = $1", org_id),
            "analysis_results": await conn.fetchval(
                "SELECT COUNT(*) FROM analysis_results WHERE org_id = $1", org_id),
            "vendor_risk_scores": await conn.fetchval(
                "SELECT COUNT(*) FROM vendor_risk_scores WHERE org_id = $1", org_id),
            "department_risk_scores": await conn.fetchval(
                "SELECT COUNT(*) FROM department_risk_scores WHERE org_id = $1", org_id),
        }

        for tbl, count in tables.items():
            logger.info("  %s: %d rows", tbl, count)

        # Sample data from each table
        logger.info("\n  Sample analysis_results:")
        sample = await conn.fetch(
            "SELECT voucher_number, leak_probability, risk_category, is_duplicate "
            "FROM analysis_results WHERE org_id = $1 ORDER BY leak_probability DESC LIMIT 5",
            org_id,
        )
        for r in sample:
            logger.info("    %s prob=%.4f cat=%s dup=%s",
                        r["voucher_number"], r["leak_probability"],
                        r["risk_category"], r["is_duplicate"])

        logger.info("\n  Sample vendor_risk_scores:")
        sample = await conn.fetch(
            "SELECT vendor_name, risk_score, anomaly_count, transaction_count "
            "FROM vendor_risk_scores WHERE org_id = $1 ORDER BY risk_score DESC LIMIT 5",
            org_id,
        )
        for r in sample:
            logger.info("    %s score=%.4f anomalies=%d txns=%d",
                        r["vendor_name"], r["risk_score"],
                        r["anomaly_count"], r["transaction_count"])

        logger.info("\n  Sample department_risk_scores:")
        sample = await conn.fetch(
            "SELECT department, spike_score, risk_category "
            "FROM department_risk_scores WHERE org_id = $1 ORDER BY spike_score DESC LIMIT 5",
            org_id,
        )
        for r in sample:
            logger.info("    %s spike=%.4f cat=%s",
                        r["department"], r["spike_score"], r["risk_category"])

        REPORT["result_tables"] = tables

    finally:
        await conn.close()


# =====================================================================
#  STEP 6 — Test Dashboard Service Layer
# =====================================================================

async def step6_test_dashboard(org_id: str):
    logger.info("=" * 60)
    logger.info("STEP 6 — Testing dashboard service layer")
    logger.info("=" * 60)

    import asyncpg
    from app.services.dashboard_service import (
        fetch_latest_run,
        fetch_run_history,
        fetch_high_risk_transactions,
        fetch_vendor_risk,
        fetch_department_risk,
    )

    conn = await asyncpg.connect(os.getenv("DATABASE_URL"))
    uid = UUID(org_id)

    dashboard_results = {}

    try:
        # ── latest-run ───────────────────────────────────────
        logger.info("\n  GET /api/v1/dashboard/latest-run")
        result = await fetch_latest_run(conn, uid)
        if result:
            data = result.model_dump()
            logger.info("    run_id: %s", data.get("run_id"))
            logger.info("    transactions_scored: %s", data.get("transactions_scored"))
            logger.info("    high_risk_count: %s", data.get("high_risk_count"))
            logger.info("    duplicates_found: %s", data.get("duplicates_found"))
            dashboard_results["latest_run"] = {"status": "OK", "data": data}
        else:
            logger.warning("    No completed run found")
            dashboard_results["latest_run"] = {"status": "NO DATA"}

        # ── run-history ──────────────────────────────────────
        logger.info("\n  GET /api/v1/dashboard/run-history")
        result = await fetch_run_history(conn, uid, limit=10)
        logger.info("    Returned %d runs", len(result))
        for r in result[:3]:
            d = r.model_dump()
            logger.info("    run_id=%s scored=%s", d.get("run_id"), d.get("transactions_scored"))
        dashboard_results["run_history"] = {
            "status": "OK" if result else "NO DATA",
            "count": len(result),
        }

        # ── high-risk-transactions ───────────────────────────
        logger.info("\n  GET /api/v1/dashboard/high-risk-transactions")
        result = await fetch_high_risk_transactions(conn, uid, limit=100)
        logger.info("    Returned %d high-risk transactions", len(result))
        for r in result[:3]:
            d = r.model_dump()
            logger.info("    voucher=%s prob=%.4f dup=%s",
                        d.get("voucher_number"), d.get("leak_probability", 0), d.get("is_duplicate"))
        dashboard_results["high_risk_transactions"] = {
            "status": "OK" if result else "NO DATA",
            "count": len(result),
        }

        # ── vendor-risk ──────────────────────────────────────
        logger.info("\n  GET /api/v1/dashboard/vendor-risk")
        result = await fetch_vendor_risk(conn, uid)
        logger.info("    Returned %d vendor risk scores", len(result))
        for r in result[:3]:
            d = r.model_dump()
            logger.info("    vendor=%s score=%.4f anomalies=%s",
                        d.get("vendor_name"), d.get("risk_score", 0), d.get("anomaly_count"))
        dashboard_results["vendor_risk"] = {
            "status": "OK" if result else "NO DATA",
            "count": len(result),
        }

        # ── department-risk ──────────────────────────────────
        logger.info("\n  GET /api/v1/dashboard/department-risk")
        result = await fetch_department_risk(conn, uid)
        logger.info("    Returned %d department risk scores", len(result))
        for r in result[:3]:
            d = r.model_dump()
            logger.info("    dept=%s spike=%.4f cat=%s",
                        d.get("department"), d.get("spike_score", 0), d.get("risk_category"))
        dashboard_results["department_risk"] = {
            "status": "OK" if result else "NO DATA",
            "count": len(result),
        }

        REPORT["dashboard_api"] = dashboard_results

    finally:
        await conn.close()


# =====================================================================
#  STEP 7-8 — Print Final Report
# =====================================================================

def print_final_report():
    w = 65

    print("\n" + "=" * w)
    print("  LEAKWATCH — FULL SYSTEM VALIDATION REPORT")
    print("=" * w)

    # ── Data Ingestion ───────────────────────────────────────
    print("\n## DATA INGESTION")
    print("-" * w)
    c = REPORT["cleaning"]
    i = REPORT["ingestion"]
    print(f"  Input rows (synthetic):      {c.get('input_rows', 'N/A')}")
    print(f"  Rows after cleaning:         {c.get('final_rows', 'N/A')}")
    print(f"  Rows removed:                {c.get('rows_removed', 'N/A')}")
    print(f"  Vouchers unique:             {c.get('vouchers_unique', 'N/A')}")
    print(f"  Rows inserted to DB:         {i.get('rows_inserted', 'N/A')}")
    print(f"  Rows skipped (duplicates):   {i.get('rows_skipped', 'N/A')}")
    print(f"  Final DB transaction count:  {i.get('final_db_count', 'N/A')}")

    # ── ML Analysis ──────────────────────────────────────────
    print("\n## ML ANALYSIS")
    print("-" * w)
    m = REPORT["ml_analysis"]
    print(f"  Run ID:                      {m.get('run_id', 'N/A')}")
    print(f"  Status:                      {m.get('status', 'N/A')}")
    print(f"  Transactions scored:         {m.get('transactions_scored', 'N/A')}")
    print(f"  High risk count:             {m.get('high_risk_count', 'N/A')}")
    print(f"  Duplicates found:            {m.get('duplicates_found', 'N/A')}")
    print(f"  Duration:                    {m.get('duration_seconds', 'N/A')}s")

    # ── Result Tables ────────────────────────────────────────
    print("\n## RESULT TABLES")
    print("-" * w)
    t = REPORT["result_tables"]
    for tbl, count in t.items():
        print(f"  {tbl:<30} {count} rows")

    # ── Dashboard API ────────────────────────────────────────
    print("\n## DASHBOARD API VALIDATION")
    print("-" * w)
    d = REPORT["dashboard_api"]
    endpoints = {
        "latest_run": "GET /api/v1/dashboard/latest-run",
        "run_history": "GET /api/v1/dashboard/run-history",
        "high_risk_transactions": "GET /api/v1/dashboard/high-risk-transactions",
        "vendor_risk": "GET /api/v1/dashboard/vendor-risk",
        "department_risk": "GET /api/v1/dashboard/department-risk",
    }
    all_ok = True
    for key, label in endpoints.items():
        info = d.get(key, {})
        status = info.get("status", "UNKNOWN")
        count = info.get("count", "")
        count_str = f" ({count} rows)" if count != "" else ""
        symbol = "PASS" if status == "OK" else "FAIL"
        if status != "OK":
            all_ok = False
        print(f"  [{symbol}]  {label}{count_str}")

    # ── Final Verdict ────────────────────────────────────────
    print("\n" + "=" * w)

    ml_ok = REPORT["ml_analysis"].get("status") == "completed"
    tables_ok = all(v > 0 for v in REPORT["result_tables"].values())
    data_ok = REPORT["ingestion"].get("final_db_count", 0) > 0

    system_pass = ml_ok and tables_ok and all_ok and data_ok

    print(f"\n  SYSTEM VALIDATION: {'PASS' if system_pass else 'FAIL'}")
    if system_pass:
        print("\n  The LeakWatch system successfully completed its first")
        print("  full pipeline execution.")
    else:
        print("\n  One or more validation checks failed.")
        if not data_ok:
            print("  - Data ingestion issue")
        if not ml_ok:
            print("  - ML analysis did not complete")
        if not tables_ok:
            print("  - Missing data in result tables")
        if not all_ok:
            print("  - Dashboard API endpoint failure")

    print("\n" + "=" * w)


# =====================================================================
#  Main orchestrator
# =====================================================================

async def main():
    total_start = time.time()

    # Step 1 — Clean
    df = step1_clean_dataset()

    # Step 2 — Load into DB
    org_id = await step2_load_into_database(df)

    # Step 3 — Train model (if needed)
    await step3_train_model(org_id)

    # Step 4 — Run ML analysis
    await step4_run_analysis(org_id)

    # Step 5 — Validate tables
    await step5_validate_tables(org_id)

    # Step 6 — Test dashboard
    await step6_test_dashboard(org_id)

    total_duration = time.time() - total_start
    logger.info("Total execution time: %.1f seconds", total_duration)

    # Step 7-8 — Report
    print_final_report()


if __name__ == "__main__":
    asyncio.run(main())
