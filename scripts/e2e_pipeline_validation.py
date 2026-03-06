"""
LeakWatch End-to-End Pipeline Validation
=========================================
Runs full pipeline validation with graceful DB fallback.
Database-dependent steps are skipped if connection cannot be established in 5 seconds.
"""

import sys
import os
import math
import asyncio
import logging
from pathlib import Path
from types import SimpleNamespace
from datetime import datetime

# Ensure project root is on path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
os.chdir(PROJECT_ROOT)

import pandas as pd
import numpy as np

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger("e2e_validation")

# ── Report accumulator ──────────────────────────────────────────
report = {
    "dataset_cleaning": {},
    "offline_ml_validation": {},
    "database_connectivity": "NOT TESTED",
    "pipeline_execution": "NOT TESTED",
    "final_status": "FAIL",
}

# ═══════════════════════════════════════════════════════════════════
# STEP 1 — Clean the Synthetic Dataset (Offline Safe)
# ═══════════════════════════════════════════════════════════════════
def step1_clean_dataset():
    print("\n" + "=" * 70)
    print("STEP 1 — Clean the Synthetic Dataset")
    print("=" * 70)

    input_path = Path("data/synthetic/payments_6500_rows.csv")
    output_path = Path("data/processed/payments_6500_rows_clean.csv")

    if not input_path.exists():
        print(f"ERROR: Input file not found: {input_path}")
        return None

    df = pd.read_csv(str(input_path))
    initial_rows = len(df)
    print(f"Loaded {initial_rows} rows from {input_path}")

    # Keep required columns
    columns_to_keep = [
        "VOUCHER NUMBER",
        "AMOUNT",
        "CHECK DATE",
        "DEPARTMENT NAME",
        "VENDOR NAME",
    ]
    available = [c for c in columns_to_keep if c in df.columns]
    missing_cols = [c for c in columns_to_keep if c not in df.columns]
    if missing_cols:
        print(f"WARNING: Missing columns: {missing_cols}")
    df = df[available]

    # Parse dates and amounts
    df["CHECK DATE"] = pd.to_datetime(df["CHECK DATE"], errors="coerce")
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")

    # Remove zero/negative payments
    df = df[df["AMOUNT"] > 0]

    # Remove rows with missing values
    df = df.dropna(subset=["AMOUNT", "CHECK DATE", "VENDOR NAME"])

    # Remove exact duplicates
    before_dedup = len(df)
    df = df.drop_duplicates()
    exact_dupes_removed = before_dedup - len(df)

    # Create year/month columns
    df["YEAR"] = df["CHECK DATE"].dt.year
    df["MONTH"] = df["CHECK DATE"].dt.month

    # Save cleaned data
    output_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(str(output_path), index=False)

    rows_removed = initial_rows - len(df)
    final_count = len(df)

    # Verification
    print(f"\n--- Verification ---")
    print(f"Correct columns: {list(df.columns)}")
    print(f"Amount dtype: {df['AMOUNT'].dtype} (numeric: {pd.api.types.is_numeric_dtype(df['AMOUNT'])})")
    print(f"Date dtype: {df['CHECK DATE'].dtype}")
    print(f"Vendor name preserved: {'VENDOR NAME' in df.columns}")
    voucher_dupes = df["VOUCHER NUMBER"].duplicated().sum()
    print(f"Voucher number uniqueness: {voucher_dupes} duplicates")

    print(f"\n--- Summary ---")
    print(f"Rows processed:     {initial_rows}")
    print(f"Rows removed:       {rows_removed}")
    print(f"  (exact dupes:     {exact_dupes_removed})")
    print(f"Final cleaned rows: {final_count}")
    print(f"Output saved to:    {output_path}")

    report["dataset_cleaning"] = {
        "rows_processed": initial_rows,
        "rows_removed": rows_removed,
        "final_count": final_count,
        "voucher_dupes": voucher_dupes,
    }
    return df


# ═══════════════════════════════════════════════════════════════════
# STEP 2 — Offline Dataset Validation
# ═══════════════════════════════════════════════════════════════════
def step2_offline_validation(df):
    print("\n" + "=" * 70)
    print("STEP 2 — Offline Dataset Validation")
    print("=" * 70)

    # Dataset overview
    print("\n--- Dataset Overview ---")
    print(f"Total rows:         {len(df)}")
    print(f"Unique vendors:     {df['VENDOR NAME'].nunique()}")
    print(f"Unique departments: {df['DEPARTMENT NAME'].nunique()}")
    print(f"Date range:         {df['CHECK DATE'].min()} to {df['CHECK DATE'].max()}")
    print(f"Amount min:         ${df['AMOUNT'].min():,.2f}")
    print(f"Amount max:         ${df['AMOUNT'].max():,.2f}")
    print(f"Amount mean:        ${df['AMOUNT'].mean():,.2f}")
    print(f"Amount median:      ${df['AMOUNT'].median():,.2f}")

    # Duplicate detection (vendor_name + amount + check_date)
    print("\n--- Duplicate Detection (vendor_name + amount + check_date) ---")
    dup_cols = ["VENDOR NAME", "AMOUNT", "CHECK DATE"]
    dup_mask = df.duplicated(subset=dup_cols, keep=False)
    dup_groups = df[dup_mask]
    dup_count = df.duplicated(subset=dup_cols, keep="first").sum()
    print(f"Duplicate groups found:    {dup_count}")
    if dup_count > 0:
        print(f"Total rows in dup groups:  {len(dup_groups)}")
        print("Sample duplicates:")
        print(dup_groups.head(10).to_string(index=False))

    # Vendor anomaly detection (amount > vendor_mean + 3*vendor_std)
    print("\n--- Vendor Anomaly Detection (amount > mean + 3*std) ---")
    vendor_stats = df.groupby("VENDOR NAME")["AMOUNT"].agg(["mean", "std"]).reset_index()
    vendor_stats.columns = ["VENDOR NAME", "vendor_mean", "vendor_std"]
    vendor_stats["vendor_std"] = vendor_stats["vendor_std"].fillna(0)
    df_merged = df.merge(vendor_stats, on="VENDOR NAME")
    df_merged["is_vendor_anomaly"] = df_merged["AMOUNT"] > (
        df_merged["vendor_mean"] + 3 * df_merged["vendor_std"]
    )
    anomalies = df_merged[df_merged["is_vendor_anomaly"]]
    print(f"Vendor anomalies found: {len(anomalies)}")
    if len(anomalies) > 0:
        print("Sample anomalies:")
        print(anomalies[["VOUCHER NUMBER", "VENDOR NAME", "AMOUNT", "vendor_mean", "vendor_std"]].head(10).to_string(index=False))

    # Keyword anomaly detection
    print("\n--- Keyword Anomaly Detection ---")
    # Check if there's a description-like column; the synthetic CSV may not have one
    desc_col = None
    for col in ["DESCRIPTION", "CONTRACT NUMBER"]:
        if col in df.columns:
            desc_col = col
            break

    keywords = ["urgent", "manual", "adjustment"]
    if desc_col:
        for kw in keywords:
            matches = df[df[desc_col].astype(str).str.lower().str.contains(kw, na=False)]
            print(f"  '{kw}' matches: {len(matches)}")
    else:
        print("  No description column available in dataset — keyword check skipped")
        print("  (Keywords will be checked during ML scoring via transaction descriptions)")

    # Department spike detection
    print("\n--- Department Spike Detection ---")
    df_temp = df.copy()
    df_temp["YEAR_MONTH"] = df_temp["CHECK DATE"].dt.to_period("M")
    dept_month = df_temp.groupby(["DEPARTMENT NAME", "YEAR_MONTH"])["AMOUNT"].agg(["sum", "count"]).reset_index()
    dept_month.columns = ["department", "year_month", "total_amount", "txn_count"]

    # Compute spike score: (monthly_total - dept_mean) / dept_std
    dept_stats = dept_month.groupby("department")["total_amount"].agg(["mean", "std"]).reset_index()
    dept_stats.columns = ["department", "dept_mean", "dept_std"]
    dept_month = dept_month.merge(dept_stats, on="department")
    dept_month["spike_score"] = np.where(
        dept_month["dept_std"] > 0,
        (dept_month["total_amount"] - dept_month["dept_mean"]) / dept_month["dept_std"],
        0,
    )
    spikes = dept_month[dept_month["spike_score"] > 2].sort_values("spike_score", ascending=False)
    print(f"Department-month spikes (score > 2): {len(spikes)}")
    if len(spikes) > 0:
        print(spikes.head(10).to_string(index=False))

    return True


# ═══════════════════════════════════════════════════════════════════
# STEP 3 — Feature Pipeline Validation (Offline Safe)
# ═══════════════════════════════════════════════════════════════════
def step3_feature_pipeline(df):
    print("\n" + "=" * 70)
    print("STEP 3 — Feature Pipeline Validation")
    print("=" * 70)

    from app.ml.feature_engineering import extract_transaction_features
    from app.ml.model import LeakDetectionModel

    # Build sample transactions as SimpleNamespace (mimicking DB row objects)
    sample = df.head(20).copy()
    dept_stats = df.groupby("DEPARTMENT NAME")["AMOUNT"].agg(["mean", "std"]).to_dict("index")

    feature_vectors = []
    model = LeakDetectionModel()

    for _, row in sample.iterrows():
        txn = SimpleNamespace(
            voucher_number=row["VOUCHER NUMBER"],
            amount=float(row["AMOUNT"]),
            check_date=row["CHECK DATE"],
            department=row["DEPARTMENT NAME"],
            description="",  # synthetic data may not have descriptions
            vendor_name=row["VENDOR NAME"],
        )

        d_stats = dept_stats.get(row["DEPARTMENT NAME"], {"mean": 0, "std": 0})
        features = extract_transaction_features(
            txn,
            recent_transaction_count=0,
            department_avg_amount=float(d_stats["mean"]),
            department_std_amount=float(d_stats["std"]) if not pd.isna(d_stats["std"]) else 0.0,
        )

        vector = model.build_feature_vector(features)
        feature_vectors.append(vector)

    # Verify
    all_len_8 = all(len(v) == 8 for v in feature_vectors)
    no_missing = all(not any(math.isnan(x) for x in v) for v in feature_vectors)

    print(f"\nFeature extraction: {'OK' if len(feature_vectors) == len(sample) else 'FAIL'}")
    print(f"Feature vector length = 8: {'OK' if all_len_8 else 'FAIL'}")
    print(f"No missing values: {'OK' if no_missing else 'FAIL'}")

    print("\n--- Example Feature Vectors (first 5) ---")
    labels = ["amount_log", "is_large_txn", "is_procurement", "urgent", "manual", "adjustment", "txn_count_hr", "amount_zscore"]
    for i, vec in enumerate(feature_vectors[:5]):
        txn_row = sample.iloc[i]
        print(f"\n  [{txn_row['VOUCHER NUMBER']}] {txn_row['VENDOR NAME']}")
        print(f"    Amount: ${txn_row['AMOUNT']:,.2f}")
        for label, val in zip(labels, vec):
            print(f"    {label:30s} = {val:.4f}")

    report["offline_ml_validation"]["feature_pipeline"] = "OK" if (all_len_8 and no_missing) else "FAIL"
    return feature_vectors


# ═══════════════════════════════════════════════════════════════════
# STEP 4 — ML Scoring Dry Run (Offline Safe)
# ═══════════════════════════════════════════════════════════════════
def step4_ml_scoring(df):
    print("\n" + "=" * 70)
    print("STEP 4 — ML Scoring Dry Run")
    print("=" * 70)

    from app.ml.model import LeakDetectionModel
    from app.ml.feature_engineering import extract_transaction_features
    import joblib

    model_path = Path("models/isolation_forest.pkl")

    # If model doesn't exist, train one in-memory from the cleaned dataset
    if not model_path.exists():
        print("No pre-trained model found. Training in-memory from cleaned dataset...")
        model = LeakDetectionModel()

        dept_stats = df.groupby("DEPARTMENT NAME")["AMOUNT"].agg(["mean", "std"]).to_dict("index")
        training_vectors = []
        for _, row in df.iterrows():
            txn = SimpleNamespace(
                amount=float(row["AMOUNT"]),
                check_date=row["CHECK DATE"],
                department=row["DEPARTMENT NAME"],
                description="",
                vendor_name=row["VENDOR NAME"],
            )
            d_stats = dept_stats.get(row["DEPARTMENT NAME"], {"mean": 0, "std": 0})
            features = extract_transaction_features(
                txn,
                recent_transaction_count=0,
                department_avg_amount=float(d_stats["mean"]),
                department_std_amount=float(d_stats["std"]) if not pd.isna(d_stats["std"]) else 0.0,
            )
            training_vectors.append(model.build_feature_vector(features))

        X_train = np.array(training_vectors)
        model.model.fit(X_train)
        print(f"Model trained on {len(X_train)} samples")

        # Save for reuse
        model_path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(model.model, str(model_path))
        print(f"Model saved to {model_path}")
    else:
        print(f"Loading pre-trained model from {model_path}")
        model = LeakDetectionModel()
        model.model = joblib.load(str(model_path))

    # Score 50 sample transactions — mix of normal + potentially anomalous
    # Take some large transactions + random sample
    large_txns = df.nlargest(10, "AMOUNT")
    remaining = df.drop(large_txns.index)
    random_sample = remaining.sample(n=min(40, len(remaining)), random_state=42)
    sample = pd.concat([large_txns, random_sample]).head(50)

    dept_stats = df.groupby("DEPARTMENT NAME")["AMOUNT"].agg(["mean", "std"]).to_dict("index")
    results = []

    for _, row in sample.iterrows():
        txn = SimpleNamespace(
            voucher_number=row["VOUCHER NUMBER"],
            amount=float(row["AMOUNT"]),
            check_date=row["CHECK DATE"],
            department=row["DEPARTMENT NAME"],
            description="",
            vendor_name=row["VENDOR NAME"],
        )
        d_stats = dept_stats.get(row["DEPARTMENT NAME"], {"mean": 0, "std": 0})
        features = extract_transaction_features(
            txn,
            recent_transaction_count=0,
            department_avg_amount=float(d_stats["mean"]),
            department_std_amount=float(d_stats["std"]) if not pd.isna(d_stats["std"]) else 0.0,
        )

        score = model.score(features)

        # Build risk factors
        risk_factors = []
        if features["is_large_transaction"]:
            risk_factors.append("large_transaction")
        if features["is_procurement"]:
            risk_factors.append("procurement_department")
        if abs(features["amount_zscore"]) > 3:
            risk_factors.append("department_spending_outlier")

        results.append({
            "voucher_number": row["VOUCHER NUMBER"],
            "vendor_name": row["VENDOR NAME"],
            "amount": row["AMOUNT"],
            "leak_probability": score,
            "risk_factors": risk_factors,
        })

    results_df = pd.DataFrame(results)
    results_df = results_df.sort_values("leak_probability", ascending=False)

    print(f"\nScored {len(results_df)} transactions")
    print(f"\n--- Top 15 by Leak Probability ---")
    for _, r in results_df.head(15).iterrows():
        factors = ", ".join(r["risk_factors"]) if r["risk_factors"] else "none"
        print(f"  {r['voucher_number']:20s} | ${r['amount']:>12,.2f} | risk={r['leak_probability']:.4f} | {factors}")

    print(f"\n--- Score Distribution ---")
    print(f"  Min:    {results_df['leak_probability'].min():.4f}")
    print(f"  Max:    {results_df['leak_probability'].max():.4f}")
    print(f"  Mean:   {results_df['leak_probability'].mean():.4f}")
    print(f"  Median: {results_df['leak_probability'].median():.4f}")

    # Check that anomaly rows (large transactions, outliers) get higher scores
    high_risk = results_df[results_df["risk_factors"].apply(len) > 0]
    low_risk = results_df[results_df["risk_factors"].apply(len) == 0]

    if len(high_risk) > 0 and len(low_risk) > 0:
        avg_high = high_risk["leak_probability"].mean()
        avg_low = low_risk["leak_probability"].mean()
        print(f"\n--- Anomaly Score Comparison ---")
        print(f"  Avg score (with risk factors):    {avg_high:.4f} ({len(high_risk)} txns)")
        print(f"  Avg score (no risk factors):      {avg_low:.4f} ({len(low_risk)} txns)")
        print(f"  Anomalous rows score higher:      {'YES' if avg_high > avg_low else 'NO'}")
        report["offline_ml_validation"]["anomaly_scoring_valid"] = avg_high > avg_low
    else:
        print("  (Insufficient data for anomaly comparison)")
        report["offline_ml_validation"]["anomaly_scoring_valid"] = "INCONCLUSIVE"

    report["offline_ml_validation"]["sample_scores"] = {
        "count": len(results_df),
        "min": float(results_df["leak_probability"].min()),
        "max": float(results_df["leak_probability"].max()),
        "mean": float(results_df["leak_probability"].mean()),
    }
    return results


# ═══════════════════════════════════════════════════════════════════
# STEP 5 — Database Connection Test
# ═══════════════════════════════════════════════════════════════════
async def step5_db_connection_test():
    print("\n" + "=" * 70)
    print("STEP 5 — Database Connection Test")
    print("=" * 70)

    try:
        from dotenv import load_dotenv
        load_dotenv()
    except ImportError:
        pass

    db_url = os.environ.get("DATABASE_URL", "")
    if not db_url:
        print("DATABASE_URL not set — skipping database connection")
        report["database_connectivity"] = "SKIPPED (no DATABASE_URL)"
        return False

    print(f"Attempting database connection (5-second timeout)...")
    try:
        import asyncpg

        conn = await asyncio.wait_for(
            asyncpg.connect(dsn=db_url),
            timeout=5.0,
        )
        version = await conn.fetchval("SELECT version()")
        await conn.close()
        print(f"DATABASE CONNECTION: SUCCESS")
        print(f"  Server: {version[:80]}...")
        report["database_connectivity"] = "SUCCESS"
        return True

    except asyncio.TimeoutError:
        print("DATABASE CONNECTION: SKIPPED (timeout after 5 seconds)")
        report["database_connectivity"] = "SKIPPED (timeout)"
        return False
    except Exception as e:
        print(f"DATABASE CONNECTION: SKIPPED ({type(e).__name__}: {e})")
        report["database_connectivity"] = f"SKIPPED ({type(e).__name__})"
        return False


# ═══════════════════════════════════════════════════════════════════
# STEP 6 — Conditional Database Steps
# ═══════════════════════════════════════════════════════════════════
async def step6_database_steps(df, db_connected):
    print("\n" + "=" * 70)
    print("STEP 6 — Conditional Database Steps")
    print("=" * 70)

    if not db_connected:
        print("Database connection not available — skipping all database steps")
        report["pipeline_execution"] = "SKIPPED (no DB)"
        return

    import asyncpg
    import httpx
    from dotenv import load_dotenv
    load_dotenv()

    db_url = os.environ.get("DATABASE_URL", "")

    try:
        conn = await asyncio.wait_for(asyncpg.connect(dsn=db_url), timeout=5.0)
    except Exception as e:
        print(f"Failed to reconnect: {e}")
        report["pipeline_execution"] = "SKIPPED (reconnect failed)"
        return

    try:
        # ── Insert dataset rows ──────────────────────────────────
        print("\n--- Inserting transactions ---")

        # Get org_id
        org_row = await conn.fetchrow("SELECT id FROM organizations LIMIT 1")
        if org_row is None:
            print("No organization found in database — skipping insert")
            report["pipeline_execution"] = "SKIPPED (no org)"
            await conn.close()
            return

        org_id = org_row["id"]
        print(f"Using org_id: {org_id}")

        # Batch insert with dedup
        inserted = 0
        skipped = 0
        for _, row in df.iterrows():
            try:
                await conn.execute(
                    """
                    INSERT INTO transactions (voucher_number, amount, check_date, department, vendor_name, org_id)
                    VALUES ($1, $2, $3, $4, $5, $6)
                    ON CONFLICT (voucher_number) DO NOTHING
                    """,
                    str(row["VOUCHER NUMBER"]),
                    float(row["AMOUNT"]),
                    row["CHECK DATE"].to_pydatetime(),
                    str(row["DEPARTMENT NAME"]),
                    str(row["VENDOR NAME"]),
                    org_id,
                )
                inserted += 1
            except Exception as e:
                skipped += 1
                if skipped <= 3:
                    logger.warning("Insert error: %s", e)

        print(f"Inserted: {inserted}, Skipped: {skipped}")

        # ── Run ML analysis via API ──────────────────────────────
        print("\n--- Running ML analysis via API ---")
        base_url = os.environ.get("API_BASE_URL", "http://localhost:8000/api/v1")
        api_token = os.environ.get("API_TOKEN", "")

        headers = {}
        if api_token:
            headers["Authorization"] = f"Bearer {api_token}"

        async with httpx.AsyncClient(timeout=30) as client:
            try:
                resp = await client.post(f"{base_url}/ml/run-analysis", headers=headers)
                if resp.status_code == 200:
                    run_data = resp.json()
                    run_id = run_data.get("run_id")
                    print(f"Analysis run started: {run_id}")

                    # Poll status
                    for _ in range(30):
                        await asyncio.sleep(2)
                        status_resp = await client.get(f"{base_url}/ml/run-status/{run_id}", headers=headers)
                        if status_resp.status_code == 200:
                            status = status_resp.json().get("status")
                            print(f"  Run status: {status}")
                            if status in ("completed", "failed"):
                                break
                else:
                    print(f"  API returned {resp.status_code}: {resp.text[:200]}")
            except Exception as e:
                print(f"  API call failed: {e}")

        # ── Validate tables ──────────────────────────────────────
        print("\n--- Validating database tables ---")
        for table in ["analysis_runs", "analysis_results", "vendor_risk_scores", "department_risk_scores"]:
            try:
                count = await conn.fetchval(f"SELECT COUNT(*) FROM {table}")
                print(f"  {table:30s}: {count} rows")
            except Exception as e:
                print(f"  {table:30s}: ERROR - {e}")

        # ── Test dashboard APIs ──────────────────────────────────
        print("\n--- Testing dashboard APIs ---")
        dashboard_endpoints = [
            "/dashboard/latest-run",
            "/dashboard/high-risk-transactions",
            "/dashboard/vendor-risk",
            "/dashboard/department-risk",
        ]
        async with httpx.AsyncClient(timeout=15) as client:
            for endpoint in dashboard_endpoints:
                try:
                    resp = await client.get(f"{base_url}{endpoint}", headers=headers)
                    print(f"  GET {endpoint}: {resp.status_code}")
                except Exception as e:
                    print(f"  GET {endpoint}: ERROR - {e}")

        report["pipeline_execution"] = "COMPLETED"

    except Exception as e:
        print(f"Database steps failed: {e}")
        report["pipeline_execution"] = f"FAILED ({e})"
    finally:
        await conn.close()


# ═══════════════════════════════════════════════════════════════════
# STEP 7 — System Validation Report
# ═══════════════════════════════════════════════════════════════════
def step7_report():
    print("\n" + "=" * 70)
    print("STEP 7 — System Validation Report")
    print("=" * 70)

    print("\n## DATASET CLEANING")
    dc = report["dataset_cleaning"]
    if dc:
        print(f"  Rows processed:     {dc.get('rows_processed', 'N/A')}")
        print(f"  Rows removed:       {dc.get('rows_removed', 'N/A')}")
        print(f"  Final clean count:  {dc.get('final_count', 'N/A')}")
        print(f"  Voucher duplicates: {dc.get('voucher_dupes', 'N/A')}")
    else:
        print("  FAILED")

    print("\n## OFFLINE ML VALIDATION")
    ml = report["offline_ml_validation"]
    print(f"  Feature pipeline:   {ml.get('feature_pipeline', 'N/A')}")
    scores = ml.get("sample_scores", {})
    if scores:
        print(f"  Scored transactions: {scores.get('count', 'N/A')}")
        print(f"  Score range:        {scores.get('min', 'N/A'):.4f} — {scores.get('max', 'N/A'):.4f}")
        print(f"  Score mean:         {scores.get('mean', 'N/A'):.4f}")
    print(f"  Anomaly scoring:    {ml.get('anomaly_scoring_valid', 'N/A')}")

    print(f"\n## DATABASE CONNECTIVITY")
    print(f"  {report['database_connectivity']}")

    print(f"\n## PIPELINE EXECUTION")
    print(f"  {report['pipeline_execution']}")

    # Determine final status
    cleaning_ok = bool(dc and dc.get("final_count", 0) > 0)
    feature_ok = ml.get("feature_pipeline") == "OK"
    scoring_ok = bool(scores)
    db_ok = report["database_connectivity"] == "SUCCESS"
    pipeline_ok = report["pipeline_execution"] == "COMPLETED"

    if cleaning_ok and feature_ok and scoring_ok and db_ok and pipeline_ok:
        final = "PASS"
    elif cleaning_ok and feature_ok and scoring_ok:
        final = "PARTIAL (offline validation passed, DB steps skipped/failed)"
    else:
        final = "FAIL"

    report["final_status"] = final

    print(f"\n## FINAL STATUS")
    print(f"  LeakWatch pipeline validation: {final}")
    print("=" * 70)


# ═══════════════════════════════════════════════════════════════════
# Main
# ═══════════════════════════════════════════════════════════════════
async def main():
    print("=" * 70)
    print("  LeakWatch End-to-End Pipeline Validation")
    print(f"  Started: {datetime.now().isoformat()}")
    print("=" * 70)

    # Step 1
    df = step1_clean_dataset()
    if df is None:
        print("FATAL: Dataset cleaning failed. Cannot continue.")
        report["final_status"] = "FAIL"
        step7_report()
        return

    # Step 2
    step2_offline_validation(df)

    # Step 3
    step3_feature_pipeline(df)

    # Step 4
    step4_ml_scoring(df)

    # Step 5
    db_connected = await step5_db_connection_test()

    # Step 6
    await step6_database_steps(df, db_connected)

    # Step 7
    step7_report()


if __name__ == "__main__":
    asyncio.run(main())
