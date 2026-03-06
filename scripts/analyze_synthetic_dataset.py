"""
Analyze the synthetic financial dataset for anomaly patterns and ML readiness.

Loads data/synthetic/payments_6500_rows.csv and performs:
  A. Basic dataset statistics
  B. Duplicate detection validation
  C. Vendor amount anomaly detection
  D. Keyword anomaly detection
  E. Department spending spike analysis

Prints a structured report and an ML-readiness verdict.

Usage:
    python scripts/analyze_synthetic_dataset.py
"""

import os
import sys

import numpy as np
import pandas as pd

DATASET_PATH = os.path.join("data", "synthetic", "payments_6500_rows.csv")
KEYWORD_PATTERNS = ["urgent", "manual", "adjustment"]
SPIKE_THRESHOLD = 2.0
VENDOR_ZSCORE_THRESHOLD = 3.0


def load_dataset() -> pd.DataFrame:
    if not os.path.exists(DATASET_PATH):
        print(f"ERROR: Dataset not found at {DATASET_PATH}")
        sys.exit(1)
    df = pd.read_csv(DATASET_PATH)
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    df["_parsed_date"] = pd.to_datetime(df["CHECK DATE"], errors="coerce")
    return df


# ── A. Basic Dataset Statistics ──────────────────────────────

def basic_statistics(df: pd.DataFrame) -> dict:
    date_col = df["_parsed_date"].dropna()
    amounts = df["AMOUNT"].dropna()

    top_vendors = (
        df["VENDOR NAME"]
        .value_counts()
        .head(10)
        .reset_index()
        .rename(columns={"index": "vendor", "count": "txn_count"})
    )

    dept_counts = (
        df["DEPARTMENT NAME"]
        .value_counts()
        .reset_index()
        .rename(columns={"index": "department", "count": "txn_count"})
    )

    return {
        "total_rows": len(df),
        "unique_vendors": df["VENDOR NAME"].nunique(),
        "unique_departments": df["DEPARTMENT NAME"].nunique(),
        "date_min": date_col.min(),
        "date_max": date_col.max(),
        "amount_min": amounts.min(),
        "amount_max": amounts.max(),
        "amount_mean": amounts.mean(),
        "amount_median": amounts.median(),
        "top_vendors": top_vendors,
        "dept_counts": dept_counts,
    }


# ── B. Duplicate Detection ──────────────────────────────────

def duplicate_analysis(df: pd.DataFrame) -> dict:
    dup_cols = ["VENDOR NAME", "AMOUNT", "CHECK DATE"]
    is_dup = df.duplicated(subset=dup_cols, keep=False)
    dup_rows = df[is_dup].copy()

    # Group duplicates to count pairs
    if len(dup_rows) > 0:
        groups = dup_rows.groupby(dup_cols).size().reset_index(name="group_size")
        num_groups = len(groups)
        example_groups = (
            dup_rows[["VOUCHER NUMBER", "VENDOR NAME", "AMOUNT", "CHECK DATE"]]
            .sort_values(dup_cols)
            .head(10)
        )
    else:
        num_groups = 0
        example_groups = pd.DataFrame()

    return {
        "total_duplicate_rows": int(is_dup.sum()),
        "duplicate_groups": num_groups,
        "example_groups": example_groups,
    }


# ── C. Vendor Amount Anomaly Detection ──────────────────────

def vendor_anomaly_analysis(df: pd.DataFrame) -> dict:
    vendor_stats = df.groupby("VENDOR NAME")["AMOUNT"].agg(["mean", "std", "count"])
    vendor_stats.columns = ["vendor_mean", "vendor_std", "txn_count"]
    # Only consider vendors with enough data points for meaningful std
    vendor_stats = vendor_stats[vendor_stats["txn_count"] >= 2].copy()
    vendor_stats["vendor_std"] = vendor_stats["vendor_std"].fillna(0)

    merged = df.merge(vendor_stats, left_on="VENDOR NAME", right_index=True, how="inner")
    merged["z_score"] = np.where(
        merged["vendor_std"] > 0,
        (merged["AMOUNT"] - merged["vendor_mean"]) / merged["vendor_std"],
        0.0,
    )
    anomalies = merged[merged["z_score"] > VENDOR_ZSCORE_THRESHOLD].copy()
    anomalies = anomalies.sort_values("z_score", ascending=False)

    top_anomalies = anomalies[
        ["VOUCHER NUMBER", "VENDOR NAME", "AMOUNT", "vendor_mean", "vendor_std", "z_score"]
    ].head(10)

    return {
        "total_anomalies": len(anomalies),
        "top_anomalies": top_anomalies,
    }


# ── D. Keyword Anomaly Detection ────────────────────────────

def keyword_analysis(df: pd.DataFrame) -> dict:
    pattern = "|".join(KEYWORD_PATTERNS)
    # Check VENDOR NAME (where urgency keywords are embedded in our synthetic data)
    mask = df["VENDOR NAME"].str.contains(pattern, case=False, na=False)
    keyword_rows = df[mask].copy()

    examples = keyword_rows[["VOUCHER NUMBER", "VENDOR NAME", "AMOUNT"]].head(10)

    return {
        "total_keyword_rows": len(keyword_rows),
        "examples": examples,
    }


# ── E. Department Spending Spike Analysis ────────────────────

def department_spike_analysis(df: pd.DataFrame) -> dict:
    work = df.dropna(subset=["_parsed_date", "DEPARTMENT NAME"]).copy()
    work["year_month"] = work["_parsed_date"].dt.to_period("M")

    monthly = (
        work.groupby(["DEPARTMENT NAME", "year_month"])["AMOUNT"]
        .sum()
        .reset_index()
        .rename(columns={"AMOUNT": "monthly_spend"})
    )

    dept_avg = (
        monthly.groupby("DEPARTMENT NAME")["monthly_spend"]
        .mean()
        .reset_index()
        .rename(columns={"monthly_spend": "dept_avg_spend"})
    )

    monthly = monthly.merge(dept_avg, on="DEPARTMENT NAME")
    monthly["spike_score"] = np.where(
        monthly["dept_avg_spend"] > 0,
        monthly["monthly_spend"] / monthly["dept_avg_spend"],
        0.0,
    )

    spikes = monthly[monthly["spike_score"] > SPIKE_THRESHOLD].copy()
    spikes = spikes.sort_values("spike_score", ascending=False)

    return {
        "total_spike_months": len(spikes),
        "spike_departments": spikes["DEPARTMENT NAME"].nunique(),
        "spikes": spikes[
            ["DEPARTMENT NAME", "year_month", "monthly_spend", "dept_avg_spend", "spike_score"]
        ].head(15),
    }


# ── Report Printer ───────────────────────────────────────────

def print_report(stats, dups, vendor_anom, keywords, spikes):
    w = 60

    print("\n" + "=" * w)
    print("  SYNTHETIC DATASET ANALYSIS REPORT")
    print("=" * w)

    # ── A ─────────────────────────────────────────────
    print("\n## DATASET OVERVIEW")
    print("-" * w)
    print(f"  Total rows:            {stats['total_rows']:,}")
    print(f"  Unique vendors:        {stats['unique_vendors']:,}")
    print(f"  Unique departments:    {stats['unique_departments']:,}")
    print(f"  Date range:            {stats['date_min'].date()} → {stats['date_max'].date()}")
    print(f"  Amount min:            ${stats['amount_min']:,.2f}")
    print(f"  Amount max:            ${stats['amount_max']:,.2f}")
    print(f"  Amount mean:           ${stats['amount_mean']:,.2f}")
    print(f"  Amount median:         ${stats['amount_median']:,.2f}")
    print(f"\n  Top 10 Vendors by Transaction Count:")
    for _, row in stats["top_vendors"].iterrows():
        print(f"    {row['VENDOR NAME']:<45} {row['txn_count']:>5}")
    print(f"\n  Transactions per Department:")
    for _, row in stats["dept_counts"].iterrows():
        print(f"    {row['DEPARTMENT NAME']:<50} {row['txn_count']:>5}")

    # ── B ─────────────────────────────────────────────
    print(f"\n## DUPLICATE ANALYSIS")
    print("-" * w)
    print(f"  Duplicate rows detected:   {dups['total_duplicate_rows']}")
    print(f"  Duplicate groups (pairs):  {dups['duplicate_groups']}")
    if len(dups["example_groups"]) > 0:
        print(f"\n  Example duplicate groups:")
        print(dups["example_groups"].to_string(index=False))

    # ── C ─────────────────────────────────────────────
    print(f"\n## VENDOR ANOMALY ANALYSIS")
    print("-" * w)
    print(f"  Extreme vendor anomalies (z > {VENDOR_ZSCORE_THRESHOLD}):  {vendor_anom['total_anomalies']}")
    if len(vendor_anom["top_anomalies"]) > 0:
        print(f"\n  Top 10 Largest Anomalies:")
        fmt = vendor_anom["top_anomalies"].copy()
        fmt["vendor_mean"] = fmt["vendor_mean"].map("${:,.2f}".format)
        fmt["AMOUNT"] = fmt["AMOUNT"].map("${:,.2f}".format)
        fmt["z_score"] = fmt["z_score"].map("{:.1f}".format)
        print(fmt.to_string(index=False))

    # ── D ─────────────────────────────────────────────
    print(f"\n## KEYWORD TRIGGER ANALYSIS")
    print("-" * w)
    print(f"  Keywords scanned:          {KEYWORD_PATTERNS}")
    print(f"  Keyword-triggered rows:    {keywords['total_keyword_rows']}")
    if len(keywords["examples"]) > 0:
        print(f"\n  Example rows:")
        print(keywords["examples"].to_string(index=False))

    # ── E ─────────────────────────────────────────────
    print(f"\n## DEPARTMENT SPIKE ANALYSIS")
    print("-" * w)
    print(f"  Spike threshold:           > {SPIKE_THRESHOLD}x department average")
    print(f"  Months with spikes:        {spikes['total_spike_months']}")
    print(f"  Departments with spikes:   {spikes['spike_departments']}")
    if len(spikes["spikes"]) > 0:
        print(f"\n  Top department spikes:")
        fmt = spikes["spikes"].copy()
        fmt["monthly_spend"] = fmt["monthly_spend"].map("${:,.2f}".format)
        fmt["dept_avg_spend"] = fmt["dept_avg_spend"].map("${:,.2f}".format)
        fmt["spike_score"] = fmt["spike_score"].map("{:.2f}x".format)
        print(fmt.to_string(index=False))

    # ── ML Readiness ─────────────────────────────────
    checks = {
        "Duplicates detected": dups["total_duplicate_rows"] > 0,
        "Vendor anomalies present": vendor_anom["total_anomalies"] > 0,
        "Keyword triggers present": keywords["total_keyword_rows"] > 0,
        "Department spikes present": spikes["total_spike_months"] > 0,
    }

    all_pass = all(checks.values())

    print(f"\n## ML READINESS VALIDATION")
    print("-" * w)
    for check, passed in checks.items():
        status = "PASS" if passed else "FAIL"
        print(f"  [{status}]  {check}")
    print()
    verdict = "YES" if all_pass else "NO"
    print(f"  Dataset ready for ML pipeline: {verdict}")
    print("=" * w)


# ── Main ─────────────────────────────────────────────────────

def main():
    print(f"Loading dataset from {DATASET_PATH} ...")
    df = load_dataset()
    print(f"  Loaded {len(df):,} rows.\n")

    stats = basic_statistics(df)
    dups = duplicate_analysis(df)
    vendor_anom = vendor_anomaly_analysis(df)
    keywords = keyword_analysis(df)
    spikes = department_spike_analysis(df)

    print_report(stats, dups, vendor_anom, keywords, spikes)


if __name__ == "__main__":
    main()
