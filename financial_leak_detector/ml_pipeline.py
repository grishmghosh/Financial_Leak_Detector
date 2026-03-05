"""
Standalone ML pipeline for batch analysis of cleaned payment data.

Uses the canonical feature engineering from app.ml to ensure consistency
between offline analysis and the FastAPI scoring service.
"""

import logging
import os

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest

from app.ml.feature_engineering import extract_transaction_features
from app.ml.model import LeakDetectionModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ── Data loading ────────────────────────────────────────────
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_PATH = os.path.normpath(
    os.path.join(PROJECT_ROOT, "data", "processed", "payments_cleaned.csv")
)

logger.info("Resolved dataset path: %s", DATA_PATH)

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Cleaned dataset not found at expected location.")

df = pd.read_csv(DATA_PATH)

# Normalize column names
df.columns = df.columns.str.lower().str.replace(" ", "_")
logger.info("Columns: %s", df.columns.tolist())

required_columns = [
    "voucher_number", "amount", "check_date",
    "department_name", "vendor_name", "year", "month",
]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

df["check_date"] = pd.to_datetime(df["check_date"])
logger.info("Dataset shape: %s", df.shape)

df["department_name"] = df["department_name"].fillna("unknown_department")

# ── Compute aggregates needed by the canonical feature pipeline ──
dept_stats = df.groupby("department_name")["amount"].agg(["mean", "std"]).fillna(0)

# Map department stats back to each row
df["_dept_avg"] = df["department_name"].map(dept_stats["mean"])
df["_dept_std"] = df["department_name"].map(dept_stats["std"]).fillna(0)

# ── Feature engineering (canonical pipeline) ─────────────────
model_cls = LeakDetectionModel()
feature_vectors = []

for _, row in df.iterrows():
    txn = type("Transaction", (), {
        "amount": row["amount"],
        "department": row["department_name"],
        "description": "",  # CSV does not contain description text
    })()

    features = extract_transaction_features(
        txn,
        recent_transaction_count=0,
        department_avg_amount=float(row["_dept_avg"]),
        department_std_amount=float(row["_dept_std"]),
    )
    feature_vectors.append(model_cls.build_feature_vector(features))

X = np.array(feature_vectors)
logger.info("Feature matrix shape: %s", X.shape)

# ── Isolation Forest ─────────────────────────────────────────
iso_model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42,
)
iso_model.fit(X)

df["anomaly_score"] = iso_model.decision_function(X)
df["anomaly_flag"] = iso_model.predict(X)
df["anomaly_flag"] = df["anomaly_flag"].map({-1: 1, 1: 0})

logger.info("Anomaly distribution:\n%s", df["anomaly_flag"].value_counts().to_string())

# ── Risk categorisation ──────────────────────────────────────
df["risk_rank"] = df["anomaly_score"].rank(method="first", ascending=True)
total_rows = len(df)
high_threshold = int(total_rows * 0.02)
medium_threshold = int(total_rows * 0.07)

conditions = [
    df["risk_rank"] <= high_threshold,
    (df["risk_rank"] > high_threshold) & (df["risk_rank"] <= medium_threshold),
]
df["risk_level"] = np.select(conditions, ["High", "Medium"], default="Low")
df.drop(columns=["risk_rank"], inplace=True)

logger.info("Risk level distribution:\n%s", df["risk_level"].value_counts().to_string())

# ── Vendor risk summary ──────────────────────────────────────
vendor_risk_summary = df.groupby("vendor_name").agg(
    total_spend=("amount", "sum"),
    total_transactions=("voucher_number", "count"),
    anomaly_count=("anomaly_flag", "sum"),
    high_risk_count=("risk_level", lambda x: (x == "High").sum()),
).reset_index()

vendor_risk_summary["vendor_risk_score"] = (
    vendor_risk_summary["anomaly_count"] / vendor_risk_summary["total_transactions"]
)
vendor_risk_summary = vendor_risk_summary.sort_values(
    by="vendor_risk_score", ascending=False,
).reset_index(drop=True)

logger.info("Vendor risk summary shape: %s", vendor_risk_summary.shape)

# ── Department monthly spend ─────────────────────────────────
department_monthly = df.groupby(
    ["department_name", "year", "month"]
).agg(monthly_spend=("amount", "sum")).reset_index()

department_monthly = department_monthly.sort_values(
    by=["department_name", "year", "month"],
).reset_index(drop=True)

dept_monthly_stats = department_monthly.groupby("department_name")["monthly_spend"].agg(
    mean_monthly_spend="mean",
    std_monthly_spend="std",
).reset_index()

department_monthly = department_monthly.merge(
    dept_monthly_stats, on="department_name", how="left",
)
department_monthly["std_monthly_spend"] = department_monthly["std_monthly_spend"].replace(0, 1)
department_monthly["spike_score"] = (
    (department_monthly["monthly_spend"] - department_monthly["mean_monthly_spend"])
    / department_monthly["std_monthly_spend"]
).fillna(0)

dept_conditions = [
    department_monthly["spike_score"] > 2,
    (department_monthly["spike_score"] > 1) & (department_monthly["spike_score"] <= 2),
]
department_monthly["department_risk"] = np.select(
    dept_conditions, ["High", "Medium"], default="Low",
)

department_summary = department_monthly[[
    "department_name", "year", "month",
    "monthly_spend", "spike_score", "department_risk",
]].sort_values(by="spike_score", ascending=False).reset_index(drop=True)

logger.info("Department summary shape: %s", department_summary.shape)

# ── Output ───────────────────────────────────────────────────
os.makedirs("outputs", exist_ok=True)
logger.info("Pipeline complete.")