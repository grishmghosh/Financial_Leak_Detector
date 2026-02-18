import pandas as pd
import numpy as np
import os
from sklearn.ensemble import IsolationForest

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.join(SCRIPT_DIR, "..")
DATA_PATH = os.path.join(PROJECT_ROOT, "data", "processed", "payments_cleaned.csv")
DATA_PATH = os.path.normpath(DATA_PATH)

# Print the full resolved dataset path
print("Resolved dataset path:", DATA_PATH)

# Check if file exists
if not os.path.exists(DATA_PATH):
    raise FileNotFoundError("Cleaned dataset not found at expected location.")

# Load dataset
df = pd.read_csv(DATA_PATH)

# Normalize column names: lowercase and replace spaces with underscores
df.columns = df.columns.str.lower().str.replace(" ", "_")
print("Normalized Columns:", df.columns.tolist())

# Validate required columns
required_columns = ["voucher_number", "amount", "check_date", "department_name", "vendor_name", "year", "month"]
missing = [col for col in required_columns if col not in df.columns]
if missing:
    raise ValueError(f"Missing required columns: {missing}")

# Parse check_date as datetime
df["check_date"] = pd.to_datetime(df["check_date"])

# Print dataset shape
print("Dataset Shape:", df.shape)

# Print column names
print("Column Names:", df.columns.tolist())

# Print null value summary
print("Null Value Summary:")
print(df.isnull().sum())

# Replace null values in department_name
df["department_name"] = df["department_name"].fillna("unknown_department")

# Print updated null summary after replacement
print("\nUpdated Null Value Summary:")
print(df.isnull().sum())

# Verify department_name has zero nulls after cleaning
assert df["department_name"].isnull().sum() == 0, "department_name still contains null values after cleaning!"
print("Null check passed: department_name has zero nulls.")

# -------------------------
# Feature Engineering
# -------------------------

# log_amount
df["log_amount"] = np.log1p(df["amount"])

# vendor_avg_amount
vendor_avg = df.groupby("vendor_name")["amount"].transform("mean")
df["vendor_avg_amount"] = vendor_avg

# vendor_std_dev
vendor_std = df.groupby("vendor_name")["amount"].transform("std")
df["vendor_std_dev"] = vendor_std.fillna(0)

# department_avg_amount
dept_avg = df.groupby("department_name")["amount"].transform("mean")
df["department_avg_amount"] = dept_avg

# monthly_vendor_total
monthly_totals = df.groupby(["vendor_name", "year", "month"])["amount"].transform("sum")
df["monthly_vendor_total"] = monthly_totals

# rolling_growth_rate
vendor_monthly = (
    df.groupby(["vendor_name", "year", "month"])["amount"]
    .sum()
    .reset_index()
    .sort_values(["vendor_name", "year", "month"])
)

vendor_monthly["rolling_growth_rate"] = (
    vendor_monthly.groupby("vendor_name")["amount"]
    .pct_change()
    .fillna(0)
)

df = df.merge(
    vendor_monthly[["vendor_name", "year", "month", "rolling_growth_rate"]],
    on=["vendor_name", "year", "month"],
    how="left"
)

# z_score_amount
df["z_score_amount"] = (
    (df["amount"] - df["vendor_avg_amount"]) /
    df["vendor_std_dev"].replace(0, 1)
)

# Print first 5 rows of selected features
print(df[[
    "voucher_number", "amount", "log_amount",
    "vendor_avg_amount", "vendor_std_dev",
    "department_avg_amount", "monthly_vendor_total",
    "rolling_growth_rate", "z_score_amount"
]].head())

# -------------------------
# Isolation Forest Model
# -------------------------

# Define feature matrix
feature_columns = [
    "log_amount",
    "vendor_avg_amount",
    "vendor_std_dev",
    "department_avg_amount",
    "monthly_vendor_total",
    "rolling_growth_rate",
    "z_score_amount"
]

X = df[feature_columns]

# Initialize model
model = IsolationForest(
    n_estimators=100,
    contamination=0.02,
    random_state=42
)

# Fit model
model.fit(X)

# Generate anomaly score
df.loc[:, "anomaly_score"] = model.decision_function(X)

# Generate anomaly flag
df.loc[:, "anomaly_flag"] = model.predict(X)

# Convert anomaly_flag: -1 -> 1 (anomaly), 1 -> 0 (normal)
df.loc[:, "anomaly_flag"] = df["anomaly_flag"].map({-1: 1, 1: 0})

# Print anomaly distribution
print("Anomaly Distribution:")
print(df["anomaly_flag"].value_counts())

# -------------------------
# Business Risk Categorization
# -------------------------

# Rank anomaly_score in ascending order (lower score = more anomalous)
df["risk_rank"] = df["anomaly_score"].rank(method="first", ascending=True)

# Compute total number of rows
total_rows = len(df)

# Compute thresholds
high_threshold = int(total_rows * 0.02)
medium_threshold = int(total_rows * 0.07)

# High = top 2%, Medium = next 5% (2% to 7%)
conditions = [
    df["risk_rank"] <= high_threshold,
    (df["risk_rank"] > high_threshold) & (df["risk_rank"] <= medium_threshold)
]

choices = ["High", "Medium"]

df["risk_level"] = np.select(conditions, choices, default="Low")

# Drop the risk_rank column
df.drop(columns=["risk_rank"], inplace=True)

# Print risk distribution
print("Risk Level Distribution:")
print(df["risk_level"].value_counts())

# Create anomaly_results dataframe
anomaly_results = df[[
    "voucher_number",
    "anomaly_score",
    "anomaly_flag",
    "risk_level"
]]

# Print first 5 rows of anomaly_results
print("Anomaly Results Preview:")
print(anomaly_results.head())

# -------------------------
# Aggregation Intelligence
# -------------------------

# Create Vendor Risk Summary
vendor_risk_summary = df.groupby("vendor_name").agg(
    total_spend=("amount", "sum"),
    total_transactions=("voucher_number", "count"),
    anomaly_count=("anomaly_flag", "sum"),
    high_risk_count=("risk_level", lambda x: (x == "High").sum())
).reset_index()

vendor_risk_summary["vendor_risk_score"] = (
    vendor_risk_summary["anomaly_count"] / vendor_risk_summary["total_transactions"]
)

vendor_risk_summary = vendor_risk_summary.sort_values(
    by="vendor_risk_score", ascending=False
).reset_index(drop=True)

print("Vendor Risk Summary Preview:")
print(vendor_risk_summary.head())

# Create Department Monthly Spend Table
department_monthly = df.groupby(
    ["department_name", "year", "month"]
).agg(
    monthly_spend=("amount", "sum")
).reset_index()

department_monthly = department_monthly.sort_values(
    by=["department_name", "year", "month"]
).reset_index(drop=True)

# Compute Department Spike Score
dept_stats = department_monthly.groupby("department_name")["monthly_spend"].agg(
    mean_monthly_spend="mean",
    std_monthly_spend="std"
).reset_index()

department_monthly = department_monthly.merge(dept_stats, on="department_name", how="left")

department_monthly["std_monthly_spend"] = department_monthly["std_monthly_spend"].replace(0, 1)

department_monthly["spike_score"] = (
    (department_monthly["monthly_spend"] - department_monthly["mean_monthly_spend"])
    / department_monthly["std_monthly_spend"]
)

department_monthly["spike_score"] = department_monthly["spike_score"].fillna(0)

# Assign Department Risk Level
dept_conditions = [
    department_monthly["spike_score"] > 2,
    (department_monthly["spike_score"] > 1) & (department_monthly["spike_score"] <= 2)
]

dept_choices = ["High", "Medium"]

department_monthly["department_risk"] = np.select(dept_conditions, dept_choices, default="Low")

# Create Final Department Summary
department_summary = department_monthly[[
    "department_name",
    "year",
    "month",
    "monthly_spend",
    "spike_score",
    "department_risk"
]].sort_values(by="spike_score", ascending=False).reset_index(drop=True)

print("Department Summary Preview:")
print(department_summary.head())

# Validation Checks
print("Vendor Risk Summary Shape:", vendor_risk_summary.shape)
print("Department Summary Shape:", department_summary.shape)

print("Total anomalies in df:", df["anomaly_flag"].sum())
print("Total anomalies in vendor summary:", vendor_risk_summary["anomaly_count"].sum())

print("Vendor Risk Score Min:", vendor_risk_summary["vendor_risk_score"].min())
print("Vendor Risk Score Max:", vendor_risk_summary["vendor_risk_score"].max())

print("Spike Score Stats:")
print(department_summary["spike_score"].describe())

print("Department Risk Distribution:")
print(department_summary["department_risk"].value_counts())

# Create outputs folder if it does not exist
os.makedirs("outputs", exist_ok=True)