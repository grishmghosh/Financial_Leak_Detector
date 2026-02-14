import pandas as pd

# ── Load raw data ────────────────────────────────────────────────
df = pd.read_csv("data/raw/payments.csv")

# ── Keep only required columns ───────────────────────────────────
columns_to_keep = [
    "VOUCHER NUMBER",
    "AMOUNT",
    "CHECK DATE",
    "DEPARTMENT NAME",
    "VENDOR NAME",
]
df = df[columns_to_keep]

# ── Convert CHECK DATE to datetime ──────────────────────────────
df["CHECK DATE"] = pd.to_datetime(df["CHECK DATE"], errors="coerce")

# ── Convert AMOUNT to numeric ───────────────────────────────────
df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")

# ── Remove zero or negative payments ─────────────────────────────
df = df[df["AMOUNT"] > 0]

# ── Remove rows with missing or invalid values ──────────────────
df = df.dropna(subset=["AMOUNT", "CHECK DATE", "VENDOR NAME"])

# ── Remove duplicate rows ───────────────────────────────────────
df = df.drop_duplicates()

# ── Create year and month columns from CHECK DATE ───────────────
df["YEAR"] = df["CHECK DATE"].dt.year
df["MONTH"] = df["CHECK DATE"].dt.month

# ── Save cleaned data ───────────────────────────────────────────
df.to_csv("data/processed/payments_cleaned.csv", index=False)

# ── Print summary statistics ────────────────────────────────────
print(f"Total rows: {len(df)}")
print(f"Unique vendors: {df['VENDOR NAME'].nunique()}")
print(f"Unique departments: {df['DEPARTMENT NAME'].nunique()}")
print(f"Date range: {df['CHECK DATE'].min()} to {df['CHECK DATE'].max()}")
print(f"Duplicate voucher count: {df['VOUCHER NUMBER'].duplicated().sum()}")
