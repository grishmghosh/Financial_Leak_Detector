"""
Generate a realistic synthetic financial dataset for testing the ML pipeline.

Reads data/raw/payments.csv, learns statistical distributions, and produces
5000–8000 transactions with injected anomaly patterns.

Output: data/synthetic/payments_large.csv

Usage:
    python scripts/generate_synthetic_payments.py
"""

import os
import random
import string
from collections import Counter

import numpy as np
import pandas as pd

# ── Configuration ────────────────────────────────────────────
SEED = 42
TARGET_ROWS = 6500  # within 5000–8000 range
ANOMALY_CFG = {
    "duplicate_payments": 120,      # exact duplicate transactions
    "vendor_anomalies": 80,         # unusually large for small-payment vendors
    "urgent_payments": 60,          # descriptions with urgency keywords
    "department_spikes": 100,       # concentrated high-spend bursts
    "new_vendor_risk": 50,          # single-appearance large-amount vendors
}
NORMAL_ROWS = TARGET_ROWS - sum(ANOMALY_CFG.values())

INPUT_PATH = os.path.join("data", "raw", "payments.csv")
OUTPUT_DIR = os.path.join("data", "synthetic")
OUTPUT_PATH = os.path.join(OUTPUT_DIR, "payments_6500_rows.csv")

COLUMNS = [
    "VOUCHER NUMBER", "AMOUNT", "CHECK DATE",
    "DEPARTMENT NAME", "CONTRACT NUMBER", "VENDOR NAME", "CASHED",
]

np.random.seed(SEED)
random.seed(SEED)


# ── Helpers ──────────────────────────────────────────────────

def load_source() -> pd.DataFrame:
    """Load and minimally prepare the source dataset."""
    df = pd.read_csv(INPUT_PATH)
    df["AMOUNT"] = pd.to_numeric(df["AMOUNT"], errors="coerce")
    df["_parsed_date"] = pd.to_datetime(df["CHECK DATE"], errors="coerce")
    df = df.dropna(subset=["AMOUNT", "_parsed_date", "VENDOR NAME"])
    df = df[df["AMOUNT"] > 0]
    return df


def learn_distributions(df: pd.DataFrame) -> dict:
    """Extract statistical distributions from the source data."""
    # --- vendor frequency (top 200 cover the bulk of transactions) ---
    vendor_counts = df["VENDOR NAME"].value_counts()
    top_vendors = vendor_counts.head(200)
    vendor_probs = (top_vendors / top_vendors.sum()).values
    vendor_names = top_vendors.index.tolist()

    # --- per-vendor amount stats ---
    vendor_amount_stats = {}
    for v in vendor_names:
        amounts = df.loc[df["VENDOR NAME"] == v, "AMOUNT"]
        vendor_amount_stats[v] = {
            "mean": float(amounts.mean()),
            "std": float(amounts.std()) if len(amounts) > 1 else float(amounts.mean()) * 0.1,
            "min": float(amounts.min()),
            "max": float(amounts.max()),
        }

    # --- department frequency ---
    dept_counts = df["DEPARTMENT NAME"].value_counts().dropna()
    dept_probs = (dept_counts / dept_counts.sum()).values
    dept_names = dept_counts.index.tolist()

    # --- amount distribution (log-normal fit on positive amounts) ---
    log_amounts = np.log(df["AMOUNT"].values)
    amount_mu = float(log_amounts.mean())
    amount_sigma = float(log_amounts.std())

    # --- date range ---
    valid_dates = df["_parsed_date"].dropna()
    date_min = valid_dates.min()
    date_max = valid_dates.max()

    # --- contract numbers ---
    contracts = df["CONTRACT NUMBER"].dropna().value_counts()
    top_contracts = contracts.head(100).index.tolist()

    # --- voucher prefixes ---
    prefixes = df["VOUCHER NUMBER"].dropna().str[:4].value_counts()
    top_prefixes = prefixes.head(6).index.tolist()

    return {
        "vendor_names": vendor_names,
        "vendor_probs": vendor_probs,
        "vendor_amount_stats": vendor_amount_stats,
        "dept_names": dept_names,
        "dept_probs": dept_probs,
        "amount_mu": amount_mu,
        "amount_sigma": amount_sigma,
        "date_min": date_min,
        "date_max": date_max,
        "top_contracts": top_contracts,
        "top_prefixes": top_prefixes,
    }


def random_voucher(prefix: str, used: set) -> str:
    """Generate a unique voucher number with a realistic prefix."""
    for _ in range(1000):
        suffix = "".join(random.choices(string.digits, k=9))
        vn = f"{prefix}{suffix}"
        if vn not in used:
            used.add(vn)
            return vn
    raise RuntimeError("Could not generate unique voucher number")


def random_date(date_min: pd.Timestamp, date_max: pd.Timestamp) -> str:
    """Return a random date formatted as MM/DD/YYYY."""
    delta = (date_max - date_min).days
    d = date_min + pd.Timedelta(days=random.randint(0, max(delta, 1)))
    return d.strftime("%m/%d/%Y")


def sample_amount_for_vendor(vendor: str, stats: dict) -> float:
    """Sample a realistic amount for a known vendor."""
    s = stats.get(vendor)
    if s is None:
        return round(np.random.lognormal(4.5, 1.5), 2)
    # Log-normal centered on vendor mean
    mu = np.log(max(s["mean"], 1.0))
    sigma = min(max(s["std"] / max(s["mean"], 1.0), 0.1), 1.5)
    amt = np.random.lognormal(mu, sigma)
    return round(max(amt, 1.0), 2)


# ── Normal transaction generator ────────────────────────────

def generate_normal(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Generate n normal (non-anomalous) transactions."""
    rows = []
    for _ in range(n):
        vendor = np.random.choice(dist["vendor_names"], p=dist["vendor_probs"])
        dept = np.random.choice(dist["dept_names"], p=dist["dept_probs"])
        amount = sample_amount_for_vendor(vendor, dist["vendor_amount_stats"])
        prefix = random.choice(dist["top_prefixes"])
        rows.append({
            "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
            "AMOUNT": amount,
            "CHECK DATE": random_date(dist["date_min"], dist["date_max"]),
            "DEPARTMENT NAME": dept,
            "CONTRACT NUMBER": random.choice(dist["top_contracts"]),
            "VENDOR NAME": vendor,
            "CASHED": random.choices([True, False], weights=[0.97, 0.03])[0],
        })
    return rows


# ── Anomaly injectors ───────────────────────────────────────

def inject_duplicate_payments(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Create exact-duplicate payment pairs (same vendor, amount, date)."""
    rows = []
    for _ in range(n // 2):
        vendor = np.random.choice(dist["vendor_names"], p=dist["vendor_probs"])
        dept = np.random.choice(dist["dept_names"], p=dist["dept_probs"])
        amount = sample_amount_for_vendor(vendor, dist["vendor_amount_stats"])
        check_date = random_date(dist["date_min"], dist["date_max"])
        contract = random.choice(dist["top_contracts"])
        cashed = random.choices([True, False], weights=[0.97, 0.03])[0]
        prefix = random.choice(dist["top_prefixes"])
        for _ in range(2):
            rows.append({
                "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
                "AMOUNT": amount,
                "CHECK DATE": check_date,
                "DEPARTMENT NAME": dept,
                "CONTRACT NUMBER": contract,
                "VENDOR NAME": vendor,
                "CASHED": cashed,
            })
    return rows


def inject_vendor_anomalies(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Large payments for vendors that normally receive small amounts."""
    # Pick vendors whose average amount is below the 25th percentile
    small_vendors = [
        v for v, s in dist["vendor_amount_stats"].items()
        if s["mean"] < 500
    ]
    if not small_vendors:
        small_vendors = dist["vendor_names"][:20]

    rows = []
    for _ in range(n):
        vendor = random.choice(small_vendors)
        normal_mean = dist["vendor_amount_stats"].get(vendor, {}).get("mean", 100)
        # 10x–50x the normal amount
        amount = round(normal_mean * random.uniform(10, 50), 2)
        dept = np.random.choice(dist["dept_names"], p=dist["dept_probs"])
        prefix = random.choice(dist["top_prefixes"])
        rows.append({
            "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
            "AMOUNT": amount,
            "CHECK DATE": random_date(dist["date_min"], dist["date_max"]),
            "DEPARTMENT NAME": dept,
            "CONTRACT NUMBER": random.choice(dist["top_contracts"]),
            "VENDOR NAME": vendor,
            "CASHED": True,
        })
    return rows


def inject_urgent_payments(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Transactions with suspicious urgency keywords in description-like fields."""
    urgent_phrases = [
        "URGENT PAYMENT - EXPEDITED PROCESSING",
        "MANUAL ADJUSTMENT - OVERRIDE APPROVED",
        "PRIORITY PAYMENT - IMMEDIATE RELEASE",
        "URGENT - SAME DAY WIRE TRANSFER",
        "MANUAL ADJUSTMENT - EMERGENCY DISBURSEMENT",
        "PRIORITY PAYMENT - EXECUTIVE APPROVAL",
        "URGENT REISSUE - REPLACEMENT CHECK",
        "MANUAL CORRECTION - AMOUNT ADJUSTMENT",
    ]
    rows = []
    for _ in range(n):
        vendor = np.random.choice(dist["vendor_names"], p=dist["vendor_probs"])
        amount = round(np.random.lognormal(8, 1.2), 2)  # tend toward larger amounts
        dept = np.random.choice(dist["dept_names"], p=dist["dept_probs"])
        prefix = random.choice(dist["top_prefixes"])
        # Embed description in VENDOR NAME field (matches how data flows to ML)
        vendor_with_desc = f"{vendor} - {random.choice(urgent_phrases)}"
        rows.append({
            "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
            "AMOUNT": amount,
            "CHECK DATE": random_date(dist["date_min"], dist["date_max"]),
            "DEPARTMENT NAME": dept,
            "CONTRACT NUMBER": random.choice(dist["top_contracts"]),
            "VENDOR NAME": vendor_with_desc,
            "CASHED": True,
        })
    return rows


def inject_department_spikes(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Concentrated high-spend bursts within a single department over a short window."""
    rows = []
    # Pick 5 departments, each gets n/5 transactions within a 3-day window
    spike_depts = random.sample(dist["dept_names"][:15], min(5, len(dist["dept_names"])))
    per_dept = n // len(spike_depts)

    for dept in spike_depts:
        # Random 3-day spike window
        delta = (dist["date_max"] - dist["date_min"]).days
        spike_start = dist["date_min"] + pd.Timedelta(days=random.randint(0, max(delta - 3, 1)))
        for _ in range(per_dept):
            vendor = np.random.choice(dist["vendor_names"], p=dist["vendor_probs"])
            # 3x–10x typical amounts
            amount = round(np.random.lognormal(9, 0.8), 2)
            day_offset = random.randint(0, 2)
            date = (spike_start + pd.Timedelta(days=day_offset)).strftime("%m/%d/%Y")
            prefix = random.choice(dist["top_prefixes"])
            rows.append({
                "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
                "AMOUNT": amount,
                "CHECK DATE": date,
                "DEPARTMENT NAME": dept,
                "CONTRACT NUMBER": random.choice(dist["top_contracts"]),
                "VENDOR NAME": vendor,
                "CASHED": True,
            })
    return rows


def inject_new_vendor_risk(dist: dict, n: int, used_vouchers: set) -> list[dict]:
    """Single-appearance vendors with suspiciously large amounts."""
    first_names = [
        "APEX", "TITAN", "NEXUS", "STELLAR", "PRIME", "VORTEX", "ZENITH",
        "OMEGA", "PINNACLE", "SUMMIT", "HORIZON", "CATALYST", "VERTEX",
        "QUANTUM", "PARAGON", "DYNAMO", "STRATOS", "CIPHER", "AEGIS", "FORGE",
    ]
    suffixes = [
        "CONSULTING LLC", "ENTERPRISES INC", "SOLUTIONS GROUP",
        "PARTNERS LP", "SERVICES CO", "GLOBAL LTD", "MANAGEMENT CORP",
        "HOLDINGS INC", "ADVISORS LLC", "TECHNOLOGIES INC",
    ]
    rows = []
    used_new_vendors = set()
    for _ in range(n):
        # Generate a unique vendor name that doesn't exist in the original data
        for _attempt in range(100):
            name = f"{random.choice(first_names)} {random.choice(suffixes)}"
            if name not in used_new_vendors and name not in dist["vendor_names"]:
                used_new_vendors.add(name)
                break

        amount = round(np.random.lognormal(10, 1.0), 2)  # large amounts (median ~22k)
        dept = np.random.choice(dist["dept_names"], p=dist["dept_probs"])
        prefix = random.choice(dist["top_prefixes"])
        rows.append({
            "VOUCHER NUMBER": random_voucher(prefix, used_vouchers),
            "AMOUNT": amount,
            "CHECK DATE": random_date(dist["date_min"], dist["date_max"]),
            "DEPARTMENT NAME": dept,
            "CONTRACT NUMBER": "DV",
            "VENDOR NAME": name,
            "CASHED": random.choices([True, False], weights=[0.90, 0.10])[0],
        })
    return rows


# ── Main ─────────────────────────────────────────────────────

def main():
    print("Loading source dataset...")
    df = load_source()
    print(f"  Source rows (after cleaning): {len(df):,}")

    print("Learning distributions...")
    dist = learn_distributions(df)
    print(f"  Top vendors: {len(dist['vendor_names'])}")
    print(f"  Departments: {len(dist['dept_names'])}")
    print(f"  Date range: {dist['date_min'].date()} → {dist['date_max'].date()}")

    used_vouchers: set[str] = set()
    all_rows: list[dict] = []
    anomaly_labels: list[str] = []

    # ── Normal transactions ──────────────────────────────────
    print(f"\nGenerating {NORMAL_ROWS} normal transactions...")
    normal = generate_normal(dist, NORMAL_ROWS, used_vouchers)
    all_rows.extend(normal)
    anomaly_labels.extend(["normal"] * len(normal))

    # ── Inject anomalies ─────────────────────────────────────
    injectors = [
        ("duplicate_payments", inject_duplicate_payments),
        ("vendor_anomalies", inject_vendor_anomalies),
        ("urgent_payments", inject_urgent_payments),
        ("department_spikes", inject_department_spikes),
        ("new_vendor_risk", inject_new_vendor_risk),
    ]

    for name, fn in injectors:
        count = ANOMALY_CFG[name]
        print(f"Injecting {count} {name.replace('_', ' ')}...")
        rows = fn(dist, count, used_vouchers)
        all_rows.extend(rows)
        anomaly_labels.extend([name] * len(rows))

    # ── Assemble DataFrame ───────────────────────────────────
    result = pd.DataFrame(all_rows, columns=COLUMNS)

    # Shuffle to mix anomalies throughout
    shuffle_idx = np.random.permutation(len(result))
    result = result.iloc[shuffle_idx].reset_index(drop=True)
    anomaly_labels = [anomaly_labels[i] for i in shuffle_idx]

    # ── Save ─────────────────────────────────────────────────
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    result.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows → {OUTPUT_PATH}")

    # ── Summary statistics ───────────────────────────────────
    label_counts = Counter(anomaly_labels)
    dup_pairs = ANOMALY_CFG["duplicate_payments"] // 2

    # Count actual duplicate rows (same vendor, amount, date)
    dup_cols = ["VENDOR NAME", "AMOUNT", "CHECK DATE"]
    actual_dups = result.duplicated(subset=dup_cols, keep=False).sum()

    print("\n" + "=" * 55)
    print("  SYNTHETIC DATASET SUMMARY")
    print("=" * 55)
    print(f"  Total rows:                {len(result):,}")
    print(f"  Unique voucher numbers:    {result['VOUCHER NUMBER'].nunique():,}")
    print(f"  Unique vendors:            {result['VENDOR NAME'].nunique():,}")
    print(f"  Unique departments:        {result['DEPARTMENT NAME'].nunique():,}")
    print(f"  Date range:                {result['CHECK DATE'].iloc[0]} → ...")
    print(f"  Amount range:              ${result['AMOUNT'].min():,.2f} – ${result['AMOUNT'].max():,.2f}")
    print(f"  Mean amount:               ${result['AMOUNT'].mean():,.2f}")
    print(f"  Median amount:             ${result['AMOUNT'].median():,.2f}")
    print()
    print("  ANOMALY DISTRIBUTION")
    print("  " + "-" * 40)
    print(f"  {'Category':<25} {'Count':>6}")
    print("  " + "-" * 40)
    for category in ["normal"] + [name for name, _ in injectors]:
        print(f"  {category:<25} {label_counts[category]:>6}")
    print("  " + "-" * 40)
    print(f"  {'TOTAL':<25} {sum(label_counts.values()):>6}")
    print()
    print(f"  Duplicate payment pairs:   {dup_pairs}")
    print(f"  Rows in dup groups:        {actual_dups}")
    print(f"  New one-time vendors:      {ANOMALY_CFG['new_vendor_risk']}")
    print("=" * 55)


if __name__ == "__main__":
    main()
