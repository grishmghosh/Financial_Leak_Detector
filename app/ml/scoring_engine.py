from app.ml.feature_engineering import extract_transaction_features


def score_transaction(
    transaction,
    recent_transaction_count: int = 0,
    department_avg_amount: float = 0.0,
    department_std_amount: float = 0.0,
) -> tuple[float, list[str]]:
    features = extract_transaction_features(
        transaction,
        recent_transaction_count,
        department_avg_amount,
        department_std_amount,
    )

    score = 0.05
    risk_factors = []

    if features["is_large_transaction"]:
        score += 0.25
        risk_factors.append("large_transaction")

    if features["is_procurement"]:
        score += 0.15
        risk_factors.append("procurement_department")

    if features["description_contains_urgent"]:
        score += 0.15
        risk_factors.append("keyword_urgent")

    if features["description_contains_manual"]:
        score += 0.15
        risk_factors.append("keyword_manual")

    if features["description_contains_adjustment"]:
        score += 0.1
        risk_factors.append("keyword_adjustment")

    if features["transaction_count_last_hour"] > 5:
        score += 0.2
        risk_factors.append("high_transaction_frequency")

    if abs(features["amount_zscore"]) > 3:
        score += 0.25
        risk_factors.append("department_spending_outlier")

    return (min(score, 0.95), risk_factors)
