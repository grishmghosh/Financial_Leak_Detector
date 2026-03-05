from app.ml.feature_engineering import extract_transaction_features


def score_transaction(
    transaction,
    recent_transaction_count: int = 0,
    department_avg_amount: float = 0.0,
    department_std_amount: float = 0.0,
) -> float:
    features = extract_transaction_features(
        transaction,
        recent_transaction_count,
        department_avg_amount,
        department_std_amount,
    )

    score = 0.05

    if features["is_large_transaction"]:
        score += 0.25

    if features["is_procurement"]:
        score += 0.15

    if features["description_contains_urgent"]:
        score += 0.15

    if features["description_contains_manual"]:
        score += 0.15

    if features["description_contains_adjustment"]:
        score += 0.1

    if features["transaction_count_last_hour"] > 5:
        score += 0.2

    if abs(features["amount_zscore"]) > 3:
        score += 0.25

    return min(score, 0.95)
