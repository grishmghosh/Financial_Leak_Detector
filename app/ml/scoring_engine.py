from app.ml.feature_engineering import extract_transaction_features


def score_transaction(transaction) -> float:
    features = extract_transaction_features(transaction)

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

    return min(score, 0.95)
