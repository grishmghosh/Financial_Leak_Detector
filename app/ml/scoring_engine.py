from app.ml.feature_engineering import extract_transaction_features
from app.ml.model_loader import score_transaction as ml_score_transaction


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

    risk_factors = []

    if features["is_large_transaction"]:
        risk_factors.append("large_transaction")

    if features["is_procurement"]:
        risk_factors.append("procurement_department")

    if features["description_contains_urgent"]:
        risk_factors.append("keyword_urgent")

    if features["description_contains_manual"]:
        risk_factors.append("keyword_manual")

    if features["description_contains_adjustment"]:
        risk_factors.append("keyword_adjustment")

    if features["transaction_count_last_hour"] > 5:
        risk_factors.append("high_transaction_frequency")

    if abs(features["amount_zscore"]) > 3:
        risk_factors.append("department_spending_outlier")

    ml_score = ml_score_transaction(features)
    score = ml_score

    return score, risk_factors
