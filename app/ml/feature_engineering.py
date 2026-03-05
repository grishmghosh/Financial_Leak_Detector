import math


def extract_transaction_features(
    transaction,
    recent_transaction_count: int = 0,
    department_avg_amount: float = 0.0,
    department_std_amount: float = 0.0,
) -> dict:
    amount = float(transaction.amount)
    description_lower = (transaction.description or "").lower()

    if department_std_amount is None or department_std_amount == 0:
        amount_zscore = 0.0
    else:
        amount_zscore = (amount - department_avg_amount) / department_std_amount

    return {
        "amount": amount,
        "amount_log": math.log10(amount + 1),
        "is_large_transaction": amount > 100_000,
        "is_procurement": (transaction.department or "").lower() == "procurement",
        "description_contains_urgent": "urgent" in description_lower,
        "description_contains_manual": "manual" in description_lower,
        "description_contains_adjustment": "adjustment" in description_lower,
        "transaction_count_last_hour": recent_transaction_count,
        "department_avg_amount": department_avg_amount,
        "department_std_amount": department_std_amount,
        "amount_zscore": amount_zscore,
    }
