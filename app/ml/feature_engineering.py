import math


def extract_transaction_features(transaction) -> dict:
    amount = float(transaction.amount)
    description_lower = (transaction.description or "").lower()

    return {
        "amount": amount,
        "amount_log": math.log10(amount + 1),
        "is_large_transaction": amount > 100_000,
        "is_procurement": (transaction.department or "").lower() == "procurement",
        "description_contains_urgent": "urgent" in description_lower,
        "description_contains_manual": "manual" in description_lower,
        "description_contains_adjustment": "adjustment" in description_lower,
    }
