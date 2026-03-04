def score_transaction(transaction) -> float:
    score = 0.05

    if transaction.amount > 100_000:
        score += 0.25

    if transaction.department and transaction.department.lower() == "procurement":
        score += 0.15

    if transaction.description:
        keywords = {"urgent", "manual", "adjustment"}
        description_lower = transaction.description.lower()
        if any(word in description_lower for word in keywords):
            score += 0.2

    return min(score, 0.95)
