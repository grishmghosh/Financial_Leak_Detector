from app.schemas.analytics import RiskSummaryResponse


async def get_risk_summary(conn) -> RiskSummaryResponse:
    total_transactions = await conn.fetchval(
        "SELECT COUNT(*) FROM transactions"
    )

    high_risk_transactions = await conn.fetchval(
        "SELECT COUNT(*) FROM transactions WHERE leak_probability >= 0.6"
    )

    average_risk_score = await conn.fetchval(
        "SELECT AVG(leak_probability) FROM transactions"
    )

    rows = await conn.fetch(
        """
        SELECT department, COUNT(*) AS count
        FROM transactions
        WHERE leak_probability >= 0.6
        GROUP BY department
        ORDER BY count DESC
        LIMIT 5
        """
    )
    top_risky_departments = [dict(row) for row in rows]

    return RiskSummaryResponse(
        total_transactions=total_transactions or 0,
        high_risk_transactions=high_risk_transactions or 0,
        average_risk_score=float(average_risk_score) if average_risk_score is not None else 0.0,
        top_risky_departments=top_risky_departments,
    )
