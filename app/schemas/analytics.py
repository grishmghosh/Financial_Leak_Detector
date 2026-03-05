from pydantic import BaseModel


class RiskSummaryResponse(BaseModel):
    total_transactions: int
    high_risk_transactions: int
    average_risk_score: float
    top_risky_departments: list[dict]
