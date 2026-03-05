from fastapi import APIRouter, Depends

from app.db.connection import get_db
from app.schemas.analytics import RiskSummaryResponse
from app.services.analytics_service import get_risk_summary

router = APIRouter(prefix="/analytics", tags=["analytics"])


@router.get("/risk-summary", response_model=RiskSummaryResponse)
async def get_risk_summary_endpoint(
    conn=Depends(get_db),
) -> RiskSummaryResponse:
    return await get_risk_summary(conn)
