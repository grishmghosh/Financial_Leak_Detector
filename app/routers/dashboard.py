import logging

from fastapi import APIRouter, Depends, HTTPException

from app.core.role_guard import get_org_context
from app.db.connection import get_db
from app.schemas.dashboard import (
    DepartmentRiskResponse,
    HighRiskTransactionResponse,
    LatestRunResponse,
    RunHistoryItem,
    VendorRiskResponse,
)
from app.services.dashboard_service import (
    fetch_department_risk,
    fetch_high_risk_transactions,
    fetch_latest_run,
    fetch_run_history,
    fetch_vendor_risk,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/dashboard", tags=["dashboard"])


@router.get("/latest-run", response_model=LatestRunResponse)
async def get_latest_run_endpoint(
    user: dict = Depends(get_org_context),
    conn=Depends(get_db),
) -> LatestRunResponse:
    result = await fetch_latest_run(conn, org_id=user["org_id"])
    if result is None:
        raise HTTPException(status_code=404, detail="No completed analysis run found")
    return result


@router.get("/run-history", response_model=list[RunHistoryItem])
async def get_run_history_endpoint(
    limit: int = 50,
    offset: int = 0,
    user: dict = Depends(get_org_context),
    conn=Depends(get_db),
) -> list[RunHistoryItem]:
    return await fetch_run_history(conn, org_id=user["org_id"], limit=limit, offset=offset)


@router.get("/high-risk-transactions", response_model=list[HighRiskTransactionResponse])
async def get_high_risk_transactions_endpoint(
    limit: int = 100,
    offset: int = 0,
    user: dict = Depends(get_org_context),
    conn=Depends(get_db),
) -> list[HighRiskTransactionResponse]:
    return await fetch_high_risk_transactions(conn, org_id=user["org_id"], limit=limit, offset=offset)


@router.get("/vendor-risk", response_model=list[VendorRiskResponse])
async def get_vendor_risk_endpoint(
    user: dict = Depends(get_org_context),
    conn=Depends(get_db),
) -> list[VendorRiskResponse]:
    return await fetch_vendor_risk(conn, org_id=user["org_id"])


@router.get("/department-risk", response_model=list[DepartmentRiskResponse])
async def get_department_risk_endpoint(
    user: dict = Depends(get_org_context),
    conn=Depends(get_db),
) -> list[DepartmentRiskResponse]:
    return await fetch_department_risk(conn, org_id=user["org_id"])