"""
Dashboard service layer — reads ML analysis results for the dashboard API.
"""

import logging
from uuid import UUID

from app.db.queries.dashboard_queries import (
    get_department_risk_scores,
    get_high_risk_transactions,
    get_latest_run,
    get_run_history,
    get_vendor_risk_scores,
)
from app.schemas.dashboard import (
    DepartmentRiskResponse,
    HighRiskTransactionResponse,
    LatestRunResponse,
    RunHistoryItem,
    VendorRiskResponse,
)

logger = logging.getLogger(__name__)


async def fetch_latest_run(conn, org_id: UUID) -> LatestRunResponse | None:
    logger.info("Fetching latest analysis run for org_id=%s", org_id)
    row = await get_latest_run(conn, org_id)
    if row is None:
        logger.info("No completed analysis run found for org_id=%s", org_id)
        return None
    return LatestRunResponse(**{k: str(v) if k == "run_id" else v for k, v in row.items()})


async def fetch_run_history(
    conn, org_id: UUID, *, limit: int = 50, offset: int = 0
) -> list[RunHistoryItem]:
    logger.info("Fetching run history for org_id=%s (limit=%d, offset=%d)", org_id, limit, offset)
    rows = await get_run_history(conn, org_id, limit=limit, offset=offset)
    logger.info("Returned %d run history records for org_id=%s", len(rows), org_id)
    return [RunHistoryItem(**{k: str(v) if k == "run_id" else v for k, v in r.items()}) for r in rows]


async def fetch_high_risk_transactions(
    conn, org_id: UUID, *, limit: int = 100, offset: int = 0
) -> list[HighRiskTransactionResponse]:
    logger.info("Fetching high-risk transactions for org_id=%s (limit=%d, offset=%d)", org_id, limit, offset)
    rows = await get_high_risk_transactions(conn, org_id, limit=limit, offset=offset)
    logger.info("Returned %d high-risk transactions for org_id=%s", len(rows), org_id)
    return [HighRiskTransactionResponse(**r) for r in rows]


async def fetch_vendor_risk(conn, org_id: UUID) -> list[VendorRiskResponse]:
    logger.info("Fetching vendor risk scores for org_id=%s", org_id)
    rows = await get_vendor_risk_scores(conn, org_id)
    logger.info("Returned %d vendor risk scores for org_id=%s", len(rows), org_id)
    return [VendorRiskResponse(**{k: float(v) if k == "total_spend" else v for k, v in r.items()}) for r in rows]


async def fetch_department_risk(conn, org_id: UUID) -> list[DepartmentRiskResponse]:
    logger.info("Fetching department risk scores for org_id=%s", org_id)
    rows = await get_department_risk_scores(conn, org_id)
    logger.info("Returned %d department risk scores for org_id=%s", len(rows), org_id)
    return [DepartmentRiskResponse(**{k: float(v) if k == "monthly_spend" else v for k, v in r.items()}) for r in rows]