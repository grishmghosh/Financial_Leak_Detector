import logging
from uuid import UUID

from fastapi import APIRouter, BackgroundTasks, Depends, HTTPException

from app.core.role_guard import require_admin
from app.db.connection import get_db
from app.schemas.ml import (
    AnalysisRunRequest,
    AnalysisRunResponse,
    RetrainResponse,
    RunStatusResponse,
)
from app.services.ml_service import (
    get_run_status,
    retrain_model,
    run_analysis_pipeline,
    start_analysis,
)

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/ml", tags=["ml"])


@router.post("/run-analysis", response_model=AnalysisRunResponse)
async def run_analysis_endpoint(
    body: AnalysisRunRequest | None = None,
    background_tasks: BackgroundTasks = BackgroundTasks(),
    user: dict = Depends(require_admin),
    conn=Depends(get_db),
) -> AnalysisRunResponse:
    filters = body or AnalysisRunRequest()
    org_id: UUID = user["org_id"]
    user_id: UUID = UUID(user["user_id"])

    run = await start_analysis(conn, org_id=org_id, user_id=user_id, filters=filters)
    run_id = run["id"]

    background_tasks.add_task(run_analysis_pipeline, run_id, org_id, filters)

    return AnalysisRunResponse(run_id=run_id, status="running")


@router.get("/run-status/{run_id}", response_model=RunStatusResponse)
async def run_status_endpoint(
    run_id: UUID,
    conn=Depends(get_db),
) -> RunStatusResponse:
    result = await get_run_status(conn, run_id)
    if result is None:
        raise HTTPException(status_code=404, detail="Analysis run not found")
    return RunStatusResponse(**result)


@router.post("/retrain", response_model=RetrainResponse)
async def retrain_endpoint(
    user: dict = Depends(require_admin),
    conn=Depends(get_db),
) -> RetrainResponse:
    logger.info("Retrain requested by user %s (org %s)", user["user_id"], user["org_id"])
    await retrain_model(conn, org_id=user["org_id"])
    return RetrainResponse(status="model_retrained")
