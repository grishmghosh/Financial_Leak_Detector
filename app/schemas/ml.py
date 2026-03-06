from datetime import date, datetime
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class AnalysisRunRequest(BaseModel):
    start_date: Optional[date] = None
    end_date: Optional[date] = None
    department: Optional[str] = None


class AnalysisRunResponse(BaseModel):
    run_id: UUID
    status: str


class RunStatusResponse(BaseModel):
    run_id: UUID
    status: str
    transactions_scored: Optional[int] = None
    high_risk_count: Optional[int] = None
    duplicates_found: Optional[int] = None
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None

    model_config = {"from_attributes": True}


class RetrainResponse(BaseModel):
    status: str
