from datetime import datetime
from typing import Optional

from pydantic import BaseModel


class LatestRunResponse(BaseModel):
    run_id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    transactions_scored: Optional[int] = None
    high_risk_count: Optional[int] = None
    duplicates_found: Optional[int] = None

    model_config = {"from_attributes": True}


class RunHistoryItem(BaseModel):
    run_id: str
    started_at: Optional[datetime] = None
    completed_at: Optional[datetime] = None
    transactions_scored: Optional[int] = None
    high_risk_count: Optional[int] = None

    model_config = {"from_attributes": True}


class HighRiskTransactionResponse(BaseModel):
    voucher_number: str
    leak_probability: Optional[float] = None
    risk_factors: Optional[list[str]] = None
    is_duplicate: Optional[bool] = None
    duplicate_of: Optional[str] = None

    model_config = {"from_attributes": True}


class VendorRiskResponse(BaseModel):
    vendor_name: str
    total_spend: Optional[float] = None
    transaction_count: Optional[int] = None
    anomaly_count: Optional[int] = None
    risk_score: Optional[float] = None

    model_config = {"from_attributes": True}


class DepartmentRiskResponse(BaseModel):
    department: str
    monthly_spend: Optional[float] = None
    spike_score: Optional[float] = None
    risk_category: Optional[str] = None

    model_config = {"from_attributes": True}