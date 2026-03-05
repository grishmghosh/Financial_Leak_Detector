from decimal import Decimal
from datetime import date
from typing import Optional
from uuid import UUID

from pydantic import BaseModel


class TransactionBase(BaseModel):
    voucher_number: str
    amount: Decimal
    check_date: date
    department: str
    description: Optional[str] = None


class TransactionCreate(TransactionBase):
    org_id: UUID


class TransactionResponse(TransactionBase):
    org_id: UUID
    leak_probability: Optional[float] = None
    risk_factors: list[str] | None = None

    model_config = {"from_attributes": True}