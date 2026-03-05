from fastapi import APIRouter, Depends, HTTPException

from app.db.connection import get_db
from app.schemas.transaction import TransactionCreate, TransactionResponse
from app.services.transaction_service import (
    create_transaction,
    get_high_risk_transactions,
    get_transaction_by_voucher,
    list_transactions,
    search_transactions,
)

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("/", response_model=TransactionResponse)
async def create_transaction_endpoint(
    transaction: TransactionCreate,
    conn=Depends(get_db),
) -> TransactionResponse:
    return await create_transaction(conn, transaction)


@router.get("/high-risk", response_model=list[TransactionResponse])
async def get_high_risk_transactions_endpoint(
    threshold: float = 0.6,
    conn=Depends(get_db),
) -> list[TransactionResponse]:
    return await get_high_risk_transactions(conn, threshold)


@router.get("/search", response_model=list[TransactionResponse])
async def search_transactions_endpoint(
    department: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_risk: float | None = None,
    conn=Depends(get_db),
) -> list[TransactionResponse]:
    return await search_transactions(
        conn,
        department=department,
        min_amount=min_amount,
        max_amount=max_amount,
        start_date=start_date,
        end_date=end_date,
        min_risk=min_risk,
    )


@router.get("/", response_model=list[TransactionResponse])
async def list_transactions_endpoint(
    conn=Depends(get_db),
) -> list[TransactionResponse]:
    return await list_transactions(conn)


@router.get("/{voucher_number}", response_model=TransactionResponse)
async def get_transaction_endpoint(
    voucher_number: str,
    conn=Depends(get_db),
) -> TransactionResponse:
    result = await get_transaction_by_voucher(conn, voucher_number)
    if result is None:
        raise HTTPException(status_code=404, detail="Transaction not found")
    return result