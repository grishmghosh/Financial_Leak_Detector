from fastapi import APIRouter, Depends

from app.db.connection import get_db
from app.schemas.transaction import TransactionCreate, TransactionResponse
from app.services.transaction_service import create_transaction

router = APIRouter(prefix="/transactions", tags=["transactions"])


@router.post("/", response_model=TransactionResponse)
async def create_transaction_endpoint(
    transaction: TransactionCreate,
    conn=Depends(get_db),
) -> TransactionResponse:
    return await create_transaction(conn, transaction)