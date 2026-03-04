from app.ml.scoring_engine import score_transaction
from app.schemas.transaction import TransactionCreate, TransactionResponse


async def create_transaction(conn, transaction: TransactionCreate) -> TransactionResponse:
    await conn.execute(
        """
        INSERT INTO transactions
        (voucher_number, amount, check_date, department, description)
        VALUES ($1, $2, $3, $4, $5)
        """,
        transaction.voucher_number,
        transaction.amount,
        transaction.check_date,
        transaction.department,
        transaction.description,
    )

    leak_probability = score_transaction(transaction)

    row = await conn.fetchrow(
        """
        UPDATE transactions
        SET leak_probability = $1
        WHERE voucher_number = $2
        RETURNING voucher_number, amount, check_date, department, description, leak_probability
        """,
        leak_probability,
        transaction.voucher_number,
    )
    return TransactionResponse(**dict(row))


async def list_transactions(conn) -> list[TransactionResponse]:
    rows = await conn.fetch(
        """
        SELECT voucher_number, amount, check_date, department, description, leak_probability
        FROM transactions
        ORDER BY check_date DESC
        """
    )
    return [TransactionResponse(**dict(row)) for row in rows]


async def get_transaction_by_voucher(conn, voucher_number: str) -> TransactionResponse | None:
    row = await conn.fetchrow(
        """
        SELECT voucher_number, amount, check_date, department, description, leak_probability
        FROM transactions
        WHERE voucher_number = $1
        """,
        voucher_number,
    )
    if row is None:
        return None
    return TransactionResponse(**dict(row))