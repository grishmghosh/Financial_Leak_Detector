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

    recent_transaction_count = await conn.fetchval(
        """
        SELECT COUNT(*)
        FROM transactions
        WHERE department = $1
        AND check_date >= NOW() - INTERVAL '1 hour'
        """,
        transaction.department,
    )

    dept_stats = await conn.fetchrow(
        """
        SELECT
            AVG(amount) AS avg_amount,
            STDDEV(amount) AS std_amount
        FROM transactions
        WHERE department = $1
        """,
        transaction.department,
    )
    department_avg_amount = float(dept_stats["avg_amount"]) if dept_stats["avg_amount"] is not None else 0.0
    department_std_amount = float(dept_stats["std_amount"]) if dept_stats["std_amount"] is not None else 0.0

    leak_probability = score_transaction(
        transaction,
        recent_transaction_count,
        department_avg_amount,
        department_std_amount,
    )

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


async def get_high_risk_transactions(conn, threshold: float = 0.6) -> list[TransactionResponse]:
    rows = await conn.fetch(
        """
        SELECT voucher_number, amount, check_date, department, description, leak_probability
        FROM transactions
        WHERE leak_probability >= $1
        ORDER BY leak_probability DESC
        """,
        threshold,
    )
    return [TransactionResponse(**dict(row)) for row in rows]