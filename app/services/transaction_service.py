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

    leak_probability, risk_factors = score_transaction(
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
    return TransactionResponse(**dict(row), risk_factors=risk_factors)


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


async def search_transactions(
    conn,
    department: str | None = None,
    min_amount: float | None = None,
    max_amount: float | None = None,
    start_date: str | None = None,
    end_date: str | None = None,
    min_risk: float | None = None,
) -> list[TransactionResponse]:
    query = """
        SELECT voucher_number, amount, check_date, department, description, leak_probability
        FROM transactions
        WHERE 1=1
    """
    params = []
    idx = 1

    if department is not None:
        query += f" AND department = ${idx}"
        params.append(department)
        idx += 1

    if min_amount is not None:
        query += f" AND amount >= ${idx}"
        params.append(min_amount)
        idx += 1

    if max_amount is not None:
        query += f" AND amount <= ${idx}"
        params.append(max_amount)
        idx += 1

    if start_date is not None:
        query += f" AND check_date >= ${idx}"
        params.append(start_date)
        idx += 1

    if end_date is not None:
        query += f" AND check_date <= ${idx}"
        params.append(end_date)
        idx += 1

    if min_risk is not None:
        query += f" AND leak_probability >= ${idx}"
        params.append(min_risk)
        idx += 1

    query += " ORDER BY check_date DESC"

    rows = await conn.fetch(query, *params)
    return [TransactionResponse(**dict(row)) for row in rows]