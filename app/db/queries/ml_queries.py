"""
Database queries for the ML analysis pipeline.
"""

from datetime import date
from typing import Optional
from uuid import UUID


async def create_analysis_run(conn, *, org_id: UUID, triggered_by: UUID | None = None,
                              trigger_source: str | None = "api") -> dict:
    row = await conn.fetchrow(
        """
        INSERT INTO analysis_runs (org_id, triggered_by, trigger_source, status)
        VALUES ($1, $2, $3, 'running')
        RETURNING id, org_id, status, started_at
        """,
        org_id, triggered_by, trigger_source,
    )
    return dict(row)


async def update_analysis_run(conn, *, run_id: UUID, org_id: UUID, status: str,
                              transactions_scored: int = 0,
                              high_risk_count: int = 0,
                              duplicates_found: int = 0) -> None:
    await conn.execute(
        """
        UPDATE analysis_runs
        SET status = $2,
            transactions_scored = $3,
            high_risk_count = $4,
            duplicates_found = $5,
            completed_at = now()
        WHERE id = $1 AND org_id = $6
        """,
        run_id, status, transactions_scored, high_risk_count, duplicates_found, org_id,
    )


async def get_analysis_run(conn, run_id: UUID) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT id AS run_id, status, transactions_scored, high_risk_count,
               duplicates_found, started_at, completed_at
        FROM analysis_runs
        WHERE id = $1
        """,
        run_id,
    )
    return dict(row) if row else None


async def fetch_transactions_for_analysis(
    conn, org_id: UUID, *,
    start_date: Optional[date] = None,
    end_date: Optional[date] = None,
    department: Optional[str] = None,
) -> list[dict]:
    query = """
        SELECT voucher_number, org_id, amount, check_date, department, description, vendor_name
        FROM transactions
        WHERE org_id = $1
    """
    params: list = [org_id]
    idx = 2

    if start_date is not None:
        query += f" AND check_date >= ${idx}"
        params.append(start_date)
        idx += 1

    if end_date is not None:
        query += f" AND check_date <= ${idx}"
        params.append(end_date)
        idx += 1

    if department is not None:
        query += f" AND department = ${idx}"
        params.append(department)
        idx += 1

    query += " ORDER BY check_date"

    rows = await conn.fetch(query, *params)
    return [dict(r) for r in rows]


async def fetch_department_stats(conn, org_id: UUID) -> dict[str, dict]:
    rows = await conn.fetch(
        """
        SELECT department,
               AVG(amount)    AS avg_amount,
               STDDEV(amount) AS std_amount,
               COUNT(*)       AS txn_count
        FROM transactions
        WHERE org_id = $1
        GROUP BY department
        """,
        org_id,
    )
    return {
        row["department"]: {
            "avg_amount": float(row["avg_amount"]) if row["avg_amount"] else 0.0,
            "std_amount": float(row["std_amount"]) if row["std_amount"] else 0.0,
            "txn_count": row["txn_count"],
        }
        for row in rows
    }


async def insert_analysis_results(conn, rows: list[dict]) -> None:
    if not rows:
        return
    await conn.executemany(
        """
        INSERT INTO analysis_results
            (org_id, run_id, voucher_number, leak_probability, risk_factors,
             risk_category, is_duplicate, duplicate_of)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
        """,
        [
            (
                r["org_id"], r["run_id"], r["voucher_number"],
                r["leak_probability"], r["risk_factors"], r["risk_category"],
                r["is_duplicate"], r["duplicate_of"],
            )
            for r in rows
        ],
    )


async def insert_vendor_risk_scores(conn, rows: list[dict]) -> None:
    if not rows:
        return
    await conn.executemany(
        """
        INSERT INTO vendor_risk_scores
            (org_id, run_id, vendor_name, total_spend, transaction_count,
             anomaly_count, risk_score)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        """,
        [
            (
                r["org_id"], r["run_id"], r["vendor_name"],
                r["total_spend"], r["transaction_count"],
                r["anomaly_count"], r["risk_score"],
            )
            for r in rows
        ],
    )


async def insert_department_risk_scores(conn, rows: list[dict]) -> None:
    if not rows:
        return
    await conn.executemany(
        """
        INSERT INTO department_risk_scores
            (org_id, run_id, department, monthly_spend, spike_score, risk_category)
        VALUES ($1, $2, $3, $4, $5, $6)
        """,
        [
            (
                r["org_id"], r["run_id"], r["department"],
                r["monthly_spend"], r["spike_score"], r["risk_category"],
            )
            for r in rows
        ],
    )
