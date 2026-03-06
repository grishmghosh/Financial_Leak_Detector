"""
Database queries for the dashboard API layer.

All queries enforce tenant isolation via explicit org_id filtering.
"""

from uuid import UUID


async def get_latest_run(conn, org_id: UUID) -> dict | None:
    row = await conn.fetchrow(
        """
        SELECT id AS run_id, started_at, completed_at,
               transactions_scored, high_risk_count, duplicates_found
        FROM analysis_runs
        WHERE org_id = $1 AND status = 'completed'
        ORDER BY started_at DESC
        LIMIT 1
        """,
        org_id,
    )
    return dict(row) if row else None


async def get_run_history(conn, org_id: UUID, *, limit: int = 50, offset: int = 0) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT id AS run_id, started_at, completed_at,
               transactions_scored, high_risk_count
        FROM analysis_runs
        WHERE org_id = $1
        ORDER BY started_at DESC
        LIMIT $2 OFFSET $3
        """,
        org_id, limit, offset,
    )
    return [dict(r) for r in rows]


async def get_high_risk_transactions(conn, org_id: UUID, *, limit: int = 100, offset: int = 0) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT voucher_number, leak_probability, risk_factors,
               is_duplicate, duplicate_of
        FROM analysis_results
        WHERE org_id = $1 AND risk_category = 'high'
        ORDER BY leak_probability DESC
        LIMIT $2 OFFSET $3
        """,
        org_id, limit, offset,
    )
    return [dict(r) for r in rows]


async def get_vendor_risk_scores(conn, org_id: UUID) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT vendor_name, total_spend, transaction_count,
               anomaly_count, risk_score
        FROM vendor_risk_scores
        WHERE org_id = $1
        ORDER BY risk_score DESC
        """,
        org_id,
    )
    return [dict(r) for r in rows]


async def get_department_risk_scores(conn, org_id: UUID) -> list[dict]:
    rows = await conn.fetch(
        """
        SELECT department, monthly_spend, spike_score, risk_category
        FROM department_risk_scores
        WHERE org_id = $1
        ORDER BY spike_score DESC
        """,
        org_id,
    )
    return [dict(r) for r in rows]