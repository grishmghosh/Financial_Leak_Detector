from fastapi import Depends, Request
import asyncpg

from app.core.security import get_current_user
from app.db.connection import get_db


async def apply_tenant_context(
    request: Request,
    user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
) -> None:
    """
    Sets PostgreSQL session variable for RLS enforcement.
    Must run before any query execution.
    """
    # ── TEMPORARY DEV BYPASS for Swagger testing ─────────────────────
    # Skip tenant context when the dev bypass user is active.
    # TODO: Remove this bypass before deploying to staging/production.
    if user.get("user_id") == "00000000-0000-0000-0000-000000000000":
        return
    # ── END TEMPORARY DEV BYPASS ─────────────────────────────────────

    user_id = user["user_id"]

    await db.execute(
        "SET app.current_user_id = $1",
        user_id
    )
