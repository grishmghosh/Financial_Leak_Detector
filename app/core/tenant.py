from fastapi import Depends
import asyncpg

from app.core.security import get_current_user
from app.db.connection import get_db


async def apply_tenant_context(
    user: dict = Depends(get_current_user),
    db: asyncpg.Connection = Depends(get_db),
) -> None:
    """
    Sets PostgreSQL session variable for RLS enforcement.
    Must run before any query execution.
    """
    user_id = user["user_id"]

    await db.execute(
        "SET app.current_user_id = $1",
        user_id
    )
