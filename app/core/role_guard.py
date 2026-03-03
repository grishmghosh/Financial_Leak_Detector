from fastapi import Depends, HTTPException
import asyncpg

from app.core.security import get_current_user
from app.db.connection import get_db


def require_role(required_role: str):
    async def role_dependency(
        user: dict = Depends(get_current_user),
        db: asyncpg.Connection = Depends(get_db),
    ) -> dict:
        row = await db.fetchrow(
            "SELECT org_id, role FROM user_organizations WHERE user_id = $1",
            user["user_id"],
        )
        if not row:
            raise HTTPException(status_code=403, detail="Forbidden")
        if row["role"] != required_role:
            raise HTTPException(status_code=403, detail="Forbidden")
        return {
            "user_id": user["user_id"],
            "org_id": row["org_id"],
            "role": row["role"],
        }

    return role_dependency


require_admin = require_role("admin")
require_member = require_role("member")
