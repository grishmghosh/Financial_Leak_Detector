"""
JWT verification for Supabase-issued tokens.

Validates the Authorization header, decodes the JWT using the shared
SUPABASE_JWT_SECRET, and returns the authenticated user's identity.
"""

from fastapi import HTTPException, Request
from jose import JWTError, jwt

from app.config import get_settings

ALGORITHM = "HS256"


async def get_current_user(request: Request) -> dict:
    """Extract and verify a Supabase JWT from the Authorization header.

    Returns:
        {"user_id": "<sub claim value>"}

    Raises:
        HTTPException 401 – missing/malformed header, invalid token,
                            or missing "sub" claim.
    """
    # ── TEMPORARY DEV BYPASS for Swagger testing ─────────────────────
    # Allows /api/v1/ml/run-analysis to be called without a JWT.
    # TODO: Remove this bypass before deploying to staging/production.
    if request.url.path.startswith("/api/v1/ml/run-analysis"):
        return {
            "user_id": "00000000-0000-0000-0000-000000000000",
            "org_id": "00000000-0000-0000-0000-000000000000",
        }
    # ── END TEMPORARY DEV BYPASS ─────────────────────────────────────

    auth_header: str | None = request.headers.get("Authorization")

    if not auth_header:
        raise HTTPException(status_code=401, detail="Missing Authorization header")

    parts = auth_header.split()

    if len(parts) != 2 or parts[0].lower() != "bearer":
        raise HTTPException(status_code=401, detail="Invalid Authorization header format")

    token = parts[1]

    try:
        payload = jwt.decode(
            token,
            get_settings().SUPABASE_JWT_SECRET,
            algorithms=[ALGORITHM],
        )
    except JWTError:
        raise HTTPException(status_code=401, detail="Invalid or expired token")

    sub: str | None = payload.get("sub")

    if not sub:
        raise HTTPException(status_code=401, detail="Token missing 'sub' claim")

    return {"user_id": sub}
