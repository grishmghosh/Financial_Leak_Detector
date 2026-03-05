"""
Application entry point.

Registers database lifecycle events and exposes a health endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, APIRouter, Request
from fastapi.responses import JSONResponse

from app.db.connection import init_db, close_db
from app.core.tenant import apply_tenant_context
from app.core.exceptions import register_exception_handlers
from app.routers.transactions import router as transactions_router
from app.routers.analytics import router as analytics_router


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown resources."""
    await init_db(app)
    yield
    await close_db(app)


app = FastAPI(
    title="Financial Leak Detector",
    lifespan=lifespan,
)

register_exception_handlers(app)

# Protected router (tenant enforced)
protected_router = APIRouter(
    dependencies=[Depends(apply_tenant_context)]
)

# API router (business routes live here)
api_router = APIRouter()
api_router.include_router(transactions_router)
api_router.include_router(analytics_router)

# Attach api_router to protected_router with prefix
protected_router.include_router(api_router, prefix="/api")

# Include protected router into main app
app.include_router(protected_router)


# Public health endpoint
@app.get("/health")
async def health():
    return {"status": "ok"}


# Public readiness endpoint
@app.get("/ready")
async def ready(request: Request):
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        return {"status": "ready"}
    except Exception:
        return JSONResponse(status_code=503, content={"status": "not_ready"})