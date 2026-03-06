"""
Application entry point.

Registers database lifecycle events and exposes a health endpoint.
"""

import logging
from contextlib import asynccontextmanager

from fastapi import Depends, FastAPI, APIRouter, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from app.config import get_settings
from app.db.connection import init_db, close_db
from app.core.tenant import apply_tenant_context
from app.core.exceptions import register_exception_handlers
from app.routers.transactions import router as transactions_router
from app.routers.analytics import router as analytics_router
from app.routers.ml import router as ml_router
from app.routers.dashboard import router as dashboard_router

logger = logging.getLogger(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Manage startup / shutdown resources."""
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    logger.info("Starting Financial Leak Detector")
    await init_db(app)
    yield
    await close_db(app)
    logger.info("Shutdown complete")


app = FastAPI(
    title="Financial Leak Detector",
    lifespan=lifespan,
)

register_exception_handlers(app)

# ── CORS middleware ──────────────────────────────────────────
settings = get_settings()

_cors_origins_by_env = {
    "development": [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5678",   # n8n
    ],
    "staging": [
        "https://staging.leakwatch.app",
        "http://localhost:5678",
    ],
    "production": [
        "https://leakwatch.app",
        "https://n8n.leakwatch.app",
    ],
}

allowed_origins = _cors_origins_by_env.get(
    settings.ENVIRONMENT,
    _cors_origins_by_env["development"],
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Protected router (tenant enforced) ──────────────────────
protected_router = APIRouter(
    dependencies=[Depends(apply_tenant_context)]
)

# ── Versioned API router ────────────────────────────────────
api_v1_router = APIRouter(prefix="/api/v1")
api_v1_router.include_router(transactions_router)
api_v1_router.include_router(analytics_router)
api_v1_router.include_router(ml_router)
api_v1_router.include_router(dashboard_router)

# Attach versioned routes behind tenant protection
protected_router.include_router(api_v1_router)

# Include protected router into main app
app.include_router(protected_router)


# ── Public health endpoint ──────────────────────────────────
@app.get("/health")
async def health():
    return {"status": "ok"}


# ── Public readiness endpoint ───────────────────────────────


@app.get("/ready")
async def ready(request: Request):
    checks = {"database": False, "model": False}
    try:
        async with request.app.state.db_pool.acquire() as conn:
            await conn.fetchval("SELECT 1")
        checks["database"] = True
    except Exception:
        logger.warning("Readiness check: database unavailable")

    try:
        from app.ml.model_loader import get_model
        get_model()
        checks["model"] = True
    except Exception:
        logger.warning("Readiness check: ML model unavailable")

    all_ready = all(checks.values())
    status_code = 200 if all_ready else 503
    return JSONResponse(
        status_code=status_code,
        content={"status": "ready" if all_ready else "not_ready", "checks": checks},
    )