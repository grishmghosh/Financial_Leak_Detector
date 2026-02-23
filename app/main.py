"""
Application entry point.

Registers database lifecycle events and exposes a health endpoint.
"""

from contextlib import asynccontextmanager

from fastapi import FastAPI

from app.db.connection import init_db, close_db


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


@app.get("/health")
async def health():
    return {"status": "ok"}
