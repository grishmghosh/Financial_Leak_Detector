"""
Async database connection pool using asyncpg.

Provides pool lifecycle helpers and a FastAPI dependency
that yields a single connection per request.
"""

from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, Request

from app.config import get_settings

_pool = None


async def init_db(app: FastAPI) -> None:
    """Create the asyncpg connection pool and attach it to app state."""
    global _pool
    settings = get_settings()
    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
    )
    _pool = app.state.db_pool


async def close_db(app: FastAPI) -> None:
    """Gracefully close the connection pool."""
    await app.state.db_pool.close()


async def get_db(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    """FastAPI dependency – acquires a connection and releases it after use."""
    async with request.app.state.db_pool.acquire() as conn:
        yield conn


def get_pool():
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool
