"""
Async database connection pool using asyncpg.

Provides pool lifecycle helpers and a FastAPI dependency
that yields a single connection per request.
"""

from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, Request

from app.config import get_settings


async def init_db(app: FastAPI) -> None:
    """Create the asyncpg connection pool and attach it to app state."""
    settings = get_settings()
    app.state.db_pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
    )


async def close_db(app: FastAPI) -> None:
    """Gracefully close the connection pool."""
    await app.state.db_pool.close()


async def get_db(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    """FastAPI dependency – acquires a connection and releases it after use."""
    async with request.app.state.db_pool.acquire() as conn:
        yield conn
