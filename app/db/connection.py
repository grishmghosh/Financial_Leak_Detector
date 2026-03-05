"""
Async database connection pool using asyncpg.

Provides pool lifecycle helpers and a FastAPI dependency
that yields a single connection per request.
"""

import logging
from typing import AsyncGenerator

import asyncpg
from fastapi import FastAPI, Request

from app.config import get_settings

logger = logging.getLogger(__name__)

_pool = None


async def init_db(app: FastAPI | None = None) -> None:
    global _pool
    settings = get_settings()
    pool = await asyncpg.create_pool(
        dsn=settings.DATABASE_URL,
        min_size=5,
        max_size=20,
        command_timeout=60,
    )
    _pool = pool
    if app is not None:
        app.state.db_pool = pool
    logger.info("Database connection pool initialised")


async def close_db(app: FastAPI | None = None) -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
        logger.info("Database connection pool closed")


async def get_db(request: Request) -> AsyncGenerator[asyncpg.Connection, None]:
    async with request.app.state.db_pool.acquire() as conn:
        yield conn


def get_pool():
    if _pool is None:
        raise RuntimeError("Database pool not initialized")
    return _pool
