"""Generic asyncpg connection-pool helper shared across products.

Reads the standard ``DB_*`` environment variables. Products needing Postgres can
build on this single pool instead of each managing their own.
"""
from __future__ import annotations

import os
from typing import Optional

import asyncpg

_pool: Optional["asyncpg.Pool"] = None


def _conn_kwargs() -> dict:
    return dict(
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
    )


async def get_pool(max_size: Optional[int] = None) -> "asyncpg.Pool":
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            min_size=1,
            max_size=max_size or int(os.getenv("DB_POOL_MAX", "5")),
            **_conn_kwargs(),
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None
