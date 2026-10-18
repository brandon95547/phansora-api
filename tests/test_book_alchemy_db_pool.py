"""One pool per process (phansora.products.book_alchemy.db.get_pool).

Opening the pool awaits, so without a lock two first callers both saw no pool and each
opened one — the API's handlers and the worker's lanes start together, which is exactly
when that happens.
"""
from __future__ import annotations

import asyncio

import pytest

db = pytest.importorskip("phansora.products.book_alchemy.db", reason="asyncpg not installed")


class FakePool:
    async def execute(self, statement):
        return None

    async def close(self):
        return None


def test_concurrent_first_callers_share_one_pool(monkeypatch):
    opened = []

    async def create_pool(**kwargs):
        await asyncio.sleep(0.01)
        pool = FakePool()
        opened.append(pool)
        return pool

    monkeypatch.setattr(db.asyncpg, "create_pool", create_pool)
    monkeypatch.setattr(db, "_pool", None)
    monkeypatch.setattr(db, "_pool_lock", asyncio.Lock())

    async def main():
        return await asyncio.gather(*(db.get_pool() for _ in range(5)))

    pools = asyncio.run(main())
    assert len(opened) == 1
    assert all(p is opened[0] for p in pools)
