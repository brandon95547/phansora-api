"""Async Postgres access layer for Book Alchemy.

State lives in the shared phansora Postgres (the same DB the Node dashboard
migrates). Connection settings come from the same env vars Node uses
(DB_USER / DB_HOST / DB_NAME / DB_PASSWORD / DB_PORT) so a single .env works
for both. A lazily-created asyncpg pool is shared across the process.
"""
from __future__ import annotations

import os
from typing import Any, Optional

import asyncpg

_pool: Optional[asyncpg.Pool] = None


async def get_pool() -> asyncpg.Pool:
    global _pool
    if _pool is None:
        _pool = await asyncpg.create_pool(
            user=os.getenv("DB_USER"),
            password=os.getenv("DB_PASSWORD"),
            database=os.getenv("DB_NAME"),
            host=os.getenv("DB_HOST", "127.0.0.1"),
            port=int(os.getenv("DB_PORT", "5432")),
            min_size=1,
            max_size=int(os.getenv("BOOK_ALCHEMY_DB_POOL", "5")),
        )
    return _pool


async def close_pool() -> None:
    global _pool
    if _pool is not None:
        await _pool.close()
        _pool = None


# --------------------------------------------------------------- projects
async def create_project(
    *,
    user_id: int,
    name: str,
    source_format: str,
    source_path: Optional[str],
    source_url: Optional[str],
    options: dict,
) -> int:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO public.book_alchemy_projects
            (user_id, name, source_format, source_path, source_url, options,
             status, phase, stage, progress)
        VALUES ($1, $2, $3, $4, $5, $6::jsonb, 'uploaded', 'uploaded', 'Uploaded', 0)
        RETURNING id
        """,
        user_id, name, source_format, source_path, source_url, _json(options),
    )
    return int(row["id"])


async def get_project(project_id: int, user_id: Optional[int] = None) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    if user_id is None:
        return await pool.fetchrow(
            "SELECT * FROM public.book_alchemy_projects WHERE id = $1", project_id
        )
    return await pool.fetchrow(
        "SELECT * FROM public.book_alchemy_projects WHERE id = $1 AND user_id = $2",
        project_id, user_id,
    )


async def list_projects(user_id: int) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(
        """
        SELECT p.*,
               (SELECT COUNT(*) FROM public.book_alchemy_sessions s
                 WHERE s.project_id = p.id AND s.status = 'complete') AS sessions_complete,
               (SELECT COUNT(*) FROM public.book_alchemy_sessions s
                 WHERE s.project_id = p.id) AS sessions_total
          FROM public.book_alchemy_projects p
         WHERE p.user_id = $1
         ORDER BY p.created_at DESC
        """,
        user_id,
    )


async def delete_project(project_id: int, user_id: int) -> Optional[asyncpg.Record]:
    """Delete a project (cascades to chunks/concepts/sessions). Returns the row
    (with source_path) so the caller can clean up files, or None if not found."""
    pool = await get_pool()
    return await pool.fetchrow(
        "DELETE FROM public.book_alchemy_projects WHERE id = $1 AND user_id = $2 RETURNING *",
        project_id, user_id,
    )


async def set_project(project_id: int, **fields: Any) -> None:
    """Patch arbitrary project columns + bump updated_at. JSONB columns are
    auto-encoded."""
    if not fields:
        return
    pool = await get_pool()
    cols, vals = [], []
    for i, (k, v) in enumerate(fields.items(), start=2):
        if k in ("options", "curriculum") and not isinstance(v, str):
            cols.append(f"{k} = ${i}::jsonb")
            vals.append(_json(v))
        else:
            cols.append(f"{k} = ${i}")
            vals.append(v)
    await pool.execute(
        f"UPDATE public.book_alchemy_projects SET {', '.join(cols)}, updated_at = NOW() WHERE id = $1",
        project_id, *vals,
    )


async def claim_next_project(worker_id: str, lease_seconds: int = 600) -> Optional[asyncpg.Record]:
    """Atomically claim one project that needs work and isn't actively leased.

    Uses FOR UPDATE SKIP LOCKED so multiple workers never grab the same row.
    A project is claimable when it's freshly uploaded or its previous lease has
    expired (crash recovery)."""
    pool = await get_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            row = await con.fetchrow(
                """
                SELECT * FROM public.book_alchemy_projects
                 WHERE status IN ('uploaded', 'processing')
                   AND phase <> 'complete'
                   AND (lease_expires_at IS NULL OR lease_expires_at < NOW())
                 ORDER BY created_at ASC
                 FOR UPDATE SKIP LOCKED
                 LIMIT 1
                """
            )
            if row is None:
                return None
            await con.execute(
                """
                UPDATE public.book_alchemy_projects
                   SET status = 'processing',
                       lease_owner = $2,
                       lease_expires_at = NOW() + ($3 || ' seconds')::interval,
                       updated_at = NOW()
                 WHERE id = $1
                """,
                row["id"], worker_id, str(lease_seconds),
            )
            return row


async def renew_lease(project_id: int, worker_id: str, lease_seconds: int = 600) -> None:
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE public.book_alchemy_projects
           SET lease_expires_at = NOW() + ($3 || ' seconds')::interval, updated_at = NOW()
         WHERE id = $1 AND lease_owner = $2
        """,
        project_id, worker_id, str(lease_seconds),
    )


async def release_lease(project_id: int) -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE public.book_alchemy_projects SET lease_owner = NULL, lease_expires_at = NULL WHERE id = $1",
        project_id,
    )


# --------------------------------------------------------------- chunks
async def insert_chunks(project_id: int, chunks: list[dict]) -> None:
    pool = await get_pool()
    await pool.executemany(
        """
        INSERT INTO public.book_alchemy_chunks
            (project_id, ordinal, text, chapter, section, page_start, page_end, char_start, char_end)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
        """,
        [
            (
                project_id, c["ordinal"], c["text"], c.get("chapter"), c.get("section"),
                c.get("page_start"), c.get("page_end"), c.get("char_start"), c.get("char_end"),
            )
            for c in chunks
        ],
    )


async def count_chunks(project_id: int) -> int:
    pool = await get_pool()
    return int(await pool.fetchval(
        "SELECT COUNT(*) FROM public.book_alchemy_chunks WHERE project_id = $1", project_id
    ))


async def get_chunk_by_ordinal(project_id: int, ordinal: int) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetchrow(
        "SELECT * FROM public.book_alchemy_chunks WHERE project_id = $1 AND ordinal = $2",
        project_id, ordinal,
    )


async def get_chunks_by_ids(project_id: int, ids: list[int]) -> list[asyncpg.Record]:
    if not ids:
        return []
    pool = await get_pool()
    return await pool.fetch(
        """
        SELECT * FROM public.book_alchemy_chunks
         WHERE project_id = $1 AND id = ANY($2::bigint[])
         ORDER BY ordinal ASC
        """,
        project_id, ids,
    )


async def get_all_chunks(project_id: int) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(
        "SELECT * FROM public.book_alchemy_chunks WHERE project_id = $1 ORDER BY ordinal ASC",
        project_id,
    )


# --------------------------------------------------------------- concepts
async def insert_concepts(project_id: int, concepts: list[dict]) -> None:
    if not concepts:
        return
    pool = await get_pool()
    await pool.executemany(
        """
        INSERT INTO public.book_alchemy_concepts (project_id, kind, content, source_chunk_ids)
        VALUES ($1, $2, $3::jsonb, $4::bigint[])
        """,
        [
            (project_id, c["kind"], _json(c["content"]), c.get("source_chunk_ids", []))
            for c in concepts
        ],
    )


async def get_concepts(project_id: int) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(
        "SELECT * FROM public.book_alchemy_concepts WHERE project_id = $1 ORDER BY id ASC",
        project_id,
    )


# --------------------------------------------------------------- sessions
async def create_session(
    *, project_id: int, ordinal: int, title: str, summary: str,
    outline: Any, source_chunk_ids: list[int],
) -> int:
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        INSERT INTO public.book_alchemy_sessions
            (project_id, ordinal, title, summary, outline, source_chunk_ids, status)
        VALUES ($1, $2, $3, $4, $5::jsonb, $6::bigint[], 'pending')
        RETURNING id
        """,
        project_id, ordinal, title, summary, _json(outline), source_chunk_ids,
    )
    return int(row["id"])


async def get_sessions(project_id: int) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(
        "SELECT * FROM public.book_alchemy_sessions WHERE project_id = $1 ORDER BY ordinal ASC",
        project_id,
    )


async def get_session(session_id: int, project_id: Optional[int] = None) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    if project_id is None:
        return await pool.fetchrow(
            "SELECT * FROM public.book_alchemy_sessions WHERE id = $1", session_id
        )
    return await pool.fetchrow(
        "SELECT * FROM public.book_alchemy_sessions WHERE id = $1 AND project_id = $2",
        session_id, project_id,
    )


async def next_session_needing(project_id: int, statuses: list[str]) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetchrow(
        """
        SELECT * FROM public.book_alchemy_sessions
         WHERE project_id = $1 AND status = ANY($2::text[])
         ORDER BY ordinal ASC LIMIT 1
        """,
        project_id, statuses,
    )


async def set_session(session_id: int, **fields: Any) -> None:
    if not fields:
        return
    pool = await get_pool()
    cols, vals = [], []
    for i, (k, v) in enumerate(fields.items(), start=2):
        if k in ("outline", "validation_notes") and not isinstance(v, str):
            cols.append(f"{k} = ${i}::jsonb")
            vals.append(_json(v))
        else:
            cols.append(f"{k} = ${i}")
            vals.append(v)
    await pool.execute(
        f"UPDATE public.book_alchemy_sessions SET {', '.join(cols)}, updated_at = NOW() WHERE id = $1",
        session_id, *vals,
    )


# --------------------------------------------------------------- util
def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)
