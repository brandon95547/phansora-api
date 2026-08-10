"""Async Postgres access layer for Book Alchemy.

State lives in the shared phansora Postgres (the same DB the Node dashboard
migrates). Connection settings come from the same env vars Node uses
(DB_USER / DB_HOST / DB_NAME / DB_PASSWORD / DB_PORT) so a single .env works
for both. A lazily-created asyncpg pool is shared across the process.
"""
from __future__ import annotations

import logging
import os
from typing import Any, Optional

import asyncpg

log = logging.getLogger("book_alchemy.db")

_pool: Optional[asyncpg.Pool] = None

# Columns added after the tables were first created by hand. This repo has no
# migration tool and the schema is not checked in, so each one is applied here,
# idempotently, the first time a pool is opened. ADD COLUMN IF NOT EXISTS on a
# nullable column with a default is a catalog-only change in Postgres 11+ — it
# does not rewrite the table, so this stays cheap however large the table grows.
_SCHEMA_ADDITIONS = (
    "ALTER TABLE public.book_alchemy_chunks "
    "ADD COLUMN IF NOT EXISTS teachable boolean NOT NULL DEFAULT true",
)


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
        await _ensure_schema(_pool)
    return _pool


async def _ensure_schema(pool: asyncpg.Pool) -> None:
    """Apply _SCHEMA_ADDITIONS. Never fatal: a read-only or lagging DB role must
    not stop the API booting, and the failure is loud enough to act on."""
    for statement in _SCHEMA_ADDITIONS:
        try:
            await pool.execute(statement)
        except Exception:  # noqa: BLE001
            log.warning("Schema addition failed (continuing): %s", statement, exc_info=True)


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
    max_projects: Optional[int] = None,
) -> Optional[int]:
    """Insert a project and return its id.

    When ``max_projects`` is set, the user's existing project count is checked
    under a per-user advisory lock and ``None`` is returned if they're already at
    the cap — so two simultaneous uploads can't both slip past the limit.
    """
    pool = await get_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            if max_projects is not None:
                # Serialize create-vs-create for this user only; released on commit.
                await con.execute(
                    "SELECT pg_advisory_xact_lock(hashtext('book_alchemy_projects'), $1)",
                    user_id,
                )
                existing = await con.fetchval(
                    "SELECT COUNT(*) FROM public.book_alchemy_projects WHERE user_id = $1",
                    user_id,
                )
                if int(existing) >= max_projects:
                    return None
            row = await con.fetchrow(
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


async def count_projects(user_id: int) -> int:
    pool = await get_pool()
    return int(await pool.fetchval(
        "SELECT COUNT(*) FROM public.book_alchemy_projects WHERE user_id = $1", user_id
    ))


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


async def reopen_project(project_id: int, *, stage: str, phase: str = "sessions") -> None:
    """Put a parked or finished project back into the worker's claimable set.

    Used by "process this phase", "process everything" and per-session regenerate
    — all three are user actions that arrive on the API process while the worker
    may be mid-book on the same row.

    A LIVE lease is deliberately preserved. A worker holding one is already
    driving this project and re-reads the row on every iteration of its loop, so
    it picks the new work up by itself; clearing the lease out from under it
    would let a second worker claim the same project and run the same lesson
    twice. An expired lease is cleared, because that is the crash-recovery case
    and nobody is coming back for it.
    """
    pool = await get_pool()
    await pool.execute(
        """
        UPDATE public.book_alchemy_projects
           SET status = 'processing',
               phase = $2,
               stage = $3,
               error_message = NULL,
               lease_owner = CASE WHEN lease_expires_at > NOW() THEN lease_owner ELSE NULL END,
               lease_expires_at = CASE WHEN lease_expires_at > NOW() THEN lease_expires_at ELSE NULL END,
               updated_at = NOW()
         WHERE id = $1
        """,
        project_id, phase, stage,
    )


async def claim_next_project(worker_id: str, lease_seconds: int = 600) -> Optional[asyncpg.Record]:
    """Atomically claim one project that needs work and isn't actively leased.

    Uses FOR UPDATE SKIP LOCKED so multiple workers never grab the same row.
    A project is claimable when it's freshly uploaded or its previous lease has
    expired (crash recovery).

    Note this needs no awareness of delivery phases: a project parked waiting for
    the listener to ask for the next phase sits at status 'awaiting_user', which
    this predicate already excludes. This query is the single point of
    correctness for claiming, and leaving it untouched is worth a lot."""
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
            (project_id, ordinal, text, chapter, section, page_start, page_end,
             char_start, char_end, teachable)
        VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9, $10)
        """,
        [
            (
                project_id, c["ordinal"], c["text"], c.get("chapter"), c.get("section"),
                c.get("page_start"), c.get("page_end"), c.get("char_start"), c.get("char_end"),
                # Default True: a chunk from before this column existed, or from a
                # path that does not classify, is material until proven otherwise.
                bool(c.get("teachable", True)),
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


async def set_chunk_teachable(chunk_id: int, teachable: bool) -> None:
    pool = await get_pool()
    await pool.execute(
        "UPDATE public.book_alchemy_chunks SET teachable = $2 WHERE id = $1",
        chunk_id, teachable,
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


async def get_concepts_for_chunks(
    project_id: int, chunk_ids: list[int]
) -> list[asyncpg.Record]:
    """Every concept indexed in the given chunks, in source order.

    This is a lesson's coverage contract: the ideas the analyze phase found in
    exactly the segments the lesson was built from. Ordered by id, which is
    insertion order, which is chunk order — so the checklist reads in the same
    sequence the lesson is meant to teach.
    """
    if not chunk_ids:
        return []
    pool = await get_pool()
    return await pool.fetch(
        """
        SELECT * FROM public.book_alchemy_concepts
         WHERE project_id = $1 AND source_chunk_ids && $2::bigint[]
         ORDER BY id ASC
        """,
        project_id, chunk_ids,
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


async def next_session_needing(
    project_id: int, statuses: list[str], max_phase: Optional[int] = None
) -> Optional[asyncpg.Record]:
    """The lowest-ordinal session in one of ``statuses``, optionally capped to a
    delivery phase.

    Ordinal order is load-bearing rather than tidy: every script is conditioned on
    what the lessons before it already covered (see pipeline._prior_coverage), so
    writing them out of order would change the output.

    ``max_phase`` is the delivery-phase ceiling from :func:`work_ceiling`. None
    means unbounded, which is how a project with no phase rows — anything that
    predates phased delivery — keeps behaving exactly as it did before.
    """
    pool = await get_pool()
    if max_phase is None:
        return await pool.fetchrow(
            """
            SELECT * FROM public.book_alchemy_sessions
             WHERE project_id = $1 AND status = ANY($2::text[])
             ORDER BY ordinal ASC LIMIT 1
            """,
            project_id, statuses,
        )
    return await pool.fetchrow(
        """
        SELECT * FROM public.book_alchemy_sessions
         WHERE project_id = $1 AND status = ANY($2::text[]) AND phase_no <= $3
         ORDER BY ordinal ASC LIMIT 1
        """,
        project_id, statuses, max_phase,
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


# --------------------------------------------------------------- delivery phases
#
# What the listener means by "Phase 3 of 7": a batch of lessons delivered
# together. Not to be confused with book_alchemy_projects.phase, which is the
# internal pipeline cursor.
#
# Lifecycle: pending -> ready -> queued -> processing -> complete | failed.
# 'ready' is exactly "the listener's button is live"; 'queued' is "they clicked,
# the worker has not reached it yet".


async def create_phases(project_id: int, phases: list[dict]) -> None:
    """Write the plan. The first phase is offered immediately; the rest wait.

    Called once, from the curriculum step. ON CONFLICT DO NOTHING makes a retried
    curriculum step harmless — the boundaries a listener has already been shown
    must never move under them.
    """
    if not phases:
        return
    pool = await get_pool()
    await pool.executemany(
        """
        INSERT INTO public.book_alchemy_phases
            (project_id, ordinal, label, status, session_start, session_end, est_seconds)
        VALUES ($1, $2, $3, $4, $5, $6, $7)
        ON CONFLICT (project_id, ordinal) DO NOTHING
        """,
        [
            (
                project_id, p["ordinal"], p["label"],
                "queued" if int(p["ordinal"]) == 1 else "pending",
                p["session_start"], p["session_end"], int(p.get("est_seconds") or 0),
            )
            for p in phases
        ],
    )
    await pool.execute(
        """
        UPDATE public.book_alchemy_sessions s
           SET phase_no = p.ordinal
          FROM public.book_alchemy_phases p
         WHERE s.project_id = $1 AND p.project_id = $1
           AND s.ordinal BETWEEN p.session_start AND p.session_end
        """,
        project_id,
    )
    await pool.execute(
        "UPDATE public.book_alchemy_phases SET requested_at = NOW(), updated_at = NOW() "
        "WHERE project_id = $1 AND ordinal = 1",
        project_id,
    )


async def get_phases(project_id: int) -> list[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetch(
        "SELECT * FROM public.book_alchemy_phases WHERE project_id = $1 ORDER BY ordinal ASC",
        project_id,
    )


async def count_phases(project_id: int) -> int:
    pool = await get_pool()
    return int(await pool.fetchval(
        "SELECT COUNT(*) FROM public.book_alchemy_phases WHERE project_id = $1", project_id
    ))


async def active_phase(project_id: int) -> Optional[asyncpg.Record]:
    """The phase the worker is (or should be) working on right now."""
    pool = await get_pool()
    return await pool.fetchrow(
        """
        SELECT * FROM public.book_alchemy_phases
         WHERE project_id = $1 AND status IN ('queued', 'processing')
         ORDER BY ordinal ASC LIMIT 1
        """,
        project_id,
    )


async def next_phase_after(project_id: int, ordinal: int) -> Optional[asyncpg.Record]:
    pool = await get_pool()
    return await pool.fetchrow(
        """
        SELECT * FROM public.book_alchemy_phases
         WHERE project_id = $1 AND ordinal > $2 AND status NOT IN ('complete', 'failed')
         ORDER BY ordinal ASC LIMIT 1
        """,
        project_id, ordinal,
    )


async def claim_phase(project_id: int, ordinal: int) -> Optional[asyncpg.Record]:
    """Take a phase the listener just asked for, or return None.

    A conditional UPDATE rather than a read-then-write, which is what makes the
    button free to double-click: two clicks, or two tabs, race into the same
    statement and exactly one of them matches ``status = 'ready'``.
    """
    pool = await get_pool()
    return await pool.fetchrow(
        """
        UPDATE public.book_alchemy_phases
           SET status = 'queued', requested_at = NOW(), error_message = NULL, updated_at = NOW()
         WHERE project_id = $1 AND ordinal = $2 AND status = 'ready'
        RETURNING *
        """,
        project_id, ordinal,
    )


async def claim_next_ready_phase(project_id: int) -> Optional[asyncpg.Record]:
    """Same claim, but for whichever phase is currently on offer. Backs
    "process everything remaining", which should not need to know the number."""
    pool = await get_pool()
    return await pool.fetchrow(
        """
        UPDATE public.book_alchemy_phases
           SET status = 'queued', requested_at = NOW(), error_message = NULL, updated_at = NOW()
         WHERE id = (
             SELECT id FROM public.book_alchemy_phases
              WHERE project_id = $1 AND status = 'ready'
              ORDER BY ordinal ASC LIMIT 1
         )
        RETURNING *
        """,
        project_id,
    )


async def retry_phase(project_id: int, ordinal: int) -> Optional[asyncpg.Record]:
    """Re-open a failed phase and reset its lessons so the worker rebuilds them."""
    pool = await get_pool()
    async with pool.acquire() as con:
        async with con.transaction():
            row = await con.fetchrow(
                """
                UPDATE public.book_alchemy_phases
                   SET status = 'queued', requested_at = NOW(), error_message = NULL,
                       updated_at = NOW()
                 WHERE project_id = $1 AND ordinal = $2 AND status = 'failed'
                RETURNING *
                """,
                project_id, ordinal,
            )
            if row is None:
                return None
            # Lessons that never finished go back to the start of the line. Ones
            # that already have audio are left alone — re-recording them would
            # burn GPU time redoing work that succeeded.
            await con.execute(
                """
                UPDATE public.book_alchemy_sessions
                   SET status = 'pending', script = NULL, updated_at = NOW()
                 WHERE project_id = $1 AND phase_no = $2 AND status <> 'complete'
                """,
                project_id, ordinal,
            )
            return row


async def set_phase(phase_id: int, **fields: Any) -> None:
    if not fields:
        return
    pool = await get_pool()
    cols = [f"{k} = ${i}" for i, k in enumerate(fields.keys(), start=2)]
    await pool.execute(
        f"UPDATE public.book_alchemy_phases SET {', '.join(cols)}, updated_at = NOW() WHERE id = $1",
        phase_id, *fields.values(),
    )


async def work_ceiling(project_id: int) -> Optional[int]:
    """The highest delivery phase the worker may touch right now.

    Everything already delivered, plus the one the listener has asked for.
    Delivered phases stay in range so a per-session Regenerate still works after
    the fact; phases nobody has clicked are out of range, which is the entire
    gate.

    Returns None — meaning unbounded — for a project with no phase rows at all.
    That is a project from before phased delivery, and it then behaves exactly as
    it did before: fail-open by construction.
    """
    pool = await get_pool()
    row = await pool.fetchrow(
        """
        SELECT COUNT(*) AS n,
               COALESCE(MAX(ordinal) FILTER (
                   WHERE status IN ('queued', 'processing', 'complete')), 0) AS ceiling
          FROM public.book_alchemy_phases WHERE project_id = $1
        """,
        project_id,
    )
    return None if int(row["n"]) == 0 else int(row["ceiling"])


async def phase_audio_seconds(project_id: int, ordinal: int) -> int:
    pool = await get_pool()
    return int(await pool.fetchval(
        """
        SELECT COALESCE(SUM(audio_seconds), 0) FROM public.book_alchemy_sessions
         WHERE project_id = $1 AND phase_no = $2
        """,
        project_id, ordinal,
    ) or 0)


async def list_phases_for_projects(project_ids: list[int]) -> list[asyncpg.Record]:
    """Every phase for a set of projects, so the list endpoint can render its
    phase rails without a query per card."""
    if not project_ids:
        return []
    pool = await get_pool()
    return await pool.fetch(
        """
        SELECT p.*,
               (SELECT COUNT(*) FROM public.book_alchemy_sessions s
                 WHERE s.project_id = p.project_id AND s.phase_no = p.ordinal) AS session_count,
               (SELECT COUNT(*) FROM public.book_alchemy_sessions s
                 WHERE s.project_id = p.project_id AND s.phase_no = p.ordinal
                   AND s.status = 'complete') AS sessions_complete
          FROM public.book_alchemy_phases p
         WHERE p.project_id = ANY($1::bigint[])
         ORDER BY p.project_id ASC, p.ordinal ASC
        """,
        project_ids,
    )


# --------------------------------------------------------------- util
def _json(value: Any) -> str:
    import json
    return json.dumps(value, ensure_ascii=False)
