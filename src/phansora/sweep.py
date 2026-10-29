"""The daily sweep: files on this API's disk that no record names.

Every product deletes its own files when something is deleted: a course, a picture, a
SpokenVerse generation. This is the backstop for what that misses: a delete whose file
step failed, a render whose caller went away, a folder from before a product cleaned up
after itself, an account that was removed. So the disk holds what the database says
exists and nothing else, however many users there are.

What goes, under the data root (PHANSORA_DATA_DIR):

  output_audio/<user>/<file>        not one of the user's SpokenVerse generations
  output_audio/<user>/.tmp_chunks/  a render's pieces, long after the render
  book_alchemy/<user>/<course>/     the course is gone
  talari/images/<user>/<picture>    no Talari row of the user's mentions it
  talari/sources/<user>/<import>    the same, with the pictures of its pages
  voices/<user>/, history/<user>/   the account is gone
  .tmp_uploads/*                    a request's scratch, long after the request

Nothing younger than a day goes, or a week for Talari (a picture is on disk before the
app records it, and a book can stay open for days), so a file is never taken from under
the request still making it. Everything else on the disk is left alone.

The API runs it once a day (start). By hand:

    phansora sweep --dry-run          list what would go
    phansora sweep --move-to DIR      move it under DIR instead of deleting it
    phansora sweep                    delete it
"""
from __future__ import annotations

import asyncio
import logging
import os
import re
import shutil
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

from phansora.shared.paths import runtime_root

log = logging.getLogger("phansora.sweep")

HOUR = 3600
DAY = 24 * HOUR
TALARI_GRACE = 7 * DAY
# When the last sweep ran, as this file's time. Counted from here rather than from start:
# a release restarts the API, several times on a busy day, and a day counted from each
# start would never come.
STAMP = ".sweep-last-run"
# Talari users are asked about this many at a time: one query per user would be ten
# thousand round trips a day at ten thousand accounts, one for all of them a single
# answer of every name they have.
BATCH = 500

USER_DIR = re.compile(r"^\d+$")
TALARI_IMAGE = re.compile(r"^[0-9a-f]{32}\.(png|jpg)$")
TALARI_ORIGINAL = re.compile(r"^[0-9a-f]{64}\.(pdf|epub|docx|txt|md)$")
TALARI_PAGE = re.compile(r"^([0-9a-f]{64})-\d+\.png$")
# A Talari file name of either shape, wherever it sits in a row's text.
TALARI_NAME_PATTERN = r"([0-9a-f]{64}\.(?:pdf|epub|docx|txt|md)|[0-9a-f]{32}\.(?:png|jpg))"


@dataclass
class Records:
    """What the database says exists, as far as the files here go."""

    users: set[str] = field(default_factory=set)
    generations: set[tuple[str, str]] = field(default_factory=set)  # (user, audio file)
    courses: set[tuple[str, str]] = field(default_factory=set)  # (user, course)
    talari: dict[str, set[str]] = field(default_factory=dict)  # user -> names their rows mention


# ── The rules: what on disk no record names ──────────────────────────────────────────


def _old(path: Path, now: float, age: float) -> bool:
    try:
        return now - path.lstat().st_mtime >= age
    except FileNotFoundError:
        return False


def _dirs(base: Path) -> Iterator[Path]:
    if base.is_dir():
        for p in base.iterdir():
            if p.is_dir() and not p.is_symlink():
                yield p


def _user_dirs(base: Path) -> Iterator[Path]:
    return (d for d in _dirs(base) if USER_DIR.match(d.name))


def spokenverse_audio(root: Path, records: Records, now: float) -> Iterator[Path]:
    for user in _dirs(root / "output_audio"):
        for p in user.iterdir():
            if p.name == ".tmp_chunks" and p.is_dir():
                yield from (c for c in p.iterdir() if _old(c, now, DAY))
            elif p.is_file() and (user.name, p.name) not in records.generations and _old(p, now, DAY):
                yield p


def book_alchemy(root: Path, records: Records, now: float) -> Iterator[Path]:
    for user in _user_dirs(root / "book_alchemy"):
        for course in _dirs(user):
            if (user.name, course.name) not in records.courses and _old(course, now, DAY):
                yield course


def talari(root: Path, records: Records, now: float) -> Iterator[Path]:
    for user in _user_dirs(root / "talari" / "images"):
        named = records.talari.get(user.name, set())
        for p in user.iterdir():
            if p.is_file() and TALARI_IMAGE.match(p.name) and p.name not in named and _old(p, now, TALARI_GRACE):
                yield p
    for user in _user_dirs(root / "talari" / "sources"):
        named = records.talari.get(user.name, set())
        for p in user.iterdir():
            if p.is_file() and TALARI_ORIGINAL.match(p.name) and p.name not in named and _old(p, now, TALARI_GRACE):
                yield p
        # A page's picture goes with its import.
        stems = {n.split(".", 1)[0] for n in named}
        pages = user / "pages"
        if pages.is_dir():
            for shot in pages.iterdir():
                m = TALARI_PAGE.match(shot.name)
                if m and shot.is_file() and m.group(1) not in stems and _old(shot, now, TALARI_GRACE):
                    yield shot


def gone_accounts(root: Path, records: Records, now: float) -> Iterator[Path]:
    for base in ("voices", "history"):
        for user in _user_dirs(root / base):
            if user.name not in records.users and _old(user, now, DAY):
                yield user


def scratch(root: Path, records: Records, now: float) -> Iterator[Path]:
    tmp = root / ".tmp_uploads"
    if tmp.is_dir():
        yield from (p for p in tmp.iterdir() if _old(p, now, DAY))


RULES = (spokenverse_audio, book_alchemy, talari, gone_accounts, scratch)


def plan(root: Path, records: Records, now: float) -> list[Path]:
    """Everything the sweep would remove, changing nothing."""
    return [p for rule in RULES for p in rule(root, records, now)]


def size_of(path: Path) -> int:
    try:
        if path.is_dir() and not path.is_symlink():
            return sum(f.lstat().st_size for f in path.rglob("*") if f.is_file())
        return path.lstat().st_size
    except FileNotFoundError:
        return 0


def apply(doomed: list[Path], root: Path, *, move_to: Optional[Path] = None, now: Optional[float] = None) -> int:
    """Remove (or move under ``move_to``, keeping their place below the root) what plan()
    found, then the per-user folders it left empty. Returns how many went."""
    gone = 0
    for path in doomed:
        try:
            if move_to is not None:
                dest = move_to / path.relative_to(root)
                dest.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(path), str(dest))
            elif path.is_dir() and not path.is_symlink():
                shutil.rmtree(path)
            else:
                path.unlink(missing_ok=True)
            gone += 1
        except OSError:
            log.warning("Sweep: could not remove %s", path, exc_info=True)
    if move_to is None:
        _prune_empty(root, time.time() if now is None else now)
    return gone


def _prune_empty(root: Path, now: float) -> None:
    """Per-user folders left empty, once nothing has been written to them for a day (a
    render recreates its own, so this never races one)."""
    for base in ("output_audio", "book_alchemy", "talari/images", "talari/sources"):
        for user in _user_dirs(root / base):
            for d in (user / ".tmp_chunks", user / "pages", user):
                try:
                    if d.is_dir() and _old(d, now, DAY) and not any(d.iterdir()):
                        d.rmdir()
                except OSError:
                    log.debug("Sweep: could not prune %s", d, exc_info=True)


# ── The records ──────────────────────────────────────────────────────────────────────


async def load_records(conn, root: Path) -> Records:
    rec = Records()
    rec.users = {str(r["id"]) for r in await conn.fetch("SELECT id FROM public.users")}
    if not rec.users:
        # A database with no accounts is the wrong database, not a site with no one on it,
        # and with nothing named everything would go.
        raise RuntimeError("The users table is empty: not sweeping against this database.")
    rec.generations = {
        (str(r["fastapi_user_id"]), str(r["audio_filename"]))
        for r in await conn.fetch("SELECT fastapi_user_id, audio_filename FROM public.spokenverse_generations")
    }
    rec.courses = {
        (str(r["user_id"]), str(r["id"]))
        for r in await conn.fetch("SELECT user_id, id FROM public.book_alchemy_projects")
    }
    users = sorted(
        {d.name for base in ("images", "sources") for d in _user_dirs(root / "talari" / base)}, key=int
    )
    rec.talari = await _talari_names(conn, users)
    return rec


async def _talari_names(conn, users: list[str]) -> dict[str, set[str]]:
    """Every file name each user's Talari rows mention, in any column of any talari_ table,
    so a table or column added later is covered without this module knowing of it. Wider
    than what the app's delete reads (src/talari/repo.js unusedApiFiles) on purpose: a file
    something still mentions is kept."""
    tables = await conn.fetch(
        """SELECT table_name, bool_or(column_name = 'user_id') AS by_user
             FROM information_schema.columns
            WHERE table_schema = 'public' AND table_name LIKE 'talari\\_%'
              AND column_name IN ('user_id', 'project_id')
            GROUP BY table_name"""
    )
    names: dict[str, set[str]] = {u: set() for u in users}
    for start in range(0, len(users), BATCH):
        batch = [int(u) for u in users[start:start + BATCH]]
        for t in tables:
            table = t["table_name"].replace('"', '""')
            owner, join = ("t.user_id", "") if t["by_user"] else (
                "p.user_id", "JOIN public.talari_projects p ON p.id = t.project_id")
            rows = await conn.fetch(
                f"""SELECT DISTINCT {owner}::text AS u, m[1] AS name
                      FROM public."{table}" t {join}, regexp_matches(t::text, $2, 'g') m
                     WHERE {owner} = ANY($1::bigint[])""",
                batch, TALARI_NAME_PATTERN,
            )
            for r in rows:
                names.setdefault(r["u"], set()).add(r["name"])
    return names


async def _connect():
    import asyncpg

    # The same settings as every other connection to the shared database (book_alchemy/db.py).
    return await asyncpg.connect(
        user=os.getenv("DB_USER"),
        password=os.getenv("DB_PASSWORD"),
        database=os.getenv("DB_NAME"),
        host=os.getenv("DB_HOST", "127.0.0.1"),
        port=int(os.getenv("DB_PORT", "5432")),
    )


# ── Running it ───────────────────────────────────────────────────────────────────────


async def sweep(
    *, root: Optional[Path] = None, dry_run: bool = False, move_to: Optional[Path] = None,
    now: Optional[float] = None,
) -> list[Path]:
    """Sweep once. Returns what went, or with ``dry_run`` what would have."""
    root = Path(root or runtime_root())
    now = time.time() if now is None else now
    conn = await _connect()
    try:
        records = await load_records(conn, root)
    finally:
        await conn.close()
    doomed = await asyncio.to_thread(plan, root, records, now)
    if not dry_run and doomed:
        await asyncio.to_thread(apply, doomed, root, move_to=move_to, now=now)
        for path in doomed:
            log.info("Sweep: removed %s", path.relative_to(root))
    return doomed


async def run_daily(root: Optional[Path] = None) -> None:
    """Sweep once a day, counted from the last sweep (STAMP). The first check is an hour
    after start, clear of a release's warm-up; a first start begins the count."""
    root = Path(root or runtime_root())
    stamp = root / STAMP
    while True:
        await asyncio.sleep(HOUR)
        try:
            if not stamp.exists():
                stamp.touch()
                continue
            if time.time() - stamp.stat().st_mtime < DAY:
                continue
            stamp.touch()
            gone = await sweep(root=root)
            log.info("Sweep: %d files and folders no record names removed", len(gone))
        except Exception:  # noqa: BLE001 — tomorrow's sweep tries again
            log.warning("Sweep failed", exc_info=True)


def start() -> Optional[asyncio.Task]:
    """The daily sweep, in this process. Not where there is no database to sweep against
    (development, the tests), or with PHANSORA_SWEEP=off."""
    if not os.getenv("DB_NAME") or os.getenv("PHANSORA_SWEEP", "on").strip().lower() in {"0", "off", "false", "no"}:
        return None
    return asyncio.create_task(run_daily(), name="phansora-sweep")
