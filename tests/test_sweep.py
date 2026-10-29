"""The daily sweep: what goes, what stays, and when it refuses to run.

Found on 2026-10-29: 247 Talari pictures (252 MB) that no row named any more, 112
narrations kept by the voice service after Narrava had taken its own copy, Book Alchemy
lessons left in output_audio/, and scratch folders from July. The rules are tested on a
data folder laid out as the server's is; the database is a stand-in.
"""
from __future__ import annotations

import asyncio
import os
import time

import pytest

from phansora import sweep
from phansora.sweep import DAY, Records

NOW = time.time()
OLD = NOW - 30 * DAY
A, B, C = "a" * 32 + ".png", "b" * 32 + ".png", "c" * 32 + ".png"
PDF_KEPT, PDF_GONE = "1" * 64 + ".pdf", "2" * 64 + ".pdf"


def _file(root, rel, *, age=OLD):
    path = root / rel
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(b"x" * 10)
    os.utime(path, (age, age))
    return path


def _age(root, rel, age=OLD):
    os.utime(root / rel, (age, age))


@pytest.fixture
def data(tmp_path):
    root = tmp_path
    _file(root, "output_audio/1/kept.mp3")
    _file(root, "output_audio/1/narration.mp3")
    _file(root, "output_audio/1/just-made.mp3", age=NOW)
    _file(root, "output_audio/1/.tmp_chunks/session/chunk_0001.wav")
    _age(root, "output_audio/1/.tmp_chunks/session")
    _file(root, "output_audio/9/session.mp3")
    _file(root, "book_alchemy/5/52/lesson.mp3")
    _file(root, "book_alchemy/5/40/lesson.mp3")
    _age(root, "book_alchemy/5/40")
    _file(root, "book_alchemy/5/41/lesson.mp3", age=NOW)
    _file(root, f"talari/images/1/{A}")
    _file(root, f"talari/images/1/{B}")
    _file(root, f"talari/images/1/{C}", age=NOW - 3 * DAY)
    _file(root, f"talari/images/cutover-test/{B}")
    _file(root, f"talari/sources/1/{PDF_KEPT}")
    _file(root, f"talari/sources/1/{PDF_GONE}")
    _file(root, f"talari/sources/1/pages/{PDF_KEPT[:64]}-1.png")
    _file(root, f"talari/sources/1/pages/{PDF_GONE[:64]}-1.png")
    _file(root, "voices/3/voices.json")
    _age(root, "voices/3")
    _file(root, "voices/99/voices.json")
    _age(root, "voices/99")
    _file(root, "voices/_defaults/voices.json")
    _file(root, "history/99/events.jsonl")
    _age(root, "history/99")
    _file(root, ".tmp_uploads/in_old/session.txt")
    _age(root, ".tmp_uploads/in_old")
    _file(root, ".tmp_uploads/in_new/session.txt", age=NOW)
    _file(root, "data/dossier_nova/embeddings.db")
    return root


RECORDS = Records(
    users={"1", "3", "5"},
    generations={("1", "kept.mp3")},
    courses={("5", "52")},
    talari={"1": {A, PDF_KEPT}},
)


def _planned(root):
    return sorted(p.relative_to(root).as_posix() for p in sweep.plan(root, RECORDS, NOW))


def test_what_no_record_names_goes_and_nothing_else(data):
    assert _planned(data) == sorted([
        "output_audio/1/narration.mp3",                   # Narrava's, kept by the API too
        "output_audio/1/.tmp_chunks/session",             # an old render's pieces
        "output_audio/9/session.mp3",                     # Book Alchemy's, outside its course
        "book_alchemy/5/40",                              # a deleted course
        f"talari/images/1/{B}",                           # a picture of a deleted book
        f"talari/sources/1/{PDF_GONE}",                   # an import no book has
        f"talari/sources/1/pages/{PDF_GONE[:64]}-1.png",  # ...and its page's picture
        "voices/99",                                      # an account that is gone
        "history/99",
        ".tmp_uploads/in_old",
    ])


def test_young_files_are_never_taken(data):
    planned = _planned(data)
    assert "output_audio/1/just-made.mp3" not in planned     # its history row is seconds away
    assert "book_alchemy/5/41" not in planned
    assert f"talari/images/1/{C}" not in planned             # inside the week a book may stay open
    assert ".tmp_uploads/in_new" not in planned


def test_folders_that_are_not_a_users_are_left_alone(data):
    planned = _planned(data)
    assert not any(p.startswith(("talari/images/cutover-test", "voices/_defaults", "data/")) for p in planned)


def test_removing_and_pruning_what_it_leaves_empty(data):
    doomed = sweep.plan(data, RECORDS, NOW)
    assert sweep.apply(doomed, data, now=NOW) == len(doomed)
    assert not (data / "book_alchemy/5/40").exists()
    assert (data / "output_audio/1/kept.mp3").exists()
    assert (data / f"talari/images/1/{A}").exists()
    # output_audio/9 held only the lesson. Emptied just now, it stays (a render may be about
    # to write to it); once it has been quiet for a day it goes.
    assert (data / "output_audio/9").is_dir()
    _age(data, "output_audio/9", NOW - 2 * DAY)
    sweep.apply([], data, now=NOW)
    assert not (data / "output_audio/9").exists()


def test_moving_instead_of_deleting_keeps_each_files_place(data, tmp_path_factory):
    aside = tmp_path_factory.mktemp("aside")
    doomed = sweep.plan(data, RECORDS, NOW)
    sweep.apply(doomed, data, move_to=aside)
    assert (aside / f"talari/images/1/{B}").read_bytes() == b"x" * 10
    assert (aside / "book_alchemy/5/40/lesson.mp3").exists()
    assert not (data / f"talari/images/1/{B}").exists()


class _Conn:
    """A stand-in database: answers by the table a query reads."""

    def __init__(self, users):
        self.users = users
        self.batches = []

    async def fetch(self, sql, *args):
        if "FROM public.users" in sql:
            return [{"id": u} for u in self.users]
        if "spokenverse_generations" in sql:
            return [{"fastapi_user_id": "1", "audio_filename": "kept.mp3"}]
        if "book_alchemy_projects" in sql:
            return [{"user_id": 5, "id": 52}]
        if "information_schema" in sql:
            return [{"table_name": "talari_projects", "by_user": True},
                    {"table_name": "talari_illustrations", "by_user": False}]
        if "regexp_matches" in sql:
            self.batches.append(args[0])
            return [{"u": "1", "name": A}] if "talari_illustrations" in sql else []
        raise AssertionError(sql)


def test_records_are_read_in_batches_of_users(data, monkeypatch):
    monkeypatch.setattr(sweep, "BATCH", 1)
    _file(data, f"talari/images/2/{A}")
    conn = _Conn(users=[1, 2, 3, 5])
    rec = asyncio.run(sweep.load_records(conn, data))
    assert rec.generations == {("1", "kept.mp3")}
    assert rec.courses == {("5", "52")}
    assert rec.talari == {"1": {A}, "2": set()}
    # Users 1 and 2 have Talari folders: one batch each, two tables per batch.
    assert conn.batches == [[1], [1], [2], [2]]


def test_an_empty_users_table_stops_the_sweep(data):
    # A wrong database names nothing, and against it everything would go.
    with pytest.raises(RuntimeError, match="users table is empty"):
        asyncio.run(sweep.load_records(_Conn(users=[]), data))


def test_it_does_not_start_without_a_database(monkeypatch):
    monkeypatch.delenv("DB_NAME", raising=False)
    assert sweep.start() is None
    monkeypatch.setenv("DB_NAME", "phansora")
    monkeypatch.setenv("PHANSORA_SWEEP", "off")
    assert sweep.start() is None
