"""The Chrono Origin trace cache.

Shared between users on purpose — a trace answers an objective question and a full run
is expensive — so the thing that has to hold is that it never answers a DIFFERENT
question with a stored one. The key used to be the title alone while the request also
carried context, depth, sources and language, so the first run of a title answered every
later variation of it and the form's controls were silently ignored.
"""
from __future__ import annotations

import time

import json

import pytest

from phansora.products.chrono_origin.services import cache as cache_mod
from phansora.products.chrono_origin.services.cache import (
    delete_cached,
    _cache_path,
    get_cached,
    normalize_title,
    request_key,
    save_cached,
)


@pytest.fixture(autouse=True)
def cache_dir(tmp_path, monkeypatch):
    """Point the cache at a temp dir, and defeat the settings lru_cache."""
    class _S:
        chrono_cache_dir = str(tmp_path)
        chrono_cache_ttl_days = 30

    monkeypatch.setattr(cache_mod, "get_settings", lambda: _S())
    return tmp_path


def key_for(title, **kw):
    return request_key(title, **kw)


def _trace(**kw):
    """A payload the cache will accept.

    Every fixture here carries a timeline, because save_cached now refuses to persist a
    trace without one — an empty timeline is a failed trace, and a failed trace served
    from disk for thirty days is how one bad run silently poisons a subject. These tests
    are about KEYING, so the timeline is the smallest thing that clears that bar.
    """
    return {"timeline": [{"id": "t1", "source_title": "A step"}], **kw}


def test_same_request_hits():
    k = key_for("Moses")
    save_cached("Moses", k, _trace(title="moses"))
    assert get_cached("Moses", k) == _trace(title="moses")


def test_title_alone_no_longer_decides():
    # The bug: these are different questions and used to share one answer.
    biblical = key_for("Moses", context="biblical figure")
    egyptian = key_for("Moses", context="Egyptian mythology")
    assert biblical != egyptian

    save_cached("Moses", biblical, _trace(which="biblical"))
    assert get_cached("Moses", egyptian) is None
    assert get_cached("Moses", biblical) == _trace(which="biblical")


@pytest.mark.parametrize("kw", [
    {"max_depth": 6},
    {"language": "fr"},
    {"context": "New Mexico, 1947"},
])
def test_every_result_changing_field_splits_the_key(kw):
    base = key_for("Roswell")
    assert key_for("Roswell", **kw) != base


def test_title_is_still_normalised_the_same_way():
    # Whitespace and punctuation must not fragment the cache for one real question.
    assert key_for("  The   Great Flood!  ") == key_for("the great flood")
    assert normalize_title("  The   Great Flood!  ") == "the great flood"


def test_delete_removes_every_variant_of_a_title():
    # Invalidation is by TITLE, because that is all the caller (the Node app, deleting a
    # user's trace) knows — so it has to clear the whole family, not one exact key.
    for kw in ({}, {"context": "biblical figure"}, {"max_depth": 6}):
        save_cached("Moses", key_for("Moses", **kw), _trace(k=str(kw)))
    assert delete_cached("Moses") == 3
    for kw in ({}, {"context": "biblical figure"}, {"max_depth": 6}):
        assert get_cached("Moses", key_for("Moses", **kw)) is None


def test_delete_leaves_other_titles_alone():
    save_cached("Moses", key_for("Moses"), _trace(a=1))
    save_cached("Pizzagate", key_for("Pizzagate"), _trace(b=2))
    delete_cached("Moses")
    assert get_cached("Pizzagate", key_for("Pizzagate")) == _trace(b=2)


def test_delete_is_idempotent():
    assert delete_cached("never-traced") == 0


def test_expired_entries_are_not_served(cache_dir, monkeypatch):
    k = key_for("Moses")
    save_cached("Moses", k, _trace(title="moses"))
    # Age the file past the TTL rather than waiting 30 days.
    path = next(cache_dir.glob("*-moses-*.json"))
    old = time.time() - (31 * 86400)
    import os
    os.utime(path, (old, old))
    assert get_cached("Moses", k) is None
    assert not path.exists()          # and it cleans up on the way past


def test_ttl_of_zero_disables_expiry(cache_dir, monkeypatch):
    class _S:
        chrono_cache_dir = str(cache_dir)
        chrono_cache_ttl_days = 0

    k = key_for("Moses")
    save_cached("Moses", k, _trace(title="moses"))
    monkeypatch.setattr(cache_mod, "get_settings", lambda: _S())
    path = next(cache_dir.glob("*-moses-*.json"))
    old = time.time() - (365 * 86400)
    import os
    os.utime(path, (old, old))
    assert get_cached("Moses", k) == _trace(title="moses")


def test_a_half_written_file_is_never_served(cache_dir):
    # save_cached writes to a temp file and moves it into place, so a reader cannot
    # observe a partial trace.
    save_cached("Moses", key_for("Moses"), _trace(title="moses"))
    assert list(cache_dir.glob("*.tmp")) == []


def test_delete_does_not_take_a_longer_title_with_it():
    """`moses` must not delete `moses parting the red sea`.

    The filenames are `{version}-{slug}-{digest}.json`, so a glob of `*-moses-*.json`
    matches BOTH — which is why deletion parses the slug out and compares it exactly.
    """
    save_cached("Moses", key_for("Moses"), _trace(a=1))
    save_cached("Moses parting the Red Sea", key_for("Moses parting the Red Sea"), _trace(b=2))

    assert delete_cached("Moses") == 1
    assert get_cached("Moses", key_for("Moses")) is None
    assert get_cached("Moses parting the Red Sea", key_for("Moses parting the Red Sea")) == _trace(b=2)


def test_schema_version_partitions_the_cache(cache_dir, monkeypatch):
    """A trace stored under an older pipeline shape is not served after a version bump.

    That is the guard against silently handing back a timeline with none of the fields a
    newer pipeline produces — it partitions rather than deletes, so a rollback still
    finds its own entries.
    """
    save_cached("Moses", key_for("Moses"), _trace(shape="old"))
    assert get_cached("Moses", key_for("Moses")) == _trace(shape="old")

    monkeypatch.setattr(cache_mod, "SCHEMA_VERSION", "v99")
    assert get_cached("Moses", key_for("Moses")) is None


def test_a_trace_with_no_timeline_is_never_cached():
    """The failure that made this necessary: a trace whose search returned nothing was
    synthesized into a well-formed "No research material was provided", stored as a
    success, and then served from cache for every later request of that title — so
    re-running the trace could not clear it. Nothing empty gets persisted now."""
    k = key_for("Jesus Christ")
    save_cached("Jesus Christ", k, {"timeline": [], "origin": {"summary": "No research material was provided."}})
    assert get_cached("Jesus Christ", k) is None


def test_an_already_poisoned_entry_heals_itself_on_read():
    """Entries written before save_cached learned to refuse them.

    A cache hit returns before every guard in the pipeline, so a stored empty trace is
    served straight back and re-running cannot clear it. Dropped on read instead, which
    needs no sweep and no manual invalidation.
    """
    k = key_for("Jesus Christ")
    path = _cache_path("Jesus Christ", k)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"timeline": [], "origin": {"summary": "Nothing to report."}}))

    assert get_cached("Jesus Christ", k) is None
    assert not path.exists(), "the poisoned entry was served again on the next read"
