"""The course download is streamed, and its length is promised in advance.

Sending Content-Length turns "12 MB downloaded" into a progress bar that fills, and it is
only possible because the entries are STORED rather than deflated: a stored archive's size
is arithmetic — 30 bytes of local header and 46 of central directory per entry, both plus
the name, a 16-byte data descriptor, the file, and 22 bytes to close.

That arithmetic is a promise to the client. If it is wrong the browser hangs waiting for
bytes that never come, or truncates the file. So two things have to hold, and neither is
guaranteed by anything in this repo:

  * zipfile has to keep emitting exactly those headers for an unseekable stream. A future
    Python that added a zip64 extra field by default would break every download silently,
    and this is the only place that would notice.
  * the compression mode has to stay STORED. Deflate makes the size unknowable until the
    bytes are written, so switching it back is not a slower download — it is a broken one.
"""

from __future__ import annotations

import os
import zipfile

import pytest

from phansora.products.book_alchemy import server as ba_server

pytestmark = pytest.mark.skipif(
    not getattr(ba_server, "_BOOK_ALCHEMY_OK", False),
    reason="book alchemy is not enabled in this environment",
)


def _predict(entries) -> int:
    """The formula the download route promises as Content-Length."""
    total = 22
    for arcname, _path, size in entries:
        n = len(arcname.encode("utf-8"))
        total += 30 + n + size + 16 + 46 + n
    return total


@pytest.fixture()
def tracks(tmp_path):
    """A handful of files standing in for audio, with names that are not plain ASCII."""
    made = []
    for i in range(1, 5):
        p = tmp_path / f"track{i}.bin"
        p.write_bytes(os.urandom(20_000 + i * 1_311))  # incompressible, like mp3
        made.append((f"{i:02d} - Résumé Träck {i}.mp3", p, p.stat().st_size))
    return made


def _collect(entries, narration: bytes) -> bytes:
    return b"".join(ba_server._ba_stream_zip(entries, narration))


def test_streamed_length_is_exactly_what_was_promised(tracks):
    narration = ("Course\n\n" + "a line of narration\n" * 200).encode("utf-8")
    entries = list(tracks) + [("Course — Narration.txt", None, len(narration))]
    body = _collect(entries, narration)
    assert len(body) == _predict(entries), (
        "Content-Length would not match the body: the browser would hang or truncate"
    )


def test_the_archive_is_valid_and_round_trips(tracks, tmp_path):
    narration = b"narration\n"
    entries = list(tracks) + [("Course — Narration.txt", None, len(narration))]
    out = tmp_path / "course.zip"
    out.write_bytes(_collect(entries, narration))

    with zipfile.ZipFile(out) as zf:
        assert zf.testzip() is None
        assert len(zf.namelist()) == len(entries)
        # Names keep their accents rather than arriving mangled.
        assert entries[0][0] in zf.namelist()
        # Every track comes back byte for byte.
        for arcname, path, _size in tracks:
            assert zf.read(arcname) == path.read_bytes()
        assert zf.read("Course — Narration.txt") == narration


def test_entries_are_stored_not_deflated(tracks):
    """Deflate would make the promised length wrong, not merely slower."""
    narration = b""
    entries = list(tracks)
    out = _collect(entries, narration)
    with zipfile.ZipFile(__import__("io").BytesIO(out)) as zf:
        for info in zf.infolist():
            assert info.compress_type == zipfile.ZIP_STORED


def test_deflate_would_break_the_promised_length(tracks, tmp_path):
    """The negative case, so the reason STORED matters cannot be optimized away by accident."""
    entries = list(tracks)
    deflated = tmp_path / "d.zip"
    with zipfile.ZipFile(deflated, "w", zipfile.ZIP_DEFLATED) as zf:
        for arcname, path, _size in entries:
            zf.write(path, arcname=arcname)
    assert deflated.stat().st_size != _predict(entries)
