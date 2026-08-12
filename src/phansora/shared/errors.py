"""Turning an exception into a response without handing the caller our internals.

Endpoints used to interpolate the exception straight into the response detail:

    raise HTTPException(status_code=500, detail=f"TXT->Audio failed: {e}")

which shipped absolute filesystem paths, database errors and — via the DeepSeek
clients, which put the upstream body in the exception message — whole provider
error payloads to whoever made the request.

`fail()` keeps both halves where they belong: the caller gets a fixed sentence
plus a short reference, and the log gets that reference alongside the traceback.
Support can still join the two, which is the only reason the detail was ever
useful to a human in the first place.
"""
from __future__ import annotations

import logging
import secrets

from fastapi import HTTPException


def fail(
    status_code: int,
    message: str,
    exc: BaseException | None = None,
    *,
    logger: logging.Logger,
    context: str | None = None,
) -> HTTPException:
    """Build the HTTPException to raise, logging `exc` against a reference id.

    Returned rather than raised so call sites keep `raise ... from e`, which is
    what preserves the cause chain in the traceback.
    """
    ref = secrets.token_hex(4)
    if exc is not None:
        logger.exception("%s [ref=%s]", context or message, ref)
    else:
        logger.error("%s [ref=%s]", context or message, ref)
    return HTTPException(status_code=status_code, detail=f"{message} (reference {ref})")
