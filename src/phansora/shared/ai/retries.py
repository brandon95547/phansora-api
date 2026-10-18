"""Which failed provider calls are worth making again.

A retry is only worth its wait when the next attempt can come out differently. A
timeout, a dropped connection, a 429 or a 5xx can: the provider was busy or briefly
broken. A 401 (bad key), 402 (no credit), 403 (no access) or 400 (a request it will
never accept) cannot, and retrying one with backoff only delays the real error — and,
in the research clients, used to bury it inside tenacity's RetryError.

Kept import-light (stdlib only) so every client can use it whatever HTTP library it
speaks; each passes its own transport-error types to ``worth_retrying``.
"""
from __future__ import annotations

from typing import Optional, Tuple, Type


class ProviderError(RuntimeError):
    """A provider call that failed, carrying the HTTP status when there was one.

    A RuntimeError, so callers that already catch RuntimeError around a provider call
    still do; ``status`` is there so they no longer have to read it out of the message.
    """

    def __init__(self, message: str, *, status: Optional[int] = None) -> None:
        super().__init__(message)
        self.status = status


def retryable_status(status: Optional[int]) -> bool:
    """Whether a provider's HTTP status might be different on the next attempt."""
    if status is None:
        return False
    return status in (408, 429) or 500 <= status < 600


def worth_retrying(exc: BaseException, transport_errors: Tuple[Type[BaseException], ...]) -> bool:
    """True for an HTTP error with a retryable status, or for one of ``transport_errors``
    (the calling client's timeouts and connection failures)."""
    status = getattr(exc, "status", None)
    if status is None:
        status = getattr(exc, "status_code", None)
    if isinstance(status, int):
        return retryable_status(status)
    return isinstance(exc, transport_errors)
