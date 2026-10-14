"""Errors that reach the caller with a message they can act on."""

from __future__ import annotations


class ServiceError(Exception):
    """ComfyUI could not do what was asked.

    Never swallowed or replaced with stand-in output: a failed picture surfaces as a
    failure, with this message. The messages are written here (comfy.py), not passed
    through from ComfyUI, so they carry no internals.
    """

    def __init__(self, message: str, status: int = 502) -> None:
        super().__init__(message)
        self.message = message
        self.status = status


class NotConfigured(ServiceError):
    """ComfyUI is not there yet, or not set up: a 503, which says "try again later"."""

    def __init__(self, message: str) -> None:
        super().__init__(message, status=503)
