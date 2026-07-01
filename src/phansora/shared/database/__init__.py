"""Shared database access."""
from .postgres import close_pool, get_pool

__all__ = ["get_pool", "close_pool"]
