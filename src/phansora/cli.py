"""Unified Phansora command-line entry point.

    phansora serve [--host H] [--port P]   Run the unified API (uvicorn).
    phansora tts   [...]                    SpokenVerse batch TTS / PDF->TXT CLI.
    phansora sweep [--dry-run] [--move-to DIR]
                                            Remove files no record names (phansora/sweep.py).

Everything after the subcommand is forwarded verbatim to that product's own
argument parser, so their existing flags keep working unchanged.
"""
from __future__ import annotations

import sys
from typing import List, Optional

# Loads .env, once, before any product module reads the environment — the same import
# that does it for the API in phansora.main.
import phansora.config  # noqa: F401


def _forward(argv: List[str], prog: str) -> None:
    """Reset sys.argv so a delegate that reads it directly sees only its args."""
    sys.argv = [prog, *argv]


def main(argv: Optional[List[str]] = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    if not argv or argv[0] in {"-h", "--help", "help"}:
        print(__doc__)
        return 0

    command, rest = argv[0], argv[1:]

    if command == "serve":
        import argparse

        from phansora.config import settings

        parser = argparse.ArgumentParser(prog="phansora serve")
        parser.add_argument("--host", default=settings.host)
        parser.add_argument("--port", type=int, default=settings.port)
        parser.add_argument("--reload", action="store_true")
        opts = parser.parse_args(rest)

        import uvicorn

        uvicorn.run(
            "phansora.main:app", host=opts.host, port=opts.port, reload=opts.reload
        )
        return 0

    if command == "tts":
        from phansora.products.spokenverse.txt_to_voice.cli import main as tts_main

        _forward(rest, "phansora-tts")
        tts_main()
        return 0

    if command == "sweep":
        import argparse
        import asyncio
        from pathlib import Path

        from phansora import sweep

        parser = argparse.ArgumentParser(prog="phansora sweep")
        parser.add_argument("--dry-run", action="store_true", help="list what would go; change nothing")
        parser.add_argument("--move-to", type=Path, help="move it under this folder instead of deleting it")
        opts = parser.parse_args(rest)

        root = sweep.runtime_root()
        # Planned first, measured, then removed: a size read after the removal is nothing.
        found = asyncio.run(sweep.sweep(root=root, dry_run=True))
        total = 0
        for path in found:
            size = sweep.size_of(path)
            total += size
            print(f"{size / 1e6:9.1f} MB  {path.relative_to(root)}")
        if not opts.dry_run:
            sweep.apply(found, root, move_to=opts.move_to)
        verb = "would go" if opts.dry_run else (f"moved to {opts.move_to}" if opts.move_to else "removed")
        print(f"{len(found)} files and folders, {total / 1e6:.1f} MB, {verb}.")
        return 0

    print(f"Unknown command: {command}\n")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
