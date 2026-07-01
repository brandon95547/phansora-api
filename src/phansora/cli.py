"""Unified Phansora command-line entry point.

    phansora serve [--host H] [--port P]   Run the unified API (uvicorn).
    phansora tts   [...]                    SpokenVerse batch TTS / PDF->TXT CLI.
    phansora dossier [...]                  Dossier Nova pipeline CLI.

Everything after the subcommand is forwarded verbatim to that product's own
argument parser, so their existing flags keep working unchanged.
"""
from __future__ import annotations

import sys
from typing import List, Optional


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

    if command == "dossier":
        from phansora.products.dossier_nova.main import main as dossier_main

        return dossier_main(rest) or 0

    print(f"Unknown command: {command}\n")
    print(__doc__)
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
