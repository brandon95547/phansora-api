#!/usr/bin/env python3
"""Preflight check for the system dependencies pip cannot install.

`pip install -r requirements.txt` only covers PyPI. Several features here are Python
wrappers around SYSTEM programs — pytesseract needs the `tesseract` binary, the audio
pipeline needs `ffmpeg`/`ffprobe` — and a host that skipped them installs perfectly, boots
perfectly, and then fails the first time a user touches that feature. That is how a scanned
PDF ended up failing in production with "tesseract is not installed or it's not in your
PATH": nothing checked until a real book hit the worker.

Run this after provisioning a box (`make doctor`) to find those gaps at deploy time
instead. Exits non-zero if anything REQUIRED is missing, so it can gate a deploy.

Stdlib only, no imports from `phansora`, and deliberately written for OLD Pythons (3.6+):
it has to run on a bare box before the 3.10 venv exists, and prod's system `python3` is
3.6. So: no `from __future__ import annotations`, and no `subprocess.run(capture_output=)`
or `text=` (both 3.7+).
"""

import shutil
import subprocess
import sys
from typing import Callable, List, NamedTuple, Optional, Tuple


class Check(NamedTuple):
    name: str
    powers: str                       # what breaks without it
    required: bool
    probe: Callable[[], Tuple[bool, str]]
    install: dict                     # distro -> command


def _distro() -> str:
    """rhel | debian | macos | unknown — picks which install hint to print."""
    if sys.platform == "darwin":
        return "macos"
    try:
        with open("/etc/os-release", encoding="utf-8") as fh:
            info = dict(
                line.rstrip("\n").split("=", 1)
                for line in fh if "=" in line
            )
    except OSError:
        return "unknown"
    ident = (info.get("ID", "") + " " + info.get("ID_LIKE", "")).replace('"', "").lower()
    if any(k in ident for k in ("rhel", "centos", "fedora")):
        return "rhel"
    if any(k in ident for k in ("debian", "ubuntu")):
        return "debian"
    return "unknown"


def _run(cmd, timeout=15):
    # stdout=/stderr=PIPE + universal_newlines, not capture_output=/text= — those are 3.7+.
    try:
        out = subprocess.run(
            cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
            universal_newlines=True, timeout=timeout,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return (out.stdout or "") + (out.stderr or "")


def _binary(name: str, version_args: List[str]) -> Tuple[bool, str]:
    path = shutil.which(name)
    if not path:
        return False, "not on PATH"
    out = _run([name, *version_args]) or ""
    first = next((ln.strip() for ln in out.splitlines() if ln.strip()), "")
    return True, f"{path} ({first[:60]})" if first else path


def check_tesseract() -> Tuple[bool, str]:
    ok, detail = _binary("tesseract", ["--version"])
    if not ok:
        return False, detail
    # The engine alone is not enough: on RHEL the base package ships an EMPTY tessdata dir,
    # so OCR installs "successfully" and then dies on the first page with
    # "Error opening data file .../eng.traineddata".
    langs = _run(["tesseract", "--list-langs"]) or ""
    have = {ln.strip() for ln in langs.splitlines() if ln.strip() and " " not in ln.strip()}
    if "eng" not in have:
        return False, f"{detail} — installed, but the 'eng' language pack is MISSING"
    return True, f"{detail}, langs: {','.join(sorted(have))[:40]}"


CHECKS: List[Check] = [
    Check(
        "tesseract", "OCR for scanned PDFs (Book Alchemy, SpokenVerse)", True, check_tesseract,
        {
            "rhel": "dnf install -y tesseract tesseract-langpack-eng",
            "debian": "apt install -y tesseract-ocr tesseract-ocr-eng",
            "macos": "brew install tesseract",
        },
    ),
    Check(
        "ffmpeg", "audio concat / transcode / loudness (every render)", True,
        lambda: _binary("ffmpeg", ["-version"]),
        {"rhel": "dnf install -y ffmpeg", "debian": "apt install -y ffmpeg", "macos": "brew install ffmpeg"},
    ),
    Check(
        "ffprobe", "media duration probing", True,
        lambda: _binary("ffprobe", ["-version"]),
        {"rhel": "dnf install -y ffmpeg", "debian": "apt install -y ffmpeg", "macos": "brew install ffmpeg"},
    ),
]

GREEN, RED, YELLOW, DIM, RESET = "\033[32m", "\033[31m", "\033[33m", "\033[2m", "\033[0m"


def main() -> int:
    distro = _distro()
    print(f"System dependencies (pip does not install these) — detected: {distro}\n")

    missing: List[Check] = []
    for chk in CHECKS:
        try:
            ok, detail = chk.probe()
        except Exception as exc:  # noqa: BLE001 — a probe must never abort the report
            ok, detail = False, f"probe failed: {exc}"
        mark = f"{GREEN}ok  {RESET}" if ok else (f"{RED}FAIL{RESET}" if chk.required else f"{YELLOW}warn{RESET}")
        print(f"  [{mark}] {chk.name:<10} {detail}")
        print(f"         {DIM}{chk.powers}{RESET}")
        if not ok:
            missing.append(chk)

    if not missing:
        print(f"\n{GREEN}All system dependencies present.{RESET}")
        return 0

    print(f"\n{RED}Missing:{RESET}")
    for chk in missing:
        cmd = chk.install.get(distro) or next(iter(chk.install.values()))
        print(f"  {chk.name:<10} {cmd}")
    blocking = [c for c in missing if c.required]
    print()
    return 1 if blocking else 0


if __name__ == "__main__":
    sys.exit(main())
