"""requirements-ci.txt is the suite's dependency set; requirements.txt is prod's.

CI cannot install requirements.txt (its torch wheels are cp310 CUDA builds), so it
installs a second list. The danger of a second list is that it tests against versions
prod does not run, so every package pinned in both has to be pinned to the same version.
"""
from __future__ import annotations

import re
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
_PIN = re.compile(r"^([A-Za-z0-9_.-]+)(?:\[[^\]]*\])?==([^\s;#]+)")


def pins(name: str) -> dict[str, str]:
    out = {}
    for raw in (ROOT / name).read_text(encoding="utf-8").splitlines():
        found = _PIN.match(raw.strip())
        if found:
            out[re.sub(r"[-_.]+", "-", found.group(1)).lower()] = found.group(2)
    return out


def test_ci_tests_the_versions_prod_runs():
    prod, ci = pins("requirements.txt"), pins("requirements-ci.txt")
    drifted = {k: (ci[k], prod[k]) for k in ci.keys() & prod.keys() if ci[k] != prod[k]}
    assert not drifted, f"requirements-ci.txt and requirements.txt disagree (ci, prod): {drifted}"


def test_ci_pins_everything_it_installs():
    lines = [
        line.split("#", 1)[0].strip()
        for line in (ROOT / "requirements-ci.txt").read_text(encoding="utf-8").splitlines()
    ]
    loose = [line for line in lines if line and not line.startswith("-r") and "==" not in line]
    assert not loose, f"unpinned in requirements-ci.txt: {loose}"
