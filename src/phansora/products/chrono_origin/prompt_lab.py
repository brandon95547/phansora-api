"""Try a Chrono-Origin prompt against the live provider, without deploying it.

    python -m phansora.products.chrono_origin.prompt_lab "Jesus Christ" -p my.txt

Prompt work on this product kept failing for a reason the app could not show: a
single run tells you almost nothing. Asked to trace backward, the same prompt on
the same model reached the material a subject descends from in roughly one run in
seven — so one good result "proved" a prompt that mostly did not work, and one bad
result condemned a prompt that mostly did. Every question worth asking here is a
question about a RATE, and answering it meant a deploy, a restart and a click
through the dashboard per sample.

So this runs a prompt N times, in parallel, and reports what actually varies:

  --expect REGEX   the fraction of runs whose answer matched. This is the metric.
                   Pick something only a correct answer would contain — the name of
                   an object you expect it to reach, not a word it says anyway.
  searches         how many queries the model chose to run. "Perform multiple
                   searches" is unfalsifiable and gets satisfied with three; a
                   number in the prompt moves this and is worth watching.
  low-auth         citations landing on forums, wikis and video. Largely NOT under
                   the prompt's control — grounding returns what the search engine
                   surfaced — so a prompt that bans Wikipedia will not change it.
  truncated        answers cut off at the output cap. Silent otherwise: a cut-off
                   answer is prose, and reads exactly like a short one.

Two prompts given, or more, are run against each other on the same subject.

``--full`` runs synthesis too and prints the timeline. Worth the extra time and
money, because the research call finding something is not the same as it reaching
the board: a corpus that named the Hebrew scriptures in three runs out of three
produced a timeline containing them in none of them.

Nothing here writes to the cache or to the trace store.
"""
from __future__ import annotations

import argparse
import os
import re
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import TraceRequest
from .pipeline import prompts as P
from .pipeline import source_policy as sp

def load_env(explicit: Optional[str] = None) -> Optional[str]:
    """Read .env if the provider is not already configured.

    The service gets its environment from systemd; a shell does not. Without this
    every invocation needs the key exported by hand, and a tool that is annoying to
    reach for is a tool nobody reaches for — which is how prompt work ended up being
    done by deploying instead.
    """
    if os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY"):
        return None
    candidates = [explicit] if explicit else [
        ".env", str(Path(__file__).resolve().parents[4] / ".env"),
        "/var/www/phansora-api/.env",
    ]
    for name in candidates:
        if not name or not Path(name).is_file():
            continue
        for raw in Path(name).read_text().splitlines():
            line = raw.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, value = line.split("=", 1)
            value = value.strip()
            if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
                value = value[1:-1]
            os.environ.setdefault(key.strip(), value)
        return name
    return None


# The file you edit between runs. Kept at the repo root rather than inside the
# package so editing it is never confused with editing the shipped prompt, and so a
# half-finished experiment cannot be imported by the service.
DEFAULT_PROMPT_FILE = "chrono-prompt.txt"


def default_prompt_path() -> Path:
    """Where the editable prompt lives: the working directory, else the repo root."""
    here = Path.cwd() / DEFAULT_PROMPT_FILE
    if here.is_file():
        return here
    return Path(__file__).resolve().parents[4] / DEFAULT_PROMPT_FILE


def ensure_prompt_file(path: Path) -> bool:
    """Seed the file from the shipped prompt the first time. Returns True if created.

    Starting from the live prompt rather than an empty file matters: the thing you
    almost always want is to change one part of what is deployed and measure the
    difference, and retyping 140 lines to do that is how people end up testing
    something subtly different from what is running.
    """
    if path.exists():
        return False
    path.write_text(P.RESEARCH_PROMPT)
    return True


LOW = {"low_authority", "reference_index"}
GOOD = {"repository", "academic", "primary", "institutional"}


def fill(template: str, subject: str, context: str = "") -> str:
    """Bind the subject into a prompt template.

    Deliberately ``str.replace`` and not ``str.format``: prompts here contain JSON
    examples full of braces, and a prompt written outside this repo has no reason to
    escape them. Formatting would raise on someone's perfectly good prompt file.
    """
    clause = f" ({context})" if context else ""
    return (template
            .replace("{title}", subject)
            .replace("[SEARCH TERM]", subject)   # tolerate the raw placeholder
            .replace("{context_clause}", clause))


@dataclass
class RunResult:
    ok: bool = False
    seconds: float = 0.0
    queries: List[str] = field(default_factory=list)
    citations: List[Dict[str, str]] = field(default_factory=list)
    text: str = ""
    timeline: List[str] = field(default_factory=list)
    origin: str = ""
    error: str = ""

    @property
    def tiers(self) -> List[str]:
        return [sp.default_tier(c.get("url") or "") for c in self.citations]

    def matched(self, pattern: Optional[re.Pattern]) -> bool:
        """Search the research text AND the timeline it produced.

        Both, always. Scoring only the corpus calls a run a success when the material
        was found and then dropped — which is the failure this tool was built to see.
        Scoring only the timeline misses that research-only runs have no timeline.
        """
        if pattern is None:
            return False
        hay = "\n".join([self.text, self.origin] + self.timeline)
        return bool(pattern.search(hay))


def _research(client, prompt: str) -> RunResult:
    started = time.time()
    answer = client.grounded_search(prompt)
    return RunResult(ok=True, seconds=time.time() - started, queries=list(answer.queries or []),
                     citations=list(answer.citations or []), text=answer.text or "")


def _full(prompt: str, subject: str, context: str) -> RunResult:
    """Research AND synthesis, so the timeline is what gets judged.

    The shipped research prompt is swapped for the candidate for the duration of
    the call. Module state, so runs are serialised by the caller.
    """
    from .pipeline import orchestrator as orch

    started = time.time()
    original = orch.RESEARCH_PROMPT
    saved_get, saved_save = orch.get_cached, orch.save_cached
    orch.RESEARCH_PROMPT = prompt
    orch.get_cached = lambda *a, **k: None      # a test tool must never serve a cached answer
    orch.save_cached = lambda *a, **k: None     # nor leave one behind
    try:
        o = orch.TraceOrchestrator()
        seen: Dict[str, Any] = {}
        real = o._synthesize
        def spy(**kw):
            seen["corpus"] = kw.get("corpus", "")
            seen["citations"] = kw.get("citations", [])
            return real(**kw)
        o._synthesize = spy
        result = o.run(TraceRequest(title=subject, context=context or None))
    finally:
        orch.RESEARCH_PROMPT = original
        orch.get_cached, orch.save_cached = saved_get, saved_save

    return RunResult(
        ok=True, seconds=time.time() - started,
        queries=list(result.queries_run or []),
        citations=[{"url": c.url, "title": c.title or ""} for c in (result.citations or [])],
        text=seen.get("corpus", ""),
        origin=f"{result.origin.source_title} ({result.origin.year}..{result.origin.year_end})",
        timeline=[f"{e.source_title} ({e.year}..{e.year_end})" for e in result.timeline],
    )


def run_many(prompt: str, subject: str, *, n: int, context: str, full: bool,
             workers: int) -> List[RunResult]:
    if full:
        # Serialised: the orchestrator's prompt is module state, so parallel runs
        # would overwrite each other's candidate and silently test the wrong thing.
        out = []
        for _ in range(n):
            try:
                out.append(_full(prompt, subject, context))
            except Exception as exc:  # noqa: BLE001 - a failed run is a data point
                out.append(RunResult(error=str(exc)[:120]))
        return out

    from phansora.shared.ai.gemini_research import GeminiResearchClient

    client = GeminiResearchClient()
    def once(_):
        try:
            return _research(client, prompt)
        except Exception as exc:  # noqa: BLE001
            return RunResult(error=str(exc)[:120])
    with ThreadPoolExecutor(max_workers=min(workers, n)) as pool:
        return list(pool.map(once, range(n)))


def summarise(label: str, runs: List[RunResult], expect: Optional[re.Pattern]) -> Dict[str, Any]:
    ok = [r for r in runs if r.ok]
    if not ok:
        return {"label": label, "ok": 0, "n": len(runs),
                "error": next((r.error for r in runs if r.error), "all runs failed")}
    tiers = [t for r in ok for t in r.tiers]
    return {
        "label": label, "ok": len(ok), "n": len(runs),
        "seconds": statistics.mean(r.seconds for r in ok),
        "searches": statistics.mean(len(r.queries) for r in ok),
        "cites": statistics.mean(len(r.citations) for r in ok),
        "low": sum(1 for t in tiers if t in LOW) / max(1, len(ok)),
        "good": sum(1 for t in tiers if t in GOOD) / max(1, len(ok)),
        "matched": sum(1 for r in ok if r.matched(expect)) if expect else None,
    }


def print_table(rows: List[Dict[str, Any]], expect_src: Optional[str]) -> None:
    head = "%-22s %-6s %-7s %-9s %-7s %-9s %-9s" % (
        "prompt", "ok", "secs", "searches", "cites", "low-auth", "repo/acad")
    if expect_src:
        head += " %s" % "matched"
    print(head)
    print("-" * len(head))
    for r in rows:
        if r.get("error"):
            print("%-22s %-6s %s" % (r["label"][:22], "0/%d" % r["n"], r["error"]))
            continue
        line = "%-22s %-6s %-7.1f %-9.1f %-7.1f %-9.1f %-9.1f" % (
            r["label"][:22], "%d/%d" % (r["ok"], r["n"]), r["seconds"],
            r["searches"], r["cites"], r["low"], r["good"])
        if expect_src:
            line += " %d/%d" % (r["matched"], r["ok"])
        print(line)
    if expect_src:
        print("\nmatched = runs whose answer contained /%s/" % expect_src)


def main(argv: Optional[List[str]] = None) -> int:
    ap = argparse.ArgumentParser(
        prog="prompt_lab",
        description="Run a Chrono-Origin research prompt against the live provider.",
        epilog="Prompts may use {title}, [SEARCH TERM] or {context_clause}.",
    )
    ap.add_argument("subject", help='what to trace, e.g. "Jesus Christ"')
    ap.add_argument("-p", "--prompt", action="append", default=[], metavar="FILE",
                    help="prompt file; repeat to compare several. '-' reads stdin. "
                         "Omit to use %s, created from the shipped prompt if absent."
                         % DEFAULT_PROMPT_FILE)
    ap.add_argument("--shipped", action="store_true",
                    help="also run the deployed prompt, as a baseline to compare against")
    ap.add_argument("-n", "--runs", type=int, default=1,
                    help="runs per prompt (default 1). One run tells you very little here.")
    ap.add_argument("-e", "--expect", metavar="REGEX",
                    help="score each run on whether its answer matches this")
    ap.add_argument("-c", "--context", default="", help="disambiguating context")
    ap.add_argument("--full", action="store_true",
                    help="run synthesis too and print the timeline (slower, costs more)")
    ap.add_argument("--show", action="store_true", help="print the first answer in full")
    ap.add_argument("--model", help="override GEMINI_MODEL for this run only")
    ap.add_argument("--workers", type=int, default=4, help="parallel runs (default 4)")
    ap.add_argument("--env", metavar="FILE", help="env file to read (default: find .env)")
    args = ap.parse_args(argv)

    loaded = load_env(args.env)
    if loaded:
        print("# config from %s\n" % loaded, file=sys.stderr)

    if args.model:
        os.environ["GEMINI_MODEL"] = args.model
        os.environ.pop("CHRONO_MODEL", None)

    specs = list(args.prompt)
    if not specs:
        path = default_prompt_path()
        if ensure_prompt_file(path):
            print("# created %s from the deployed prompt — edit it and run again\n" % path,
                  file=sys.stderr)
        else:
            print("# using %s\n" % path, file=sys.stderr)
        specs = [str(path)]
    # Added, never substituted: the question is almost always "is my edit better
    # than what is deployed", and that needs both in the same table.
    if args.shipped:
        specs.append("<shipped>")

    sources: List[Any] = []
    for spec in specs:
        if spec == "<shipped>":
            sources.append(("shipped", P.RESEARCH_PROMPT))
        elif spec == "-":
            sources.append(("<stdin>", sys.stdin.read()))
        else:
            path = Path(spec)
            if not path.is_file():
                print("no such prompt file: %s" % spec, file=sys.stderr)
                return 2
            sources.append((path.name, path.read_text()))

    expect = re.compile(args.expect, re.I) if args.expect else None
    rows: List[Dict[str, Any]] = []
    first: Optional[RunResult] = None
    for label, template in sources:
        prompt = fill(template, args.subject, args.context)
        runs = run_many(prompt, args.subject, n=args.runs, context=args.context,
                        full=args.full, workers=args.workers)
        rows.append(summarise(label, runs, expect))
        ok = [r for r in runs if r.ok]
        if first is None and ok:
            first = ok[0]
        if args.full:
            for i, r in enumerate(runs, 1):
                if not r.ok:
                    print("  %s run %d FAILED: %s" % (label, i, r.error)); continue
                print("  %s run %d -> ORIGIN %s" % (label, i, r.origin))
                for step in r.timeline[:6]:
                    print("      %s" % step)
        print()

    print_table(rows, args.expect)

    if first is not None:
        print("\nqueries the model chose (first run):")
        for q in first.queries or ["(none)"]:
            print("   -", q[:100])
        if first.citations:
            print("\nsources (first run):")
            for c in first.citations[:12]:
                print("   %-9s %s" % (sp.default_tier(c.get("url") or ""),
                                      (c.get("url") or "")[:88]))
        if args.show:
            print("\n" + "=" * 70 + "\n")
            print(first.text)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
