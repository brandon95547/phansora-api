"""The prompt lab.

Its job is to answer questions about RATES. A prompt asked to trace backward
reached the right material in roughly one run in seven, so a single sample proved
nothing in either direction — which is exactly how a prompt that mostly did not
work survived being "tested" repeatedly.

These pin the parts that would quietly corrupt a measurement: substitution that
drops the subject, a run that writes to the cache, and scoring that counts wrong.
"""
from __future__ import annotations

import os
import re

import pytest

from phansora.products.chrono_origin import prompt_lab as L


class TestFillingInTheSubject:
    def test_binds_every_placeholder_form(self):
        """Prompts arrive from a chat window, a file, or the repo. All three shapes."""
        for tpl in ('trace "{title}" now', 'trace "[SEARCH TERM]" now'):
            assert L.fill(tpl, "flying kites") == 'trace "flying kites" now'

    def test_binds_every_occurrence(self):
        out = L.fill("{title} ... {title} ... {title}", "X")
        assert out == "X ... X ... X"

    def test_context_disambiguates(self):
        assert L.fill('"{title}"{context_clause}', "Mercury", "the planet") == '"Mercury" (the planet)'

    def test_no_context_leaves_no_empty_parens(self):
        assert L.fill('"{title}"{context_clause}', "Mercury") == '"Mercury"'

    def test_braces_in_a_prompt_do_not_blow_up(self):
        """str.format would raise here, and this is not a rare prompt.

        The synthesis prompt is largely a JSON schema, and a prompt pasted from
        anywhere else has no reason to escape its braces.
        """
        tpl = 'trace "{title}" and return {"origin": {"year": 1}}'
        out = L.fill(tpl, "X")
        assert '"origin"' in out and out.startswith('trace "X"')


class TestScoring:
    def _run(self, text="", timeline=(), origin="", cites=()):
        return L.RunResult(ok=True, text=text, timeline=list(timeline), origin=origin,
                           citations=[{"url": u} for u in cites])

    def test_expect_matches_against_the_answer(self):
        pat = re.compile("septuagint", re.I)
        assert self._run(text="...the Septuagint...").matched(pat) is True
        assert self._run(text="...nothing relevant...").matched(pat) is False

    def test_expect_with_no_pattern_never_matches(self):
        """No --expect means no claim, not a free pass."""
        assert self._run(text="anything").matched(None) is False

    def test_full_mode_scores_the_timeline_not_just_the_research(self):
        """The distinction the whole tool exists for.

        A corpus naming the Hebrew scriptures in 3 runs of 3 produced a timeline
        containing them in 0 of 3. Scoring the research text alone would have called
        that a success.
        """
        pat = re.compile("hebrew", re.I)
        r = self._run(text="research mentions hebrew scriptures",
                      origin="The Septuagint", timeline=["Paul's letters"])
        assert r.matched(pat) is True          # the corpus is in scope too
        r2 = self._run(text="", origin="The Hebrew Scriptures", timeline=[])
        assert r2.matched(pat) is True         # and so is the timeline

    def test_tiers_come_from_the_resolved_url(self):
        r = self._run(cites=["https://en.wikipedia.org/wiki/X", "https://www.jstor.org/stable/1"])
        assert "low_authority" in r.tiers

    def test_a_failed_run_is_counted_not_hidden(self):
        rows = L.summarise("p", [L.RunResult(ok=True, text="x"), L.RunResult(error="boom")], None)
        assert rows["ok"] == 1 and rows["n"] == 2

    def test_all_failing_reports_the_error_rather_than_zeroes(self):
        """Zeroes across the board read as 'this prompt scored badly'. It did not run."""
        rows = L.summarise("p", [L.RunResult(error="401 unauthorised")], None)
        assert rows.get("error")
        assert "401" in rows["error"]

    def test_match_count_is_out_of_runs_that_actually_RAN(self):
        pat = re.compile("hit", re.I)
        rows = L.summarise("p", [L.RunResult(ok=True, text="hit"), L.RunResult(error="boom")], pat)
        assert rows["matched"] == 1 and rows["ok"] == 1


class TestItCannotContaminateProduction:
    @pytest.fixture(autouse=True)
    def _configured(self, monkeypatch):
        """_full builds a real orchestrator, which refuses to exist unconfigured."""
        monkeypatch.setenv("GEMINI_API_KEY", "k")
        monkeypatch.setenv("GEMINI_MODEL", "gemini-3.5-flash-lite")
        monkeypatch.delenv("CHRONO_MODEL", raising=False)
        monkeypatch.setenv("CHRONO_LLM_PROVIDER", "gemini")

    def test_full_mode_neither_reads_nor_writes_the_cache(self, monkeypatch):
        """A cached answer would be served instantly and score the OLD prompt.

        The cache key does not hash the prompt text, so without this every run after
        the first would silently grade whatever was cached.
        """
        from phansora.products.chrono_origin.pipeline import orchestrator as orch

        calls = {"get": 0, "save": 0}
        monkeypatch.setattr(orch, "get_cached", lambda *a, **k: calls.__setitem__("get", 1))
        monkeypatch.setattr(orch, "save_cached", lambda *a, **k: calls.__setitem__("save", 1))

        class Boom(Exception):
            pass

        def explode(self, req):
            # far enough in that the cache functions would already have been swapped
            assert orch.get_cached("t", "k") is None, "cache read was not disabled"
            orch.save_cached("t", "k", {})
            raise Boom()

        monkeypatch.setattr(orch.TraceOrchestrator, "run", explode)
        with pytest.raises(Boom):
            L._full("prompt", "Kites", "")
        assert calls == {"get": 0, "save": 0}, "the lab touched the real cache"

    def test_full_mode_puts_the_shipped_prompt_back(self, monkeypatch):
        """Module state. Leaving a candidate installed would poison every later run,
        including the ones the app serves from this same process."""
        from phansora.products.chrono_origin.pipeline import orchestrator as orch

        before = orch.RESEARCH_PROMPT

        def explode(self, req):
            assert orch.RESEARCH_PROMPT == "candidate"
            raise RuntimeError("stop here")

        monkeypatch.setattr(orch.TraceOrchestrator, "run", explode)
        with pytest.raises(RuntimeError):
            L._full("candidate", "Kites", "")
        assert orch.RESEARCH_PROMPT == before


class TestTheCommandLine:
    def test_a_missing_prompt_file_fails_loudly(self, capsys):
        assert L.main(["X", "-p", "/nope/does-not-exist.txt"]) == 2
        assert "no such prompt file" in capsys.readouterr().err

    def test_model_override_clears_the_product_override_too(self, monkeypatch):
        """CHRONO_MODEL outranks GEMINI_MODEL, so setting only the latter would be
        silently ignored and the run would grade the wrong model."""
        monkeypatch.setenv("CHRONO_MODEL", "some-other-model")
        monkeypatch.setattr(L, "run_many", lambda *a, **k: [L.RunResult(error="stopped")])
        L.main(["X", "--model", "gemini-3.5-flash"])
        import os
        assert os.environ["GEMINI_MODEL"] == "gemini-3.5-flash"
        assert "CHRONO_MODEL" not in os.environ


class TestFindingItsConfiguration:
    """A shell has none of the environment systemd hands the service."""

    def test_reads_an_env_file_when_the_key_is_absent(self, tmp_path, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        monkeypatch.delenv("GEMINI_MODEL", raising=False)
        env = tmp_path / ".env"
        env.write_text('GEMINI_API_KEY=secret\nGEMINI_MODEL="gemini-3.5-flash-lite"\n# note\n')
        assert L.load_env(str(env)) == str(env)
        assert os.environ["GEMINI_API_KEY"] == "secret"
        assert os.environ["GEMINI_MODEL"] == "gemini-3.5-flash-lite"   # quotes stripped

    def test_an_already_set_key_wins(self, tmp_path, monkeypatch):
        """Never override a key the caller exported deliberately."""
        monkeypatch.setenv("GEMINI_API_KEY", "from-the-shell")
        env = tmp_path / ".env"
        env.write_text("GEMINI_API_KEY=from-the-file\n")
        assert L.load_env(str(env)) is None
        assert os.environ["GEMINI_API_KEY"] == "from-the-shell"

    def test_a_missing_env_file_is_not_fatal(self, monkeypatch):
        monkeypatch.delenv("GEMINI_API_KEY", raising=False)
        monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
        assert L.load_env("/nope/.env") is None


class TestTheEditableFile:
    """The point of the tool: edit a file, run, repeat — no deploy, no restart."""

    def test_it_is_seeded_from_the_deployed_prompt(self, tmp_path):
        """An empty file would mean retyping 140 lines to change one of them,
        which is how you end up measuring something other than what is running."""
        from phansora.products.chrono_origin.pipeline import prompts as P

        f = tmp_path / "chrono-prompt.txt"
        assert L.ensure_prompt_file(f) is True
        assert f.read_text() == P.RESEARCH_PROMPT

    def test_an_existing_file_is_never_overwritten(self, tmp_path):
        """Clobbering it would silently discard the experiment in progress."""
        f = tmp_path / "chrono-prompt.txt"
        f.write_text("my careful draft")
        assert L.ensure_prompt_file(f) is False
        assert f.read_text() == "my careful draft"

    def test_the_working_directory_wins(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)
        (tmp_path / L.DEFAULT_PROMPT_FILE).write_text("local")
        assert L.default_prompt_path() == tmp_path / L.DEFAULT_PROMPT_FILE

    def test_it_falls_back_to_the_repo_root(self, tmp_path, monkeypatch):
        monkeypatch.chdir(tmp_path)          # no file here
        assert L.default_prompt_path().name == L.DEFAULT_PROMPT_FILE
        assert L.default_prompt_path().parent != tmp_path


def test_shipped_is_a_baseline_not_a_replacement(tmp_path, monkeypatch, capsys):
    """--shipped answers "is my edit better than what is deployed?", which needs both."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / L.DEFAULT_PROMPT_FILE).write_text("my candidate {title}")
    seen = []
    monkeypatch.setattr(L, "run_many",
                        lambda prompt, *a, **k: seen.append(prompt) or [L.RunResult(ok=True, text="")])
    L.main(["Kites", "--shipped"])
    assert len(seen) == 2, "only one prompt was run"
    assert seen[0] == "my candidate Kites"
    assert "{title}" not in seen[1] and len(seen[1]) > 200   # the real shipped prompt, filled
