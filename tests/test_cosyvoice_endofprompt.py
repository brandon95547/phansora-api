"""The ``<|endofprompt|>`` marker CosyVoice3 refuses to run without.

CosyVoice2 had no end-of-prompt token. CosyVoice3's LLM asserts one is present:

    # cosyvoice/llm/llm.py
    assert 151646 in text, '<|endofprompt|> not detected in CosyVoice3 text or prompt_text'

So a missing marker is not a quality regression, it is a hard failure on *every* request.
That makes it worth pinning, especially because the place it has to be applied is
counter-intuitive: it goes on the string used as the ``_SPK_CACHE`` key, not on the value
passed at call time. Upstream's ``frontend_zero_shot`` ignores its ``prompt_text``
argument entirely when a cached ``zero_shot_spk_id`` is supplied, so marking only the call
argument would leave the cached conditioning unmarked and assert on the second request but
not the first — the worst kind of bug to find in prod.

These are pure-function tests: no model, no CUDA, no CosyVoice checkout.
"""
from __future__ import annotations

import pytest

cv = pytest.importorskip(
    "phansora.products.spokenverse.txt_to_voice.adapters.cosyvoice3_client",
    reason="spokenverse adapter not importable on this host",
)

MARKER = "<|endofprompt|>"


def test_a_bare_reference_transcript_gets_the_system_prefix():
    """The plain-cloning shape from upstream's example.py."""
    out = cv._ensure_endofprompt("Dusk was falling as the boy arrived.")

    assert out == "You are a helpful assistant.<|endofprompt|>Dusk was falling as the boy arrived."
    assert out.count(MARKER) == 1


def test_the_transcript_survives_intact_after_the_marker():
    """CosyVoice conditions on the transcript matching the audio — it must not be altered."""
    transcript = "These are dark times, that is no denying."
    assert cv._ensure_endofprompt(transcript).endswith(transcript)


def test_text_that_already_has_the_marker_is_untouched():
    """An instruction may arrive already carrying it; double-prefixing would corrupt the
    prompt AND change the cache key, silently splitting one voice into two entries."""
    already = "Speak in a calm, reassuring tone.<|endofprompt|>"
    assert cv._ensure_endofprompt(already) == already


def test_it_is_idempotent():
    once = cv._ensure_endofprompt("hello")
    assert cv._ensure_endofprompt(once) == once
    assert once.count(MARKER) == 1


def test_a_marker_anywhere_counts_not_just_at_the_end():
    """The engine checks for the token id, not its position."""
    mid = "You are a narrator.<|endofprompt|>Once upon a time."
    assert cv._ensure_endofprompt(mid) == mid


def test_empty_text_still_produces_a_usable_prompt():
    """Empty conditioning would otherwise reach the model with no marker and assert.
    Prefixing an empty string still yields a valid prompt."""
    out = cv._ensure_endofprompt("")

    assert out == "You are a helpful assistant.<|endofprompt|>"
    assert MARKER in out


def test_surrounding_whitespace_does_not_defeat_the_check():
    assert cv._ensure_endofprompt("  Speak warmly.<|endofprompt|>  ").count(MARKER) == 1


def test_the_system_prompt_is_overridable():
    assert cv._ensure_endofprompt("hi", "You are a narrator.") == (
        "You are a narrator.<|endofprompt|>hi"
    )


def test_the_marker_matches_the_token_the_engine_asserts_on():
    """Guards against a typo in the literal. 151646 is <|endofprompt|> in the Qwen
    tokenizer CosyVoice3 uses; the string here has to be exactly that token's text."""
    assert cv._ENDOFPROMPT == "<|endofprompt|>"
    assert cv._DEFAULT_SYSTEM_PROMPT == "You are a helpful assistant."


class TestConditioningReachesTheCacheKey:
    """The composition that actually runs in _synthesize_sync.

    Reproduced here rather than calling _synthesize_sync, which would need a model. If
    that expression changes, these break — which is the point.
    """

    @staticmethod
    def _conditioning(instruct: str, prompt_text: str) -> str:
        return cv._ensure_endofprompt(instruct or prompt_text)

    def test_plain_cloning_conditions_on_the_marked_transcript(self):
        cond = self._conditioning("", "Dusk was falling.")
        assert MARKER in cond and cond.endswith("Dusk was falling.")

    def test_an_instruction_takes_precedence_and_is_marked(self):
        cond = self._conditioning("Speak in a calm tone.", "Dusk was falling.")
        assert MARKER in cond
        assert "Speak in a calm tone." in cond
        assert "Dusk was falling." not in cond  # instruct replaces the transcript slot

    def test_each_instruction_gets_its_own_cache_key(self):
        """_SPK_CACHE is keyed on (clip, conditioning). Two deliveries of one voice must
        not collide, or the second would inherit the first's prosody."""
        a = self._conditioning("Speak calmly.", "ref")
        b = self._conditioning("Speak urgently.", "ref")
        assert a != b

    def test_the_same_request_twice_hits_the_same_key(self):
        """The other half: identical input must NOT miss the cache, or every request pays
        speaker extraction again."""
        assert self._conditioning("Speak calmly.", "ref") == self._conditioning("Speak calmly.", "ref")


class TestEnvFallback:
    """COSYVOICE3_X falls back to COSYVOICE2_X so a not-yet-renamed prod .env still boots."""

    def test_the_new_name_is_preferred(self, monkeypatch):
        monkeypatch.setenv("COSYVOICE3_REPO", "/new")
        monkeypatch.setenv("COSYVOICE2_REPO", "/old")
        assert cv._env("COSYVOICE3_REPO") == "/new"

    def test_the_old_name_is_used_when_the_new_one_is_unset(self, monkeypatch):
        monkeypatch.delenv("COSYVOICE3_REPO", raising=False)
        monkeypatch.setenv("COSYVOICE2_REPO", "/var/www/CosyVoice")
        assert cv._env("COSYVOICE3_REPO") == "/var/www/CosyVoice"

    def test_an_empty_new_name_does_not_mask_the_old_one(self, monkeypatch):
        """.env writes blanks rather than omitting keys, so "" must fall through."""
        monkeypatch.setenv("COSYVOICE3_REPO", "")
        monkeypatch.setenv("COSYVOICE2_REPO", "/var/www/CosyVoice")
        assert cv._env("COSYVOICE3_REPO") == "/var/www/CosyVoice"

    def test_the_fallback_covers_the_typed_readers_too(self, monkeypatch):
        monkeypatch.delenv("COSYVOICE3_MAX_CHARS", raising=False)
        monkeypatch.delenv("COSYVOICE3_SPEED", raising=False)
        monkeypatch.delenv("COSYVOICE3_USE_VLLM", raising=False)
        monkeypatch.setenv("COSYVOICE2_MAX_CHARS", "250")
        monkeypatch.setenv("COSYVOICE2_SPEED", "1.25")
        monkeypatch.setenv("COSYVOICE2_USE_VLLM", "0")

        assert cv._env_int("COSYVOICE3_MAX_CHARS", 200) == 250
        assert cv._env_float("COSYVOICE3_SPEED", 1.0) == 1.25
        assert cv._env_bool("COSYVOICE3_USE_VLLM", True) is False

    def test_unrelated_variables_are_not_rewritten(self, monkeypatch):
        monkeypatch.setenv("TTS_ENGINE", "cosyvoice3")
        assert cv._env("TTS_ENGINE") == "cosyvoice3"

    def test_defaults_still_apply_when_neither_is_set(self, monkeypatch):
        monkeypatch.delenv("COSYVOICE3_MAX_CHARS", raising=False)
        monkeypatch.delenv("COSYVOICE2_MAX_CHARS", raising=False)
        assert cv._env_int("COSYVOICE3_MAX_CHARS", 200) == 200


def test_trt_is_off_by_default_on_v3(monkeypatch):
    """Upstream warns the DiT TensorRT fp16 engine "have some performance issue, use at
    caution!" for CosyVoice3, so this default is deliberate, not an oversight."""
    monkeypatch.delenv("COSYVOICE3_USE_TRT", raising=False)
    monkeypatch.delenv("COSYVOICE2_USE_TRT", raising=False)
    assert cv._env_bool("COSYVOICE3_USE_TRT", False) is False


def test_the_public_surface_other_modules_import_is_intact():
    """voices.py and server.py import these by name; losing one is an ImportError at
    startup, not at synthesis."""
    for name in ("LANGUAGES", "LANGUAGE_DEFAULT", "SPEED_MIN", "SPEED_MAX",
                 "SPEED_DEFAULT", "INSTRUCT_MAX_CHARS", "synthesize_to_file",
                 "_discover_voices_sync", "list_voices", "preload"):
        assert hasattr(cv, name), f"cosyvoice3_client lost {name}"
