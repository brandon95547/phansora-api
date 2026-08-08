"""Narrava Studio "Enhance Narration" tests.

The LLM boundary is stubbed, so these run without an API key and without importing any
provider SDK — the same constraint the other narrava tests work under. What they check is
the part that is ours: which voice a request resolves to, and what the model is actually
told once it has.
"""
import pytest


@pytest.fixture()
def captured(monkeypatch):
    """Stub the DeepSeek call and hand back what it was asked."""
    from phansora.products.narrava_studio.services import script

    seen = {}

    def fake(system, user, *, max_output_tokens=2000):
        seen["system"] = system
        seen["user"] = user
        return user  # unchanged is a valid answer, and keeps the ratio guard happy

    monkeypatch.setattr(script.llm, "deepseek_text", fake)
    return seen


def test_the_picked_voice_reaches_the_prompt(captured):
    from phansora.products.narrava_studio.services import script

    script.enhance_narration("The records list the same three names.", style="angry")

    assert "THE VOICE — Angry / Confrontational." in captured["system"]
    # And only that one: six voices in one prompt is six contradictory instructions.
    assert "Cinematic / Dramatic" not in captured["system"]
    assert "Calm / Documentary" not in captured["system"]


def test_an_unknown_or_missing_voice_polishes_calmly(captured):
    """A creative preference this build does not know is not worth failing a request over."""
    from phansora.products.narrava_studio.services import script

    script.enhance_narration("The records list the same three names.", style="retro")
    assert "THE VOICE — Calm / Documentary." in captured["system"]

    script.enhance_narration("The records list the same three names.")
    assert "THE VOICE — Calm / Documentary." in captured["system"]


def test_the_voice_never_licenses_new_content(captured):
    """The guard rails ride in the SAME prompt as the voice, on every style.

    The failure this protects against is a "more dramatic" narration that quietly says
    something the writer did not: the voice decides how a fact is worded and nothing else.
    """
    from phansora.products.narrava_studio.services import script

    for style in ("dark", "angry", "investigative", "cinematic", "calm", "light"):
        script.enhance_narration("The records list the same three names.", style=style)
        system = captured["system"]
        assert "Add any fact, name, number, claim, opinion or example" in system
        assert "Change the order in which the ideas are presented." in system
        assert "atmosphere, drama or emphasis" in system


def test_the_example_is_labelled_as_register_not_content(captured):
    """The one-line sample is a demonstration of voice — the model is told not to borrow it.

    Without that sentence the example is just text in the context window, and a short
    narration handed the suspense example came back talking about organizations.
    """
    from phansora.products.narrava_studio.services import script

    script.enhance_narration("A quiet street on a Tuesday morning.", style="dark")

    assert "A line written in this voice sounds like this:" in captured["system"]
    assert "It is not content" in captured["system"]


def test_empty_narration_never_reaches_the_model(captured):
    from phansora.products.narrava_studio.services import script

    assert script.enhance_narration("   ", style="dark") == ""
    assert "system" not in captured


def test_a_summary_instead_of_a_rewrite_is_refused(monkeypatch):
    """Every voice may reword freely; none of them may hand back a shorter piece.

    A reply well under the original length is a summary or a truncation, and passing it on
    would replace the writer's script with a shorter one under the name of a polish.
    """
    from phansora.products.narrava_studio.services import script

    monkeypatch.setattr(script.llm, "deepseek_text", lambda *a, **k: "They were connected.")

    with pytest.raises(RuntimeError, match="shortened"):
        script.enhance_narration(
            "Over a period of several years, the same handful of names appeared in the "
            "filings of one organization, and the records show they were there together "
            "more than once.",
            style="light",
        )


def test_the_ids_match_the_browsers_list():
    """Pinned on both sides (assets/js/admin/studio/util/enhance-styles.js).

    A rename here does not error — entry() falls back to calm — so every narration would
    quietly come back in the wrong voice with nothing in the logs to say why.
    """
    from phansora.products.narrava_studio.services.enhance_styles import ENHANCE_STYLES

    assert list(ENHANCE_STYLES) == [
        "dark", "angry", "investigative", "cinematic", "calm", "light",
    ]
    assert [s["label"] for s in ENHANCE_STYLES.values()] == [
        "Dark / Suspenseful",
        "Angry / Confrontational",
        "Investigative / Serious",
        "Cinematic / Dramatic",
        "Calm / Documentary",
        "Light / Conversational",
    ]
