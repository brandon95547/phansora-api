"""Narrava Studio storyboard tests.

Only the LLM boundary is stubbed, so these run without an API key and without importing
any provider SDK — the same constraint test_smoke.py works under.
"""


def test_suggest_for_span_unwraps_the_ideas_envelope(monkeypatch):
    """The model answers {"ideas": [...]}; the envelope must not swallow the answer.

    Regression: 8a330ea routed the ideas call through _clean_ideas, which took a bare list,
    while _ask_ideas handed it the whole dict. Every scene came back with zero ideas and
    degraded=True however good the model's reply was, which the UI reported as
    "Could not write more ideas for this scene."
    """
    from phansora.products.narrava_studio.services import storyboard

    payload = {"ideas": [
        {"visual": "Archival newsreel of a harbour at dawn",
         "media_type": "image",
         "search_terms": ["harbour dawn", "archival newsreel", "1940s port"]},
        {"visual": "Slow aerial over a modern container terminal",
         "media_type": "video",
         "search_terms": ["container terminal aerial", "cargo port drone"]},
    ]}
    monkeypatch.setattr(storyboard.llm, "generate_json", lambda *a, **k: payload)

    out = storyboard.suggest_for_span("The port never truly sleeps.", count=2)

    assert len(out["ideas"]) == 2
    assert out["degraded"] is False
    # The flat fields mirror ideas[0] for callers written before ideas existed.
    assert out["visual_prompt"]
    assert out["media_type"] == "image"
    assert out["search_terms"]


def test_clean_ideas_takes_either_shape():
    """Both the bare list and the envelope normalize to the same ideas."""
    from phansora.products.narrava_studio.services import storyboard

    items = [{"visual": "A lighthouse in fog", "media_type": "image",
              "search_terms": ["lighthouse fog", "coastal beacon"]}]

    from_list = storyboard._clean_ideas(items, span="x", limit=3)
    from_envelope = storyboard._clean_ideas({"ideas": items}, span="x", limit=3)

    assert len(from_list) == 1
    assert from_list == from_envelope


def test_suggest_for_span_degrades_when_the_model_fails(monkeypatch):
    """The honest-failure signal has to survive the fix.

    Without it a total LLM outage returns the same shape as a good answer, and the client
    records "this scene only has one idea" permanently.
    """
    from phansora.products.narrava_studio.services import storyboard

    monkeypatch.setattr(storyboard.llm, "generate_json", lambda *a, **k: {})

    out = storyboard.suggest_for_span("The port never truly sleeps.", count=2)

    assert out["ideas"] == []
    assert out["degraded"] is True


def test_suggest_for_span_degrades_when_fewer_ideas_than_asked(monkeypatch):
    """Coming up short is degraded too — that is what lets the client try again."""
    from phansora.products.narrava_studio.services import storyboard

    payload = {"ideas": [
        {"visual": "A lighthouse in fog", "media_type": "image",
         "search_terms": ["lighthouse fog", "coastal beacon"]},
    ]}
    monkeypatch.setattr(storyboard.llm, "generate_json", lambda *a, **k: payload)

    out = storyboard.suggest_for_span("The port never truly sleeps.", count=5)

    assert len(out["ideas"]) == 1
    assert out["degraded"] is True
