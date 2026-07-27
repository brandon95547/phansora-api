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


def test_style_block_falls_back_to_unstyled(monkeypatch):
    """An unknown style builds unstyled rather than failing the request.

    The style is a creative preference; refusing to build a storyboard because a client sent
    a name this version does not know would be a worse answer than building it plain.
    """
    from phansora.products.narrava_studio.services import storyboard

    for value in (None, "", "   ", "nonsense"):
        assert storyboard._style_block(value) == ""
    assert storyboard._style_block("CINEMATIC").startswith("DOCUMENTARY STYLE — Cinematic")


def test_style_reaches_the_ideas_prompt(monkeypatch):
    from phansora.products.narrava_studio.services import storyboard

    seen = {}

    def fake(system, user, **kwargs):
        seen["user"] = user
        return {"ideas": [{"visual": "A lighthouse in fog", "media_type": "image",
                           "search_terms": ["lighthouse fog"]}]}

    monkeypatch.setattr(storyboard.llm, "generate_json", fake)
    storyboard.suggest_for_span("The port never truly sleeps.", count=2, style="dark")

    assert seen["user"].startswith("DOCUMENTARY STYLE — Dark / Suspense")
    # The style may shape the pictures and how fast they turn over, but the spans are what
    # every placeholder's position is anchored to — so it is told where its authority ends.
    assert "must still be the narration verbatim" in seen["user"]


def test_narration_and_storyboard_read_the_same_style(monkeypatch):
    """One choice, two prompts. A film written investigative and shot cinematic is the
    failure the shared registry exists to prevent, so both blocks must name the same style
    from the same entry."""
    from phansora.products.narrava_studio.services import styles

    assert styles.visual_block("historical").startswith("DOCUMENTARY STYLE — Historical")
    assert styles.narration_block("historical").startswith("DOCUMENTARY STYLE — Historical")
    # Unknown and missing write unstyled on both sides rather than raising.
    for value in (None, "", "   ", "nonsense"):
        assert styles.visual_block(value) == ""
        assert styles.narration_block(value) == ""


def test_doc_style_reaches_the_script_writer(monkeypatch):
    """The narration is written in the house style, not just shot in it."""
    from phansora.products.narrava_studio.services import script

    seen = {}

    def fake(system, user, **kwargs):
        seen["user"] = user
        return "The port never truly sleeps."

    monkeypatch.setattr(script.llm, "generate_text", fake)
    script.generate_script("A port at night", style="documentary", doc_style="dark")

    assert seen["user"].startswith("DOCUMENTARY STYLE — Dark / Suspense")
    # The brief still reaches it — the style frames the request, it does not replace it.
    assert "A port at night" in seen["user"]


def test_style_does_not_move_the_placeholders(monkeypatch):
    """The style reaches the model, never the layout.

    Same scenes in, same spans out — where a placeholder starts and ends is arithmetic over
    the model's spans and the measured clock, and the style must not enter it.
    """
    from phansora.products.narrava_studio.services import storyboard

    text = ("The port never truly sleeps. Steam gave way to diesel, and diesel to the "
            "container. And by the century end the cargo moved itself.")
    word_times = [(i * 0.5, i * 0.5 + 0.5) for i in range(len(text.split()))]
    total = len(text.split()) * 0.5

    scenes = {"scenes": [
        {"text": "The port never truly sleeps.", "visual": "A harbour at dawn.",
         "rationale": "r", "media_type": "image", "search_terms": ["harbour"]},
        {"text": "Steam gave way to diesel, and diesel to the container.",
         "visual": "A container ship.", "rationale": "r", "media_type": "image",
         "search_terms": ["container ship"]},
        {"text": "And by the century end the cargo moved itself.",
         "visual": "Automated cranes.", "rationale": "r", "media_type": "image",
         "search_terms": ["crane"]},
    ]}
    monkeypatch.setattr(storyboard.llm, "generate_json", lambda *a, **k: scenes)

    plain = storyboard.build_storyboard(text, total, max_scenes=6, word_times=word_times)
    styled = storyboard.build_storyboard(text, total, max_scenes=6, word_times=word_times,
                                         style="historical")

    assert [(s.start_sec, s.end_sec) for s in plain] == [(s.start_sec, s.end_sec) for s in styled]
    assert [s.text for s in plain] == [s.text for s in styled]


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
