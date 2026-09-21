"""Lyrics off a music track, without a speech model — for music videos, which have no script.

Whisper's output is handed in as stand-in segment objects, so these run anywhere. What is
under test is what the transcript becomes: that its words are positional over its text the
same way a narration's timings are over its script (the caption builder and the storyboard
both depend on it), that whisper's notes for sounds never become lyrics, and that a word the
model was unsure of comes back marked for checking.
"""
from types import SimpleNamespace

import pytest


def _seg(*words):
    """A whisper segment: ``(text, start, end[, probability])`` per word."""
    return SimpleNamespace(
        text="".join(w[0] for w in words),
        words=[
            SimpleNamespace(word=w[0], start=w[1], end=w[2], probability=w[3] if len(w) > 3 else 0.9)
            for w in words
        ],
    )


def _transcribe(*segments, total=None):
    from phansora.products.narrava_studio.services import align

    return align._transcript(align._sung_pieces(segments), total)


def _tokens(text):
    from phansora.products.narrava_studio.services import align

    return [m.group(0) for m in align.WORD_RE.finditer(text)]


def test_words_are_positional_over_the_text_and_lines_break_per_segment():
    out = _transcribe(
        _seg((" Hello,", 1.0, 1.4), (" world.", 1.5, 2.0)),
        _seg((" I", 3.0, 3.1), ("'m", 3.1, 3.3), (" here", 3.4, 3.9)),
    )
    assert out["text"] == "Hello, world.\nI'm here"
    # "I" + "'m" arrive as two pieces and are ONE word of the text — which is how the caption
    # builder and the storyboard tokenize it, so it is how the timings have to be counted.
    assert _tokens(out["text"]) == ["Hello", "world", "I'm", "here"]
    assert out["words"] == [(1.0, 1.4), (1.5, 2.0), (3.0, 3.3), (3.4, 3.9)]
    assert out["guessed"] == [False, False, False, False]


def test_notes_for_sounds_never_become_lyrics():
    out = _transcribe(
        _seg((" [Music]", 0.0, 6.0)),
        # The same note split across pieces, which whisper does as readily.
        _seg((" [", 6.0, 6.1), ("Music", 6.1, 6.5), ("]", 6.5, 6.6)),
        _seg((" (upbeat", 7.0, 7.5), (" music)", 7.5, 8.0)),
        _seg((" ♪", 8.0, 8.2), (" Take", 8.5, 8.8), (" me", 8.9, 9.1), (" home", 9.2, 9.9)),
    )
    assert _tokens(out["text"]) == ["Take", "me", "home"]
    assert "Music" not in out["text"] and "♪" not in out["text"]
    assert out["words"] == [(8.5, 8.8), (8.9, 9.1), (9.2, 9.9)]


def test_a_backing_vocal_in_parentheses_is_sung_and_stays():
    out = _transcribe(_seg((" Hold", 1.0, 1.3), (" on", 1.4, 1.6), (" (yeah)", 1.7, 2.1)))
    assert _tokens(out["text"]) == ["Hold", "on", "yeah"]
    assert len(out["words"]) == 3


def test_a_word_the_model_was_unsure_of_is_marked():
    out = _transcribe(_seg((" Starlight", 0.0, 0.6, 0.97), (" dewdrops", 0.7, 1.2, 0.21)))
    assert out["guessed"] == [False, True]


def test_timestamps_that_run_backwards_are_made_monotonic_and_clamped():
    out = _transcribe(
        _seg((" one", 5.0, 5.5), (" two", 4.8, 5.9)),
        _seg((" three", 9.0, 12.0)),
        total=10.0,
    )
    starts = [s for s, _ in out["words"]]
    assert starts == sorted(starts)
    assert max(e for _, e in out["words"]) <= 10.0


def test_a_rest_whisper_folded_into_the_next_word_does_not_light_it_early():
    """Whisper leaves no gap inside a segment, so a musical rest lands in the FRONT of the
    word after it. That word then starts where the rest starts, and the karaoke highlight —
    which lights the last word to have STARTED — names it through the whole silence."""
    out = _transcribe(
        # "memory" ends at 30.2 and "Holding" runs straight on to 32.18: a two-syllable word
        # given two seconds, because the rest between the two phrases went into it.
        _seg((" a", 29.6, 29.68), (" memory", 29.68, 30.2), (" Holding", 30.2, 32.18),
             (" him", 32.18, 32.42)),
    )
    starts = [s for s, _ in out["words"]]
    # Started near its end instead — measured against forced alignment, the real onset of
    # that word is 31.83, where whisper's own answer was 1.63s early.
    assert starts[2] == pytest.approx(31.68, abs=0.01)
    # Its neighbours are plausibly sized and are left exactly as whisper heard them.
    assert starts[0] == 29.6 and starts[1] == 29.68 and starts[3] == 32.18


def test_a_rest_whisper_reported_is_believed():
    """A gap the model actually left is one it heard, so the start after it is honest."""
    out = _transcribe(_seg((" wait", 1.0, 1.4)), _seg((" for", 4.0, 4.9), (" me", 5.0, 5.2)))
    # "for" is given 0.9s after a three-second gap — long for one syllable, but whisper
    # placed that silence itself, so the onset is left alone.
    assert [s for s, _ in out["words"]] == [1.0, 4.0, 5.0]


def test_an_instrumental_is_refused_with_a_reason():
    from phansora.products.narrava_studio.services import align

    with pytest.raises(align.AlignmentFailed, match="vocals"):
        _transcribe(_seg((" [Music]", 0.0, 30.0)), _seg((" ♪", 30.0, 31.0)))
    with pytest.raises(align.AlignmentFailed, match="vocals"):
        _transcribe()


def test_lyric_words_asks_for_wording_and_the_files_own_timeline(monkeypatch):
    """Beam 5 because nothing corrects the wording afterwards; VAD off so times are the
    file's; no cross-window prompt so a chorus cannot cascade into a hallucinated verse."""
    from phansora.products.narrava_studio.services import align

    seen = {}

    class Model:
        def transcribe(self, path, **kwargs):
            seen.update(kwargs, path=path)
            return iter([_seg((" Run", 0.5, 0.9), (" away", 1.0, 1.6))]), None

    monkeypatch.setattr(align, "_load_model", lambda: Model())
    out = align.lyric_words("song.m4a", language="en", total_duration_sec=4.0)

    assert out["text"] == "Run away"
    assert out["words"] == [(0.5, 0.9), (1.0, 1.6)]
    assert seen["path"] == "song.m4a"
    assert seen["word_timestamps"] is True
    assert seen["vad_filter"] is False
    assert seen["beam_size"] == 5
    assert seen["condition_on_previous_text"] is False
    assert seen["language"] == "en"


def test_a_file_whisper_cannot_read_is_a_user_error_not_a_crash(monkeypatch):
    from phansora.products.narrava_studio.services import align

    class Model:
        def transcribe(self, path, **kwargs):
            raise RuntimeError("Invalid data found when processing input")

    monkeypatch.setattr(align, "_load_model", lambda: Model())
    with pytest.raises(align.AlignmentFailed, match="could not be read"):
        align.lyric_words("broken.mp3")


# ── the storyboard of a music video ─────────────────────────────────────────────

def test_the_lyrics_editor_never_talks_about_narration():
    from phansora.products.narrava_studio.services import storyboard

    for prompt in (storyboard._LYRICS_SYSTEM, storyboard._LYRICS_REFINE_SYSTEM):
        lowered = prompt.lower()
        assert "narration" not in lowered
        assert "documentary" not in lowered
        assert "script" not in lowered
        assert "music video" in lowered


def test_lyrics_reach_the_music_video_editor_and_land_on_the_measured_words(monkeypatch):
    from phansora.products.narrava_studio.services import storyboard

    text = "Take me home tonight\nwhere the river runs\nand the lights go down"
    words = text.split()
    word_times = [(2.0 + i * 0.6, 2.0 + i * 0.6 + 0.5) for i in range(len(words))]
    total = 12.0
    asked = {}

    def fake(system, user, **kwargs):
        asked.update(system=system, user=user)
        return {"scenes": [
            {"text": "Take me home tonight", "visual": "A car on a night road.", "rationale": "r",
             "media_type": "video", "search_terms": ["night road"]},
            {"text": "where the river runs", "visual": "A river at dusk.", "rationale": "r",
             "media_type": "video", "search_terms": ["river"]},
            {"text": "and the lights go down", "visual": "City lights dimming.", "rationale": "r",
             "media_type": "video", "search_terms": ["city lights"]},
        ]}

    monkeypatch.setattr(storyboard.llm, "generate_json", fake)
    scenes = storyboard.build_storyboard(text, total, max_scenes=6, word_times=word_times,
                                         source="lyrics")

    assert asked["system"] == storyboard._LYRICS_SYSTEM
    assert "LYRICS:" in asked["user"] and "song" in asked["user"]
    assert [s.text for s in scenes] == ["Take me home tonight", "where the river runs",
                                       "and the lights go down"]
    # Scene 2 opens on "where" — word 4 — a hair before it is sung, but never back past the
    # end of the word before it (the clock only leads into silence).
    lead = max(word_times[3][1], word_times[4][0] - storyboard._CUT_LEAD_SEC)
    assert scenes[1].start_sec == pytest.approx(lead, abs=0.001)
    assert scenes[0].start_sec == 0.0 and scenes[-1].end_sec == total


def test_narration_still_gets_the_documentary_editor(monkeypatch):
    from phansora.products.narrava_studio.services import storyboard

    asked = {}

    def fake(system, user, **kwargs):
        asked["system"] = system
        return {"scenes": [{"text": "The port never sleeps.", "visual": "A harbour.",
                            "rationale": "r", "media_type": "image", "search_terms": ["harbour"]}]}

    monkeypatch.setattr(storyboard.llm, "generate_json", fake)
    text = "The port never sleeps."
    storyboard.build_storyboard(text, 3.0, word_times=[(i * 0.5, i * 0.5 + 0.4) for i in range(4)])
    assert asked["system"] == storyboard._SYSTEM


# ── Known lyrics ──────────────────────────────────────────────────────────────
# The words come from the user's sheet and only the timing comes off the recording, so what
# these pin down is the join: that the sheet's furniture never reaches a caption, that a
# word the singing did not yield is still placed and marked, and that lyrics for a
# different song are refused rather than spread across this one.


def _aligned(monkeypatch, lyrics, transcript, *, total=None):
    """``lyric_alignment`` over a transcript handed in instead of decoded."""
    from phansora.products.narrava_studio.services import align

    monkeypatch.setattr(align, "lyric_words", lambda *a, **k: transcript)
    return align.lyric_alignment("song.mp3", lyrics, total_duration_sec=total)


def _heard(*words):
    """A transcript in ``lyric_words``' shape: ``(text, start, end)`` per word."""
    text = " ".join(w[0] for w in words)
    return {
        "text": text,
        "words": [(w[1], w[2]) for w in words],
        "guessed": [False] * len(words),
    }


def test_a_sheets_section_markers_and_blank_lines_never_reach_a_caption():
    from phansora.products.narrava_studio.services import align

    assert align.clean_lyrics(
        "[Verse 1]\nTake me home\n\n\n(Chorus)\n  where the river runs  \n"
    ) == "Take me home\nwhere the river runs"


def test_the_sheets_words_win_and_the_recording_only_says_when(monkeypatch):
    # Whisper misheard "river" as "rubber" and "runs" as "once" — what a band does to it.
    out = _aligned(
        monkeypatch,
        "[Verse 1]\nTake me home\nwhere the river runs\n",
        _heard(("Take", 1.0, 1.4), ("me", 1.4, 1.6), ("home", 1.7, 2.3),
               ("where", 3.0, 3.2), ("the", 3.2, 3.4), ("rubber", 3.5, 4.0),
               ("once", 4.1, 4.6)),
        total=6.0,
    )
    # The sheet is captioned, not the transcript — and its line breaks survive, because a
    # lyric line is a caption card.
    assert out["text"] == "Take me home\nwhere the river runs"
    # Every word the singing yielded keeps the second it was sung at.
    assert out["words"][0] == (1.0, 1.4)
    assert out["words"][3] == (3.0, 3.2)
    # The two it did not are placed between the words either side of them, and marked.
    assert out["guessed"] == [False, False, False, False, False, True, True]
    assert 3.4 <= out["words"][5][0] <= out["words"][6][0] <= 6.0


def test_lyrics_for_a_different_song_are_refused_rather_than_spread_over_this_one(monkeypatch):
    from phansora.products.narrava_studio.services import align

    with pytest.raises(align.AlignmentFailed) as exc:
        _aligned(
            monkeypatch,
            "totally different words about a harbour at dawn",
            _heard(("Take", 1.0, 1.4), ("me", 1.4, 1.6), ("home", 1.7, 2.3),
                   ("tonight", 2.4, 3.0)),
        )
    assert "do not look like the same track" in str(exc.value)


def test_an_empty_sheet_says_so_instead_of_timing_nothing(monkeypatch):
    from phansora.products.narrava_studio.services import align

    with pytest.raises(align.AlignmentFailed) as exc:
        _aligned(monkeypatch, "[Chorus]\n\n", _heard(("Take", 1.0, 1.4)))
    assert "no words in the lyrics" in str(exc.value)


def test_a_bare_sound_name_on_its_own_line_is_not_a_lyric():
    # Whisper writes "[Music]" over an intro, and writes it WITHOUT the brackets just as
    # readily — which put a caption card reading "Music" eleven seconds before the first
    # word of the prod song.
    out = _transcribe(
        _seg((" Music", 16.6, 18.0)),
        _seg((" I'll", 28.1, 28.8), (" wait", 28.8, 29.0)),
    )
    assert _tokens(out["text"]) == ["I'll", "wait"]
    assert out["words"] == [(28.1, 28.8), (28.8, 29.0)]


def test_a_song_that_sings_the_word_music_mid_line_keeps_it():
    out = _transcribe(_seg((" The", 1.0, 1.2), (" music", 1.2, 1.6), (" plays", 1.6, 2.0)))
    assert _tokens(out["text"]) == ["The", "music", "plays"]
