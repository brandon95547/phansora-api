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


def _aligned(monkeypatch, lyrics, transcript, *, total=None, measured=None):
    """``lyric_alignment`` over a transcript handed in instead of decoded.

    The forced aligner is stubbed rather than left to the machine: with torchaudio and the
    MMS weights present it would really run, and these are about what the TRANSCRIPT path
    infers. ``measured`` stands in for it where a test wants the other branch.
    """
    from phansora.products.narrava_studio.services import align

    monkeypatch.setattr(align, "lyric_words", lambda *a, **k: transcript)
    monkeypatch.setattr(align, "_measured", lambda *a, **k: measured)
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


def test_a_line_the_transcript_missed_is_not_smeared_across_the_intro(monkeypatch):
    # Whisper missed the first sung line of the prod song entirely, so its words had nothing
    # to anchor to — and were spread from second zero across a 28-second instrumental intro,
    # which put the first caption card on screen through the whole introduction.
    out = _aligned(
        monkeypatch,
        "Take me home tonight\nwhere the river runs",
        _heard(("where", 30.0, 30.2), ("the", 30.2, 30.4), ("river", 30.5, 31.0),
               ("runs", 31.1, 31.6)),
        total=60.0,
    )
    assert out["guessed"][:4] == [True, True, True, True]
    # Placed by how long they take to sing, backwards from the first word that WAS heard —
    # "Take me home tonight" is five syllables, so a little over a second before 30.0.
    assert 28.0 <= out["words"][0][0] <= 29.5
    assert out["words"][3][1] <= 30.0
    # And the words that were heard keep exactly the seconds they were heard at.
    assert out["words"][4] == (30.0, 30.2)


def test_a_sheet_ending_past_what_was_heard_is_not_stretched_to_the_end_of_the_song(monkeypatch):
    out = _aligned(
        monkeypatch,
        "Take me home\nand the lights go down",
        _heard(("Take", 10.0, 10.4), ("me", 10.4, 10.6), ("home", 10.7, 11.3)),
        total=240.0,
    )
    # The five words nobody heard follow the singing rather than reaching for 4:00.
    assert out["words"][-1][1] < 20.0
    assert out["words"][3][0] >= 11.3


def test_a_narration_still_starts_at_the_top_of_its_own_recording():
    from phansora.products.narrava_studio.services import align

    # No paces: a narration's voice starts when the recording does, so second zero is a real
    # edge there and the leading run is interpolated from it exactly as before.
    filled = align._fill_gaps([None, None, (4.0, 4.5)], 10.0, weights=[4, 4, 4])
    assert filled[0][0] == 0.0


def test_a_whole_line_of_stage_direction_is_not_a_lyric():
    # "Outro music playing" landed at 27.9s on the prod song, right on top of the first line
    # actually sung.
    out = _transcribe(
        _seg((" Outro", 27.8, 28.0), (" music", 28.0, 28.4), (" playing", 28.4, 29.0)),
        _seg((" Holding", 31.6, 32.1), (" the", 32.1, 32.4), (" cross", 32.4, 32.8)),
    )
    assert _tokens(out["text"]) == ["Holding", "the", "cross"]


def test_a_song_that_sings_about_music_keeps_the_line():
    for line in ("The music plays", "Music is my life"):
        out = _transcribe(_seg(*[(f" {w}", i, i + 0.4) for i, w in enumerate(line.split())]))
        assert _tokens(out["text"]) == line.split(), line


# ── Reading the audio instead of the transcript ───────────────────────────────
# Everything above infers timing from what whisper wrote down. The forced aligner measures
# it off the recording, which on singing is a different order of accuracy — but it is a
# refinement of an answer that already exists, and it must never be the thing standing
# between a user and their captions.


def test_the_measured_timings_win_when_the_audio_can_be_read(monkeypatch):
    out = _aligned(
        monkeypatch,
        "Take me home",
        _heard(("Take", 1.0, 1.4), ("me", 1.4, 1.6), ("home", 1.7, 2.3)),
        total=10.0,
        measured={"words": [(5.0, 5.4), (5.4, 5.6), (5.7, 6.9)], "guessed": [False, False, True]},
    )
    assert out["words"] == [(5.0, 5.4), (5.4, 5.6), (5.7, 6.9)]
    assert out["guessed"] == [False, False, True]
    # The transcript still reports how much of the sheet it corroborated, which is what says
    # whether these are even the same song.
    assert out["heard_ratio"] == 1.0


def test_a_host_that_cannot_read_the_audio_still_gets_captions(monkeypatch):
    out = _aligned(
        monkeypatch,
        "Take me home",
        _heard(("Take", 1.0, 1.4), ("me", 1.4, 1.6), ("home", 1.7, 2.3)),
        total=10.0,
        measured=None,
    )
    assert out["words"] == [(1.0, 1.4), (1.4, 1.6), (1.7, 2.3)]


def test_every_way_the_aligner_can_fail_keeps_the_inferred_answer(monkeypatch):
    from phansora.products.narrava_studio.services import align, forced_align

    for boom in (forced_align.ForcedAlignUnavailable("no weights"), RuntimeError("out of memory")):
        monkeypatch.setattr(forced_align, "enabled", lambda: True)
        monkeypatch.setattr(forced_align, "align", lambda *a, **k: (_ for _ in ()).throw(boom))
        assert align._measured("song.mp3", "Take me home", [(1.0, 1.4), (1.4, 1.6), (1.7, 2.3)]) is None


def test_a_word_the_alphabet_cannot_spell_keeps_the_time_it_had(monkeypatch):
    from phansora.products.narrava_studio.services import align, forced_align

    monkeypatch.setattr(forced_align, "enabled", lambda: True)
    # "1969" has nothing the model's 26 letters can hold, so the aligner hands back None
    # for it and the transcript's own time stands.
    monkeypatch.setattr(forced_align, "align",
                        lambda *a, **k: [(5.0, 5.4, 0.9), None, (6.0, 6.4, 0.9)])
    out = align._measured("song.mp3", "home 1969 again", [(1.0, 1.4), (1.5, 1.9), (2.0, 2.4)])
    # Placed between the two words either side of it — which were MEASURED, so the whole
    # answer stays in one frame of reference — and marked for reading.
    assert out["words"][0] == (5.0, 5.4) and out["words"][2] == (6.0, 6.4)
    assert 5.4 <= out["words"][1][0] <= out["words"][1][1] <= 6.0
    assert out["guessed"] == [False, True, False]


def test_the_alphabet_keeps_what_it_can_and_drops_what_it_cannot():
    from phansora.products.narrava_studio.services import forced_align

    assert forced_align.spellable("Holding") == "holding"
    assert forced_align.spellable("don’t") == "don't"
    assert forced_align.spellable("well-worn") == "well-worn"
    assert forced_align.spellable("1969") == ""
    assert forced_align.spellable("—") == ""


def test_a_line_the_model_could_not_find_is_marked_for_reading():
    from phansora.products.narrava_studio.services import forced_align

    # Measured on the prod song: lines that match score 0.22-0.82, lines the singer does
    # not sing score 0.038-0.081.
    assert forced_align.unsure(0.05) is True
    assert forced_align.unsure(0.22) is False


def test_confidence_is_judged_a_line_at_a_time_not_a_word_at_a_time(monkeypatch):
    from phansora.products.narrava_studio.services import align, forced_align

    monkeypatch.setattr(forced_align, "enabled", lambda: True)
    # Line one matches; line two does not. "the" scores badly on BOTH — a short function
    # word under singing always does — and that must not be what decides either line.
    monkeypatch.setattr(forced_align, "align", lambda *a, **k: [
        (1.0, 1.4, 0.80), (1.4, 1.6, 0.02), (1.7, 2.3, 0.75),
        (3.0, 3.4, 0.04), (3.4, 3.6, 0.03), (3.7, 4.3, 0.05),
    ])
    out = align._measured(
        "song.mp3", "hold the cross\nfull of grace",
        [(1.0, 1.4)] * 6, total_duration_sec=10.0,
    )
    assert out["guessed"] == [False, False, False, True, True, True]
