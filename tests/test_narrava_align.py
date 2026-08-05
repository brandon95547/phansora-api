"""Alignment matching, without a speech model.

Everything the model produces is handed in as plain tuples, so these run anywhere. What is
under test is the editorial machinery: that digits and spoken numbers anchor to each other
instead of opening interpolation gaps, that the gaps which do remain are filled sanely, and
that audio which is not this script is refused rather than guessed at.
"""
import pytest


def _heard(words, step=0.5, start=0.0):
    """(word, start, end) tuples at a steady pace — a stand-in transcript."""
    out = []
    t = start
    for w in words:
        out.append((w, t, t + step * 0.8))
        t += step
    return out


# ── spoken forms ─────────────────────────────────────────────────────────────

def test_years_speak_in_pairs_not_cardinals():
    from phansora.products.narrava_studio.services import align

    assert align._spoken_forms("1969") == ["nineteen", "sixty", "nine"]
    assert align._spoken_forms("2025") == ["twenty", "twenty", "five"]
    assert align._spoken_forms("1900") == ["nineteen", "hundred"]
    assert align._spoken_forms("2000") == ["two", "thousand"]
    assert align._spoken_forms("2005") == ["two", "thousand", "five"]
    assert align._spoken_forms("1905") == ["nineteen", "oh", "five"]


def test_plain_numbers_speak_as_cardinals():
    from phansora.products.narrava_studio.services import align

    assert align._spoken_forms("42") == ["forty", "two"]
    assert align._spoken_forms("7") == ["seven"]
    assert align._spoken_forms("300") == ["three", "hundred"]
    assert align._spoken_forms("12000") == ["twelve", "thousand"]
    assert align._spoken_forms("0") == ["zero"]


def test_ordinals_speak_as_ordinals():
    from phansora.products.narrava_studio.services import align

    assert align._spoken_forms("21st") == ["twenty", "first"]
    assert align._spoken_forms("3rd") == ["third"]
    assert align._spoken_forms("20th") == ["twentieth"]
    assert align._spoken_forms("12th") == ["twelfth"]


def test_ordinary_words_and_id_numbers_pass_through():
    from phansora.products.narrava_studio.services import align

    assert align._spoken_forms("earth") == ["earth"]
    assert align._spoken_forms("didn't") == ["didn't"]
    # A 13+ digit run is a serial, not something anyone reads as a cardinal.
    assert align._spoken_forms("1234567890123") == ["1234567890123"]


# ── anchoring ────────────────────────────────────────────────────────────────

def test_digits_in_the_script_anchor_to_a_spelled_out_transcript():
    from phansora.products.narrava_studio.services import align

    said = ["in", "1969", "three", "men", "left"]
    heard = _heard(["in", "nineteen", "sixty", "nine", "three", "men", "left"])
    times, ratio = align._anchored(said, heard)

    assert ratio == 1.0
    # The year spans from the first spoken piece to the last — when it was being said.
    assert times[1][0] == heard[1][1]
    assert times[1][1] == heard[3][2]
    # And the words after it carry their own real times, not interpolations.
    assert times[2] == (heard[4][1], heard[4][2])


def test_spelled_numbers_in_the_script_anchor_to_a_digit_transcript():
    from phansora.products.narrava_studio.services import align

    said = ["nineteen", "sixty", "nine", "was", "loud"]
    heard = _heard(["1969", "was", "loud"])
    times, ratio = align._anchored(said, heard)

    assert ratio == 1.0
    # All three script words share the one heard token's span.
    assert times[0] == times[1] == times[2] == (heard[0][1], heard[0][2])


def test_a_mismatched_recording_is_refused_not_guessed(monkeypatch):
    from phansora.products.narrava_studio.services import align

    monkeypatch.setattr(
        align, "_heard",
        lambda path, language: _heard(["entirely", "different", "words", "here"]),
    )
    with pytest.raises(align.AlignmentFailed):
        align.word_times("audio.mp3", "the moon landing of nineteen sixty nine")


def test_a_misheard_stretch_is_interpolated_between_real_anchors(monkeypatch):
    from phansora.products.narrava_studio.services import align

    # "quantum flux" mis-heard as garbage; neighbours heard clean.
    said_text = "the machine used quantum flux to fly"
    heard = _heard(["the", "machine", "used", "kwanza", "flocks", "to", "fly"])
    monkeypatch.setattr(align, "_heard", lambda path, language: heard)

    words = align.word_times("audio.mp3", said_text)
    assert len(words) == 7
    # The interpolated pair sits strictly between its real neighbours.
    assert words[2][1] <= words[3][0] < words[4][1] <= words[5][0] + 1e-9
    # And the clock never runs backwards.
    for prev, cur in zip(words, words[1:]):
        assert cur[0] >= prev[0]


def test_interpolation_shares_a_gap_by_word_length():
    from phansora.products.narrava_studio.services import align

    # One anchored word each side of a two-word hole: "a" (1 char) then "extraordinarily"
    # (15 chars). The long word should get the lion's share of the gap.
    times = [(0.0, 1.0), None, None, (5.0, 5.5)]
    out = align._fill_gaps(times, None, weights=[4, 1, 15, 4])
    a_width = out[1][1] - out[1][0]
    big_width = out[2][1] - out[2][0]
    assert big_width > a_width * 10
    # The pieces tile the gap exactly.
    assert out[1][0] == pytest.approx(1.0)
    assert out[2][1] == pytest.approx(5.0)
