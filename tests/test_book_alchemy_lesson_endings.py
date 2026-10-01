"""How a lesson ends, and how long it runs.

The report that prompted this: lessons ended halfway through a sentence, and some
ran for one to three minutes.

Both had the same shape. The script writer's reply was capped at
`max_words * 1.7 + 400` tokens, where `max_words` is a pacing hint the writer is
free to exceed (it has a note list to teach in full) — and `chat()` threw away the
`finish_reason` that says the cap was hit. So a reply that ran out of room was
recorded as it stood. The cap was smallest on the smallest lessons, and nothing
stopped the segmentation model proposing a lesson out of one short section, so
the shortest lessons were also the ones cut off.

Pinned here: a script is allowed to finish and never ends on half a sentence, a
lesson fills a sitting wherever the source has that much to teach, and it stops
where the source stops rather than wherever a chunk happened to end.
"""

from __future__ import annotations

import asyncio
import random
from types import SimpleNamespace

from phansora.products.book_alchemy import prompts
from phansora.products.book_alchemy.chunking import build_chunks
from phansora.products.book_alchemy.deepseek_client import (
    MAX_CONTINUATIONS,
    MAX_OUTPUT_TOKENS,
    DeepSeekClient,
)
from phansora.products.book_alchemy.parsers import Block, ParsedDoc
from phansora.products.book_alchemy.pipeline import (
    DEPTHS,
    MAX_LESSON_MINUTES,
    MIN_LESSON_MINUTES,
    MIN_LESSON_WORDS,
    WORDS_PER_MINUTE,
    _plan_cuts,
    _resolve_lessons,
    lesson_word_budget,
    planned_lesson_words,
)
from phansora.products.book_alchemy.sentences import (
    complete_ending,
    ends_complete,
    last_sentence_end,
)

STANDARD = DEPTHS["standard"]


# ── Where a sentence ends ────────────────────────────────────────────────────

def test_a_finished_text_is_recognized():
    assert ends_complete("The treaty was signed in 1648.")
    assert ends_complete('She asked, "Why now?"')
    assert ends_complete("It held (for a while).  \n")
    assert ends_complete("And then…")


def test_a_cut_off_text_is_recognized():
    assert not ends_complete("The treaty was signed in 1648, and the")
    assert not ends_complete("The treaty was signed in")
    assert not ends_complete("")


def test_the_last_finished_sentence_is_found():
    text = "One idea. A second idea. And a thi"
    assert text[: last_sentence_end(text)] == "One idea. A second idea."
    assert last_sentence_end("no ending here") == 0


def test_a_title_or_an_initial_is_not_the_end_of_a_sentence():
    """Otherwise a script trimmed to its 'last sentence' ends on "Dr."."""
    text = "The method was sound. It was refined by Dr. Hale and J. R. Okoro, who"
    assert text[: last_sentence_end(text)] == "The method was sound."


def test_a_period_at_the_very_end_always_counts():
    """Nothing follows it, so it cannot be abbreviating anything."""
    assert ends_complete("The work was finished by Dr.")
    assert complete_ending("She left it to the U.S.") == ("She left it to the U.S.", "")


def test_a_ragged_ending_is_trimmed_to_the_last_sentence():
    kept, dropped = complete_ending("First point. Second point. The third poi")
    assert kept == "First point. Second point."
    assert dropped == "The third poi"


def test_a_complete_script_is_left_alone():
    script = "First point.\n\nSecond point."
    assert complete_ending(script) == (script, "")


def test_a_text_with_no_sentence_at_all_is_kept():
    """An empty script is a worse failure than an unpunctuated one."""
    assert complete_ending("just a heading") == ("just a heading", "")


# ── A reply that is allowed to finish ────────────────────────────────────────

class ScriptedClient(DeepSeekClient):
    """A client whose model replies are a fixed list of (text, finish_reason)."""

    def __init__(self, replies):
        self.cfg = SimpleNamespace(timeout_s=180)
        self.replies = list(replies)
        self.calls: list[dict] = []

    async def _completion(self, **kwargs):
        self.calls.append(kwargs)
        return self.replies.pop(0)


def _write(client: DeepSeekClient):
    return asyncio.run(client.chat_to_end(system="sys", user="usr"))


def test_a_reply_that_finishes_is_returned_as_is():
    client = ScriptedClient([("All of it. Every word.", "stop")])
    assert _write(client) == ("All of it. Every word.", True)
    assert len(client.calls) == 1


def test_the_writer_is_given_the_whole_output_ceiling():
    """The cap that used to end lessons was sized from a word target. The length
    of a lesson is the prompt's business; the cap must never be what decides it."""
    client = ScriptedClient([("Done.", "stop")])
    _write(client)
    assert client.calls[0]["max_output_tokens"] == MAX_OUTPUT_TOKENS


def test_a_cut_off_reply_is_carried_on_from_its_last_sentence():
    """The failure itself: finish_reason 'length' used to be thrown away, and the
    lesson was recorded ending on 'and the'."""
    client = ScriptedClient([
        ("The first idea is settled. The second idea begins with the", "length"),
        ("The second idea is now complete. So is the third.", "stop"),
    ])
    text, finished = _write(client)

    assert finished
    assert text == (
        "The first idea is settled. "
        "The second idea is now complete. So is the third."
    )
    # The half sentence is not handed back: the model resumes from a clean edge.
    assert client.calls[1]["continue_from"] == "The first idea is settled."


def test_a_paragraph_break_survives_the_join():
    client = ScriptedClient([
        ("One paragraph ends here.\n\nThe next one starts and", "length"),
        ("The next one, whole.", "stop"),
    ])
    text, _ = _write(client)
    assert text == "One paragraph ends here.\n\nThe next one, whole."


def test_a_reply_that_never_finishes_says_so():
    """Bounded: the caller is told, and trims the ragged end itself."""
    replies = [(f"Sentence {i}. And then the", "length") for i in range(MAX_CONTINUATIONS + 1)]
    client = ScriptedClient(replies)
    text, finished = _write(client)

    assert not finished
    assert len(client.calls) == MAX_CONTINUATIONS + 1
    assert complete_ending(text)[0].endswith(f"Sentence {MAX_CONTINUATIONS}.")


def test_the_continuation_turn_reaches_the_model_after_its_own_words():
    """The invariant, not the wording: the reply so far goes back as the model's
    own turn, followed by the ask to carry on."""
    seen: dict = {}

    class Recorder:
        status = 200

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        async def json(self):
            return {"choices": [{"message": {"content": "More."}, "finish_reason": "stop"}]}

    class Session:
        def __init__(self, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *exc):
            return False

        def post(self, url, json, headers):
            seen.update(json)
            return Recorder()

    import phansora.products.book_alchemy.deepseek_client as mod

    client = DeepSeekClient(SimpleNamespace(
        base_url="http://x", api_key="k", model="m", timeout_s=5,
        max_retries=0, min_retry_sleep_s=0, max_retry_sleep_s=0,
    ))
    real = mod.aiohttp.ClientSession
    mod.aiohttp.ClientSession = Session
    try:
        asyncio.run(client._completion(
            system="sys", user="usr", max_output_tokens=10, temperature=0.0,
            json_mode=False, continue_from="Said so far.",
        ))
    finally:
        mod.aiohttp.ClientSession = real

    assert [m["role"] for m in seen["messages"]] == ["system", "user", "assistant", "user"]
    assert seen["messages"][2]["content"] == "Said so far."
    assert seen["messages"][3]["content"] == prompts.CONTINUE_USER


# ── How long a lesson is planned to run ──────────────────────────────────────

# 715 source words of ordinary prose plan to two minutes of narration at the
# default depth (715 * 0.42 = 300 words at 150 wpm), which keeps the arithmetic
# below readable: n segments is 2n minutes.
def seg(chapter: str | None = None, *, words: int = 715, concepts: int = 10) -> dict:
    return {"chapter": chapter, "words": words, "topics": ["t"], "concepts": concepts}


def book(n: int, chapter: str | None = None) -> list[dict]:
    return [{**seg(chapter), "ordinal": i} for i in range(n)]


def part(start: int, title: str = "") -> dict:
    return {"start": start, "title": title, "summary": "", "topics": [f"{title} topic"]}


def minutes(lesson: dict) -> float:
    return lesson["planned_words"] / WORDS_PER_MINUTE


def spans(lessons: list[dict]) -> list[tuple[int, int]]:
    return [(l["start"], l["end"]) for l in lessons]


def test_a_short_part_is_folded_into_its_neighbor():
    """The one-to-three-minute lesson. The model is free to propose a part out of
    a single short section; it no longer becomes a lesson of its own."""
    lessons = _resolve_lessons([part(0), part(1), part(8)], book(15), STANDARD)

    assert spans(lessons) == [(0, 7), (8, 14)]
    assert all(minutes(l) >= MIN_LESSON_MINUTES for l in lessons)


def test_a_short_last_part_is_folded_back():
    lessons = _resolve_lessons([part(0), part(7), part(14)], book(15), STANDARD)
    assert spans(lessons) == [(0, 6), (7, 14)]


def test_lessons_end_where_the_subject_changes():
    """Parts that already fill a sitting are left exactly where the model put them."""
    lessons = _resolve_lessons([part(0), part(6), part(12)], book(18), STANDARD)
    assert spans(lessons) == [(0, 5), (6, 11), (12, 17)]


def test_lessons_sit_near_the_floor_when_the_stopping_points_allow():
    """Five-minute sections become ten-minute lessons, not twenty-minute ones:
    as close to the floor as the places the source stops will allow."""
    # Sections of four and six minutes, alternating.
    entries = [part(s) for s in (0, 2, 5, 7, 10, 12, 15, 17)]
    lessons = _resolve_lessons(entries, book(20), STANDARD)
    assert spans(lessons) == [(0, 4), (5, 9), (10, 14), (15, 19)]


def test_a_chapter_change_is_a_stopping_point_the_model_did_not_have_to_name():
    digests = [{**seg("One"), "ordinal": i} for i in range(6)]
    digests += [{**seg("Two"), "ordinal": 6 + i} for i in range(6)]
    lessons = _resolve_lessons([part(0)], digests, STANDARD)
    assert spans(lessons) == [(0, 5), (6, 11)]


def test_a_topic_a_little_over_the_cap_is_not_cut_mid_thought():
    """Twenty-two minutes on one subject, with nowhere natural to stop. Running
    two minutes long beats ending a lesson in the middle of it."""
    lessons = _resolve_lessons([part(0)], book(11), STANDARD)
    assert spans(lessons) == [(0, 10)]
    assert minutes(lessons[0]) > MAX_LESSON_MINUTES


def test_a_topic_far_over_the_cap_is_split_evenly():
    lessons = _resolve_lessons([part(0, "Long")], book(16), STANDARD)
    assert spans(lessons) == [(0, 7), (8, 15)]
    assert [l["title"] for l in lessons] == ["Long", "Long (part 2)"]


def test_a_source_with_no_structure_gets_as_few_cuts_as_possible():
    """Every cut here lands mid-topic, so make as few as the cap allows."""
    lessons = _resolve_lessons([part(0)], book(30), STANDARD)
    assert len(lessons) == 3
    assert all(MIN_LESSON_MINUTES <= minutes(l) <= MAX_LESSON_MINUTES + 0.5 for l in lessons)


def test_a_work_that_fits_one_sitting_is_one_lesson():
    """However many chapters it has — still decided in code."""
    digests = [{**seg(f"Ch {i}"), "ordinal": i} for i in range(9)]
    lessons = _resolve_lessons([part(0), part(3), part(6)], digests, STANDARD)
    assert spans(lessons) == [(0, 8)]


def test_a_stretch_with_few_ideas_is_planned_by_what_it_will_be_asked_for():
    """A genealogy is thousands of words and two concepts. Planned by its word
    count it would be cut into "ten-minute" lessons the writer is then asked to
    fill with three minutes of material."""
    sparse = planned_lesson_words(concept_count=2, source_words=5000, depth=STANDARD)
    dense = planned_lesson_words(concept_count=60, source_words=5000, depth=STANDARD)
    assert sparse < MIN_LESSON_WORDS < dense

    # Twelve sparse segments are 24 minutes by word count but only ~9 by content,
    # so they are one lesson rather than two thin ones.
    digests = [{**seg(concepts=0), "ordinal": i} for i in range(12)]
    digests += [{**seg(), "ordinal": 12 + i} for i in range(12)]
    lessons = _resolve_lessons([part(0), part(6), part(12)], digests, STANDARD)
    assert lessons[0]["start"] == 0 and lessons[0]["end"] >= 11


def test_merged_parts_keep_both_names_and_all_their_topics():
    lessons = _resolve_lessons(
        [part(0, "Origins"), part(2, "Growth"), part(8, "Decline")], book(15), STANDARD
    )
    assert [l["title"] for l in lessons] == ["Origins – Growth", "Decline"]
    assert lessons[0]["topics"] == ["Origins topic", "Growth topic"]
    assert [l["ordinal"] for l in lessons] == [1, 2]


def test_every_segment_lands_in_exactly_one_lesson():
    """The contract the rest of the pipeline indexes by, over many random books."""
    rng = random.Random(7)
    for _ in range(200):
        n = rng.randint(1, 60)
        digests = [
            {**seg(f"c{rng.randint(0, 6)}" if rng.random() < 0.5 else None,
                   words=rng.randint(40, 900), concepts=rng.randint(0, 14)),
             "ordinal": i}
            for i in range(n)
        ]
        starts = sorted({0, *(rng.randrange(n) for _ in range(rng.randint(0, 8)))})
        lessons = _resolve_lessons([part(s, f"p{s}") for s in starts], digests, STANDARD)

        covered = [i for l in lessons for i in range(l["start"], l["end"] + 1)]
        assert covered == list(range(n))
        assert [l["ordinal"] for l in lessons] == list(range(1, len(lessons) + 1))
        assert all(l["title"] and l["planned_words"] >= 0 for l in lessons)


def test_no_runt_survives_beside_a_neighbor_that_could_take_it():
    """A lesson well under the floor exists only where joining either neighbor
    would run past the cap."""
    rng = random.Random(11)
    for _ in range(200):
        n = rng.randint(2, 50)
        digests = [{**seg(words=rng.randint(100, 900)), "ordinal": i} for i in range(n)]
        natural = {i for i in range(1, n) if rng.random() < 0.4}
        cuts = _plan_cuts(digests, natural, STANDARD) + [n]

        def length(a: int, b: int) -> float:
            words = sum(d["words"] for d in digests[a:b])
            concepts = sum(d["concepts"] for d in digests[a:b])
            return planned_lesson_words(
                concept_count=concepts, source_words=words, depth=STANDARD
            ) / WORDS_PER_MINUTE

        for i in range(len(cuts) - 1):
            if length(cuts[i], cuts[i + 1]) >= MIN_LESSON_MINUTES - 2:
                continue
            if i > 0:
                assert length(cuts[i - 1], cuts[i + 1]) > MAX_LESSON_MINUTES
            if i + 2 < len(cuts):
                assert length(cuts[i], cuts[i + 2]) > MAX_LESSON_MINUTES


# ── What the writer is asked for ─────────────────────────────────────────────

def test_a_lesson_planned_to_fill_a_sitting_is_never_asked_for_less():
    """The undershoot allowance used to let a ten-minute plan ask for seven and a
    half. The bottom of the range is the floor whenever the lesson can reach it."""
    lo, hi = lesson_word_budget(concept_count=40, source_words=2600, depth=STANDARD)
    assert lo == MIN_LESSON_WORDS
    assert hi > lo


def test_a_lesson_with_little_to_teach_is_still_allowed_to_be_short():
    """"If there's enough content": a thin source is not padded to ten minutes."""
    lo, hi = lesson_word_budget(concept_count=3, source_words=900, depth=STANDARD)
    assert hi < MIN_LESSON_WORDS


# ── Chunk boundaries are the only places a lesson can end ────────────────────

SENTENCE = "The river rose through the night and the lower fields were lost by morning. "


def _doc(blocks: list[Block]) -> ParsedDoc:
    return ParsedDoc(title="t", blocks=blocks)


def test_a_chunk_does_not_end_on_half_a_sentence():
    """A PDF paragraph that runs over the page arrives as two blocks. A chunk that
    fills up between them used to end mid-sentence, and so could a lesson."""
    page_one = SENTENCE * 40 + "The bridge at the north crossing was the last to"
    page_two = "give way, a little before dawn. " + SENTENCE * 40
    chunks = build_chunks(_doc([
        Block(text=page_one, page_start=1, page_end=1),
        Block(text=page_two, page_start=2, page_end=2),
    ]))

    assert len(chunks) == 2
    assert ends_complete(chunks[0]["text"])
    assert chunks[1]["text"].startswith(
        "The bridge at the north crossing was the last to\n\ngive way"
    )


def test_nothing_is_lost_when_a_tail_is_held_back():
    blocks = [Block(text=SENTENCE * 30 + f"and block {i} trails off without") for i in range(6)]
    before = " ".join(b.text for b in blocks).split()
    after = " ".join(c["text"] for c in build_chunks(_doc(blocks))).split()
    assert after == before


def test_text_with_no_sentences_is_not_shuffled_between_chunks():
    """Verse, tables, lists: a long unpunctuated tail is not a sentence caught by a
    page break, and is left where it is."""
    lines = "\n".join(f"line {i} of a long list with no full stop" for i in range(60))
    chunks = build_chunks(_doc([Block(text=lines), Block(text=lines)]))
    assert [c["text"] for c in chunks] == [lines, lines]


def test_a_chapter_change_closes_a_chunk_that_is_full_enough():
    """So the start of a chapter is a place a lesson can begin."""
    chunks = build_chunks(_doc([
        Block(text=SENTENCE * 30, chapter="One"),
        Block(text=SENTENCE * 5, chapter="Two"),
    ]))
    assert [c["chapter"] for c in chunks] == ["One", "Two"]


def test_tiny_sections_do_not_each_become_a_chunk():
    """A heading every hundred words must not turn into hundreds of model calls."""
    blocks = [Block(text=SENTENCE * 2, chapter=f"Section {i}") for i in range(40)]
    assert len(build_chunks(_doc(blocks))) < 6
