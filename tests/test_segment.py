"""Tests for narrator.segment.

The splitter is exercised in two modes:

* ``max_chars=1`` forces one sentence per beat, so the beat list *is* the
  sentence list. This is how boundary decisions (abbreviations, dialogue) are
  asserted without packing getting in the way.
* a realistic ``max_chars`` exercises the packing.
"""

from __future__ import annotations

import re

from narrator.beats import Beat
from narrator.segment import segment

TERMINAL = ".!?…"
CLOSERS = "\"'”’)]»"


def ends_on_terminal(text: str) -> bool:
    stripped = text.rstrip(CLOSERS)
    return bool(stripped) and stripped[-1] in TERMINAL


def words(text: str) -> list[str]:
    return re.findall(r"\S+", text)


STORY = (
    "The house had been empty for nine years. Rain had got in through the roof "
    "and taken the ceiling with it. Mara stopped at the gate and counted the "
    "windows, the way she always had. There were seven. There had always been "
    "seven, and that was the trouble, because the house she remembered had eight."
)


# --- degenerate input -------------------------------------------------------


def test_empty_string_returns_empty_list():
    assert segment("") == []


def test_whitespace_only_returns_empty_list():
    for blank in ("   ", "\n", "\n\n\t  \n", " \r\n "):
        assert segment(blank) == [], f"expected no beats for {blank!r}"


# --- shape ------------------------------------------------------------------


def test_single_sentence_returns_one_beat():
    beats = segment("The house stood empty.")
    assert len(beats) == 1
    assert beats[0].text == "The house stood empty."


def test_returns_beat_objects_with_only_index_and_text_filled():
    for beat in segment(STORY):
        assert isinstance(beat, Beat)
        assert beat.visual_query == ""
        assert beat.audio_path is None
        assert beat.duration is None
        assert beat.words == []
        assert beat.asset_path is None


def test_indices_are_contiguous_from_zero_and_match_position():
    beats = segment(STORY, max_chars=60)
    assert len(beats) > 1, "fixture should produce several beats"
    assert [b.index for b in beats] == list(range(len(beats)))


def test_default_max_chars_is_220():
    import inspect

    assert inspect.signature(segment).parameters["max_chars"].default == 220


# --- packing ----------------------------------------------------------------


def test_no_beat_exceeds_max_chars():
    # STORY's longest sentence is 98 chars, so at these limits nothing has an
    # excuse to overflow.
    for max_chars in (100, 150, 220):
        for beat in segment(STORY, max_chars=max_chars):
            assert len(beat.text) <= max_chars, f"{len(beat.text)} > {max_chars}: {beat.text!r}"


def test_a_beat_over_max_chars_is_always_a_single_sentence():
    # The only licence to exceed max_chars is a sentence that cannot be split.
    for max_chars in (10, 25, 40, 60):
        for beat in segment(STORY, max_chars=max_chars):
            if len(beat.text) > max_chars:
                assert len(segment(beat.text, max_chars=1)) == 1, (
                    f"oversized beat holds more than one sentence: {beat.text!r}"
                )


def test_packing_fills_beats_rather_than_one_sentence_each():
    # STORY has 5 sentences; at 220 chars they must combine into fewer beats.
    assert len(segment(STORY, max_chars=220)) < 5


def test_single_unsplittable_sentence_exceeds_max_chars_without_raising():
    sentence = "She walked " + "on and on " * 40 + "until the road ended."
    assert len(sentence) > 220

    beats = segment(sentence)

    assert len(beats) == 1
    assert len(beats[0].text) > 220
    assert beats[0].text == sentence


# --- boundaries -------------------------------------------------------------


def test_every_beat_ends_on_terminal_punctuation():
    for beat in segment(STORY, max_chars=80):
        assert ends_on_terminal(beat.text), f"does not end on terminal punctuation: {beat.text!r}"


def test_one_sentence_per_beat_when_max_chars_forces_it():
    beats = segment(STORY, max_chars=1)
    assert [b.text for b in beats] == [
        "The house had been empty for nine years.",
        "Rain had got in through the roof and taken the ceiling with it.",
        "Mara stopped at the gate and counted the windows, the way she always had.",
        "There were seven.",
        "There had always been seven, and that was the trouble, because the house she "
        "remembered had eight.",
    ]


def test_splits_on_question_and_exclamation_marks():
    beats = segment("Where had it gone? Nobody knew! The street stayed quiet.", max_chars=1)
    assert [b.text for b in beats] == [
        "Where had it gone?",
        "Nobody knew!",
        "The street stayed quiet.",
    ]


def test_ellipsis_and_repeated_punctuation_do_not_produce_empty_beats():
    beats = segment("She waited... Nothing came. Really?! Nothing at all.", max_chars=1)
    assert all(b.text.strip() for b in beats)
    assert [b.text for b in beats] == [
        "She waited...",
        "Nothing came.",
        "Really?!",
        "Nothing at all.",
    ]


# --- dialogue ---------------------------------------------------------------


def test_dialogue_with_question_mark_stays_in_one_beat():
    beats = segment('"Is anyone there?" she called into the dark.', max_chars=1)
    assert [b.text for b in beats] == ['"Is anyone there?" she called into the dark.']


def test_dialogue_with_exclamation_mark_stays_in_one_beat():
    beats = segment('"Get out of the house!" he shouted from the stairs.', max_chars=1)
    assert [b.text for b in beats] == ['"Get out of the house!" he shouted from the stairs.']


def test_dialogue_with_curly_quotes_stays_in_one_beat():
    beats = segment("“Is anyone there?” she called into the dark.", max_chars=1)
    assert [b.text for b in beats] == ["“Is anyone there?” she called into the dark."]


def test_quote_marks_are_balanced_within_every_beat():
    text = (
        '"Is anyone there?" she called into the dark. Nothing answered. '
        '"I know you can hear me!" she said, louder this time. The house stayed shut.'
    )
    for max_chars in (1, 40, 80, 220):
        for beat in segment(text, max_chars=max_chars):
            assert beat.text.count('"') % 2 == 0, f"quote split across beats: {beat.text!r}"


def test_dialogue_followed_by_new_sentence_still_splits():
    beats = segment('"Get out!" The door slammed behind her.', max_chars=1)
    assert [b.text for b in beats] == ['"Get out!"', "The door slammed behind her."]


# --- abbreviations ----------------------------------------------------------


def test_abbreviations_do_not_trigger_a_split():
    text = (
        "Mr. Holloway met Dr. Reyes at the gate. "
        "The letter had come from the U.S. that morning. "
        "He wanted something simple, e.g. tea and quiet. "
        "The rain, i.e. the reason for all of it, had not stopped."
    )
    beats = segment(text, max_chars=1)
    assert [b.text for b in beats] == [
        "Mr. Holloway met Dr. Reyes at the gate.",
        "The letter had come from the U.S. that morning.",
        "He wanted something simple, e.g. tea and quiet.",
        "The rain, i.e. the reason for all of it, had not stopped.",
    ]


def test_initials_do_not_trigger_a_split():
    beats = segment("She had read J. R. R. Tolkien twice. Then she stopped.", max_chars=1)
    assert [b.text for b in beats] == [
        "She had read J. R. R. Tolkien twice.",
        "Then she stopped.",
    ]


# --- text preservation ------------------------------------------------------


def test_no_text_is_lost():
    beats = segment(STORY, max_chars=50)
    assert words(" ".join(b.text for b in beats)) == words(STORY)


def test_unterminated_final_fragment_is_still_emitted():
    beats = segment("The house stood empty. Then nothing at all")
    joined = " ".join(b.text for b in beats)
    assert "Then nothing at all" in joined


def test_internal_whitespace_and_newlines_are_normalised():
    beats = segment("The house\n  stood   empty.\nRain came in.", max_chars=220)
    for beat in beats:
        assert "\n" not in beat.text
        assert "  " not in beat.text
        assert beat.text == beat.text.strip()


def test_paragraph_break_starts_a_new_beat():
    text = "The house stood empty.\n\nRain came in through the roof."
    beats = segment(text, max_chars=220)
    assert [b.text for b in beats] == [
        "The house stood empty.",
        "Rain came in through the roof.",
    ]


# --- purity -----------------------------------------------------------------


def test_is_deterministic_and_does_not_mutate_input():
    text = STORY
    first = [b.text for b in segment(text, max_chars=70)]
    second = [b.text for b in segment(text, max_chars=70)]
    assert first == second
    assert text == STORY
