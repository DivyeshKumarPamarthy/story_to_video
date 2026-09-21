"""Tests for narrator.captions.

The generated .ass is parsed back and inspected; nothing here string-compares
a rendered file, because a formatting change that breaks nothing would fail
such a test while a timing bug that breaks everything would pass it.
"""

from __future__ import annotations

import pytest

from narrator import captions
from narrator.beats import Beat, Word
from narrator.captions import CaptionError, build_ass, parse_ass
from narrator.config import CaptionConfig

CFG = CaptionConfig(width=1920, height=1080)


def words(*specs: tuple[str, float, float]) -> list[Word]:
    return [Word(text=text, start=start, end=end) for text, start, end in specs]


def beat(index: int, *specs: tuple[str, float, float]) -> Beat:
    spoken = words(*specs)
    return Beat(
        index=index,
        text=" ".join(w.text for w in spoken),
        visual_query="",
        duration=spoken[-1].end if spoken else 0.0,
        words=spoken,
    )


SIMPLE = [
    beat(0, ("The", 0.0, 0.3), ("house", 0.3, 0.7), ("stood", 0.7, 1.0), ("empty.", 1.0, 1.4)),
    beat(1, ("Rain", 1.5, 1.8), ("came", 1.8, 2.1), ("in.", 2.1, 2.4)),
]


# --- structure --------------------------------------------------------------


def test_file_parses_back(tmp_path):
    path = build_ass(SIMPLE, CFG, tmp_path / "captions.ass")
    parsed = parse_ass(path)

    assert parsed.script_info
    assert parsed.styles
    assert parsed.events


def test_dialogue_line_count_matches_the_groups(tmp_path):
    path = build_ass(SIMPLE, CFG, tmp_path / "captions.ass")
    parsed = parse_ass(path)

    # 7 words, grouped two at a time: 2 + 2 + 2 + 1.
    assert len(parsed.events) == 4


def test_one_word_per_line_when_configured(tmp_path):
    cfg = CaptionConfig(width=1920, height=1080, words_per_line=1)
    parsed = parse_ass(build_ass(SIMPLE, cfg, tmp_path / "c.ass"))
    assert len(parsed.events) == 7


def test_resolution_is_written_from_config(tmp_path):
    cfg = CaptionConfig(width=1080, height=1920)
    parsed = parse_ass(build_ass(SIMPLE, cfg, tmp_path / "c.ass"))

    assert parsed.script_info["PlayResX"] == "1080"
    assert parsed.script_info["PlayResY"] == "1920"


def test_style_is_centred_in_the_lower_third(tmp_path):
    parsed = parse_ass(build_ass(SIMPLE, CFG, tmp_path / "c.ass"))
    style = parsed.styles["Karaoke"]

    assert style["Alignment"] == "2", "2 is bottom-centre in ASS"
    assert int(style["MarginV"]) > 0


def test_empty_beats_produce_a_valid_file_with_no_events(tmp_path):
    parsed = parse_ass(build_ass([], CFG, tmp_path / "c.ass"))
    assert parsed.events == []
    assert parsed.styles


def test_beat_without_words_raises(tmp_path):
    unaligned = Beat(index=0, text="The house stood empty.", visual_query="", duration=1.0)

    with pytest.raises(CaptionError, match="no word timings"):
        build_ass([unaligned], CFG, tmp_path / "c.ass")


# --- timings ----------------------------------------------------------------


def test_timings_match_the_words_to_the_centisecond(tmp_path):
    parsed = parse_ass(build_ass(SIMPLE, CFG, tmp_path / "c.ass"))

    assert parsed.events[0].start == pytest.approx(0.0, abs=0.005)
    assert parsed.events[0].end == pytest.approx(0.7, abs=0.005)
    assert parsed.events[1].start == pytest.approx(0.7, abs=0.005)
    assert parsed.events[1].end == pytest.approx(1.4, abs=0.005)


def test_timings_are_monotonic_and_never_overlap(tmp_path):
    parsed = parse_ass(build_ass(SIMPLE, CFG, tmp_path / "c.ass"))

    for earlier, later in zip(parsed.events, parsed.events[1:], strict=False):
        assert earlier.end <= later.start


def test_centisecond_rounding_is_exact(tmp_path):
    odd = [beat(0, ("One", 0.0, 0.333), ("two", 0.333, 1.267))]
    parsed = parse_ass(build_ass(odd, CFG, tmp_path / "c.ass"))

    assert parsed.events[0].raw_start == "0:00:00.00"
    assert parsed.events[0].raw_end == "0:00:01.27"


def test_hours_are_formatted_correctly(tmp_path):
    late = [beat(0, ("Late", 3671.5, 3672.0))]
    parsed = parse_ass(build_ass(late, CFG, tmp_path / "c.ass"))

    assert parsed.events[0].raw_start == "1:01:11.50"


def test_karaoke_tags_carry_per_word_durations(tmp_path):
    parsed = parse_ass(build_ass(SIMPLE, CFG, tmp_path / "c.ass"))

    # \k is in centiseconds: "The" is 0.30s -> 30, "house" is 0.40s -> 40.
    assert r"\k30" in parsed.events[0].text
    assert r"\k40" in parsed.events[0].text


# --- escaping ---------------------------------------------------------------


@pytest.mark.parametrize(
    ("raw", "must_not_contain"),
    [
        ("{curly}", "{curly}"),
        ("back\\slash", "back\\slash"),
    ],
)
def test_special_characters_are_escaped(tmp_path, raw, must_not_contain):
    tricky = [beat(0, (raw, 0.0, 0.5), ("after", 0.5, 1.0))]
    path = build_ass(tricky, CFG, tmp_path / "c.ass")

    parsed = parse_ass(path)
    assert must_not_contain not in parsed.events[0].text


def test_braces_cannot_open_an_override_block(tmp_path):
    # A literal { in a caption would be read as the start of an ASS override
    # block and swallow the rest of the line.
    tricky = [beat(0, ("{\\an8}gotcha", 0.0, 0.5))]
    parsed = parse_ass(build_ass(tricky, CFG, tmp_path / "c.ass"))

    # The property is that no brace is left unescaped: with our own karaoke
    # tags and every escaped brace removed, none may remain to open a block.
    import re

    text = parsed.events[0].text
    without_tags = re.sub(r"\{\\k\d+\}", "", text)
    leftover = without_tags.replace(r"\{", "").replace(r"\}", "")
    assert "{" not in leftover and "}" not in leftover


def test_newlines_do_not_break_the_dialogue_line(tmp_path):
    tricky = [beat(0, ("two\nlines", 0.0, 0.5))]
    path = build_ass(tricky, CFG, tmp_path / "c.ass")

    dialogue_lines = [
        line for line in path.read_text().splitlines() if line.startswith("Dialogue:")
    ]
    assert len(dialogue_lines) == 1


def test_commas_in_text_do_not_shift_the_fields(tmp_path):
    # Dialogue: is comma separated with the text last, so a comma in a word
    # must not be read as a new field.
    tricky = [beat(0, ("well,", 0.0, 0.5), ("maybe", 0.5, 1.0))]
    parsed = parse_ass(build_ass(tricky, CFG, tmp_path / "c.ass"))

    assert "well," in parsed.events[0].text


# --- wrapping ---------------------------------------------------------------


def test_no_rendered_line_exceeds_the_wrap_width(tmp_path):
    long_words = [
        beat(
            0,
            *[(f"supercalifragilistic{i}", i * 0.3, i * 0.3 + 0.3) for i in range(8)],
        )
    ]
    cfg = CaptionConfig(width=1920, height=1080, wrap_width=30)
    parsed = parse_ass(build_ass(long_words, cfg, tmp_path / "c.ass"))

    for event in parsed.events:
        for line in captions.visible_text(event.text).split(r"\N"):
            assert len(line) <= 30, f"{line!r} is longer than the wrap width"


def test_grouping_never_splits_across_beats(tmp_path):
    parsed = parse_ass(build_ass(SIMPLE, CFG, tmp_path / "c.ass"))

    # Beat 0 ends at 1.4 and beat 1 starts at 1.5; no event may span both.
    for event in parsed.events:
        assert not (event.start < 1.4 < event.end)


# --- config -----------------------------------------------------------------


def test_defaults_are_sane():
    cfg = CaptionConfig()
    assert cfg.words_per_line in (1, 2)
    assert cfg.font_size > 0
    assert cfg.wrap_width > 0


def test_invalid_words_per_line_raises():
    with pytest.raises(ValueError, match="words_per_line"):
        CaptionConfig(words_per_line=0)


def test_vertical_preset_uses_a_bigger_font(tmp_path):
    landscape = CaptionConfig.preset("landscape")
    vertical = CaptionConfig.preset("vertical")
    assert vertical.font_size > landscape.font_size
    assert (vertical.width, vertical.height) == (1080, 1920)
