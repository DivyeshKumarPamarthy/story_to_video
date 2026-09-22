"""Tests for narrator.align.

The default run never loads faster-whisper: `_transcribe` is the seam, and it
is fed tests/fixtures/whisper_words.json -- output recorded from the real model
running on real Kokoro audio, not invented. The wav files the tests point at
are lavfi tones of the recorded duration, because alignment only reads the
file's length once the transcription is mocked.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from unittest.mock import Mock, patch

import pytest

from narrator import align as align_module
from narrator.align import (
    END_TOLERANCE,
    AlignmentError,
    _merge_continuations,
    align,
    normalise_words,
)
from narrator.beats import Beat
from narrator.config import AlignConfig

FIXTURE = json.loads(Path(__file__).parent.joinpath("fixtures/whisper_words.json").read_text())
PLAIN = FIXTURE["plain"]
TRICKY = FIXTURE["tricky"]


def tone(path: Path, seconds: float) -> Path:
    """A wav of exactly `seconds`, via lavfi -- alignment only needs a length."""
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"sine=frequency=440:duration={seconds}",
            "-ar",
            "24000",
            str(path),
        ],
        check=True,
    )
    return path


def beat_for(case: dict, tmp_path: Path, index: int = 0) -> Beat:
    wav = tone(tmp_path / f"beat_{index}.wav", case["audio_duration"])
    return Beat(
        index=index,
        text=case["text"],
        visual_query="",
        audio_path=wav,
        duration=case["audio_duration"],
    )


def recorded(case: dict) -> Mock:
    return Mock(return_value=[dict(w) for w in case["words"]])


# --- normalisation ----------------------------------------------------------


def test_normalise_words_counts_whitespace_tokens():
    assert normalise_words("The house had been empty for nine years.") == [
        "The",
        "house",
        "had",
        "been",
        "empty",
        "for",
        "nine",
        "years.",
    ]


def test_normalise_words_drops_punctuation_only_tokens():
    assert normalise_words("She stopped — and waited.") == ["She", "stopped", "and", "waited."]


def test_normalise_words_keeps_numerals_currency_and_contractions():
    assert normalise_words("It cost $40 and she didn't care at 3 a.m.") == [
        "It",
        "cost",
        "$40",
        "and",
        "she",
        "didn't",
        "care",
        "at",
        "3",
        "a.m.",
    ]


def test_normalise_words_collapses_newlines_and_runs_of_space():
    assert normalise_words("The house\n  stood   empty.") == ["The", "house", "stood", "empty."]


# --- the contract -----------------------------------------------------------


def test_fills_words_and_touches_nothing_else(tmp_path):
    original = Beat(
        index=2,
        text=PLAIN["text"],
        visual_query="empty house",
        audio_path=tone(tmp_path / "a.wav", PLAIN["audio_duration"]),
        duration=PLAIN["audio_duration"],
    )

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        out = align([original])[0]

    assert out.index == 2
    assert out.text == PLAIN["text"]
    assert out.visual_query == "empty house"
    assert out.audio_path == original.audio_path
    assert out.duration == original.duration
    assert out.words != []


def test_does_not_mutate_the_input_beats(tmp_path):
    beats = [beat_for(PLAIN, tmp_path)]

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        align(beats)

    assert beats[0].words == []


def test_empty_beat_list_returns_empty_list():
    spy = Mock()
    with patch.object(align_module, "_transcribe", spy):
        assert align([]) == []
    assert spy.call_count == 0


def test_preserves_order_and_indices(tmp_path):
    beats = [beat_for(PLAIN, tmp_path, 0), beat_for(PLAIN, tmp_path, 1)]

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        out = align(beats)

    assert [b.index for b in out] == [0, 1]
    assert all(b.words for b in out)


# --- word counts ------------------------------------------------------------


def test_aligned_word_count_equals_normalised_reference_count(tmp_path):
    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    assert len(out.words) == len(normalise_words(PLAIN["text"])) == 8


def test_numerals_and_punctuation_do_not_desync_the_count(tmp_path):
    # Recorded whisper output splits "a.m." into " a" and ".m." -- 14 tokens
    # against 13 reference words. The continuation must be merged back.
    assert len(TRICKY["words"]) == 14
    assert len(normalise_words(TRICKY["text"])) == 13

    with patch.object(align_module, "_transcribe", recorded(TRICKY)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    assert len(out.words) == 13
    assert [w.text for w in out.words] == [
        "It",
        "was",
        "3",
        "a.m.",
        "and",
        "the",
        "ticket",
        "cost",
        "$40,",
        "but",
        "she",
        "didn't",
        "care.",
    ]


def test_merged_continuation_spans_both_fragments(tmp_path):
    with patch.object(align_module, "_transcribe", recorded(TRICKY)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    fragment_a = next(w for w in TRICKY["words"] if w["text"] == " a")
    fragment_m = next(w for w in TRICKY["words"] if w["text"] == ".m.")
    merged = next(w for w in out.words if w.text == "a.m.")

    assert merged.start == fragment_a["start"]
    assert merged.end == fragment_m["end"]


def test_word_text_is_stripped_of_whisper_leading_spaces(tmp_path):
    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    for word in out.words:
        assert word.text == word.text.strip()
        assert word.text != ""


# --- timing invariants ------------------------------------------------------


@pytest.mark.parametrize("case", [PLAIN, TRICKY], ids=["plain", "tricky"])
def test_timestamps_are_monotonic_without_overlaps(tmp_path, case):
    with patch.object(align_module, "_transcribe", recorded(case)):
        out = align([beat_for(case, tmp_path)])[0]

    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start, f"{earlier} then {later} is not monotonic"
        assert earlier.end <= later.start, f"{earlier} overlaps {later}"


@pytest.mark.parametrize("case", [PLAIN, TRICKY], ids=["plain", "tricky"])
def test_each_word_ends_after_it_starts(tmp_path, case):
    with patch.object(align_module, "_transcribe", recorded(case)):
        out = align([beat_for(case, tmp_path)])[0]

    for word in out.words:
        assert word.end > word.start


@pytest.mark.parametrize("case", [PLAIN, TRICKY], ids=["plain", "tricky"])
def test_timings_stay_inside_the_audio(tmp_path, case):
    with patch.object(align_module, "_transcribe", recorded(case)):
        out = align([beat_for(case, tmp_path)])[0]

    assert out.words[0].start >= 0
    assert out.words[-1].end <= case["audio_duration"] + 0.05


def test_word_ending_past_the_audio_raises(tmp_path):
    words = [dict(w) for w in PLAIN["words"]]
    words[-1]["end"] = PLAIN["audio_duration"] + 0.5

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError, match="past the end"):
            align([beat_for(PLAIN, tmp_path)])


def test_negative_start_raises(tmp_path):
    words = [dict(w) for w in PLAIN["words"]]
    words[0]["start"] = -0.2

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError, match="before the start"):
            align([beat_for(PLAIN, tmp_path)])


def test_overlapping_words_raise_rather_than_being_clamped(tmp_path):
    words = [dict(w) for w in PLAIN["words"]]
    words[2]["end"] = words[3]["end"]  # word 2 now runs into word 3

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError, match="overlap"):
            align([beat_for(PLAIN, tmp_path)])


def test_two_swapped_words_are_absorbed_rather_than_raising(tmp_path):
    # Sequence matching finds ordered blocks, so a swap leaves one word
    # matched and the other interpolated into the gap. The output is still
    # monotonic, which is the guarantee that matters downstream.
    words = [dict(w) for w in PLAIN["words"]]
    words[3], words[4] = words[4], words[3]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    assert [w.text for w in out.words] == normalise_words(PLAIN["text"])
    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start
        assert earlier.end <= later.start


def test_non_monotonic_matched_timings_still_raise(tmp_path):
    # Text order untouched, so every word matches and the timings come
    # straight from whisper -- a backwards timestamp must not be returned.
    words = [dict(w) for w in PLAIN["words"]]
    words[4]["start"] = words[2]["start"]
    words[4]["end"] = words[2]["end"]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError):
            align([beat_for(PLAIN, tmp_path)])


# --- mismatch is never smoothed over ----------------------------------------


def test_extra_transcribed_word_is_tolerated(tmp_path):
    # Whisper hearing one word too many no longer fails the beat: the extra
    # token simply matches nothing, and every reference word keeps its timing.
    words = [dict(w) for w in PLAIN["words"]]
    words.append({"text": " again", "start": 2.1, "end": 2.3})

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    assert [w.text for w in out.words] == normalise_words(PLAIN["text"])


def test_missing_transcribed_word_is_interpolated(tmp_path):
    # Drop "empty" from the middle of the transcript. The reference word must
    # still come back, with a timing between its surviving neighbours.
    words = [dict(w) for w in PLAIN["words"]]
    dropped = words.pop(4)
    assert dropped["text"].strip() == "empty"

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    assert [w.text for w in out.words] == normalise_words(PLAIN["text"])
    interpolated = out.words[4]
    assert interpolated.text == "empty"
    assert out.words[3].end <= interpolated.start
    assert interpolated.end <= out.words[5].start


def test_a_dropped_word_between_touching_neighbours_still_gets_a_slot(tmp_path):
    # The hard case: whisper's words touch exactly, so removing one leaves a
    # zero-width gap. Time has to be borrowed from a neighbour rather than
    # producing a zero-length or overlapping word.
    words = [dict(w) for w in PLAIN["words"]]
    words[3]["end"] = words[4]["end"]  # word 3 now runs up to where word 5 starts
    words.pop(4)

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        out = align([beat_for(PLAIN, tmp_path)])[0]

    assert len(out.words) == 8
    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start
        assert earlier.end <= later.start
    for word in out.words:
        assert word.end > word.start


def test_garbage_transcript_still_raises(tmp_path):
    garbage = [
        {"text": f" {token}", "start": i * 0.2, "end": i * 0.2 + 0.15}
        for i, token in enumerate("koala bicycle tuesday marmalade quantum".split())
    ]

    with patch.object(align_module, "_transcribe", Mock(return_value=garbage)):
        with pytest.raises(AlignmentError, match="match ratio"):
            align([beat_for(PLAIN, tmp_path)])


def test_empty_transcription_raises_rather_than_returning_no_words(tmp_path):
    with patch.object(align_module, "_transcribe", Mock(return_value=[])):
        with pytest.raises(AlignmentError, match="match ratio"):
            align([beat_for(PLAIN, tmp_path)])


def test_error_names_the_beat_and_reports_the_ratio(tmp_path):
    garbage = [{"text": " koala", "start": 0.0, "end": 0.3}]

    with patch.object(align_module, "_transcribe", Mock(return_value=garbage)):
        with pytest.raises(AlignmentError) as exc:
            align([beat_for(PLAIN, tmp_path, index=4)])

    message = str(exc.value)
    assert "beat 4" in message
    assert "koala" in message, "the error should show what was heard"
    assert "years." in message, "the error should show what was expected"


# --- the collapsed-numeral case, which used to fail the whole beat ----------


def collapsed_tricky() -> list[dict]:
    """TRICKY's recorded words with " 3", " a", ".m." collapsed into " 3am".

    This is what the live model produces on a different rendering of the same
    sentence -- observed, not invented (see the p3 log entry).
    """
    words = [dict(w) for w in TRICKY["words"]]
    three = next(w for w in words if w["text"] == " 3")
    dot_m = next(w for w in words if w["text"] == ".m.")
    collapsed = {"text": " 3am", "start": three["start"], "end": dot_m["end"]}
    keep = [w for w in words if w["text"] not in (" 3", " a", ".m.")]
    index = words.index(three)
    return keep[:index] + [collapsed] + keep[index:]


def test_collapsed_numeral_aligns_instead_of_raising(tmp_path):
    transcript = collapsed_tricky()
    assert len(transcript) == 12
    assert len(normalise_words(TRICKY["text"])) == 13

    with patch.object(align_module, "_transcribe", Mock(return_value=transcript)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    assert [w.text for w in out.words] == normalise_words(TRICKY["text"])


def test_words_carry_reference_text_not_whisper_text(tmp_path):
    transcript = collapsed_tricky()

    with patch.object(align_module, "_transcribe", Mock(return_value=transcript)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    rendered = [w.text for w in out.words]
    assert "3am" not in rendered, "whisper's rendering leaked into the captions"
    assert "3" in rendered and "a.m." in rendered


def test_interpolated_timings_stay_monotonic_and_inside_the_audio(tmp_path):
    transcript = collapsed_tricky()

    with patch.object(align_module, "_transcribe", Mock(return_value=transcript)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    assert out.words[0].start >= 0
    assert out.words[-1].end <= TRICKY["audio_duration"] + 0.05
    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start
        assert earlier.end <= later.start
    for word in out.words:
        assert word.end > word.start


def test_interpolated_words_sit_between_their_matched_neighbours(tmp_path):
    # The gap divided is the one between the surviving matched words, which is
    # wider than the collapsed token itself -- the difference is the silence
    # around it, and putting a caption there is harmless.
    transcript = collapsed_tricky()

    with patch.object(align_module, "_transcribe", Mock(return_value=transcript)):
        out = align([beat_for(TRICKY, tmp_path)])[0]

    was = next(w for w in transcript if w["text"] == " was")
    and_ = next(w for w in transcript if w["text"] == " and")
    three = next(w for w in out.words if w.text == "3")
    am = next(w for w in out.words if w.text == "a.m.")

    assert three.start >= was["end"] - 0.001
    assert am.end <= and_["start"] + 0.001
    assert three.end == pytest.approx(am.start), "the gap should be divided, not overlapped"


def test_threshold_is_configurable(tmp_path):
    transcript = collapsed_tricky()

    with patch.object(align_module, "_transcribe", Mock(return_value=transcript)):
        with pytest.raises(AlignmentError, match="match ratio"):
            align([beat_for(TRICKY, tmp_path)], cfg=AlignConfig(min_match_ratio=0.99))

        out = align([beat_for(TRICKY, tmp_path)], cfg=AlignConfig(min_match_ratio=0.5))[0]

    assert len(out.words) == 13


def test_default_threshold_is_085():
    assert AlignConfig().min_match_ratio == 0.85


@pytest.mark.parametrize("ratio", [0.0, -0.1, 1.5])
def test_invalid_threshold_raises(ratio):
    with pytest.raises(ValueError, match="min_match_ratio"):
        AlignConfig(min_match_ratio=ratio)


# --- preconditions ----------------------------------------------------------


def test_beat_without_audio_path_raises(tmp_path):
    unsynthesised = Beat(index=0, text=PLAIN["text"], visual_query="")

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        with pytest.raises(AlignmentError, match="no audio"):
            align([unsynthesised])


def test_missing_audio_file_raises(tmp_path):
    beat = Beat(
        index=0,
        text=PLAIN["text"],
        visual_query="",
        audio_path=tmp_path / "does_not_exist.wav",
        duration=1.0,
    )

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        with pytest.raises(AlignmentError, match="does not exist"):
            align([beat])


# --- config -----------------------------------------------------------------


def test_defaults_are_base_int8_on_cpu():
    cfg = AlignConfig()
    assert cfg.model_size == "base"
    assert cfg.compute_type == "int8"
    assert cfg.device == "cpu"


def test_mps_is_rejected_with_an_explanation():
    with pytest.raises(ValueError, match="CTranslate2"):
        AlignConfig(device="mps")


def test_unknown_device_raises():
    with pytest.raises(ValueError, match="unknown device"):
        AlignConfig(device="tpu")


def test_config_reaches_the_transcriber(tmp_path):
    spy = recorded(PLAIN)
    cfg = AlignConfig(model_size="small")

    with patch.object(align_module, "_transcribe", spy):
        align([beat_for(PLAIN, tmp_path)], cfg=cfg)

    assert spy.call_args.args[1] is cfg


def test_default_suite_does_not_load_faster_whisper(tmp_path):
    import sys

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        align([beat_for(PLAIN, tmp_path)])

    assert "faster_whisper" not in sys.modules


# --- slow: the real model on real narration ---------------------------------


@pytest.mark.slow
def test_real_alignment_of_generated_narration(tmp_path):
    """End to end for this stage: Kokoro speaks it, whisper aligns it back."""
    from narrator.speech import synthesize

    spoken = synthesize(
        [Beat(index=0, text=PLAIN["text"], visual_query="")],
        "af_heart",
        tmp_path / "cache",
    )
    out = align(spoken)[0]

    assert len(out.words) == len(normalise_words(PLAIN["text"]))
    assert out.words[0].start >= 0
    assert out.words[-1].end <= out.duration + END_TOLERANCE
    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start
        assert earlier.end <= later.start


@pytest.mark.slow
def test_real_alignment_of_numerals_succeeds_whichever_rendering_is_heard(tmp_path):
    """The case that motivated tolerant alignment.

    Kokoro's output is not stable across process state, so whisper hears
    "3 a.m." as either " 3"+" a"+".m." or " 3am" for the same sentence. Before
    sequence matching, the second rendering failed the whole beat. Now both
    align to the 13 reference words.
    """
    from narrator.speech import synthesize

    spoken = synthesize(
        [Beat(index=0, text=TRICKY["text"], visual_query="")],
        "af_heart",
        tmp_path / "cache",
    )
    out = align(spoken)[0]

    assert [w.text for w in out.words] == normalise_words(TRICKY["text"])
    assert out.words[0].start >= 0
    assert out.words[-1].end <= out.duration + END_TOLERANCE
    for earlier, later in zip(out.words, out.words[1:], strict=False):
        assert earlier.start < later.start
        assert earlier.end <= later.start


@pytest.mark.slow
def test_recorded_fixture_still_matches_the_live_model(tmp_path):
    """Guards against the fixture silently going stale.

    Compares word counts, not exact tokens: the synthesised audio is not
    reproducible across process state, so the tokens for a borderline sentence
    legitimately vary. A change in how whisper chunks a plain sentence is what
    would invalidate the merge rule, and that is what this catches.
    """
    from narrator.speech import synthesize

    spoken = synthesize(
        [Beat(index=0, text=PLAIN["text"], visual_query="")],
        "af_heart",
        tmp_path / "cache",
    )
    live = align_module._transcribe(spoken[0].audio_path, AlignConfig())

    assert len(_merge_continuations(live)) == len(normalise_words(PLAIN["text"])), (
        f"live model gives {[w['text'] for w in live]} for a sentence the fixture "
        f"records as {[w['text'] for w in PLAIN['words']]}: "
        f"re-record tests/fixtures/whisper_words.json"
    )


# --- silent failures 5-6 (p8 audit) -----------------------------------------


def test_dropped_transcript_tokens_are_reported(tmp_path, caplog):
    """Item 5: tokens that normalise to nothing vanished without a word."""
    import logging

    words = [dict(w) for w in PLAIN["words"]]
    words.insert(3, {"text": " —", "start": 0.74, "end": 0.75})

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with caplog.at_level(logging.DEBUG, logger="narrator.align"):
            out = align([beat_for(PLAIN, tmp_path)])[0]

    assert len(out.words) == 8
    assert any("dropped" in r.message for r in caplog.records), (
        "a discarded transcript token left no trace"
    )


def test_interpolated_words_are_reported(tmp_path, caplog):
    """Item 6: interpolated timings are guesses and were never announced."""
    import logging

    words = [dict(w) for w in PLAIN["words"]]
    words.pop(4)

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with caplog.at_level(logging.WARNING, logger="narrator.align"):
            align([beat_for(PLAIN, tmp_path)])

    assert any("interpolat" in r.message for r in caplog.records), (
        "a guessed timing was presented as a measured one"
    )


def test_a_clean_alignment_warns_about_nothing(tmp_path, caplog):
    import logging

    with patch.object(align_module, "_transcribe", recorded(PLAIN)):
        with caplog.at_level(logging.WARNING, logger="narrator.align"):
            align([beat_for(PLAIN, tmp_path)])

    assert not [r for r in caplog.records if r.levelno >= logging.WARNING]


def test_a_low_but_passing_ratio_is_reported(tmp_path, caplog):
    """Item 6: word-order damage under the threshold was absorbed in silence."""
    import logging

    words = [dict(w) for w in PLAIN["words"]]
    words[3], words[4] = words[4], words[3]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with caplog.at_level(logging.WARNING, logger="narrator.align"):
            align([beat_for(PLAIN, tmp_path)])

    assert any("ratio" in r.message for r in caplog.records)
