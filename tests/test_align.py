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


def test_out_of_order_words_raise(tmp_path):
    words = [dict(w) for w in PLAIN["words"]]
    words[3], words[4] = words[4], words[3]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError):
            align([beat_for(PLAIN, tmp_path)])


# --- mismatch is never smoothed over ----------------------------------------


def test_extra_transcribed_word_raises_alignment_error(tmp_path):
    words = [dict(w) for w in PLAIN["words"]]
    words.append({"text": " again", "start": 2.1, "end": 2.3})

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError, match="9 aligned words.*8 reference"):
            align([beat_for(PLAIN, tmp_path)])


def test_missing_transcribed_word_raises_alignment_error(tmp_path):
    words = [dict(w) for w in PLAIN["words"]][:-1]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError, match="7 aligned words.*8 reference"):
            align([beat_for(PLAIN, tmp_path)])


def test_empty_transcription_raises_rather_than_returning_no_words(tmp_path):
    with patch.object(align_module, "_transcribe", Mock(return_value=[])):
        with pytest.raises(AlignmentError):
            align([beat_for(PLAIN, tmp_path)])


def test_error_names_the_beat_and_shows_both_texts(tmp_path):
    words = [dict(w) for w in PLAIN["words"]][:-1]

    with patch.object(align_module, "_transcribe", Mock(return_value=words)):
        with pytest.raises(AlignmentError) as exc:
            align([beat_for(PLAIN, tmp_path, index=4)])

    message = str(exc.value)
    assert "beat 4" in message
    assert "years." in message, "the error should show what was expected"


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
def test_real_alignment_of_numerals_is_correct_or_loud_never_wrong(tmp_path):
    """Kokoro's output for a given sentence is not stable across process
    state: the same text, voice and config produce a different wav depending
    on what ran earlier in the process (verified by hashing). Whisper then
    hears "3 a.m." as either " 3"+" a"+".m." or " 3am", so the aligned count
    is 13 or 12 for the same input.

    The guarantee is therefore not "it always aligns" but "it never returns
    timings that do not match the text". Both branches are asserted; the
    deterministic merge behaviour is pinned by the recorded fixture in the
    fast suite.
    """
    from narrator.speech import synthesize

    spoken = synthesize(
        [Beat(index=0, text=TRICKY["text"], visual_query="")],
        "af_heart",
        tmp_path / "cache",
    )

    try:
        out = align(spoken)[0]
    except AlignmentError as exc:
        assert "reference words" in str(exc), "a mismatch must say what did not line up"
        return

    assert len(out.words) == len(normalise_words(TRICKY["text"]))
    for earlier, later in zip(out.words, out.words[1:], strict=False):
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
