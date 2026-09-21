"""Narrated wav + its own text -> word timestamps.

Fills ``Beat.words``. The trick this stage rests on: the audio was generated
by ``speech.py`` from text we still have, so the transcription is not an
unknown to be trusted -- it is a claim to be checked. If the aligned words do
not match the reference, the timings are wrong in ways that would only show up
as captions drifting off the narration, so this raises instead of returning
them.

``_transcribe`` is the seam the default test suite mocks; faster-whisper is not
imported until it runs.
"""

from __future__ import annotations

import re
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

from narrator.beats import Beat, Word
from narrator.config import AlignConfig
from narrator.speech import SynthesisError, probe_duration


class AlignmentError(RuntimeError):
    """Transcription did not match the text it was supposed to align."""


#: A word may end this far past the end of the audio before it is an error.
#: Whisper rounds to centiseconds and the last word often lands exactly on the
#: final sample.
END_TOLERANCE = 0.05

_WHITESPACE = re.compile(r"\s+")


def align(beats: list[Beat], cfg: AlignConfig | None = None) -> list[Beat]:
    """Fill ``words`` on each beat from its own audio. Input is not mutated."""
    cfg = cfg or AlignConfig()
    return [replace(beat, words=_align_beat(beat, cfg)) for beat in beats]


def normalise_words(text: str) -> list[str]:
    """Reference words, as whisper would count them.

    Whitespace tokens, minus anything with no alphanumeric character: an em
    dash on its own is punctuation the model will not emit as a word, while
    "$40", "didn't" and "a.m." are single words and must stay that way.
    """
    return [token for token in _WHITESPACE.split(text.strip()) if _is_word(token)]


def _is_word(token: str) -> bool:
    return any(char.isalnum() for char in token)


def _align_beat(beat: Beat, cfg: AlignConfig) -> list[Word]:
    audio_path = _require_audio(beat)
    words = _merge_continuations(_transcribe(audio_path, cfg))
    reference = normalise_words(beat.text)

    if len(words) != len(reference):
        raise AlignmentError(
            f"beat {beat.index}: {len(words)} aligned words but {len(reference)} reference "
            f"words. expected {' '.join(reference)!r}, "
            f"transcribed {' '.join(word.text for word in words)!r}"
        )

    _check_timings(beat, words, _audio_duration(beat, audio_path))
    return words


def _require_audio(beat: Beat) -> Path:
    if beat.audio_path is None:
        raise AlignmentError(
            f"beat {beat.index} has no audio to align; run speech.synthesize first"
        )
    audio_path = Path(beat.audio_path)
    if not audio_path.exists():
        raise AlignmentError(f"beat {beat.index} audio does not exist: {audio_path}")
    return audio_path


def _audio_duration(beat: Beat, audio_path: Path) -> float:
    try:
        return probe_duration(audio_path)
    except SynthesisError as exc:
        raise AlignmentError(f"beat {beat.index}: cannot measure {audio_path}: {exc}") from exc


def _merge_continuations(raw_words: list[dict[str, Any]]) -> list[Word]:
    """Whisper splits some words and marks the continuation by omitting the
    leading space: "a.m." comes back as " a" then ".m.".

    Merging on that signal is what keeps the word count equal to the
    reference's; without it every abbreviation desyncs the whole beat.
    """
    words: list[Word] = []
    for raw in raw_words:
        text = str(raw["text"])
        start, end = float(raw["start"]), float(raw["end"])

        if words and text[:1] and not text[:1].isspace():
            previous = words[-1]
            words[-1] = Word(
                text=previous.text + text.strip(),
                start=previous.start,
                end=max(previous.end, end),
            )
            continue

        stripped = text.strip()
        if stripped:
            words.append(Word(text=stripped, start=start, end=end))
    return words


def _check_timings(beat: Beat, words: list[Word], audio_duration: float) -> None:
    if not words:
        raise AlignmentError(f"beat {beat.index}: alignment produced no words")

    if words[0].start < 0:
        raise AlignmentError(
            f"beat {beat.index}: first word {words[0].text!r} starts before the start "
            f"of the audio ({words[0].start:.3f}s)"
        )

    if words[-1].end > audio_duration + END_TOLERANCE:
        raise AlignmentError(
            f"beat {beat.index}: last word {words[-1].text!r} ends at {words[-1].end:.3f}s, "
            f"past the end of {audio_duration:.3f}s of audio"
        )

    for word in words:
        if word.end <= word.start:
            raise AlignmentError(
                f"beat {beat.index}: word {word.text!r} ends at or before it starts "
                f"({word.start:.3f}s -> {word.end:.3f}s)"
            )

    for earlier, later in zip(words, words[1:], strict=False):
        if later.start <= earlier.start:
            raise AlignmentError(
                f"beat {beat.index}: {earlier.text!r} and {later.text!r} are out of order "
                f"({earlier.start:.3f}s then {later.start:.3f}s)"
            )
        if earlier.end > later.start:
            raise AlignmentError(
                f"beat {beat.index}: {earlier.text!r} and {later.text!r} overlap "
                f"({earlier.end:.3f}s > {later.start:.3f}s)"
            )


def _transcribe(audio_path: Path, cfg: AlignConfig) -> list[dict[str, Any]]:
    """wav -> raw word dicts. The seam the default test suite mocks."""
    model = _model(cfg.model_size, cfg.device, cfg.compute_type)
    segments, _info = model.transcribe(
        str(audio_path),
        word_timestamps=True,
        language=cfg.language,
        beam_size=cfg.beam_size,
    )
    return [
        {"text": word.word, "start": float(word.start), "end": float(word.end)}
        for segment in segments
        for word in (segment.words or [])
    ]


@lru_cache(maxsize=2)
def _model(model_size: str, device: str, compute_type: str):
    """Load the whisper model. Cached: loading costs seconds."""
    from faster_whisper import WhisperModel

    return WhisperModel(model_size, device=device, compute_type=compute_type)
