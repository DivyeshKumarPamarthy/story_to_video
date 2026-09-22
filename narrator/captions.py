"""Word timings -> an .ass subtitle file, karaoke style.

One or two words at a time, centred in the lower third, with the word being
spoken highlighted. The file is generated rather than templated so the timings
come straight from ``Beat.words``.

This module only writes the file. Burning it into video is the assembler's
job, and needs an ffmpeg built with libass -- see CLAUDE.md gotchas.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

from narrator.beats import Beat, Word
from narrator.config import CaptionConfig

STYLE_NAME = "Karaoke"


class CaptionError(RuntimeError):
    """Captions could not be built from these beats."""


@dataclass
class Event:
    """One Dialogue line, read back off disk."""

    raw_start: str
    raw_end: str
    style: str
    text: str

    @property
    def start(self) -> float:
        return _parse_timestamp(self.raw_start)

    @property
    def end(self) -> float:
        return _parse_timestamp(self.raw_end)


@dataclass
class Script:
    script_info: dict[str, str] = field(default_factory=dict)
    styles: dict[str, dict[str, str]] = field(default_factory=dict)
    events: list[Event] = field(default_factory=list)


def build_ass(beats: list[Beat], cfg: CaptionConfig, out_path: Path) -> Path:
    """Write captions for every beat's words and return the path."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    lines = [_script_info(cfg), _styles(cfg), _events(beats, cfg)]
    out_path.write_text("\n".join(lines), encoding="utf-8")
    return out_path


def visible_text(text: str) -> str:
    """The text with ASS override blocks removed -- what a viewer reads."""
    return re.sub(r"\{[^}]*\}", "", text)


def _script_info(cfg: CaptionConfig) -> str:
    return "\n".join(
        [
            "[Script Info]",
            "ScriptType: v4.00+",
            "WrapStyle: 0",
            "ScaledBorderAndShadow: yes",
            f"PlayResX: {cfg.width}",
            f"PlayResY: {cfg.height}",
            "",
        ]
    )


def _styles(cfg: CaptionConfig) -> str:
    fields = [
        "Name",
        "Fontname",
        "Fontsize",
        "PrimaryColour",
        "SecondaryColour",
        "OutlineColour",
        "BackColour",
        "Bold",
        "Italic",
        "Underline",
        "StrikeOut",
        "ScaleX",
        "ScaleY",
        "Spacing",
        "Angle",
        "BorderStyle",
        "Outline",
        "Shadow",
        "Alignment",
        "MarginL",
        "MarginR",
        "MarginV",
        "Encoding",
    ]
    values = [
        STYLE_NAME,
        cfg.font_name,
        str(cfg.font_size),
        cfg.highlight_colour,  # PrimaryColour is the *sung* colour in karaoke
        cfg.primary_colour,  # SecondaryColour is the not-yet-sung colour
        cfg.outline_colour,
        "&H64000000",
        "-1",
        "0",
        "0",
        "0",
        "100",
        "100",
        "0",
        "0",
        "1",
        str(cfg.outline),
        str(cfg.shadow),
        "2",
        "60",
        "60",
        str(cfg.margin_v),
        "1",
    ]
    return "\n".join(
        [
            "[V4+ Styles]",
            "Format: " + ", ".join(fields),
            "Style: " + ",".join(values),
            "",
        ]
    )


def _events(beats: list[Beat], cfg: CaptionConfig) -> str:
    lines = [
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]
    # Beat.words are timed against that beat's own audio, so each beat has to
    # be pushed along by everything that plays before it. Without this every
    # beat's captions start at zero and pile up on the opening shot.
    offset = 0.0
    for beat in beats:
        if not beat.words:
            raise CaptionError(
                f"beat {beat.index} has no word timings; run align before building captions"
            )
        _check_words_fit(beat)
        if beat.duration is None:
            raise CaptionError(
                f"beat {beat.index} has no duration, so the beats after it cannot be "
                f"placed; run speech.synthesize before building captions"
            )
        # Grouping never crosses a beat: a caption spanning two beats would sit
        # over a visual cut.
        for group in _group(beat.words, cfg.words_per_line):
            lines.append(_dialogue(group, cfg, offset))
        offset += beat.duration
    return "\n".join(lines) + "\n"


#: Whisper rounds to centiseconds, so a word may end a hair past its beat.
BEAT_TOLERANCE = 0.05


def _check_words_fit(beat: Beat) -> None:
    """Are these timings in the beat's own timeline?

    A beat whose words run past its duration is carrying whole-video times,
    which is the bug that made every caption pile up on the opening shot. It
    is cheap to detect and impossible to see by eye in the output.
    """
    if beat.duration is None:
        return

    last = beat.words[-1]
    if last.end > beat.duration + BEAT_TOLERANCE or beat.words[0].start < -BEAT_TOLERANCE:
        raise CaptionError(
            f"word timings fall outside beat {beat.index}: "
            f"{beat.words[0].start:.3f}s-{last.end:.3f}s against a beat of "
            f"{beat.duration:.3f}s. Beat.words are timed against the beat's own "
            f"audio, not the whole video"
        )

    for earlier, later in zip(beat.words, beat.words[1:], strict=False):
        if later.start < earlier.start:
            raise CaptionError(
                f"beat {beat.index} word timings are out of order: "
                f"{earlier.text!r} at {earlier.start:.3f}s then {later.text!r} at "
                f"{later.start:.3f}s"
            )


#: A word ending a sentence, allowing for a closing quote or bracket.
_SENTENCE_END = re.compile(r"[.!?\u2026][\"\'\u201d\u2019)\]]*$")


def _group(words: list[Word], size: int) -> list[list[Word]]:
    """Fixed-size groups, flushed early at a sentence end.

    Holding the end of one sentence and the start of the next on screen
    together reads as a mistake, however well the timings line up.
    """
    groups: list[list[Word]] = []
    current: list[Word] = []
    for word in words:
        current.append(word)
        if len(current) == size or _ends_sentence(word.text):
            groups.append(current)
            current = []
    if current:
        groups.append(current)
    return groups


def _ends_sentence(text: str) -> bool:
    if not _SENTENCE_END.search(text):
        return False
    # "Dr." and "U.S." are not sentence ends; the same asymmetry as segment.py.
    stem = _SENTENCE_END.sub("", text)
    return not (stem.lower() in _ABBREVIATIONS or len(stem) == 1)


#: Kept in step with segment.py's list, for the same reason: a false split
#: here only shortens a caption, a missed one puts two sentences on screen.
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms mx dr prof rev fr sr jr st mt capt sgt lt col gen gov hon messrs
    vs etc al cf ibid viz inc ltd dept fig vol ed eds pp approx
    """.split()
)


def _dialogue(group: list[Word], cfg: CaptionConfig, offset: float = 0.0) -> str:
    start = _timestamp(group[0].start + offset)
    end = _timestamp(group[-1].end + offset)
    return f"Dialogue: 0,{start},{end},{STYLE_NAME},,0,0,0,,{_karaoke(group, cfg)}"


def _karaoke(group: list[Word], cfg: CaptionConfig) -> str:
    """`\\k` durations in centiseconds, so libass highlights word by word."""
    rendered: list[str] = []
    line_length = 0
    for index, word in enumerate(group):
        text = _escape(word.text)
        centiseconds = max(int(round((word.end - word.start) * 100)), 1)

        if index and line_length + 1 + len(word.text) > cfg.wrap_width:
            rendered.append(r"\N")
            line_length = 0
        elif index:
            rendered.append(" ")
            line_length += 1

        rendered.append(f"{{\\k{centiseconds}}}{text}")
        line_length += len(word.text)
    return "".join(rendered)


#: Order matters: backslashes first, or the escapes get escaped.
_ESCAPES = (
    ("\\", "\\\\"),
    ("{", "\\{"),
    ("}", "\\}"),
)


def _escape(text: str) -> str:
    for raw, replacement in _ESCAPES:
        text = text.replace(raw, replacement)
    # A literal newline would end the Dialogue line; \N is the ASS line break.
    return re.sub(r"\s*\n\s*", r"\\N", text)


def _timestamp(seconds: float) -> str:
    """ASS wants h:mm:ss.cc with exactly two centisecond digits."""
    if seconds < 0:
        raise CaptionError(f"negative caption timestamp: {seconds}")
    centiseconds = int(round(seconds * 100))
    hours, remainder = divmod(centiseconds, 360_000)
    minutes, remainder = divmod(remainder, 6_000)
    secs, cents = divmod(remainder, 100)
    return f"{hours}:{minutes:02d}:{secs:02d}.{cents:02d}"


def _parse_timestamp(value: str) -> float:
    hours, minutes, seconds = value.split(":")
    return int(hours) * 3600 + int(minutes) * 60 + float(seconds)


def parse_ass(path: Path) -> Script:
    """Read an .ass file back into its parts, for tests and inspection."""
    script = Script()
    section = ""
    style_fields: list[str] = []

    for line in Path(path).read_text(encoding="utf-8").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("[") and stripped.endswith("]"):
            section = stripped
            continue

        key, _, value = stripped.partition(":")
        value = value.strip()

        if section == "[Script Info]":
            script.script_info[key] = value
        elif section == "[V4+ Styles]":
            if key == "Format":
                style_fields = [f.strip() for f in value.split(",")]
            elif key == "Style":
                values = value.split(",")
                style = dict(zip(style_fields, values, strict=False))
                script.styles[style["Name"]] = style
        elif section == "[Events]" and key == "Dialogue":
            # Text is the last field and may itself contain commas.
            parts = value.split(",", 9)
            script.events.append(
                Event(raw_start=parts[1], raw_end=parts[2], style=parts[3], text=parts[9])
            )
    return script
