"""Beats + captions + music -> one mp4.

A single ffmpeg invocation: the per-beat visuals are concatenated, the
narration is concatenated, the music bed is mixed underneath, and captions are
burned in if this ffmpeg can. The command is built as a list and logged before
it runs, and ffmpeg's own stderr is what comes back in the exception.
"""

from __future__ import annotations

import functools
import logging
import subprocess
from pathlib import Path

from narrator.beats import Beat
from narrator.config import AssembleConfig

#: A visual may fall this far short of its narration before it is an error.
VISUAL_TOLERANCE = 0.05

LOG = logging.getLogger(__name__)


class AssembleError(RuntimeError):
    """The final render could not be produced."""


@functools.lru_cache(maxsize=1)
def has_ass_filter() -> bool:
    """Can this ffmpeg burn subtitles? Needs a build with libass."""
    result = subprocess.run(
        ["ffmpeg", "-nostdin", "-hide_banner", "-filters"],
        capture_output=True,
        text=True,
    )
    return any(line.split()[1:2] == ["ass"] for line in result.stdout.splitlines() if line.strip())


def assemble(
    beats: list[Beat],
    out_path: Path,
    cfg: AssembleConfig,
    *,
    captions: Path | None = None,
    music: Path | None = None,
) -> Path:
    """Render the finished video and return its path."""
    _check(beats, captions, music)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    burn = captions is not None and has_ass_filter()
    if captions is not None and not burn:
        if cfg.require_burned_captions:
            raise AssembleError(
                "captions cannot be burned: this ffmpeg has no libass "
                "(ffmpeg -filters shows no 'ass' filter). Install an ffmpeg built "
                "with --enable-libass, or set require_burned_captions=False."
            )
        LOG.warning(
            "this ffmpeg has no libass, so captions cannot be burned; "
            "muxing %s as a soft subtitle track instead",
            captions.name,
        )

    command = _command(beats, out_path, cfg, captions, music, burn)
    LOG.debug("ffmpeg: %s", " ".join(command))

    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=cfg.ffmpeg_timeout)
    except subprocess.TimeoutExpired as exc:
        raise AssembleError(f"ffmpeg timed out after {cfg.ffmpeg_timeout}s") from exc

    if result.returncode != 0:
        raise AssembleError(
            f"ffmpeg failed (exit {result.returncode}) assembling {out_path}:\n"
            f"{result.stderr.strip()[-2000:]}"
        )
    return out_path


def _check(beats: list[Beat], captions: Path | None, music: Path | None) -> None:
    if not beats:
        raise AssembleError("no beats to assemble")

    for beat in beats:
        if beat.asset_path is None:
            raise AssembleError(f"beat {beat.index} has no visual; run visuals.fetch_all first")
        if beat.audio_path is None:
            raise AssembleError(f"beat {beat.index} has no audio; run speech.synthesize first")
        for label, path in (("visual", beat.asset_path), ("audio", beat.audio_path)):
            if not Path(path).exists():
                raise AssembleError(f"beat {beat.index} {label} does not exist: {path}")

        # concat does not care that the picture ends before the narration; it
        # just runs the audio on over the next beat's footage, and every
        # caption after that point sits on the wrong shot.
        if beat.duration is not None:
            visual_seconds = _probe_seconds(beat.asset_path)
            if visual_seconds < beat.duration - VISUAL_TOLERANCE:
                raise AssembleError(
                    f"beat {beat.index} visual is shorter than its narration: "
                    f"{visual_seconds:.3f}s of picture for {beat.duration:.3f}s of audio "
                    f"({beat.asset_path})"
                )

    if captions is not None and not Path(captions).exists():
        raise AssembleError(f"captions file does not exist: {captions}")
    if music is not None and not Path(music).exists():
        raise AssembleError(f"music file does not exist: {music}")


def _probe_seconds(path: Path) -> float:
    result = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-show_entries",
            "format=duration",
            "-of",
            "default=noprint_wrappers=1:nokey=1",
            str(path),
        ],
        capture_output=True,
        text=True,
    )
    if result.returncode != 0:
        raise AssembleError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise AssembleError(f"ffprobe gave no duration for {path}") from exc


def _command(
    beats: list[Beat],
    out_path: Path,
    cfg: AssembleConfig,
    captions: Path | None,
    music: Path | None,
    burn: bool,
) -> list[str]:
    command = ["ffmpeg", "-nostdin", "-y", "-v", "error"]

    for beat in beats:
        command += ["-i", str(beat.asset_path), "-i", str(beat.audio_path)]

    soft_captions = captions is not None and not burn
    if soft_captions:
        command += ["-i", str(captions)]
    if music is not None:
        # Loop the bed: a two-minute track under a ten-minute story would
        # otherwise leave most of it silent.
        command += ["-stream_loop", "-1", "-i", str(music)]

    command += ["-filter_complex", _filter_complex(beats, cfg, captions, music, burn)]
    command += ["-map", "[vout]", "-map", "[aout]"]
    if soft_captions:
        command += ["-map", f"{2 * len(beats)}", "-c:s", "mov_text"]

    command += [
        "-c:v",
        cfg.video_codec,
        "-crf",
        str(cfg.crf),
        "-preset",
        cfg.x264_preset,
        "-pix_fmt",
        "yuv420p",
        "-c:a",
        cfg.audio_codec,
        "-ar",
        str(cfg.sample_rate),
        "-movflags",
        "+faststart",
        str(out_path),
    ]
    return command


def _filter_complex(
    beats: list[Beat],
    cfg: AssembleConfig,
    captions: Path | None,
    music: Path | None,
    burn: bool,
) -> str:
    parts: list[str] = []

    # Normalise every clip before concat: concat demands identical geometry,
    # and a single mismatched asset would otherwise fail the whole render.
    for position in range(len(beats)):
        scale = (
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
            f"crop={cfg.width}:{cfg.height},setsar=1,fps={cfg.fps}"
        )
        # Trim each clip to its own beat. Without this a long asset stretches
        # the video past its narration and everything after it slips.
        seconds = beats[position].duration
        if seconds is not None:
            scale += f",trim=duration={seconds:.3f},setpts=PTS-STARTPTS"
        parts.append(f"[{2 * position}:v]{scale}[v{position}]")
        parts.append(f"[{2 * position + 1}:a]aresample={cfg.sample_rate}[a{position}]")

    video_inputs = "".join(f"[v{i}]" for i in range(len(beats)))
    audio_inputs = "".join(f"[a{i}]" for i in range(len(beats)))
    parts.append(f"{video_inputs}concat=n={len(beats)}:v=1:a=0[vcat]")
    parts.append(f"{audio_inputs}concat=n={len(beats)}:v=0:a=1[acat]")

    if burn and captions is not None:
        parts.append(f"[vcat]ass={_escape_path(captions)}[vout]")
    else:
        parts.append("[vcat]null[vout]")

    if music is not None:
        music_index = 2 * len(beats) + (1 if captions is not None and not burn else 0)
        parts.append(
            f"[{music_index}:a]aresample={cfg.sample_rate},volume={cfg.music_gain_db}dB[music]"
        )
        # normalize=0 matters: amix divides by the number of inputs by
        # default, which would quietly halve the narration.
        parts.append(
            "[acat][music]amix=inputs=2:duration=first:dropout_transition=0:normalize=0[aout]"
        )
    else:
        parts.append("[acat]anull[aout]")

    return ";".join(parts)


def _escape_path(path: Path) -> str:
    """Filter arguments take colons and backslashes as syntax, not text."""
    return str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
