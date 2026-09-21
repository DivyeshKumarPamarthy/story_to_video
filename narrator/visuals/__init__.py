"""Beat -> a background clip. Pexels first, generated stills as the fallback.

One interface, ``fetch(beat, cfg, out_dir)``. Whatever the source, the result
is conformed to the configured resolution, fps and the beat's own duration, so
the assembler never has to think about where a clip came from.
"""

from __future__ import annotations

import logging
import subprocess
from dataclasses import replace
from pathlib import Path

from narrator.beats import Beat
from narrator.config import VisualsConfig
from narrator.visuals import pexels, stills
from narrator.visuals.query import add_queries, build_query

__all__ = ["VisualsError", "add_queries", "build_query", "fetch", "fetch_all"]

LOG = logging.getLogger(__name__)


class VisualsError(RuntimeError):
    """A beat's visual could not be produced at all."""


def fetch(beat: Beat, cfg: VisualsConfig, out_dir: Path, *, exclude: tuple[str, ...] = ()) -> Path:
    """Produce this beat's background clip and return its path."""
    return _fetch(beat, cfg, Path(out_dir), exclude)[0]


def fetch_all(beats: list[Beat], cfg: VisualsConfig, out_dir: Path) -> list[Beat]:
    """Fill ``asset_path`` on every beat. Input is not mutated.

    Each beat excludes the asset the one before it used, so the same footage
    is never shown twice in a row.
    """
    out_dir = Path(out_dir)
    fetched: list[Beat] = []
    previous: tuple[str, ...] = ()
    for beat in beats:
        path, asset_id = _fetch(beat, cfg, out_dir, previous)
        fetched.append(replace(beat, asset_path=path))
        previous = (asset_id,)
    return fetched


def _fetch(
    beat: Beat, cfg: VisualsConfig, out_dir: Path, exclude: tuple[str, ...]
) -> tuple[Path, str]:
    out_dir.mkdir(parents=True, exist_ok=True)
    dest = out_dir / f"beat_{beat.index:03d}.mp4"
    seconds = beat.duration or cfg.default_seconds

    try:
        source, asset_id = pexels.fetch_clip(beat, cfg, out_dir / "stock", exclude=exclude)
    except pexels.PexelsError as exc:
        if cfg.require_pexels:
            raise
        # Never silent: a run that quietly stopped using stock footage looks
        # like a style choice rather than a missing key or a dead API.
        LOG.warning("pexels unavailable for beat %d (%s); using a still", beat.index, exc)
        stills.render(beat, cfg, dest, exclude=exclude, seconds=seconds)
        return dest, stills.variant_for(beat, cfg, exclude)

    _conform(source, dest, seconds, cfg)
    return dest, asset_id


def _conform(source: Path, dest: Path, seconds: float, cfg: VisualsConfig) -> None:
    """Crop, scale, loop and trim a clip to exactly what the beat needs.

    ``-stream_loop -1`` covers a clip shorter than the beat; ``-t`` trims one
    that is longer. Audio is dropped: narration is the only sound.
    """
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-stream_loop",
        "-1",
        "-i",
        str(source),
        "-t",
        f"{seconds:.3f}",
        "-vf",
        (
            f"scale={cfg.width}:{cfg.height}:force_original_aspect_ratio=increase,"
            f"crop={cfg.width}:{cfg.height},fps={cfg.fps}"
        ),
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(dest),
    ]
    LOG.debug("ffmpeg: %s", " ".join(command))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=cfg.ffmpeg_timeout)
    except subprocess.TimeoutExpired as exc:
        raise VisualsError(
            f"ffmpeg timed out after {cfg.ffmpeg_timeout}s conforming {source}"
        ) from exc
    if result.returncode != 0:
        raise VisualsError(
            f"ffmpeg failed conforming {source} (exit {result.returncode}):\n"
            f"{result.stderr.strip()[-800:]}"
        )
