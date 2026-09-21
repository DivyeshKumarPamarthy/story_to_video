"""A still image pushed in slowly -- the cheap way to look animated.

ffmpeg runs as a subprocess. Nothing here loads a model, and this module is
never imported into a process holding one.
"""

from __future__ import annotations

import hashlib
import logging
import subprocess
from pathlib import Path

from narrator.beats import Beat
from narrator.config import VisualsConfig

LOG = logging.getLogger(__name__)


class StillsError(RuntimeError):
    """Rendering a still to video failed."""


#: Background gradients used when no stills directory is supplied. Picked by a
#: hash of the beat's query so the same beat always looks the same.
_PALETTE = (
    ("0x101820", "0x2a4157"),
    ("0x1b1b2f", "0x4a2545"),
    ("0x0f2027", "0x2c5364"),
    ("0x232526", "0x414345"),
    ("0x24243e", "0x302b63"),
    ("0x1f1c18", "0x5b4636"),
)


def variant_for(beat: Beat, cfg: VisualsConfig, exclude: tuple[str, ...] = ()) -> str:
    """Which background this beat gets, as a stable id.

    Deterministic in the query, so a rerun renders the same thing, but it
    steps past anything the previous beat used.
    """
    options = _options(cfg)
    start = int(hashlib.sha256(beat.visual_query.encode("utf-8")).hexdigest(), 16) % len(options)
    for offset in range(len(options)):
        candidate = options[(start + offset) % len(options)]
        if candidate not in exclude:
            return candidate
    return options[start]


def _options(cfg: VisualsConfig) -> tuple[str, ...]:
    if cfg.stills_dir is not None and cfg.stills_dir.exists():
        images = sorted(
            p.name
            for p in cfg.stills_dir.iterdir()
            if p.suffix.lower() in {".png", ".jpg", ".jpeg", ".webp"}
        )
        if images:
            return tuple(images)
    return tuple(f"gradient:{i}" for i in range(len(_PALETTE)))


def render(
    beat: Beat,
    cfg: VisualsConfig,
    out_path: Path,
    *,
    exclude: tuple[str, ...] = (),
    seconds: float | None = None,
) -> Path:
    """Render this beat's background to ``out_path`` and return it."""
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    duration = seconds if seconds is not None else (beat.duration or cfg.default_seconds)
    variant = variant_for(beat, cfg, exclude)

    still = out_path.with_suffix(".still.png")
    if variant.startswith("gradient:"):
        write_generated_still(still, variant, cfg)
    else:
        still = (cfg.stills_dir or Path()) / variant

    _run(_zoompan_command(still, out_path, duration, cfg), out_path, cfg.ffmpeg_timeout)
    return out_path


def write_generated_still(path: Path, seed: str, cfg: VisualsConfig) -> Path:
    """A procedural gradient, so the fallback needs neither network nor art."""
    index = int(hashlib.sha256(seed.encode("utf-8")).hexdigest(), 16) % len(_PALETTE)
    first, second = _PALETTE[index]
    path.parent.mkdir(parents=True, exist_ok=True)
    command = [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-f",
        "lavfi",
        "-i",
        f"gradients=s={cfg.width}x{cfg.height}:c0={first}:c1={second}:n=2:d=1",
        "-frames:v",
        "1",
        str(path),
    ]
    _run(command, path, cfg.ffmpeg_timeout)
    return path


def _zoompan_command(still: Path, out_path: Path, seconds: float, cfg: VisualsConfig) -> list[str]:
    frames = max(int(round(seconds * cfg.fps)), 1)
    # Upscaling before zoompan is what stops the push-in juddering: zoompan
    # samples from the scaled input, so a bigger input means sub-pixel steps.
    step = (cfg.zoom - 1.0) / frames if frames else 0.0
    filters = (
        f"scale={cfg.width * 2}:{cfg.height * 2},"
        f"zoompan=z='min(1+{step:.8f}*on,{cfg.zoom})'"
        f":x='iw/2-(iw/zoom/2)':y='ih/2-(ih/zoom/2)'"
        f":d={frames}:s={cfg.width}x{cfg.height}:fps={cfg.fps}"
    )
    return [
        "ffmpeg",
        "-y",
        "-v",
        "error",
        "-loop",
        "1",
        "-i",
        str(still),
        "-vf",
        filters,
        "-t",
        f"{seconds:.3f}",
        "-an",
        "-c:v",
        "libx264",
        "-pix_fmt",
        "yuv420p",
        str(out_path),
    ]


def _run(command: list[str], target: Path, timeout: float) -> None:
    LOG.debug("ffmpeg: %s", " ".join(command))
    try:
        result = subprocess.run(command, capture_output=True, text=True, timeout=timeout)
    except subprocess.TimeoutExpired as exc:
        # `-loop 1` on a file ffmpeg cannot decode never returns, so a bad
        # still would otherwise hang the whole render.
        raise StillsError(f"ffmpeg timed out after {timeout}s producing {target}") from exc
    if result.returncode != 0:
        raise StillsError(
            f"ffmpeg failed producing {target} (exit {result.returncode}):\n"
            f"{result.stderr.strip()[-800:]}"
        )
