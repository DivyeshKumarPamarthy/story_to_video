"""ffprobe helpers for media assertions.

Deliberately independent of narrator's own ffprobe call, so a bug there cannot
make the tests agree with it.
"""

from __future__ import annotations

import json
import subprocess
from pathlib import Path


def probe(path: Path) -> dict:
    out = subprocess.run(
        [
            "ffprobe",
            "-v",
            "error",
            "-print_format",
            "json",
            "-show_format",
            "-show_streams",
            str(path),
        ],
        capture_output=True,
        text=True,
        check=True,
    )
    return json.loads(out.stdout)


def duration(path: Path) -> float:
    return float(probe(path)["format"]["duration"])


def audio_streams(path: Path) -> list[dict]:
    return [s for s in probe(path)["streams"] if s["codec_type"] == "audio"]


def sample_rate(path: Path) -> int:
    return int(audio_streams(path)[0]["sample_rate"])


def video_streams(path: Path) -> list[dict]:
    return [s for s in probe(path)["streams"] if s["codec_type"] == "video"]


def resolution(path: Path) -> tuple[int, int]:
    stream = video_streams(path)[0]
    return int(stream["width"]), int(stream["height"])


def fps(path: Path) -> float:
    rate = video_streams(path)[0]["avg_frame_rate"]
    numerator, _, denominator = rate.partition("/")
    return float(numerator) / float(denominator or 1)


def mean_volume(path: Path) -> float:
    """Mean volume in dBFS, via volumedetect."""
    out = subprocess.run(
        ["ffmpeg", "-hide_banner", "-i", str(path), "-af", "volumedetect", "-f", "null", "-"],
        capture_output=True,
        text=True,
    )
    for line in out.stderr.splitlines():
        if "mean_volume:" in line:
            return float(line.split("mean_volume:")[1].strip().split()[0])
    raise AssertionError(f"volumedetect reported no mean_volume for {path}: {out.stderr[-400:]}")
