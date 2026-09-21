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
