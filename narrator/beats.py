"""Core data contract.

Every pipeline stage takes ``list[Beat]`` and returns ``list[Beat]`` with more
fields filled in. Nothing in this module does I/O.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path


@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Beat:
    index: int
    text: str
    visual_query: str  # keyword for stock lookup
    audio_path: Path | None = None
    duration: float | None = None
    words: list[Word] = field(default_factory=list)
    asset_path: Path | None = None
