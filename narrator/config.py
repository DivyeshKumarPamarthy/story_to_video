"""Configuration values shared across pipeline stages.

Stages read config; they never hardcode a device, a sample rate, a resolution
or an fps. Configs are frozen dataclasses, so a stage can be handed a variant
with :func:`dataclasses.replace` without anyone worrying about aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

#: Torch devices this project knows how to ask for. "cpu" is the default
#: everywhere: it is the only one available on every machine, and for an 82M
#: parameter model the gap is small enough that correctness beats guessing.
#: Run `pytest -m slow -k benchmark` to get the numbers on this machine.
DEVICES = ("cpu", "mps", "cuda")


@dataclass(frozen=True)
class SpeechConfig:
    """How narration is synthesised.

    ``model_version`` is part of the cache key, so bumping the model
    invalidates previously cached audio instead of silently mixing voices.
    """

    device: str = "cpu"
    sample_rate: int = 24000  # Kokoro's native rate; resampling costs quality
    lang_code: str = "a"  # "a" American English, "b" British English
    speed: float = 1.0
    model_version: str = "kokoro-82m-v1.0"

    def __post_init__(self) -> None:
        if self.device not in DEVICES:
            raise ValueError(
                f"unknown device {self.device!r}; expected one of {', '.join(DEVICES)}"
            )
        if self.sample_rate <= 0:
            raise ValueError(f"sample_rate must be positive, got {self.sample_rate}")
        if self.speed <= 0:
            raise ValueError(f"speed must be positive, got {self.speed}")


#: faster-whisper runs through CTranslate2, which has no Metal backend, so
#: "mps" is not merely slow -- it does not exist. Apple Silicon means cpu.
ALIGN_DEVICES = ("cpu", "cuda")


@dataclass(frozen=True)
class AlignConfig:
    """How narration is aligned back to its own text.

    ``base``/``int8`` is the default because alignment runs on audio this
    project generated from text it already knows: the transcription only has
    to be good enough to carry timings, and a larger model buys accuracy that
    the word-count check would reject anyway.
    """

    model_size: str = "base"
    compute_type: str = "int8"
    device: str = "cpu"
    language: str = "en"
    beam_size: int = 5
    #: Below this, the transcription is not a rendering of the reference text
    #: and its timings mean nothing. 0.85 tolerates a model that collapses
    #: "3 a.m." into one token while still rejecting the wrong audio entirely.
    min_match_ratio: float = 0.85

    def __post_init__(self) -> None:
        if self.device == "mps":
            raise ValueError(
                "faster-whisper cannot run on mps: CTranslate2 has no Metal backend. "
                "Use AlignConfig(device='cpu')"
            )
        if self.device not in ALIGN_DEVICES:
            raise ValueError(
                f"unknown device {self.device!r}; expected one of {', '.join(ALIGN_DEVICES)}"
            )
        if self.beam_size <= 0:
            raise ValueError(f"beam_size must be positive, got {self.beam_size}")
        if not 0 < self.min_match_ratio <= 1:
            raise ValueError(f"min_match_ratio must be in (0, 1], got {self.min_match_ratio}")


#: Aspect presets. CLAUDE.md is the contract: nothing hardcodes a resolution.
VIDEO_PRESETS = {
    "landscape": (1920, 1080),
    "vertical": (1080, 1920),
}


@dataclass(frozen=True)
class VisualsConfig:
    """How each beat's background is sourced and rendered."""

    width: int = 1920
    height: int = 1080
    fps: int = 30
    #: Total push-in across a still's clip. 1.0 would be a static frame.
    zoom: float = 1.12
    #: Used when a beat has no duration yet (no narration synthesised).
    default_seconds: float = 4.0
    #: Directory of images to use instead of generated backgrounds.
    stills_dir: Path | None = None
    api_key_env: str = "PEXELS_API_KEY"
    request_timeout: float = 10.0
    #: ffmpeg is killed after this many seconds. Not paranoia: `-loop 1`
    #: on an undecodable image never returns (see CLAUDE.md gotchas).
    ffmpeg_timeout: float = 120.0
    #: True makes a missing key or a failed API call fatal instead of falling
    #: back to stills. The pipeline stays usable offline by default, but a run
    #: that is meant to use stock footage can insist on it.
    require_pexels: bool = False

    def __post_init__(self) -> None:
        for name in ("width", "height", "fps"):
            if getattr(self, name) <= 0:
                raise ValueError(f"{name} must be positive, got {getattr(self, name)}")
        if self.zoom < 1.0:
            raise ValueError(f"zoom must be at least 1.0, got {self.zoom}")
        if self.default_seconds <= 0:
            raise ValueError(f"default_seconds must be positive, got {self.default_seconds}")
        if self.ffmpeg_timeout <= 0:
            raise ValueError(f"ffmpeg_timeout must be positive, got {self.ffmpeg_timeout}")

    @classmethod
    def preset(cls, name: str, **overrides: object) -> VisualsConfig:
        if name not in VIDEO_PRESETS:
            raise ValueError(f"unknown preset {name!r}; expected one of {', '.join(VIDEO_PRESETS)}")
        width, height = VIDEO_PRESETS[name]
        return cls(width=width, height=height, **overrides)  # type: ignore[arg-type]
