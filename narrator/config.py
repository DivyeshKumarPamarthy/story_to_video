"""Configuration values shared across pipeline stages.

Stages read config; they never hardcode a device, a sample rate, a resolution
or an fps. Configs are frozen dataclasses, so a stage can be handed a variant
with :func:`dataclasses.replace` without anyone worrying about aliasing.
"""

from __future__ import annotations

from dataclasses import dataclass

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
