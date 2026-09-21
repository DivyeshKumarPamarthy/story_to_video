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
