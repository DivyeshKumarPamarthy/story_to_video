"""Beat text -> narrated wav, via Kokoro-82M.

Fills ``audio_path`` and ``duration``. Synthesis is cached by a hash of the
inputs that change the sound, so reruns of the pipeline are free.

Two seams exist for testing, and neither imports torch until it is called:
``_pipeline`` (loads the model onto a device) and ``_synthesize_audio``
(text -> samples). The default test suite mocks them.
"""

from __future__ import annotations

import hashlib
import subprocess
import wave
from dataclasses import replace
from functools import lru_cache
from pathlib import Path
from typing import Any

import numpy as np

from narrator.beats import Beat
from narrator.config import SpeechConfig


class SynthesisError(RuntimeError):
    """Speech synthesis ran but produced something unusable."""


#: Kokoro v1.0 English voices. Hardcoded on purpose: an unknown voice id must
#: fail loudly rather than fall back to a default and narrate a whole story in
#: the wrong voice.
VOICES = frozenset(
    """
    af_alloy af_aoede af_bella af_heart af_jessica af_kore af_nicole af_nova
    af_river af_sarah af_sky
    am_adam am_echo am_eric am_fenrir am_liam am_michael am_onyx am_puck am_santa
    bf_alice bf_emma bf_isabella bf_lily
    bm_daniel bm_fable bm_george bm_lewis
    """.split()
)

#: Narration sits around 10-22 characters per second. These bounds are wider
#: than that -- they are not a style check, they catch the failure where the
#: model emits a fraction of a second of nothing and the run continues with a
#: silent beat that only shows up in the final render.
MIN_CHARS_PER_SECOND = 4.0
MAX_CHARS_PER_SECOND = 40.0

#: Below this, character count says nothing useful about duration.
PLAUSIBILITY_MIN_CHARS = 20


def synthesize(
    beats: list[Beat],
    voice: str,
    cache_dir: Path,
    cfg: SpeechConfig | None = None,
) -> list[Beat]:
    """Synthesise narration for each beat, returning new beats.

    The input beats are not mutated. Beats whose audio is already cached are
    not re-synthesised.
    """
    cfg = cfg or SpeechConfig()
    if voice not in VOICES:
        raise ValueError(f"unknown voice {voice!r}; expected one of {', '.join(sorted(VOICES))}")

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)

    narrated: list[Beat] = []
    for beat in beats:
        path = _cache_path(cache_dir, beat.text, voice, cfg)
        if path.exists():
            duration = probe_duration(path)
        else:
            duration = _render(path, beat.text, voice, cfg)
        narrated.append(replace(beat, audio_path=path, duration=duration))
    return narrated


def probe_duration(path: Path) -> float:
    """Seconds of audio in ``path``, via ffprobe -- the file is never loaded."""
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
        raise SynthesisError(f"ffprobe failed for {path}: {result.stderr.strip()}")
    try:
        return float(result.stdout.strip())
    except ValueError as exc:
        raise SynthesisError(f"ffprobe gave no duration for {path}: {result.stdout!r}") from exc


def _render(path: Path, text: str, voice: str, cfg: SpeechConfig) -> float:
    """Synthesise to a temporary file, validate, then move into place.

    Nothing reaches the cache until it has been probed and found plausible, so
    a crash or a silent model cannot leave a bad file to be reused forever.
    """
    partial = path.with_name(path.name + ".part")
    try:
        audio = _synthesize_audio(text, voice, cfg)
        _write_wav(partial, audio, cfg.sample_rate)
        duration = probe_duration(partial)
        _check_plausible(text, duration)
        partial.replace(path)
    except BaseException:
        partial.unlink(missing_ok=True)
        raise
    return duration


def _cache_path(cache_dir: Path, text: str, voice: str, cfg: SpeechConfig) -> Path:
    """Cache key: everything that changes the sound.

    ``device`` is deliberately excluded -- the same model saying the same
    sentence is the same narration wherever it was computed, and including it
    would throw away the cache every time the device changed.
    """
    key = "\x00".join(
        [
            text,
            voice,
            cfg.model_version,
            cfg.lang_code,
            f"{cfg.speed:.4f}",
            str(cfg.sample_rate),
        ]
    )
    return cache_dir / f"{hashlib.sha256(key.encode('utf-8')).hexdigest()[:16]}.wav"


def _check_plausible(text: str, duration: float) -> None:
    if len(text) < PLAUSIBILITY_MIN_CHARS:
        return
    rate = len(text) / duration if duration > 0 else float("inf")
    if not MIN_CHARS_PER_SECOND <= rate <= MAX_CHARS_PER_SECOND:
        raise SynthesisError(
            f"implausible narration rate: {len(text)} characters in {duration:.2f}s "
            f"({rate:.1f} chars/sec) for {_excerpt(text)}"
        )


def _write_wav(path: Path, audio: Any, sample_rate: int) -> None:
    samples = np.asarray(audio, dtype=np.float32).reshape(-1)
    if samples.size == 0:
        raise SynthesisError("synthesis returned no audio samples")

    pcm = (np.clip(samples, -1.0, 1.0) * 32767.0).astype("<i2")
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


def _synthesize_audio(text: str, voice: str, cfg: SpeechConfig) -> np.ndarray:
    """text -> mono float32 samples. The seam the default test suite mocks."""
    pipeline = _pipeline(cfg.device, cfg.lang_code)

    chunks = []
    for result in pipeline(text, voice=voice, speed=cfg.speed):
        audio = getattr(result, "audio", None)
        if audio is not None:
            chunks.append(_to_numpy(audio))

    if not chunks:
        raise SynthesisError(f"kokoro produced no audio for {_excerpt(text)}")
    return np.concatenate(chunks)


@lru_cache(maxsize=4)
def _pipeline(device: str, lang_code: str):
    """Load Kokoro onto ``device``. Cached: loading costs seconds."""
    _require_device(device)
    from kokoro import KPipeline

    return KPipeline(lang_code=lang_code, device=device)


def _require_device(device: str) -> None:
    if device == "cpu":
        return

    import torch

    available = {
        "mps": torch.backends.mps.is_available,
        "cuda": torch.cuda.is_available,
    }[device]
    if not available():
        raise SynthesisError(
            f"device {device!r} is not available on this machine; use SpeechConfig(device='cpu')"
        )


def _to_numpy(audio: Any) -> np.ndarray:
    if hasattr(audio, "detach"):  # a torch tensor, possibly on mps or cuda
        audio = audio.detach().cpu().numpy()
    return np.asarray(audio, dtype=np.float32).reshape(-1)


def _excerpt(text: str, limit: int = 60) -> str:
    return repr(text if len(text) <= limit else text[: limit - 1] + "…")
