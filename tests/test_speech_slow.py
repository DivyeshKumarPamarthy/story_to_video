"""Slow speech tests: real model, real inference.

Excluded from the default run. Nothing here may be imported at module scope --
collection happens before deselection, so a top-level `import torch` would pull
the whole stack into every fast run.

    uv run pytest -m slow tests/test_speech_slow.py -s
"""

from __future__ import annotations

import time
from dataclasses import replace

import pytest

from narrator import speech
from narrator.beats import Beat
from narrator.config import SpeechConfig
from tests import ffprobe

VOICE = "af_heart"

#: One fixed sentence for every timing run, so numbers are comparable.
BENCHMARK_SENTENCE = (
    "The house had been empty for nine years, and the rain had taken the ceiling with it."
)


def beat(text: str = BENCHMARK_SENTENCE) -> Beat:
    return Beat(index=0, text=text, visual_query="")


def mps_available() -> bool:
    import torch

    return torch.backends.mps.is_available()


@pytest.mark.slow
def test_real_synthesis_produces_narration(tmp_path):
    cfg = SpeechConfig()  # cpu
    out = speech.synthesize([beat()], VOICE, tmp_path, cfg=cfg)[0]

    assert out.audio_path.exists()
    assert ffprobe.sample_rate(out.audio_path) == cfg.sample_rate
    assert len(ffprobe.audio_streams(out.audio_path)) == 1

    probed = ffprobe.duration(out.audio_path)
    assert out.duration == pytest.approx(probed, abs=0.05)

    # Real speech, not a click or a silence.
    rate = len(BENCHMARK_SENTENCE) / probed
    assert 8 <= rate <= 25, f"{rate:.1f} chars/sec does not sound like narration"


@pytest.mark.slow
def test_real_synthesis_is_cached(tmp_path):
    first = speech.synthesize([beat()], VOICE, tmp_path)[0]
    mtime = first.audio_path.stat().st_mtime_ns

    second = speech.synthesize([beat()], VOICE, tmp_path)[0]

    assert second.audio_path == first.audio_path
    assert second.audio_path.stat().st_mtime_ns == mtime, "cache hit rewrote the file"


@pytest.mark.slow
def test_unavailable_device_raises_rather_than_falling_back(tmp_path):
    import torch

    if torch.cuda.is_available():
        pytest.skip("cuda is available on this machine, so it cannot be the missing device")

    with pytest.raises(speech.SynthesisError, match="not available"):
        speech.synthesize([beat()], VOICE, tmp_path, cfg=SpeechConfig(device="cuda"))


@pytest.mark.slow
def test_benchmark_cpu_vs_mps(tmp_path, capsys):
    """Time the same sentence on each device so the default can be chosen from
    a number rather than a guess.

    Reports load time and synthesis time separately: loading is paid once per
    process, synthesis is paid per beat, and only the second scales with story
    length. The first synthesis on a device is discarded as a warm-up.
    """
    devices = ["cpu"]
    if mps_available():
        devices.append("mps")
    else:
        print("\nmps unavailable on this machine; timing cpu only")

    results = {}
    for device in devices:
        cfg = replace(SpeechConfig(), device=device)

        speech._pipeline.cache_clear()
        load_started = time.perf_counter()
        speech._pipeline(cfg.device, cfg.lang_code)
        load_seconds = time.perf_counter() - load_started

        # Warm-up: first call on a device pays lazy kernel compilation.
        speech._synthesize_audio(BENCHMARK_SENTENCE, VOICE, cfg)

        runs = []
        for _ in range(3):
            started = time.perf_counter()
            audio = speech._synthesize_audio(BENCHMARK_SENTENCE, VOICE, cfg)
            runs.append(time.perf_counter() - started)

        audio_seconds = len(audio) / cfg.sample_rate
        results[device] = {
            "load": load_seconds,
            "best": min(runs),
            "runs": runs,
            "audio_seconds": audio_seconds,
            "realtime_factor": audio_seconds / min(runs),
        }

    with capsys.disabled():
        print(f"\n\nKokoro-82M, {len(BENCHMARK_SENTENCE)} chars, best of 3 after warm-up\n")
        print(f"{'device':<8}{'load':>9}{'synth':>10}{'xRT':>9}  {'runs (s)'}")
        for device, r in results.items():
            runs = ", ".join(f"{x:.3f}" for x in r["runs"])
            print(
                f"{device:<8}{r['load']:>8.2f}s{r['best']:>9.3f}s"
                f"{r['realtime_factor']:>8.1f}x  {runs}"
            )
        if len(results) == 2:
            speedup = results["cpu"]["best"] / results["mps"]["best"]
            verdict = "mps faster" if speedup > 1 else "cpu faster"
            print(f"\n{verdict}: mps is {speedup:.2f}x cpu on synthesis")
            print(
                f"(mps pays {results['mps']['load'] - results['cpu']['load']:+.2f}s more to load)"
            )
        print()

    for device, r in results.items():
        assert r["realtime_factor"] > 1.0, f"{device} is slower than real time: {r}"
