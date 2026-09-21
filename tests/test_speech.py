"""Tests for narrator.speech.

The default run never imports torch or kokoro and never writes a real model's
output: `_synthesize_audio` is the seam, and it is mocked. Everything below the
seam -- wav writing, cache keys, ffprobe, validation -- is the real code.

Audio assertions go through ffprobe (tests/ffprobe.py), per CLAUDE.md.
"""

from __future__ import annotations

import hashlib
import os
from dataclasses import replace
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, patch

import numpy as np
import pytest

from narrator import speech
from narrator.beats import Beat
from narrator.config import SpeechConfig
from narrator.speech import SynthesisError, synthesize
from tests import ffprobe

CFG = SpeechConfig()
VOICE = "af_heart"

SENTENCE = "The house had been empty for nine years."


def beat(text: str = SENTENCE, index: int = 0) -> Beat:
    return Beat(index=index, text=text, visual_query="")


def fake_audio(text: str, cfg: SpeechConfig = CFG, chars_per_second: float = 15.0) -> np.ndarray:
    """A tone whose length tracks character count, like real narration does."""
    seconds = max(len(text) / chars_per_second, 0.2)
    samples = int(seconds * cfg.sample_rate)
    t = np.linspace(0.0, seconds, samples, endpoint=False, dtype=np.float32)
    return (0.2 * np.sin(2 * np.pi * 220.0 * t)).astype(np.float32)


def synth_spy(chars_per_second: float = 15.0) -> Mock:
    return Mock(side_effect=lambda text, voice, cfg: fake_audio(text, cfg, chars_per_second))


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# --- env --------------------------------------------------------------------


def test_mps_fallback_is_enabled_in_the_test_environment():
    # conftest sets this before anything can import torch.
    assert os.environ["PYTORCH_ENABLE_MPS_FALLBACK"] == "1"


def test_default_suite_does_not_load_torch_or_kokoro(tmp_path):
    import sys

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        synthesize([beat()], VOICE, tmp_path)

    assert "torch" not in sys.modules, "default suite must not import torch"
    assert "kokoro" not in sys.modules, "default suite must not import kokoro"


# --- contract ---------------------------------------------------------------


def test_fills_audio_path_and_duration(tmp_path):
    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, tmp_path)

    assert len(out) == 1
    assert out[0].audio_path is not None
    assert out[0].audio_path.exists()
    assert out[0].duration is not None
    assert out[0].duration > 0


def test_leaves_every_other_field_untouched(tmp_path):
    original = Beat(index=3, text=SENTENCE, visual_query="empty house")

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([original], VOICE, tmp_path)

    assert out[0].index == 3
    assert out[0].text == SENTENCE
    assert out[0].visual_query == "empty house"
    assert out[0].words == []
    assert out[0].asset_path is None


def test_does_not_mutate_the_input_beats(tmp_path):
    beats = [beat()]

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        synthesize(beats, VOICE, tmp_path)

    assert beats[0].audio_path is None
    assert beats[0].duration is None


def test_empty_beat_list_returns_empty_list(tmp_path):
    spy = synth_spy()
    with patch.object(speech, "_synthesize_audio", spy):
        assert synthesize([], VOICE, tmp_path) == []
    assert spy.call_count == 0


def test_preserves_order_and_indices(tmp_path):
    beats = [beat("First sentence here.", 0), beat("Second sentence here.", 1)]

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize(beats, VOICE, tmp_path)

    assert [b.index for b in out] == [0, 1]
    assert [b.text for b in out] == [b.text for b in beats]
    assert out[0].audio_path != out[1].audio_path


# --- the wav file itself (ffprobe) ------------------------------------------


def test_output_exists_and_is_non_empty(tmp_path):
    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, tmp_path)

    assert out[0].audio_path.stat().st_size > 0


def test_sample_rate_matches_config(tmp_path):
    cfg = replace(CFG, sample_rate=16000)

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, tmp_path, cfg=cfg)

    assert ffprobe.sample_rate(out[0].audio_path) == 16000


def test_writes_exactly_one_mono_audio_stream(tmp_path):
    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, tmp_path)

    streams = ffprobe.audio_streams(out[0].audio_path)
    assert len(streams) == 1
    assert streams[0]["channels"] == 1


def test_reported_duration_matches_ffprobe(tmp_path):
    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, tmp_path)

    assert out[0].duration == pytest.approx(ffprobe.duration(out[0].audio_path), abs=0.05)


def test_duration_lands_in_a_sane_band_for_the_character_count(tmp_path):
    text = "The house had been empty for nine years, and the rain had taken the ceiling."

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat(text)], VOICE, tmp_path)

    chars_per_second = len(text) / out[0].duration
    assert 10 <= chars_per_second <= 22, f"{chars_per_second:.1f} chars/sec is not narration"


def test_silently_empty_synthesis_raises_rather_than_caching_garbage(tmp_path):
    # The failure this guards: a model that emits a fraction of a second of
    # nothing, which is only visible as an absurd chars-per-second rate.
    long_text = "The house had been empty for nine years and nobody ever went inside it."
    stub = Mock(return_value=fake_audio("x" * 2, CFG))  # ~0.2s for 71 chars

    with patch.object(speech, "_synthesize_audio", stub):
        with pytest.raises(SynthesisError, match="implausible"):
            synthesize([beat(long_text)], VOICE, tmp_path)


def test_zero_length_audio_raises(tmp_path):
    stub = Mock(return_value=np.zeros(0, dtype=np.float32))

    with patch.object(speech, "_synthesize_audio", stub):
        with pytest.raises(SynthesisError, match="no audio"):
            synthesize([beat()], VOICE, tmp_path)


def test_failed_synthesis_leaves_no_cache_file_behind(tmp_path):
    stub = Mock(side_effect=RuntimeError("model exploded"))

    with patch.object(speech, "_synthesize_audio", stub):
        with pytest.raises(RuntimeError):
            synthesize([beat()], VOICE, tmp_path)

    assert list(tmp_path.glob("*.wav")) == []
    assert list(tmp_path.glob("*.part")) == []


# --- determinism and caching ------------------------------------------------


def test_same_text_voice_and_config_give_an_identical_file(tmp_path):
    first_dir = tmp_path / "one"
    second_dir = tmp_path / "two"

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        a = synthesize([beat()], VOICE, first_dir)[0]
        b = synthesize([beat()], VOICE, second_dir)[0]

    assert a.audio_path.name == b.audio_path.name, "cache key is not deterministic"
    assert sha256(a.audio_path) == sha256(b.audio_path)


def test_second_call_with_identical_input_hits_the_cache(tmp_path):
    spy = synth_spy()

    with patch.object(speech, "_synthesize_audio", spy):
        first = synthesize([beat()], VOICE, tmp_path)
        second = synthesize([beat()], VOICE, tmp_path)

    assert spy.call_count == 1, "second call re-synthesised instead of using the cache"
    assert first[0].audio_path == second[0].audio_path
    assert first[0].duration == pytest.approx(second[0].duration, abs=0.001)


def test_cache_key_ignores_beat_index(tmp_path):
    spy = synth_spy()

    with patch.object(speech, "_synthesize_audio", spy):
        a = synthesize([beat(SENTENCE, index=0)], VOICE, tmp_path)[0]
        b = synthesize([beat(SENTENCE, index=7)], VOICE, tmp_path)[0]

    assert a.audio_path == b.audio_path
    assert spy.call_count == 1


@pytest.mark.parametrize(
    "change",
    [
        pytest.param({"text": "Different words entirely."}, id="text"),
        pytest.param({"voice": "am_michael"}, id="voice"),
        pytest.param({"cfg": replace(CFG, speed=1.3)}, id="speed"),
        pytest.param({"cfg": replace(CFG, model_version="kokoro-82m-v9.9")}, id="model_version"),
        pytest.param({"cfg": replace(CFG, sample_rate=16000)}, id="sample_rate"),
    ],
)
def test_cache_key_changes_when_an_input_changes(tmp_path, change):
    text = change.get("text", SENTENCE)
    voice = change.get("voice", VOICE)
    cfg = change.get("cfg", CFG)

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        baseline = synthesize([beat()], VOICE, tmp_path, cfg=CFG)[0]
        changed = synthesize([beat(text)], voice, tmp_path, cfg=cfg)[0]

    assert baseline.audio_path != changed.audio_path


def test_cache_key_ignores_device(tmp_path):
    # Same model and text on a different device is the same narration; the
    # cache should not be invalidated by where it was computed.
    spy = synth_spy()

    with patch.object(speech, "_synthesize_audio", spy):
        a = synthesize([beat()], VOICE, tmp_path, cfg=replace(CFG, device="cpu"))[0]
        b = synthesize([beat()], VOICE, tmp_path, cfg=replace(CFG, device="mps"))[0]

    assert a.audio_path == b.audio_path
    assert spy.call_count == 1


def test_creates_the_cache_directory_if_missing(tmp_path):
    cache_dir = tmp_path / "nested" / "cache"

    with patch.object(speech, "_synthesize_audio", synth_spy()):
        out = synthesize([beat()], VOICE, cache_dir)

    assert out[0].audio_path.parent == cache_dir


# --- validation -------------------------------------------------------------


def test_unknown_voice_raises_value_error_without_synthesising(tmp_path):
    spy = synth_spy()

    with patch.object(speech, "_synthesize_audio", spy):
        with pytest.raises(ValueError, match="unknown voice"):
            synthesize([beat()], "af_nonexistent", tmp_path)

    assert spy.call_count == 0, "silently fell back instead of raising"
    assert list(tmp_path.glob("*.wav")) == []


def test_known_voices_are_accepted(tmp_path):
    with patch.object(speech, "_synthesize_audio", synth_spy()):
        for voice in ("af_heart", "am_michael", "bf_emma"):
            synthesize([beat()], voice, tmp_path)


def test_unknown_device_raises_value_error():
    with pytest.raises(ValueError, match="unknown device"):
        SpeechConfig(device="tpu")


def test_default_device_is_cpu():
    assert SpeechConfig().device == "cpu"


@pytest.mark.parametrize("device", ["cpu", "mps", "cuda"])
def test_supported_devices_are_accepted(device):
    assert SpeechConfig(device=device).device == device


# --- device plumbing --------------------------------------------------------


def fake_pipeline(cfg: SpeechConfig = CFG):
    """Stands in for a loaded KPipeline: called with text, yields Results."""

    def pipeline(text, voice, speed=1.0, **kwargs):
        yield SimpleNamespace(audio=fake_audio(text, cfg))

    return pipeline


def test_configured_device_is_passed_to_the_model_loader(tmp_path):
    loader = Mock(return_value=fake_pipeline())
    cfg = replace(CFG, device="mps")

    with patch.object(speech, "_pipeline", loader):
        synthesize([beat()], VOICE, tmp_path, cfg=cfg)

    loader.assert_called_once_with("mps", cfg.lang_code)


def test_device_is_not_hardcoded(tmp_path):
    loader = Mock(return_value=fake_pipeline())

    for device in ("cpu", "mps", "cuda"):
        with patch.object(speech, "_pipeline", loader):
            synthesize([beat()], VOICE, tmp_path / device, cfg=replace(CFG, device=device))

    assert [call.args[0] for call in loader.call_args_list] == ["cpu", "mps", "cuda"]


def test_voice_and_speed_reach_the_pipeline(tmp_path):
    calls = []

    def recording_pipeline(text, voice, speed=1.0, **kwargs):
        calls.append((text, voice, speed))
        yield SimpleNamespace(audio=fake_audio(text))

    with patch.object(speech, "_pipeline", Mock(return_value=recording_pipeline)):
        synthesize([beat()], VOICE, tmp_path, cfg=replace(CFG, speed=1.25))

    assert calls == [(SENTENCE, VOICE, 1.25)]


def test_pipeline_yielding_no_audio_raises(tmp_path):
    def silent_pipeline(text, voice, speed=1.0, **kwargs):
        yield SimpleNamespace(audio=None)

    with patch.object(speech, "_pipeline", Mock(return_value=silent_pipeline)):
        with pytest.raises(SynthesisError, match="no audio"):
            synthesize([beat()], VOICE, tmp_path)


def test_multiple_chunks_are_concatenated(tmp_path):
    def two_chunk_pipeline(text, voice, speed=1.0, **kwargs):
        half = fake_audio(text)
        yield SimpleNamespace(audio=half)
        yield SimpleNamespace(audio=half)

    with patch.object(speech, "_pipeline", Mock(return_value=two_chunk_pipeline)):
        out = synthesize([beat()], VOICE, tmp_path)

    single = len(SENTENCE) / 15.0
    assert out[0].duration == pytest.approx(2 * single, abs=0.05)
