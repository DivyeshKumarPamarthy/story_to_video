"""Tests for narrator.pipeline.

Speech and alignment are mocked: the point here is the wiring, the run
directory and resumption, not the models, which have their own suites.
"""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import Mock

import pytest

from narrator import pipeline as pipeline_module
from narrator.beats import Word
from narrator.config import PipelineConfig
from narrator.pipeline import PipelineError, build, plan
from tests import ffprobe

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_story.txt"

# max_chars=60 keeps the fixture's three sentences as three beats; at the
# 220 default they pack into one, which would not exercise concatenation.
CFG = PipelineConfig.preset(
    "landscape", width=320, height=240, fps=12, sample_rate=24000, max_chars=60
)


def fake_synthesize(beats, voice, cache_dir, cfg=None):
    """Write a real, short wav per beat so ffprobe and ffmpeg have something."""
    import subprocess
    from dataclasses import replace

    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    out = []
    for beat in beats:
        path = cache_dir / f"beat_{beat.index:03d}.wav"
        if not path.exists():
            subprocess.run(
                [
                    "ffmpeg",
                    "-nostdin",
                    "-y",
                    "-v",
                    "error",
                    "-f",
                    "lavfi",
                    "-i",
                    "sine=frequency=330:duration=0.6:sample_rate=24000",
                    str(path),
                ],
                check=True,
            )
        out.append(replace(beat, audio_path=path, duration=0.6))
    return out


def fake_align(beats, cfg=None):
    from dataclasses import replace

    aligned = []
    for beat in beats:
        words = beat.text.split()
        step = (beat.duration or 0.6) / max(len(words), 1)
        aligned.append(
            replace(
                beat,
                words=[
                    Word(text=w, start=i * step, end=(i + 1) * step) for i, w in enumerate(words)
                ],
            )
        )
    return aligned


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(pipeline_module.speech, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline_module.align_module, "align", fake_align)
    return None


# --- the golden path --------------------------------------------------------


def test_builds_a_video_from_the_fixture(tmp_path, mocked):
    out = build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")

    assert out.exists()
    assert len(ffprobe.video_streams(out)) == 1
    assert len(ffprobe.audio_streams(out)) == 1
    assert ffprobe.resolution(out) == (320, 240)
    assert ffprobe.fps(out) == pytest.approx(12, abs=0.1)


def test_duration_is_the_sum_of_the_beats(tmp_path, mocked):
    out = build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")

    # tiny_story.txt is three sentences; the stub gives each 0.6s.
    assert ffprobe.duration(out) == pytest.approx(1.8, abs=0.2)


def test_narration_starts_at_zero(tmp_path, mocked):
    out = build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")
    assert ffprobe.leading_silence(out) <= 0.05


def test_run_directory_holds_each_stage(tmp_path, mocked):
    run_dir = tmp_path / "run"
    build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=run_dir)

    assert (run_dir / "manifest.json").exists()
    assert (run_dir / "captions.ass").exists()
    assert list((run_dir / "audio").glob("*.wav"))
    assert list((run_dir / "visuals").glob("*.mp4"))


# --- dry run ----------------------------------------------------------------


def test_dry_run_writes_a_manifest_and_renders_nothing(tmp_path, mocked):
    run_dir = tmp_path / "run"
    out_path = tmp_path / "out.mp4"

    manifest = plan(FIXTURE, out_path, cfg=CFG, run_dir=run_dir)

    assert manifest.exists()
    assert not out_path.exists()
    assert not list(run_dir.glob("**/*.mp4"))


def test_dry_run_manifest_describes_the_beats(tmp_path, mocked):
    manifest = json.loads(
        plan(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run").read_text()
    )

    assert manifest["beat_count"] == 3
    assert len(manifest["beats"]) == 3
    assert manifest["beats"][0]["index"] == 0
    assert manifest["beats"][0]["text"].startswith("The house")
    assert manifest["beats"][0]["visual_query"]
    assert manifest["out"].endswith("out.mp4")


def test_dry_run_does_not_synthesise(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    spy = Mock(side_effect=fake_synthesize)
    monkeypatch.setattr(pipeline_module.speech, "synthesize", spy)

    plan(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")

    assert spy.call_count == 0


# --- resumption -------------------------------------------------------------


def test_rerun_reuses_cached_audio(tmp_path, monkeypatch, mocked):
    run_dir = tmp_path / "run"
    out_path = tmp_path / "out.mp4"
    build(FIXTURE, out_path, cfg=CFG, run_dir=run_dir)

    spy = Mock(side_effect=fake_synthesize)
    monkeypatch.setattr(pipeline_module.speech, "synthesize", spy)
    out_path.unlink()

    build(FIXTURE, out_path, cfg=CFG, run_dir=run_dir)

    assert out_path.exists()
    assert spy.call_count == 0, "re-synthesised narration that was already on disk"


def test_rerun_reuses_visuals(tmp_path, monkeypatch, mocked):
    run_dir = tmp_path / "run"
    build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=run_dir)

    spy = Mock()
    monkeypatch.setattr(pipeline_module.visuals, "fetch_all", spy)
    build(FIXTURE, tmp_path / "out2.mp4", cfg=CFG, run_dir=run_dir)

    assert spy.call_count == 0


def test_force_ignores_the_cache(tmp_path, monkeypatch, mocked):
    run_dir = tmp_path / "run"
    build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=run_dir)

    spy = Mock(side_effect=fake_synthesize)
    monkeypatch.setattr(pipeline_module.speech, "synthesize", spy)
    build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=run_dir, force=True)

    assert spy.call_count == 1


def test_a_changed_story_is_not_served_from_the_old_run(tmp_path, mocked):
    run_dir = tmp_path / "run"
    build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=run_dir)

    changed = tmp_path / "changed.txt"
    changed.write_text("A completely different sentence entirely. And a second one here.")
    build(changed, tmp_path / "out2.mp4", cfg=CFG, run_dir=run_dir)

    manifest = json.loads((run_dir / "manifest.json").read_text())
    assert manifest["beat_count"] == 2
    assert "different sentence" in manifest["beats"][0]["text"]


# --- failure ----------------------------------------------------------------


def test_missing_input_raises_pipeline_error(tmp_path):
    with pytest.raises(PipelineError, match="does not exist"):
        build(tmp_path / "nope.txt", tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")


def test_empty_story_raises(tmp_path, mocked):
    empty = tmp_path / "empty.txt"
    empty.write_text("   \n\n  ")

    with pytest.raises(PipelineError, match="no beats"):
        build(empty, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")


def test_alignment_failure_is_not_swallowed(tmp_path, monkeypatch, mocked):
    from narrator.align import AlignmentError

    monkeypatch.setattr(
        pipeline_module.align_module,
        "align",
        Mock(side_effect=AlignmentError("beat 0: match ratio 0.10 is below 0.85")),
    )

    with pytest.raises(AlignmentError):
        build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run")


def test_music_is_passed_through_to_the_mix(tmp_path, mocked):
    import subprocess

    bed = tmp_path / "bed.wav"
    subprocess.run(
        [
            "ffmpeg",
            "-nostdin",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            "sine=frequency=660:duration=3:sample_rate=24000",
            str(bed),
        ],
        check=True,
    )

    out = build(FIXTURE, tmp_path / "out.mp4", cfg=CFG, run_dir=tmp_path / "run", music=bed)

    assert ffprobe.duration(out) == pytest.approx(1.8, abs=0.2)
