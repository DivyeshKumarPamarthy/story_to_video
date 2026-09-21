"""Tests for narrator.visuals.stills -- a still pushed in with zoompan.

Every assertion about the rendered file goes through ffprobe, per CLAUDE.md.
Resolutions and durations are kept tiny so the suite stays fast.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from narrator.beats import Beat
from narrator.config import VisualsConfig
from narrator.visuals import stills
from tests import ffprobe

CFG = VisualsConfig(width=320, height=240, fps=12)


def beat(query: str = "empty house", index: int = 0, duration: float | None = 1.0) -> Beat:
    return Beat(index=index, text="The house stood empty.", visual_query=query, duration=duration)


def test_renders_a_playable_video(tmp_path):
    out = stills.render(beat(), CFG, tmp_path / "beat.mp4")

    assert out.exists()
    assert out.stat().st_size > 0
    assert len(ffprobe.video_streams(out)) == 1


def test_resolution_matches_config(tmp_path):
    out = stills.render(beat(), CFG, tmp_path / "beat.mp4")
    assert ffprobe.resolution(out) == (320, 240)


def test_vertical_resolution_is_honoured(tmp_path):
    cfg = VisualsConfig(width=180, height=320, fps=12)
    out = stills.render(beat(), cfg, tmp_path / "vertical.mp4")
    assert ffprobe.resolution(out) == (180, 320)


def test_fps_matches_config(tmp_path):
    out = stills.render(beat(), CFG, tmp_path / "beat.mp4")
    assert ffprobe.fps(out) == pytest.approx(12, abs=0.1)


@pytest.mark.parametrize("seconds", [0.5, 1.0, 2.0])
def test_duration_matches_the_beat(tmp_path, seconds):
    out = stills.render(beat(duration=seconds), CFG, tmp_path / f"{seconds}.mp4")
    assert ffprobe.duration(out) == pytest.approx(seconds, abs=0.15)


def test_beat_without_duration_uses_the_configured_default(tmp_path):
    cfg = VisualsConfig(width=320, height=240, fps=12, default_seconds=1.5)
    out = stills.render(beat(duration=None), cfg, tmp_path / "default.mp4")
    assert ffprobe.duration(out) == pytest.approx(1.5, abs=0.15)


def test_has_no_audio_stream(tmp_path):
    # Narration is the only audio; a stock clip's own track must not leak in.
    out = stills.render(beat(), CFG, tmp_path / "beat.mp4")
    assert ffprobe.audio_streams(out) == []


def test_different_queries_render_different_backgrounds(tmp_path):
    first = stills.render(beat("empty house", 0), CFG, tmp_path / "a.mp4")
    second = stills.render(beat("frozen river", 1), CFG, tmp_path / "b.mp4")
    assert first.read_bytes() != second.read_bytes()


def test_same_query_is_reproducible(tmp_path):
    first = stills.render(beat("empty house"), CFG, tmp_path / "a.mp4")
    second = stills.render(beat("empty house"), CFG, tmp_path / "b.mp4")
    assert stills.variant_for(beat("empty house"), CFG, exclude=()) == stills.variant_for(
        beat("empty house"), CFG, exclude=()
    )
    assert first.exists() and second.exists()


def test_excluded_variant_is_not_reused(tmp_path):
    used = stills.variant_for(beat("empty house"), CFG, exclude=())
    other = stills.variant_for(beat("empty house"), CFG, exclude=(used,))
    assert other != used


def test_a_supplied_still_is_used_when_one_is_available(tmp_path):
    stills_dir = tmp_path / "stills"
    stills_dir.mkdir()
    image = stills_dir / "backdrop.png"
    stills.write_generated_still(image, "seeded", VisualsConfig(width=320, height=240))
    cfg = VisualsConfig(width=320, height=240, fps=12, stills_dir=stills_dir)

    out = stills.render(beat(), cfg, tmp_path / "beat.mp4")

    assert ffprobe.resolution(out) == (320, 240)
    assert ffprobe.duration(out) == pytest.approx(1.0, abs=0.15)


def test_ffmpeg_failure_surfaces_stderr(tmp_path):
    broken = VisualsConfig(
        width=320, height=240, fps=12, stills_dir=tmp_path / "missing", ffmpeg_timeout=5
    )
    (tmp_path / "missing").mkdir()
    (tmp_path / "missing" / "not_an_image.png").write_text("this is not a png")

    with pytest.raises(stills.StillsError) as exc:
        stills.render(beat(), broken, tmp_path / "beat.mp4")

    message = str(exc.value).lower()
    assert "ffmpeg" in message
    # -loop 1 on an undecodable image hangs rather than erroring, so the
    # timeout is the failure path here.
    assert "timed out" in message or "failed" in message


def test_output_directory_is_created(tmp_path):
    out = stills.render(beat(), CFG, tmp_path / "nested" / "deep" / "beat.mp4")
    assert out.exists()


def test_command_is_a_list_and_is_logged(tmp_path, caplog):
    import logging

    with caplog.at_level(logging.DEBUG, logger="narrator.visuals.stills"):
        stills.render(beat(), CFG, tmp_path / "beat.mp4")

    assert any("ffmpeg" in record.message for record in caplog.records)


def test_path_is_returned_not_a_string(tmp_path):
    assert isinstance(stills.render(beat(), CFG, tmp_path / "beat.mp4"), Path)
