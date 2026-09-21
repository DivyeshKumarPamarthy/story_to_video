"""Tests for narrator.assemble.

Fixtures are 1-second lavfi sources, so the suite stays fast while every
assertion is made against a real encoded file through ffprobe.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

import pytest

from narrator import assemble as assemble_module
from narrator.assemble import AssembleError, assemble, has_ass_filter
from narrator.beats import Beat, Word
from narrator.config import AssembleConfig
from tests import ffprobe

CFG = AssembleConfig(width=320, height=240, fps=12, sample_rate=24000)


def lavfi(path: Path, spec: str, extra: list[str] | None = None) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    subprocess.run(
        ["ffmpeg", "-nostdin", "-y", "-v", "error", "-f", "lavfi", "-i", spec]
        + (extra or [])
        + [str(path)],
        check=True,
    )
    return path


def make_beat(tmp_path: Path, index: int, seconds: float = 1.0, silent: bool = False) -> Beat:
    video = lavfi(
        tmp_path / f"v{index}.mp4",
        f"testsrc=size=320x240:rate=12:duration={seconds}",
        ["-pix_fmt", "yuv420p"],
    )
    spec = (
        f"anullsrc=r=24000:cl=mono:d={seconds}"
        if silent
        else f"sine=frequency={220 + index * 110}:duration={seconds}:sample_rate=24000"
    )
    audio = lavfi(tmp_path / f"a{index}.wav", spec)
    return Beat(
        index=index,
        text=f"Beat {index}.",
        visual_query="",
        audio_path=audio,
        duration=seconds,
        words=[Word(text=f"Beat{index}", start=0.0, end=seconds)],
        asset_path=video,
    )


def music(tmp_path: Path, seconds: float = 3.0) -> Path:
    return lavfi(tmp_path / "music.wav", f"sine=frequency=660:duration={seconds}:sample_rate=24000")


# --- streams and shape ------------------------------------------------------


def test_produces_exactly_one_video_and_one_audio_stream(tmp_path):
    beats = [make_beat(tmp_path, 0), make_beat(tmp_path, 1)]
    out = assemble(beats, tmp_path / "out.mp4", CFG)

    assert len(ffprobe.video_streams(out)) == 1
    assert len(ffprobe.audio_streams(out)) == 1


def test_duration_is_the_sum_of_the_beats(tmp_path):
    beats = [make_beat(tmp_path, 0, 1.0), make_beat(tmp_path, 1, 0.5)]
    out = assemble(beats, tmp_path / "out.mp4", CFG)

    assert ffprobe.duration(out) == pytest.approx(1.5, abs=0.2)


def test_resolution_and_fps_match_config(tmp_path):
    beats = [make_beat(tmp_path, 0)]
    out = assemble(beats, tmp_path / "out.mp4", CFG)

    assert ffprobe.resolution(out) == (320, 240)
    assert ffprobe.fps(out) == pytest.approx(12, abs=0.1)


def test_vertical_config_is_honoured(tmp_path):
    cfg = AssembleConfig(width=180, height=320, fps=12, sample_rate=24000)
    beats = [make_beat(tmp_path, 0)]
    out = assemble(beats, tmp_path / "out.mp4", cfg)

    assert ffprobe.resolution(out) == (180, 320)


def test_output_directory_is_created(tmp_path):
    beats = [make_beat(tmp_path, 0)]
    out = assemble(beats, tmp_path / "nested" / "deep" / "out.mp4", CFG)
    assert out.exists()


# --- sync -------------------------------------------------------------------


def test_narration_starts_within_50ms_of_zero(tmp_path):
    # The sync regression test: a leading gap here desyncs every caption.
    beats = [make_beat(tmp_path, 0), make_beat(tmp_path, 1)]
    out = assemble(beats, tmp_path / "out.mp4", CFG)

    assert ffprobe.leading_silence(out) <= 0.05


def test_narration_still_starts_at_zero_with_a_music_bed(tmp_path):
    beats = [make_beat(tmp_path, 0)]
    out = assemble(beats, tmp_path / "out.mp4", CFG, music=music(tmp_path))

    assert ffprobe.leading_silence(out) <= 0.05


# --- the music bed ----------------------------------------------------------


def test_music_is_measurably_quieter_than_narration(tmp_path):
    # Narration silent, music present: what reaches the mix is the music bed
    # alone, so its level can be measured directly against the source.
    beats = [make_beat(tmp_path, 0, silent=True)]
    bed = music(tmp_path)

    out = assemble(beats, tmp_path / "out.mp4", CFG, music=bed)

    source_level = ffprobe.mean_volume(bed)
    mixed_level = ffprobe.mean_volume(out)
    attenuation = source_level - mixed_level

    assert attenuation == pytest.approx(abs(CFG.music_gain_db), abs=3.0), (
        f"music bed is {attenuation:.1f}dB down, expected about {abs(CFG.music_gain_db)}dB"
    )


def test_narration_is_not_attenuated_by_the_mix(tmp_path):
    # amix normalises by default, which would quietly halve the narration.
    beats = [make_beat(tmp_path, 0)]
    without = assemble(beats, tmp_path / "without.mp4", CFG)
    with_music = assemble(beats, tmp_path / "with.mp4", CFG, music=music(tmp_path))

    assert ffprobe.mean_volume(with_music) == pytest.approx(ffprobe.mean_volume(without), abs=2.0)


def test_music_is_looped_to_cover_a_longer_story(tmp_path):
    beats = [make_beat(tmp_path, 0, 1.0), make_beat(tmp_path, 1, 1.0)]
    short_bed = lavfi(tmp_path / "short.wav", "sine=frequency=660:duration=0.4:sample_rate=24000")

    out = assemble(beats, tmp_path / "out.mp4", CFG, music=short_bed)

    assert ffprobe.duration(out) == pytest.approx(2.0, abs=0.2)


def test_music_never_outlasts_the_narration(tmp_path):
    beats = [make_beat(tmp_path, 0, 0.5)]
    out = assemble(beats, tmp_path / "out.mp4", CFG, music=music(tmp_path, seconds=5.0))

    assert ffprobe.duration(out) == pytest.approx(0.5, abs=0.2)


# --- captions ---------------------------------------------------------------


def caption_file(tmp_path: Path, beats: list[Beat]) -> Path:
    from narrator.captions import build_ass
    from narrator.config import CaptionConfig

    return build_ass(beats, CaptionConfig(width=320, height=240), tmp_path / "c.ass")


def test_captions_are_included_somehow(tmp_path, caplog):
    beats = [make_beat(tmp_path, 0)]
    subtitles = caption_file(tmp_path, beats)

    with caplog.at_level(logging.WARNING, logger="narrator.assemble"):
        out = assemble(beats, tmp_path / "out.mp4", CFG, captions=subtitles)

    if has_ass_filter():
        assert ffprobe.subtitle_streams(out) == [], "burned captions leave no stream"
    else:
        assert len(ffprobe.subtitle_streams(out)) == 1, "captions were dropped entirely"
        assert any("libass" in record.message for record in caplog.records), (
            "falling back to soft subtitles must be logged, not silent"
        )


def test_no_subtitle_stream_when_no_captions_are_given(tmp_path):
    beats = [make_beat(tmp_path, 0)]
    out = assemble(beats, tmp_path / "out.mp4", CFG)
    assert ffprobe.subtitle_streams(out) == []


def test_requiring_burned_captions_fails_loudly_without_libass(tmp_path):
    beats = [make_beat(tmp_path, 0)]
    subtitles = caption_file(tmp_path, beats)
    cfg = AssembleConfig(
        width=320, height=240, fps=12, sample_rate=24000, require_burned_captions=True
    )

    if has_ass_filter():
        pytest.skip("this ffmpeg has libass, so the requirement is satisfiable")

    with pytest.raises(AssembleError, match="libass"):
        assemble(beats, tmp_path / "out.mp4", cfg, captions=subtitles)


def test_missing_caption_file_raises(tmp_path):
    beats = [make_beat(tmp_path, 0)]

    with pytest.raises(AssembleError, match="captions"):
        assemble(beats, tmp_path / "out.mp4", CFG, captions=tmp_path / "nope.ass")


# --- preconditions ----------------------------------------------------------


def test_no_beats_raises(tmp_path):
    with pytest.raises(AssembleError, match="no beats"):
        assemble([], tmp_path / "out.mp4", CFG)


def test_beat_without_asset_raises(tmp_path):
    beat = make_beat(tmp_path, 0)
    beat = Beat(**{**beat.__dict__, "asset_path": None})

    with pytest.raises(AssembleError, match="no visual"):
        assemble([beat], tmp_path / "out.mp4", CFG)


def test_beat_without_audio_raises(tmp_path):
    beat = make_beat(tmp_path, 0)
    beat = Beat(**{**beat.__dict__, "audio_path": None})

    with pytest.raises(AssembleError, match="no audio"):
        assemble([beat], tmp_path / "out.mp4", CFG)


def test_missing_asset_file_raises(tmp_path):
    beat = make_beat(tmp_path, 0)
    beat.asset_path.unlink()

    with pytest.raises(AssembleError, match="does not exist"):
        assemble([beat], tmp_path / "out.mp4", CFG)


# --- the command itself -----------------------------------------------------


def test_command_is_built_as_a_list_and_logged(tmp_path, caplog):
    beats = [make_beat(tmp_path, 0)]

    with caplog.at_level(logging.DEBUG, logger="narrator.assemble"):
        assemble(beats, tmp_path / "out.mp4", CFG)

    assert any("ffmpeg" in record.message for record in caplog.records)


def test_ffmpeg_failure_surfaces_stderr(tmp_path, monkeypatch):
    beats = [make_beat(tmp_path, 0)]
    monkeypatch.setattr(
        assemble_module, "_filter_complex", lambda *a, **k: "this is not a filtergraph"
    )

    with pytest.raises(AssembleError) as exc:
        assemble(beats, tmp_path / "out.mp4", CFG)

    message = str(exc.value)
    assert "ffmpeg" in message.lower()
    assert len(message) > 60, "the exception should carry ffmpeg's own words"
