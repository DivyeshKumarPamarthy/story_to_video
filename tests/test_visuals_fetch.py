"""Tests for the fetch interface: Pexels first, stills as the fallback."""

from __future__ import annotations

import logging

import pytest

from narrator import visuals
from narrator.beats import Beat
from narrator.config import VisualsConfig
from narrator.visuals import pexels
from tests import ffprobe

CFG = VisualsConfig(width=320, height=240, fps=12)


def beat(index: int = 0, query: str = "empty house", duration: float = 1.0) -> Beat:
    return Beat(index=index, text="The house stood empty.", visual_query=query, duration=duration)


def fake_clip(tmp_path, seconds=2.0, width=640, height=480):
    """A real video file standing in for a downloaded Pexels clip."""
    import subprocess

    path = tmp_path / "stock.mp4"
    subprocess.run(
        [
            "ffmpeg",
            "-y",
            "-v",
            "error",
            "-f",
            "lavfi",
            "-i",
            f"testsrc=size={width}x{height}:rate=15:duration={seconds}",
            "-pix_fmt",
            "yuv420p",
            str(path),
        ],
        check=True,
    )
    return path


# --- happy path -------------------------------------------------------------


def test_uses_pexels_when_it_works(tmp_path, monkeypatch):
    clip = fake_clip(tmp_path)
    monkeypatch.setattr(pexels, "fetch_clip", lambda *a, **k: (clip, "101"))

    out = visuals.fetch(beat(), CFG, tmp_path / "out")

    assert out.exists()
    assert ffprobe.resolution(out) == (320, 240)
    assert ffprobe.fps(out) == pytest.approx(12, abs=0.1)


def test_conformed_clip_matches_the_beat_duration(tmp_path, monkeypatch):
    clip = fake_clip(tmp_path, seconds=3.0)
    monkeypatch.setattr(pexels, "fetch_clip", lambda *a, **k: (clip, "101"))

    out = visuals.fetch(beat(duration=1.0), CFG, tmp_path / "out")

    assert ffprobe.duration(out) == pytest.approx(1.0, abs=0.15)


def test_a_clip_shorter_than_the_beat_is_looped_to_cover_it(tmp_path, monkeypatch):
    clip = fake_clip(tmp_path, seconds=0.5)
    monkeypatch.setattr(pexels, "fetch_clip", lambda *a, **k: (clip, "101"))

    out = visuals.fetch(beat(duration=2.0), CFG, tmp_path / "out")

    assert ffprobe.duration(out) == pytest.approx(2.0, abs=0.15)


def test_stock_audio_is_stripped(tmp_path, monkeypatch):
    clip = fake_clip(tmp_path)
    monkeypatch.setattr(pexels, "fetch_clip", lambda *a, **k: (clip, "101"))

    out = visuals.fetch(beat(), CFG, tmp_path / "out")

    assert ffprobe.audio_streams(out) == []


# --- fallback ---------------------------------------------------------------


def test_api_error_falls_back_to_stills_without_raising(tmp_path, monkeypatch):
    def boom(*args, **kwargs):
        raise pexels.PexelsError("502 from upstream")

    monkeypatch.setattr(pexels, "fetch_clip", boom)

    out = visuals.fetch(beat(), CFG, tmp_path / "out")

    assert out.exists()
    assert ffprobe.resolution(out) == (320, 240)


def test_zero_results_falls_back_to_stills(tmp_path, monkeypatch):
    def empty(*args, **kwargs):
        raise pexels.PexelsError("no results for 'empty house'")

    monkeypatch.setattr(pexels, "fetch_clip", empty)

    out = visuals.fetch(beat(), CFG, tmp_path / "out")
    assert ffprobe.duration(out) == pytest.approx(1.0, abs=0.15)


def test_missing_key_falls_back_but_says_so_out_loud(tmp_path, monkeypatch, caplog):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    with caplog.at_level(logging.WARNING, logger="narrator.visuals"):
        out = visuals.fetch(beat(), CFG, tmp_path / "out")

    assert out.exists(), "a missing key must not stop the pipeline producing video"
    assert any("PEXELS_API_KEY" in record.message for record in caplog.records), (
        "falling back for lack of a key must be logged, not silent"
    )


def test_missing_key_is_fatal_when_pexels_is_required(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    cfg = VisualsConfig(width=320, height=240, fps=12, require_pexels=True)

    with pytest.raises(pexels.MissingApiKey):
        visuals.fetch(beat(), cfg, tmp_path / "out")


def test_api_failure_is_still_fatal_when_pexels_is_required(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    monkeypatch.setattr(
        pexels, "fetch_clip", lambda *a, **k: (_ for _ in ()).throw(pexels.PexelsError("502"))
    )
    cfg = VisualsConfig(width=320, height=240, fps=12, require_pexels=True)

    with pytest.raises(pexels.PexelsError):
        visuals.fetch(beat(), cfg, tmp_path / "out")


# --- as a pipeline stage ----------------------------------------------------


def test_fetch_all_fills_asset_path_only(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    beats = [beat(0), beat(1, "frozen river")]

    out = visuals.fetch_all(beats, CFG, tmp_path / "out")

    for original, result in zip(beats, out, strict=True):
        assert result.asset_path is not None
        assert result.asset_path.exists()
        assert result.text == original.text
        assert result.visual_query == original.visual_query
        assert result.duration == original.duration
    assert beats[0].asset_path is None, "input was mutated"


def test_fetch_all_returns_empty_for_empty_input(tmp_path):
    assert visuals.fetch_all([], CFG, tmp_path / "out") == []


def test_consecutive_beats_never_share_an_asset(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "k")
    clip = fake_clip(tmp_path)
    seen_excludes = []
    ids = iter(["101", "202", "303"])

    def fetch_clip(beat_, cfg, dest, exclude):
        seen_excludes.append(set(exclude))
        return clip, next(ids)

    monkeypatch.setattr(pexels, "fetch_clip", fetch_clip)

    visuals.fetch_all([beat(0), beat(1), beat(2)], CFG, tmp_path / "out")

    assert seen_excludes[0] == set()
    assert "101" in seen_excludes[1]
    assert "202" in seen_excludes[2]


def test_consecutive_still_fallbacks_differ(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)

    out = visuals.fetch_all([beat(0), beat(1), beat(2)], CFG, tmp_path / "out")
    renders = [b.asset_path.read_bytes() for b in out]

    assert renders[0] != renders[1]
    assert renders[1] != renders[2]


def test_each_beat_gets_its_own_file(tmp_path, monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    out = visuals.fetch_all([beat(0), beat(1)], CFG, tmp_path / "out")
    assert out[0].asset_path != out[1].asset_path
