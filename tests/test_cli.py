"""Tests for the narrator CLI."""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from narrator import pipeline as pipeline_module
from narrator.cli import main
from tests.test_pipeline import fake_align, fake_synthesize

FIXTURE = Path(__file__).parent / "fixtures" / "tiny_story.txt"


@pytest.fixture
def mocked(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    monkeypatch.setattr(pipeline_module.speech, "synthesize", fake_synthesize)
    monkeypatch.setattr(pipeline_module.align_module, "align", fake_align)


def test_builds_a_video(tmp_path, mocked):
    out = tmp_path / "out.mp4"
    result = CliRunner().invoke(
        main,
        [
            "build",
            str(FIXTURE),
            "--out",
            str(out),
            "--voice",
            "af_heart",
            "--run-dir",
            str(tmp_path / "run"),
            "--preset",
            "landscape",
            "--width",
            "320",
            "--height",
            "240",
            "--fps",
            "12",
        ],
    )

    assert result.exit_code == 0, result.output
    assert out.exists()


def test_dry_run_renders_nothing(tmp_path, mocked):
    out = tmp_path / "out.mp4"
    result = CliRunner().invoke(
        main,
        ["build", str(FIXTURE), "--out", str(out), "--dry-run", "--run-dir", str(tmp_path / "run")],
    )

    assert result.exit_code == 0, result.output
    assert not out.exists()
    assert (tmp_path / "run" / "manifest.json").exists()
    assert "manifest" in result.output.lower()


def test_missing_input_exits_one_with_a_readable_message(tmp_path):
    result = CliRunner().invoke(
        main, ["build", str(tmp_path / "nope.txt"), "--out", str(tmp_path / "out.mp4")]
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "nope.txt" in result.output


def test_unknown_voice_exits_one_without_a_traceback(tmp_path, mocked):
    result = CliRunner().invoke(
        main,
        [
            "build",
            str(FIXTURE),
            "--out",
            str(tmp_path / "out.mp4"),
            "--voice",
            "af_nonexistent",
            "--run-dir",
            str(tmp_path / "run"),
        ],
    )

    assert result.exit_code == 1
    assert "Traceback" not in result.output
    assert "voice" in result.output.lower()


def test_unknown_preset_is_rejected_by_the_parser(tmp_path):
    result = CliRunner().invoke(
        main, ["build", str(FIXTURE), "--out", str(tmp_path / "o.mp4"), "--preset", "square"]
    )
    assert result.exit_code != 0


def test_help_lists_the_options():
    result = CliRunner().invoke(main, ["build", "--help"])

    assert result.exit_code == 0
    for flag in ("--voice", "--out", "--dry-run", "--preset", "--music"):
        assert flag in result.output


def test_version_is_available():
    result = CliRunner().invoke(main, ["--version"])
    assert result.exit_code == 0
