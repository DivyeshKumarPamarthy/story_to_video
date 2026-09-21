"""Tests for the installed package, not the importable source tree.

Every other CLI test uses click's CliRunner, which imports in-process and so
cannot see a broken install. p8 found the console script raising
ModuleNotFoundError while the whole CLI suite passed.

These tests build the project and install it into a throwaway venv, then run
the `narrator` executable as a subprocess. That is deliberately not the
development venv: `uv run` re-syncs before every command, and uv's editable
install leaves the script unable to import the package on this machine (see
CLAUDE.md). What matters is that someone who installs the project gets a
working command, which is what this checks.
"""

from __future__ import annotations

import json
import os
import subprocess
from pathlib import Path

import pytest

REPO = Path(__file__).resolve().parent.parent
FIXTURE = REPO / "tests" / "fixtures" / "tiny_story.txt"


@pytest.fixture(scope="session")
def installed_script(tmp_path_factory) -> Path:
    """Build and install the project into a fresh venv, once per session."""
    venv = tmp_path_factory.mktemp("install") / "venv"
    subprocess.run(["uv", "venv", "--python", "3.12", str(venv)], check=True, capture_output=True)
    install = subprocess.run(
        ["uv", "pip", "install", "--python", str(venv / "bin" / "python"), str(REPO)],
        capture_output=True,
        text=True,
    )
    assert install.returncode == 0, install.stderr

    script = venv / "bin" / "narrator"
    assert script.exists(), "installing the project produced no console script"
    assert os.access(script, os.X_OK), f"{script} is not executable"
    return script


def run(script: Path, args: list[str], cwd: Path) -> subprocess.CompletedProcess:
    """Run the console script with no help from the source tree.

    PYTHONPATH is cleared and the working directory is elsewhere, so the
    package must come from the install rather than from the repo happening to
    be importable.
    """
    env = dict(os.environ)
    env.pop("PYTHONPATH", None)
    return subprocess.run(
        [str(script), *args],
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=300,
        env=env,
    )


def test_help_runs_from_an_unrelated_directory(installed_script, tmp_path):
    result = run(installed_script, ["build", "--help"], cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "ModuleNotFoundError" not in result.stderr
    for flag in ("--voice", "--out", "--dry-run"):
        assert flag in result.stdout


def test_version_runs_from_an_unrelated_directory(installed_script, tmp_path):
    result = run(installed_script, ["--version"], cwd=tmp_path)

    assert result.returncode == 0, result.stderr
    assert "narrator" in result.stdout


def test_dry_run_works_through_the_installed_script(installed_script, tmp_path):
    out = tmp_path / "out.mp4"
    run_dir = tmp_path / "run"

    result = run(
        installed_script,
        ["build", str(FIXTURE), "--out", str(out), "--run-dir", str(run_dir), "--dry-run"],
        cwd=tmp_path,
    )

    assert result.returncode == 0, f"stdout={result.stdout}\nstderr={result.stderr}"
    manifest = run_dir / "manifest.json"
    assert manifest.exists()
    assert json.loads(manifest.read_text())["beat_count"] >= 1
    assert not out.exists(), "a dry run rendered something"


def test_a_missing_story_exits_one_without_a_traceback(installed_script, tmp_path):
    result = run(
        installed_script,
        ["build", str(tmp_path / "nope.txt"), "--out", str(tmp_path / "o.mp4")],
        cwd=tmp_path,
    )

    assert result.returncode == 1
    assert "Traceback" not in result.stderr
    assert "nope.txt" in result.stdout + result.stderr


def test_the_package_does_not_depend_on_the_repo_being_present(installed_script, tmp_path):
    # Imports the installed copy directly, with the repo nowhere in sight.
    python = installed_script.parent / "python"
    result = subprocess.run(
        [str(python), "-c", "import narrator, narrator.cli; print(narrator.__file__)"],
        cwd=str(tmp_path),
        capture_output=True,
        text=True,
    )

    assert result.returncode == 0, result.stderr
    assert str(REPO) not in result.stdout, "the install points back at the source tree"
