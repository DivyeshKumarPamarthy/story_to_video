"""The narrator command line.

    narrator build story.txt --voice af_heart --out out.mp4 [--dry-run]

Errors are reported as one readable line and exit 1. A traceback in a user's
terminal is a bug report about us, not about their story.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

import click

from narrator import __version__
from narrator.align import AlignmentError
from narrator.assemble import AssembleError
from narrator.captions import CaptionError
from narrator.config import VIDEO_PRESETS, PipelineConfig
from narrator.pipeline import PipelineError
from narrator.pipeline import build as run_build
from narrator.pipeline import plan as run_plan
from narrator.speech import SynthesisError
from narrator.visuals import VisualsError

KNOWN_FAILURES = (
    PipelineError,
    AlignmentError,
    SynthesisError,
    CaptionError,
    AssembleError,
    VisualsError,
    ValueError,
    FileNotFoundError,
)


@click.group()
@click.version_option(__version__)
def main() -> None:
    """Turn a written story into a narrated video."""


@main.command()
@click.argument("story", type=click.Path(path_type=Path))
@click.option(
    "--out",
    type=click.Path(path_type=Path),
    default=Path("out.mp4"),
    show_default=True,
    help="Where to write the finished video.",
)
@click.option("--voice", default="af_heart", show_default=True, help="Kokoro voice id.")
@click.option(
    "--preset",
    type=click.Choice(sorted(VIDEO_PRESETS)),
    default="landscape",
    show_default=True,
    help="Aspect preset.",
)
@click.option("--width", type=int, default=None, help="Override the preset width.")
@click.option("--height", type=int, default=None, help="Override the preset height.")
@click.option("--fps", type=int, default=None, help="Override the preset frame rate.")
@click.option(
    "--music",
    type=click.Path(path_type=Path),
    default=None,
    help="Audio file to mix under the narration.",
)
@click.option(
    "--run-dir",
    type=click.Path(path_type=Path),
    default=None,
    help="Where stage output is cached. Defaults to .narrator_<story> next to it.",
)
@click.option("--dry-run", is_flag=True, help="Write manifest.json and render nothing.")
@click.option("--force", is_flag=True, help="Ignore cached stages and redo everything.")
@click.option("--verbose", "-v", is_flag=True, help="Log what each stage is doing.")
def build(
    story: Path,
    out: Path,
    voice: str,
    preset: str,
    width: int | None,
    height: int | None,
    fps: int | None,
    music: Path | None,
    run_dir: Path | None,
    dry_run: bool,
    force: bool,
    verbose: bool,
) -> None:
    """Build a narrated video from STORY."""
    logging.basicConfig(
        level=logging.INFO if verbose else logging.WARNING,
        format="%(levelname)s %(name)s: %(message)s",
    )

    overrides = {k: v for k, v in (("width", width), ("height", height), ("fps", fps)) if v}
    cfg = PipelineConfig.preset(preset, **overrides)

    try:
        if dry_run:
            manifest = run_plan(story, out, cfg=cfg, run_dir=run_dir)
            click.echo(f"dry run: wrote manifest {manifest}")
            return

        result = run_build(
            story, out, cfg=cfg, run_dir=run_dir, voice=voice, music=music, force=force
        )
    except KNOWN_FAILURES as exc:
        click.echo(f"error: {exc}", err=True)
        sys.exit(1)

    click.echo(f"wrote {result}")
