"""story.txt -> out.mp4.

Every stage writes into a run directory, so a rerun picks up where the last
one stopped instead of paying for narration again. The manifest is the record
of what a run decided; `--dry-run` writes it and stops.
"""

from __future__ import annotations

import json
import logging
import subprocess
from dataclasses import replace
from pathlib import Path

from narrator import align as align_module
from narrator import assemble as assemble_module
from narrator import captions as captions_module
from narrator import segment as segment_module
from narrator import speech, visuals
from narrator.beats import Beat, Word
from narrator.config import PipelineConfig

LOG = logging.getLogger(__name__)


class PipelineError(RuntimeError):
    """The build could not run."""


def plan(
    story_path: Path,
    out_path: Path,
    cfg: PipelineConfig | None = None,
    run_dir: Path | None = None,
) -> Path:
    """Work out what would be built and write the manifest. Renders nothing."""
    cfg = cfg or PipelineConfig()
    beats, run_dir = _prepare(story_path, run_dir, cfg)
    return _write_manifest(run_dir, story_path, out_path, beats, rendered=False)


def build(
    story_path: Path,
    out_path: Path,
    cfg: PipelineConfig | None = None,
    run_dir: Path | None = None,
    *,
    voice: str = "af_heart",
    music: Path | None = None,
    force: bool = False,
) -> Path:
    """Run the whole pipeline and return the finished video."""
    cfg = cfg or PipelineConfig()
    if voice not in speech.VOICES:
        # Checked here so a typo costs nothing: segmentation and alignment
        # would otherwise run before speech ever looked at the voice.
        raise PipelineError(
            f"unknown voice {voice!r}; try one of {', '.join(sorted(speech.VOICES)[:6])}, ..."
        )
    beats, run_dir = _prepare(story_path, run_dir, cfg)
    out_path = Path(out_path)

    audio_dir = run_dir / "audio"
    visuals_dir = run_dir / "visuals"

    beats = _narrate(beats, voice, audio_dir, cfg, force)
    beats = _align(beats, run_dir, cfg, force)
    beats = _visuals(beats, visuals_dir, cfg, force)

    subtitle_path = captions_module.build_ass(beats, cfg.captions(), run_dir / "captions.ass")
    _write_manifest(run_dir, story_path, out_path, beats, rendered=True)

    return assemble_module.assemble(
        beats,
        out_path,
        cfg.assemble(),
        captions=subtitle_path,
        music=music,
    )


def _prepare(
    story_path: Path, run_dir: Path | None, cfg: PipelineConfig
) -> tuple[list[Beat], Path]:
    story_path = Path(story_path)
    if not story_path.exists():
        raise PipelineError(f"story file does not exist: {story_path}")

    beats = segment_module.segment(story_path.read_text(encoding="utf-8"), cfg.max_chars)
    if not beats:
        raise PipelineError(f"{story_path} produced no beats; is it empty?")

    beats = visuals.add_queries(beats)
    run_dir = Path(run_dir or story_path.parent / f".narrator_{story_path.stem}")
    run_dir.mkdir(parents=True, exist_ok=True)
    return beats, run_dir


def _narrate(
    beats: list[Beat], voice: str, audio_dir: Path, cfg: PipelineConfig, force: bool
) -> list[Beat]:
    cached = _cached_audio(beats, voice, audio_dir, cfg)
    if cached is not None and not force:
        LOG.info("reusing narration from %s", audio_dir)
        return cached
    return speech.synthesize(beats, voice, audio_dir, cfg=cfg.speech)


def _cached_audio(
    beats: list[Beat], voice: str, audio_dir: Path, cfg: PipelineConfig
) -> list[Beat] | None:
    """Narration from a previous run, if all of it is still usable.

    Existence is not enough: a run killed mid-write leaves a truncated or
    empty wav, and reusing it shortens the narration without anything
    noticing. Every file is probed, and anything unusable sends the whole
    stage back to synthesis with a warning rather than being patched around.
    """
    if not audio_dir.exists():
        return None

    index = {path.stem: path for path in audio_dir.glob("*.wav")}
    restored: list[Beat] = []
    for beat in beats:
        key = speech._cache_path(audio_dir, beat.text, voice, cfg.speech).stem
        path = index.get(key) or index.get(f"beat_{beat.index:03d}")
        if path is None:
            return None

        problem = _unusable(path)
        if problem is not None:
            # Delete it as well as reporting it. speech.synthesize caches on
            # the same filename, so leaving a bad file in place would have it
            # handed straight back and the recovery would be imaginary.
            LOG.warning(
                "cached narration %s is unusable (%s); deleting it and re-synthesising",
                path.name,
                problem,
            )
            path.unlink(missing_ok=True)
            return None
        duration = speech.probe_duration(path)

        restored.append(replace(beat, audio_path=path, duration=duration))
    return restored


def _unusable(path: Path) -> str | None:
    """Why this cached wav cannot be trusted, or None if it can."""
    try:
        duration = speech.probe_duration(path)
    except speech.SynthesisError:
        return "ffprobe cannot read it"
    if duration < speech.MIN_BEAT_SECONDS:
        return f"only {duration:.3f}s long"
    return None


def _align(beats: list[Beat], run_dir: Path, cfg: PipelineConfig, force: bool) -> list[Beat]:
    cache = run_dir / "words.json"
    if cache.exists() and not force:
        restored = _load_words(beats, cache)
        if restored is not None:
            LOG.info("reusing word timings from %s", cache)
            return restored

    aligned = align_module.align(beats, cfg=cfg.align)
    cache.write_text(
        json.dumps(
            {
                str(beat.index): {
                    "text": beat.text,
                    "words": [{"text": w.text, "start": w.start, "end": w.end} for w in beat.words],
                }
                for beat in aligned
            },
            indent=2,
        )
    )
    return aligned


def _load_words(beats: list[Beat], cache: Path) -> list[Beat] | None:
    try:
        stored = json.loads(cache.read_text())
    except (OSError, json.JSONDecodeError) as exc:
        # Realigning is the right recovery, but a corrupt cache is worth
        # knowing about: it usually means a run was killed part way.
        LOG.warning("discarding unreadable %s (%s); realigning", cache.name, exc)
        return None

    restored: list[Beat] = []
    for beat in beats:
        entry = stored.get(str(beat.index))
        if entry is None or entry.get("text") != beat.text:
            return None
        restored.append(
            replace(
                beat,
                words=[
                    Word(text=w["text"], start=w["start"], end=w["end"]) for w in entry["words"]
                ],
            )
        )
    return restored


def _visuals(beats: list[Beat], visuals_dir: Path, cfg: PipelineConfig, force: bool) -> list[Beat]:
    """Reuse this run's visuals, but only the ones that still fit.

    A file called beat_003.mp4 from a run at another preset is the wrong
    size, the wrong frame rate, or too short for the narration it now has to
    cover. Reusing it produced a video that looked deliberately odd.
    """
    existing = {path.stem: path for path in visuals_dir.glob("beat_*.mp4")}
    wanted = [f"beat_{beat.index:03d}" for beat in beats]

    if not force and all(name in existing for name in wanted):
        stale = [
            name
            for beat, name in zip(beats, wanted, strict=True)
            if not _asset_fits(existing[name], beat, cfg)
        ]
        if not stale:
            LOG.info("reusing visuals from %s", visuals_dir)
            return [replace(beat, asset_path=existing[f"beat_{beat.index:03d}"]) for beat in beats]
        LOG.warning("cached visuals no longer match this run (%s); re-fetching", ", ".join(stale))
    return visuals.fetch_all(beats, cfg.visuals(), visuals_dir)


def _asset_fits(path: Path, beat: Beat, cfg: PipelineConfig) -> bool:
    try:
        probe = json.loads(
            subprocess.run(
                [
                    "ffprobe",
                    "-v",
                    "error",
                    "-print_format",
                    "json",
                    "-show_format",
                    "-show_streams",
                    str(path),
                ],
                capture_output=True,
                text=True,
                check=True,
            ).stdout
        )
    except (OSError, ValueError, subprocess.CalledProcessError):
        return False

    video = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if video is None:
        return False
    if (int(video["width"]), int(video["height"])) != (cfg.width, cfg.height):
        return False

    numerator, _, denominator = video["avg_frame_rate"].partition("/")
    fps = float(numerator) / float(denominator or 1)
    if abs(fps - cfg.fps) > 0.5:
        return False

    if beat.duration is not None:
        seconds = float(probe["format"]["duration"])
        if seconds < beat.duration - 0.05:
            return False
    return True


def _write_manifest(
    run_dir: Path, story_path: Path, out_path: Path, beats: list[Beat], rendered: bool
) -> Path:
    manifest = run_dir / "manifest.json"
    manifest.write_text(
        json.dumps(
            {
                "story": str(story_path),
                "out": str(out_path),
                "rendered": rendered,
                "beat_count": len(beats),
                "total_seconds": sum(beat.duration or 0.0 for beat in beats),
                "beats": [
                    {
                        "index": beat.index,
                        "text": beat.text,
                        "visual_query": beat.visual_query,
                        "duration": beat.duration,
                        "audio": str(beat.audio_path) if beat.audio_path else None,
                        "asset": str(beat.asset_path) if beat.asset_path else None,
                        "word_count": len(beat.words),
                    }
                    for beat in beats
                ],
            },
            indent=2,
        )
    )
    return manifest
