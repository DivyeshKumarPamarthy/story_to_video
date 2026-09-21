"""story.txt -> out.mp4.

Every stage writes into a run directory, so a rerun picks up where the last
one stopped instead of paying for narration again. The manifest is the record
of what a run decided; `--dry-run` writes it and stops.
"""

from __future__ import annotations

import json
import logging
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
    cached = _cached_audio(beats, audio_dir)
    if cached is not None and not force:
        LOG.info("reusing narration from %s", audio_dir)
        return cached
    return speech.synthesize(beats, voice, audio_dir, cfg=cfg.speech)


def _cached_audio(beats: list[Beat], audio_dir: Path) -> list[Beat] | None:
    """Narration from a previous run, if it is all still there.

    Keyed by the beat text, so editing the story invalidates only what
    changed -- and the speech cache is keyed the same way underneath.
    """
    if not audio_dir.exists():
        return None

    index = {path.stem: path for path in audio_dir.glob("*.wav")}
    restored: list[Beat] = []
    for beat in beats:
        key = speech._cache_path(audio_dir, beat.text, "", speech.SpeechConfig()).stem
        path = index.get(key) or index.get(f"beat_{beat.index:03d}")
        if path is None:
            return None
        restored.append(replace(beat, audio_path=path, duration=speech.probe_duration(path)))
    return restored


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
    except (OSError, json.JSONDecodeError):
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
    existing = {path.stem: path for path in visuals_dir.glob("beat_*.mp4")}
    wanted = [f"beat_{beat.index:03d}" for beat in beats]

    if not force and all(name in existing for name in wanted):
        LOG.info("reusing visuals from %s", visuals_dir)
        return [replace(beat, asset_path=existing[f"beat_{beat.index:03d}"]) for beat in beats]
    return visuals.fetch_all(beats, cfg.visuals(), visuals_dir)


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
