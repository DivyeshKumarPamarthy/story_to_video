"""Pexels video search. The HTTP layer is two functions, both mocked in tests."""

from __future__ import annotations

import json
import os
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any

from narrator.beats import Beat
from narrator.config import VisualsConfig

SEARCH_URL = "https://api.pexels.com/videos/search"


class PexelsError(RuntimeError):
    """Pexels could not give us a usable clip."""


class MissingApiKey(PexelsError):
    """No API key configured.

    A subclass so callers can fall back on ordinary API failures while still
    treating an absent key as something the operator should know about.
    """


def api_key(cfg: VisualsConfig) -> str:
    value = (os.environ.get(cfg.api_key_env) or "").strip()
    if not value:
        raise MissingApiKey(
            f"{cfg.api_key_env} is not set. Export a Pexels API key, or run with "
            f"stills only (VisualsConfig.require_pexels=False)."
        )
    return value


def fetch_clip(
    beat: Beat, cfg: VisualsConfig, dest_dir: Path, exclude: tuple[str, ...] = ()
) -> tuple[Path, str]:
    """Search, choose and download. Returns the file and the Pexels video id."""
    key = api_key(cfg)
    needed = beat.duration or cfg.default_seconds
    payload = _search(beat.visual_query, cfg, key)
    chosen = choose(payload, cfg, needed_seconds=needed, exclude=exclude)
    chosen_file = best_file(chosen, cfg)

    dest_dir = Path(dest_dir)
    dest = dest_dir / f"pexels_{chosen['id']}.mp4"
    try:
        _download(chosen_file["link"], dest, cfg.request_timeout)
    except OSError as exc:
        raise PexelsError(f"downloading {chosen_file['link']} failed: {exc}") from exc
    return dest, str(chosen["id"])


def _search(query: str, cfg: VisualsConfig, key: str) -> dict[str, Any]:
    orientation = "portrait" if cfg.height > cfg.width else "landscape"
    url = f"{SEARCH_URL}?" + urllib.parse.urlencode(
        {"query": query, "per_page": 15, "orientation": orientation}
    )
    try:
        return _get_json(url, {"Authorization": key}, cfg.request_timeout)
    except OSError as exc:
        raise PexelsError(f"pexels search for {query!r} failed: {exc}") from exc


def choose(
    payload: dict[str, Any],
    cfg: VisualsConfig,
    needed_seconds: float,
    exclude: tuple[str, ...] = (),
) -> dict[str, Any]:
    """Pick a clip: long enough if possible, never one the last beat used."""
    videos = payload.get("videos")
    if not isinstance(videos, list):
        raise PexelsError(f"unexpected pexels payload: {sorted(payload)[:5]}")

    usable = [
        video for video in videos if str(video.get("id")) not in exclude and _files(video, cfg)
    ]
    if not usable:
        raise PexelsError("no results left to choose from (zero results, or all excluded)")

    long_enough = [v for v in usable if float(v.get("duration", 0)) >= needed_seconds]
    if long_enough:
        return long_enough[0]
    # Nothing is long enough; the longest needs the least looping.
    return max(usable, key=lambda v: float(v.get("duration", 0)))


def best_file(video: dict[str, Any], cfg: VisualsConfig) -> dict[str, Any]:
    """The rendition closest to the target, preferring one large enough."""
    files = _files(video, cfg)
    if not files:
        raise PexelsError(f"pexels video {video.get('id')} has no usable files")

    big_enough = [f for f in files if f["width"] >= cfg.width and f["height"] >= cfg.height]
    pool = big_enough or files
    return min(pool, key=lambda f: abs(f["width"] - cfg.width) + abs(f["height"] - cfg.height))


def _files(video: dict[str, Any], cfg: VisualsConfig) -> list[dict[str, Any]]:
    return [
        f
        for f in video.get("video_files", [])
        if f.get("link") and f.get("width") and f.get("height")
    ]


def _get_json(url: str, headers: dict[str, str], timeout: float) -> dict[str, Any]:
    request = urllib.request.Request(url, headers=headers)
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


def _download(url: str, dest: Path, timeout: float) -> Path:
    dest.parent.mkdir(parents=True, exist_ok=True)
    partial = dest.with_suffix(dest.suffix + ".part")
    with urllib.request.urlopen(url, timeout=timeout) as response:  # noqa: S310
        partial.write_bytes(response.read())
    partial.replace(dest)
    return dest
