"""Tests for narrator.visuals.pexels.

No network: `_get_json` and `_download` are the seams and both are mocked.
"""

from __future__ import annotations

import pytest

from narrator.beats import Beat
from narrator.config import VisualsConfig
from narrator.visuals import pexels
from narrator.visuals.pexels import MissingApiKey, PexelsError

CFG = VisualsConfig(width=1920, height=1080, fps=30)
KEY = "test-key-not-a-real-one"


def beat(query: str = "empty house", duration: float | None = 4.0) -> Beat:
    return Beat(index=0, text="The house stood empty.", visual_query=query, duration=duration)


def video(video_id: int, seconds: int, sizes: list[tuple[int, int]]) -> dict:
    return {
        "id": video_id,
        "duration": seconds,
        "width": sizes[0][0],
        "height": sizes[0][1],
        "video_files": [
            {
                "id": video_id * 100 + i,
                "quality": "hd",
                "file_type": "video/mp4",
                "width": w,
                "height": h,
                "link": f"https://videos.example/{video_id}_{w}x{h}.mp4",
            }
            for i, (w, h) in enumerate(sizes)
        ],
    }


PAYLOAD = {
    "total_results": 2,
    "videos": [
        video(101, 10, [(1920, 1080), (640, 360)]),
        video(202, 3, [(3840, 2160), (1280, 720)]),
    ],
}


# --- the key ----------------------------------------------------------------


def test_missing_api_key_raises_a_named_error(monkeypatch):
    monkeypatch.delenv("PEXELS_API_KEY", raising=False)
    with pytest.raises(MissingApiKey, match="PEXELS_API_KEY"):
        pexels.api_key(CFG)


def test_blank_api_key_is_treated_as_missing(monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", "   ")
    with pytest.raises(MissingApiKey):
        pexels.api_key(CFG)


def test_api_key_is_read_from_the_configured_variable(monkeypatch):
    monkeypatch.setenv("SOMEWHERE_ELSE", KEY)
    assert pexels.api_key(VisualsConfig(api_key_env="SOMEWHERE_ELSE")) == KEY


def test_missing_key_is_not_a_pexels_error_subclass_accident(monkeypatch):
    # MissingApiKey must be distinguishable so callers can choose to hard-fail
    # on it while still falling back on ordinary API failures.
    assert issubclass(MissingApiKey, PexelsError)


# --- choosing ---------------------------------------------------------------


def test_chooses_a_clip_long_enough_for_the_beat():
    chosen = pexels.choose(PAYLOAD, CFG, needed_seconds=5.0, exclude=())
    assert chosen["id"] == 101, "picked a clip shorter than the beat"


def test_falls_back_to_the_longest_clip_when_none_are_long_enough():
    chosen = pexels.choose(PAYLOAD, CFG, needed_seconds=60.0, exclude=())
    assert chosen["id"] == 101


def test_excluded_ids_are_not_chosen():
    chosen = pexels.choose(PAYLOAD, CFG, needed_seconds=2.0, exclude=("101",))
    assert chosen["id"] == 202


def test_zero_results_raises():
    with pytest.raises(PexelsError, match="no results"):
        pexels.choose({"videos": []}, CFG, needed_seconds=2.0, exclude=())


def test_everything_excluded_raises():
    with pytest.raises(PexelsError, match="no results"):
        pexels.choose(PAYLOAD, CFG, needed_seconds=2.0, exclude=("101", "202"))


def test_picks_the_file_closest_to_the_target_resolution():
    chosen = pexels.choose(PAYLOAD, CFG, needed_seconds=5.0, exclude=())
    link = pexels.best_file(chosen, CFG)["link"]
    assert "1920x1080" in link


def test_prefers_a_file_at_least_as_large_as_the_target():
    cfg = VisualsConfig(width=1280, height=720, fps=30)
    chosen = pexels.choose(PAYLOAD, cfg, needed_seconds=2.0, exclude=("101",))
    assert pexels.best_file(chosen, cfg)["width"] >= 1280


def test_a_video_with_no_usable_files_is_skipped():
    payload = {"videos": [{"id": 1, "duration": 9, "video_files": []}, video(7, 9, [(1920, 1080)])]}
    assert pexels.choose(payload, CFG, needed_seconds=2.0, exclude=())["id"] == 7


# --- fetching ---------------------------------------------------------------


def test_fetch_clip_downloads_the_chosen_file(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", KEY)
    downloaded = {}

    def fake_download(url, dest, timeout):
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"fake video bytes")
        downloaded["url"] = url
        return dest

    monkeypatch.setattr(pexels, "_get_json", lambda *a, **k: PAYLOAD)
    monkeypatch.setattr(pexels, "_download", fake_download)

    path, asset_id = pexels.fetch_clip(beat(), CFG, tmp_path, exclude=())

    assert path.exists()
    assert asset_id == "101"
    assert "1920x1080" in downloaded["url"]


def test_fetch_clip_sends_the_key_as_authorization(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", KEY)
    seen = {}

    def fake_get_json(url, headers, timeout):
        seen["headers"] = headers
        seen["url"] = url
        return PAYLOAD

    monkeypatch.setattr(pexels, "_get_json", fake_get_json)
    monkeypatch.setattr(
        pexels, "_download", lambda url, dest, timeout: dest.write_bytes(b"x") or dest
    )

    pexels.fetch_clip(beat(), CFG, tmp_path, exclude=())

    assert seen["headers"]["Authorization"] == KEY
    assert "empty%20house" in seen["url"] or "empty+house" in seen["url"]


def test_http_failure_raises_pexels_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", KEY)

    def boom(*args, **kwargs):
        raise OSError("connection reset")

    monkeypatch.setattr(pexels, "_get_json", boom)

    with pytest.raises(PexelsError, match="connection reset"):
        pexels.fetch_clip(beat(), CFG, tmp_path, exclude=())


def test_malformed_payload_raises_pexels_error(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", KEY)
    monkeypatch.setattr(pexels, "_get_json", lambda *a, **k: {"unexpected": True})

    with pytest.raises(PexelsError):
        pexels.fetch_clip(beat(), CFG, tmp_path, exclude=())


def test_no_network_in_the_default_suite(tmp_path, monkeypatch):
    monkeypatch.setenv("PEXELS_API_KEY", KEY)
    monkeypatch.setattr(pexels, "_get_json", lambda *a, **k: PAYLOAD)
    monkeypatch.setattr(
        pexels, "_download", lambda url, dest, timeout: dest.write_bytes(b"x") or dest
    )

    import socket

    def no_sockets(*args, **kwargs):
        raise AssertionError("the default suite must not open a socket")

    monkeypatch.setattr(socket, "socket", no_sockets)
    pexels.fetch_clip(beat(), CFG, tmp_path, exclude=())
