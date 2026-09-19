# narrator

Turns a text story into a narrated video: TTS narration, stock or still-image
visuals with motion, karaoke captions, music bed. Local and free — no paid APIs
in the critical path.

## The data contract

`narrator/beats.py` defines the only shared types. Treat it as frozen; changing
it means changing every module.

```python
@dataclass
class Word:
    text: str
    start: float
    end: float


@dataclass
class Beat:
    index: int
    text: str
    visual_query: str  # keyword for stock lookup
    audio_path: Path | None = None
    duration: float | None = None
    words: list[Word] = field(default_factory=list)
    asset_path: Path | None = None
```

**The rule: every pipeline stage takes `list[Beat]` and returns `list[Beat]`
with more fields filled in.** No stage reaches back into an earlier stage, and
no stage depends on an earlier stage having *run* — only on the fields it needs
being populated. That is what makes each one testable from a hand-built fixture.

Which stage fills what:

| Stage | Fills |
|---|---|
| `segment.py` | `index`, `text` |
| `visuals/` (query step) | `visual_query` |
| `speech.py` | `audio_path`, `duration` |
| `align.py` | `words` |
| `visuals/` (fetch step) | `asset_path` |
| `captions.py`, `assemble.py` | nothing — they consume |

## Testing media

**Assert on `ffprobe` metadata, never on bytes or waveform equality.** Encoders
are not deterministic across versions, machines or thread counts; byte
comparison of media produces tests that pass on the machine that wrote them and
nowhere else.

Assert instead on: stream count, codec, duration (with a tolerance), sample
rate, resolution, fps, and measured loudness via `volumedetect`. Durations get a
stated tolerance — never `==` on a float.

The default suite must not hit the network or run model inference. Mock the
Kokoro, whisper and HTTP calls; put anything real behind
`@pytest.mark.slow`, which is excluded by default (`pytest -m slow` to run it).

ffmpeg fixtures come from `lavfi` (`testsrc`, `sine`) at ~1 second, so the suite
stays fast.

## Environment

- macOS, Apple Silicon (M4, 16 GB). No CUDA — torch runs on CPU or MPS.
- Python 3.12 via `uv`. (The plan said 3.11; 3.12 is what has wheels here and
  the system Python 3.14 is ahead of torch support.)
- `ffmpeg` and `ffprobe` from Homebrew, on PATH.
- Run things with `uv run pytest`, `uv run narrator ...`.

## Output formats

Both aspect ratios are supported, driven by config — never hardcode a
resolution, fps or caption position in a module.

| Preset | Resolution | fps | Captions |
|---|---|---|---|
| `landscape` | 1920x1080 | 30 | lower third, 1-2 lines |
| `vertical` | 1080x1920 | 30 | centered, large karaoke words |

Consequences for later modules: `visuals/` must crop/pad stock footage to the
target aspect (stock is mostly landscape, so vertical needs a centre crop or a
blurred-pad), `zoompan` needs the target resolution passed in, and `captions.py`
takes wrap width and vertical position from the preset.

## ffmpeg gotchas

Filter-graph syntax is the biggest time sink in this project. Every quirk solved
gets written down here immediately, with the failing command and the fix, so it
is not rediscovered three modules later.

_(empty — fill as we hit them)_
