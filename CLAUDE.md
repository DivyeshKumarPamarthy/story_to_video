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
- TTS deps are an extra: `uv sync --extra tts` (pulls torch, ~2 GB). The first
  slow run also downloads Kokoro-82M and a spacy model from the network and
  caches them under `~/.cache/huggingface`. The default suite needs none of it.

## Workflow

Work directly on `main`. No per-module branches.

- Commit as often as useful while a module is in progress — small, green-ish
  steps, message says what changed.
- When a module is implemented, its tests pass, and it has been reviewed, tag
  it: `p1` for `segment.py`, `p2` for `speech.py`, and so on through `p8`.
- The tag is the checkpoint. If a later module breaks something, `git diff p3..`
  is the question to ask.
- One module per session. Start a fresh session for each prompt (P1, P2, ...)
  so context from finished modules does not leak into the next one — that is
  what causes stray edits to files that were already done.

| Tag | Module |
|---|---|
| `p0` | scaffold, Beat contract, tooling |
| `p1` | `segment.py` |
| `p2` | `speech.py` |
| `p3` | `align.py` |
| `p4` | `visuals/` |
| `p5` | `captions.py` |
| `p6` | `assemble.py` |
| `p7` | `pipeline.py` + `cli.py` |
| `p8` | end-to-end verification pass |

## Config and devices

Config lives in `narrator/config.py` as frozen dataclasses, one per stage.
**No module hardcodes a device, a sample rate, a resolution or an fps** — it
takes a config object and a caller may hand it a variant via
`dataclasses.replace`.

`SpeechConfig` fields: `device`, `sample_rate` (24000, Kokoro's native rate),
`lang_code`, `speed`, `model_version`.

**`device` defaults to `"cpu"`.** Supported: `cpu`, `mps`, `cuda`. An
unsupported name raises `ValueError` at construction; a supported one that the
machine cannot provide raises `SynthesisError` at model-load time. Neither
falls back silently — a silent fallback to CPU is how a "GPU" benchmark quietly
measures nothing.

Measured on this machine (M4, 16 GB), Kokoro-82M, best of 3 after warm-up:

| device | model load | synth, best of 3 | vs real time |
|---|---|---|---|
| `cpu` | 1.92s | 0.506s | 9.6x |
| `mps` | 2.38s | 0.385s | 12.6x |

84-character sentence. MPS is 1.3-1.4x faster on synthesis across runs, and
pays about 0.5s more to load. On a 60-second story (~15-20 beats) that is
roughly 2 seconds saved on a 10-second job, against a CPU that already runs
~10x real time. **The default stays `cpu`**; pass `SpeechConfig(device="mps")`
when batch-rendering something long enough for it to matter.

Reproduce with `uv run pytest -m slow -k benchmark -s`.

`PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `tests/conftest.py` at import time,
before anything can import torch. MPS does not implement every operator; without
it an unimplemented op aborts the process rather than falling back to CPU, and
it has to be set before the first MPS tensor exists, which is why it is not a
fixture.

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
