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
- `ffmpeg` and `ffprobe` from the `homebrew-ffmpeg/ffmpeg` tap (the stock
  formula lacks libass), on PATH.
- **To run the CLI from this venv, sync with `uv sync --no-editable`.** uv's
  editable install writes `_editable_impl_narrator.pth` into site-packages and
  this interpreter does not honour it, so the `narrator` console script raises
  `ModuleNotFoundError` while every in-process test passes. An identical file
  under a different name *is* honoured, which makes no sense and was not worth
  chasing further. A plain `uv sync` — including the one `uv run` performs
  before every command — reverts to editable and breaks the script again.
  `tests/test_packaging.py` deliberately does not depend on this: it installs
  the project into a throwaway venv and runs the executable there, which is
  what a user actually gets.
- Run tests with `uv run pytest`. Run the CLI as `narrator ...` from the venv.
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

## Gotchas

Every quirk solved gets written down here immediately, with the failing command
and the fix, so it is not rediscovered three modules later.

### ffmpeg

- **`-loop 1` on a file ffmpeg cannot decode hangs forever.** It does not
  error; it sits there. Found by a test that deliberately fed it a text file
  named `.png`, which wedged the suite until it was killed. Every ffmpeg call
  in this project therefore passes `-nostdin` and runs under a subprocess
  timeout (`VisualsConfig.ffmpeg_timeout`), and a timeout is reported as a
  failure with the command in the message.
- **Upscale before `zoompan` or the push-in judders.** zoompan samples from
  its input, so feeding it 2x the output size gives sub-pixel steps instead of
  whole-pixel jumps. The cost is a scale filter; the benefit is the difference
  between "slow push-in" and "stuttering".
- **`zoompan` counts in frames, not seconds.** `d` is the number of output
  frames per input frame, so it is `seconds * fps`, and the zoom step has to
  be divided by that same frame count to land on the target zoom at the end.
- **`amix` divides by the number of inputs unless you say otherwise.**
  Mixing narration with a music bed at the default settings quietly halves the
  narration. `amix=inputs=2:duration=first:normalize=0` is the form that
  leaves the first input alone; `duration=first` also stops a long music track
  outlasting the story.
- **`concat` does not clip anything to length.** A clip longer than its beat
  stretches the video past its narration and everything after it slips; a
  shorter one lets the audio run on over the next beat's picture. Each branch
  gets `trim=duration=...,setpts=PTS-STARTPTS`, and `assemble` refuses a
  visual shorter than its narration outright.
- **`concat` demands identical geometry**, so every clip is put through
  `scale`/`crop`/`setsar=1`/`fps` before it reaches the concat filter. One
  mismatched asset would otherwise fail the entire render.
- **Filter arguments treat `:` and `\` as syntax**, so a file path handed to
  a filter (the `ass` filter, for one) has to be escaped or a path with a
  colon in it silently becomes two arguments.
- **The stock Homebrew ffmpeg has no libass.** Its formula does not even
  declare it, so reinstalling does not help. `brew uninstall ffmpeg` then
  `brew install homebrew-ffmpeg/ffmpeg/ffmpeg` gives a build with
  `--enable-libass`, which is what this machine now runs: `ffmpeg -filters |
  grep -w ass` prints a line and `assemble.py` burns captions rather than
  muxing a soft track. Verified by rendering text onto black and comparing
  frames, not by trusting the filter list.
- **`assemble.py` still has both paths.** `has_ass_filter()` decides at
  runtime, so the soft-subtitle fallback is no longer exercised here and its
  test skips. Anyone on the stock ffmpeg gets soft subtitles and a warning.

### torch and environment

- **An MPS tensor cannot go straight to numpy.** `np.asarray(tensor)` raises
  for a tensor on `mps`; it needs `.detach().cpu().numpy()` first. `_to_numpy`
  in `speech.py` does this by duck-typing on `detach`, so the module still does
  not import torch.
- **`pytest -q` stacks.** `addopts` already contains `-q`, so passing `-q` on
  the command line makes it `-qq`, which silently suppresses the
  "N passed" summary line. Run `uv run pytest -m slow -p no:warnings` with no
  extra `-q` when you want the count.
- **`KPipeline(lang_code="a", model=False)`** reports `.repo_id` without
  downloading any weights — useful for checking metadata in a test without
  paying for a model load.
- **Kokoro's output is not reproducible across process state.** The same text,
  voice and config produce a wav with a different sha256 depending on what ran
  earlier in the process (confirmed by hashing: a clean process and one that
  had already reloaded the pipeline and touched MPS give different bytes).
  Seeding torch and pinning it to one thread does **not** fix this — the two
  orderings still diverge — though it does change the output, so there is RNG
  in the pipeline somewhere. Two identical fresh processes do agree. Measured,
  not assumed; see the p3b log entry for the hashes.
  Consequences: never assert on exact audio or on an exact transcription of it
  in a slow test, and note that the synthesis cache stores whichever rendering
  was generated first. Deterministic assertions belong in the fast suite,
  against recorded fixtures.
- **Kokoro pads ~0.3s of silence before it speaks**, and a little after.
  Whisper then reports the first word at 0.000 regardless, so untrimmed audio
  puts every caption in the beat early and adds a pause at every beat
  boundary. `speech.py` trims to an amplitude threshold with a short retained
  pad, and the trim settings are part of the cache key.
- **`Beat.words` are timed against that beat's own audio, not the video.**
  Anything writing into whole-video time has to add the durations of the
  beats before it. This is the bug p8 found: with one beat the offset is
  zero, so it hides in every single-beat fixture.
- **faster-whisper cannot use MPS.** It runs through CTranslate2, which has no
  Metal backend, so Apple Silicon means `device="cpu"` with
  `compute_type="int8"`. This is not a performance choice.
- **Whisper marks an intra-word split by omitting the leading space.** Every
  word it emits normally starts with a space; `a.m.` comes back as `" a"` then
  `".m."`. Word counts must merge on that signal or every abbreviation
  desyncs the beat. See `_merge_continuations` in `align.py`.
- **`uv sync` prunes kokoro's runtime-installed spacy model.** `en_core_web_sm`
  is fetched by misaki on first real synthesis, and any later `uv sync` removes
  it as an unmanaged package; the next real run re-downloads it.
