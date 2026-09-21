# Build Plan — autonomous mode

This file drives the remaining build. Each session: read CLAUDE.md, then this
file, find the first step below whose tag does not exist yet, and do that one
step only. Then stop.

## Rules for every step

**Proceed without asking** when all of these hold:

1. Tests for the step were written before the implementation.
2. Full fast suite green (`pytest`), slow suite green (`pytest -m slow`).
3. `ruff check` and `ruff format --check` clean.
4. Only the files listed for the step were created or changed, plus
   CLAUDE.md, narrator/config.py, pyproject.toml, uv.lock, and
   tests/fixtures/. Adding a per-stage config dataclass to narrator/config.py
   is not a deviation.
5. Self-review passed (below).

Then: commit with a message whose body explains *why* for any non-obvious
decision, tag the step, push `main` and tags, append the report to
`docs/BUILD_LOG.md`, and stop.

**Stop and ask instead** if any of these happen:

- Tests still fail after three genuine fix attempts.
- The step would change the Beat/Word contract or the public signature of a
  module that is already tagged.
- A dependency is needed that this file doesn't name.
- An environment variable or API key is missing (e.g. PEXELS_API_KEY).
- ffmpeg lacks a filter the step needs.
- Anything outside this repository would have to be touched.

When stopping, leave the work uncommitted and explain what is blocking.

### Self-review before tagging

- Could any test be passing for the wrong reason (fixture choice, a mock that
  bypasses the logic under test, an assertion that's trivially true)?
- Is there any silent fallback — a failure that degrades instead of raising?
- Does any cache key omit an input that changes the output?
- Did an earlier module's tests change? If so, why is that legitimate?
- Is anything learned about ffmpeg, torch or the environment recorded in the
  CLAUDE.md gotchas section?

### Report format (append to docs/BUILD_LOG.md)

```
## pN — <module> — <date>
Tests: X fast, Y slow, all passing
Files: <list>
Decisions: <each non-obvious call, one line with the reason>
Deviations from this plan: <or "none">
Known limitations: <or "none">
```

Keep it short. The reader should be able to decide in two minutes whether to
open the diff.

---

## Step 0 — close out P1/P2 (tag: `p2`)

1. Amend the P2 commit body with the benchmark rationale (why CPU is the
   default, the MPS fallback env var, the numbers).
2. If absent, add to segment.py: `St.`, `Mt.`, `vs.` to the abbreviation list;
   restate the terminal-punctuation test as "every beat except possibly the
   last"; add a test that `I thought… maybe.` does not split. Commit as
   `P1 follow-up`. Do not move the `p1` tag.
3. Confirm the speech cache key includes `speed` and `lang_code`. Add a test
   for each: changing it must produce a cache miss.
4. Add a slow test asserting the hardcoded voice list is a subset of the voices
   the installed Kokoro package ships.
5. Tag `p2` on the last commit. Create docs/BUILD_LOG.md with entries for p1
   and p2 reconstructed from the commit history.

## Step 3 — aligner (tag: `p3`)

Files: `narrator/align.py`, `tests/test_align.py`

Fill `Beat.words` from `Beat.audio_path` using faster-whisper with
`word_timestamps=True`.

- faster-whisper is CPU-only on Apple Silicon (CTranslate2 has no Metal
  backend). Use `compute_type="int8"`, model `base` by default, configurable.
  Do not attempt MPS.
- The reference text is known. If the aligned word count differs from the
  normalized reference, raise `AlignmentError` — never return bad timings.
- Mock whisper in the default suite with a recorded fixture output. One slow
  test runs it for real on a wav produced by `speech.py`.

Tests:
- aligned word count equals normalized reference word count
- timestamps strictly monotonic, no overlaps
- first start ≥ 0; last end ≤ audio duration + 0.05s
- numerals and punctuation ("3 a.m.", "$40", "don't") don't desync the count
- mismatch raises AlignmentError

## Step 3b — tolerant alignment (tag: `p3b`)

Files: `narrator/align.py`, `narrator/speech.py`, their tests.

a. Replace the strict word-count rule in align.py with sequence alignment:
   match normalized whisper tokens to normalized reference words
   (difflib.SequenceMatcher or equivalent), give each reference word its
   match's timing, interpolate timings for unmatched reference words between
   matched neighbours, and raise AlignmentError only when the match ratio
   falls below a configurable threshold (default 0.85). Beat.words must carry
   the reference text, not whisper's. The align() signature does not change.
b. Test: the "3am" vs "3 a.m." case aligns instead of raising; garbage audio
   (a mismatched transcript fixture) still raises; interpolated timings stay
   monotonic and inside the audio duration.
c. Investigate Kokoro determinism: seed torch and pin the thread count before
   each synthesis, then hash the output under both test orderings as you did
   before. If that makes it reproducible, keep it and add a slow test. If not,
   record the result and move on.

## Step 4 — visuals (tag: `p4`)

Files: `narrator/visuals/__init__.py`, `pexels.py`, `stills.py`,
`narrator/visuals/query.py`, `tests/test_visuals*.py`

- `query.py` fills `Beat.visual_query` from the beat text (simple keyword
  extraction, no model). Paragraph-initial beats are a hint to change scene.
- `pexels.py`: Pexels video API, key from `PEXELS_API_KEY`. If the variable is
  unset, stop and ask rather than skipping.
- `stills.py`: still image → video via ffmpeg `zoompan` (slow push-in;
  configurable fps, resolution, duration). Runs as a subprocess, never inside
  the Python process that holds the TTS model.
- One interface, `fetch(beat, cfg) -> Path`. Pexels first; any failure falls
  back to stills. Consecutive beats never reuse an asset.
- HTTP mocked in the default suite. No network in fast tests.

Tests:
- mocked Pexels response → correct asset chosen
- API error and zero results both fall back to stills without raising
- asset duration ≥ beat duration, or loop/pad covers the gap
- stills output: ffprobe reports configured fps, resolution, duration
- consecutive beats never share an asset

## Step 5 — captions (tag: `p5`)

Files: `narrator/captions.py`, `tests/test_captions.py`

Precondition: `ffmpeg -filters | grep -w ass` returns a line. If not, stop.

Build an `.ass` file from all beats' word timings. Karaoke style, one or two
words highlighted, centered lower third.

Tests (parse the file back, don't string-compare):
- dialogue line count as expected
- timings match input to the centisecond
- `{`, `}`, `\` and newlines escaped
- no rendered line exceeds wrap width

## Step 6 — assembler (tag: `p6`)

Files: `narrator/assemble.py`, `tests/test_assemble.py`

Concat per-beat visuals, burn the `.ass` subtitles, concat narration, mix a
music bed at −20 dB relative to narration. Build the ffmpeg command as a list,
log it, surface stderr on failure.

Fixtures: 1-second synthetic clips via lavfi `testsrc` and `sine` — keep the
suite fast.

Tests:
- exactly one video and one audio stream
- duration == sum of beat durations ± 0.2s
- resolution and fps match config
- music measurably quieter than narration (`volumedetect` on stems)
- narration starts within 50ms of t=0

## Step 7 — pipeline and CLI (tag: `p7`)

Files: `narrator/pipeline.py`, `narrator/cli.py`, `tests/test_pipeline.py`,
`tests/test_cli.py`, `tests/fixtures/tiny_story.txt`

```
narrator build story.txt --voice af_heart --out out.mp4 [--dry-run]
```

Each stage writes to a run directory; rerun resumes from the last completed
stage. `--dry-run` writes manifest.json and renders nothing.

Tests:
- golden: 3-sentence fixture → mp4 passing the p6 assertions
- `--dry-run` renders nothing
- delete out.mp4, rerun → zero TTS calls
- missing input → exit 1, readable message, no traceback

## Step 8 — verification (tag: `p8`)

No new modules. Run everything including slow tests, then the CLI end to end
on tiny_story.txt and on one real story of ~300 words.

Report in BUILD_LOG.md: output durations vs expected, and anything that looks
wrong in the actual video even with green tests — sync drift, caption timing,
repeated visuals, abrupt audio. Then list every place a failure could be
swallowed silently. List only; don't fix. That list is the input to the next
round of work.

---

# Round 2

Written after the p8 verification. The pipeline produces video end to end, but
the captions are misplaced, the console script does not work, and nine places
can swallow a failure. These steps close that gap. Normal rules apply.

## Step 9 — caption timing (tag: `p9`)

a. Offset each beat's words by the cumulative duration of the beats before it
   when building captions. Keep `Beat.words` beat-relative; the shift happens
   at caption-build time.
b. Trim leading and trailing silence in `speech.py` (amplitude threshold,
   configurable, with a short retained pad). Add the trim settings to the
   cache key, with a test that changing them causes a cache miss.
c. Caption groups never cross a sentence boundary.
d. New test that would have caught the offset bug: two beats with real
   (recorded-fixture) alignment, concatenated. Assert caption times are
   monotonic across the whole file, the last caption ends within 0.3s of the
   total duration, and each beat's first caption starts within 0.1s of that
   beat's speech onset.

## Step 10 — packaging (tag: `p10`)

Fix the console script. Add a test that runs the installed `narrator`
executable as a subprocess (`--help` and a `--dry-run`), not via `CliRunner`.

## Step 11 — silent failures (tag: `p11`)

Work through the nine silent-failure items in `docs/BUILD_LOG.md` in the
logged order. Each gets a test proving it now raises or warns.

## Step 12 — re-verify (tag: `p12`)

Re-run Step 8's verification (`tiny_story.txt` and the ~300-word story) with
`require_pexels=True` and `require_burned_captions=True`. Report the same
measurements as Step 8, plus caption offset from speech onset on three
sampled beats, plus anything that still looks wrong in the video.

Prerequisites, to be satisfied before this step runs:

- an ffmpeg built with libass, so `ffmpeg -filters | grep -w ass` prints a
  line;
- `PEXELS_API_KEY` set in the environment. Its value is never printed, logged
  or committed.
