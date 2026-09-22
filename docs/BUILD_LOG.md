# Build log

One entry per tagged step. Reconstructed entries for p1 and p2 are marked as
such — they were written after the fact from the commit history.

## p1 — segment.py — 2026-09-19 (reconstructed)

Tests: 29 fast, 0 slow, all passing (26 at the tag; 3 added by the follow-up)
Files: `narrator/segment.py`, `tests/test_segment.py`

Decisions:
- Sentence boundaries are vetoed by two rules rather than detected by one: a
  lowercase following character means continuation, which covers dialogue tags
  (`"...?" she called`) and `e.g. apples` with the same line; then, for a lone
  `.` only, a single-letter token (initials, `U.S.`) or a known abbreviation.
- The abbreviation list deliberately omits `no`, `am`, `sat`, `may` and the
  weekdays. A missed split only makes a beat longer; a false split cuts a
  sentence in half mid-narration, so words that are also ordinary words stay
  out.
- A sentence longer than `max_chars` is emitted whole rather than cut or
  raised on — a beat that stops mid-sentence sounds broken once narrated.
- Paragraph breaks are hard beat boundaries. Not requested; keeps narration
  pacing aligned with the prose.
- `max_chars=1` in tests forces one sentence per beat, so the beat list is the
  sentence list and packing cannot obscure a boundary bug.

Deviations from this plan: none.

Known limitations:
- `She left the U.S. He stayed.` does not split — the single-letter rule cannot
  distinguish a dotted acronym from a sentence ending in one. Fails safe (a
  longer beat, never a severed sentence).
- `test_a_beat_over_max_chars_is_always_a_single_sentence` uses `segment()` to
  check `segment()`'s own output. It verifies an invariant, not ground truth.

## p2 — speech.py — 2026-09-21 (reconstructed)

Tests: 39 fast, 5 slow, all passing
Files: `narrator/speech.py`, `narrator/config.py`, `tests/test_speech.py`,
`tests/test_speech_slow.py`, `tests/conftest.py`, `tests/ffprobe.py`,
`pyproject.toml`, `uv.lock`, `CLAUDE.md`

Decisions:
- Torch device is a `SpeechConfig` field defaulting to `cpu`, never a hardcoded
  string. Benchmarked rather than guessed (M4, 84-char sentence, best of 3):
  cpu 0.506s / 9.6x real time, mps 0.385s / 12.6x, so mps is 1.3-1.4x faster on
  synthesis and ~0.5s slower to load. On a 60-second story that is ~2s saved on
  a ~10s job — not worth making the default.
- An unsupported device name raises `ValueError` at construction; a supported
  one the machine lacks raises `SynthesisError` at load. No silent fallback to
  cpu — that is how a "GPU" benchmark ends up measuring nothing.
- Nothing enters the cache unvalidated: synthesis writes `<key>.wav.part`,
  which is probed and checked against a 4-40 chars/sec plausibility band before
  being renamed into place. A silent model cannot leave a bad file to be reused
  forever.
- The cache key covers text, voice, model_version, lang_code, speed and
  sample_rate, and deliberately excludes device: the same model saying the same
  sentence is the same narration wherever it ran.
- The plausibility band is far wider than real narration (10-22 chars/sec). It
  is not a style check — it catches the "0.1s of nothing" failure only.
- Voice list is hardcoded so an unknown id fails loudly instead of narrating a
  whole story in a default voice. A slow test checks it against the voices the
  installed package's repo actually publishes.
- `PYTORCH_ENABLE_MPS_FALLBACK=1` is set in `tests/conftest.py` at import time,
  not in a fixture — it must precede the first MPS tensor.
- The wav writer is stdlib `wave`; numpy is a base dependency so the default
  suite needs neither torch nor soundfile.

Deviations from this plan: none.

Known limitations:
- ~~`_synthesize_audio` skips result chunks whose `audio` is `None` and only
  raises when every chunk is empty.~~ **Resolved** after tagging: a chunk with
  graphemes or phonemes that returns no audio now raises `SynthesisError`. A
  chunk with neither is padding and is still skipped. The `p2` tag was left in
  place; see `P2 fix: raise on partial chunk drop`.
- The fast duration-band test measures a synthetic fixture generated at 15
  chars/sec, so it exercises the wav-writing and ffprobe path rather than the
  model's real pacing. The slow `test_real_synthesis_produces_narration` is
  what actually checks that.
- `p1`'s terminal-punctuation test was changed during this step: it asserted
  every beat ends on `.`/`!`/`?`, which is only true when the source text does.
  Now every beat except possibly the last. The `p1` tag was left in place.


## p3 — align.py — 2026-09-21

Tests: 33 fast, 3 slow, all passing (104 fast / 8 slow across the project)
Files: `narrator/align.py`, `tests/test_align.py`,
`tests/fixtures/whisper_words.json`, `narrator/config.py`, `CLAUDE.md`

Decisions:
- The transcription is treated as a claim to check, not a source of truth: the
  audio was generated from text we still have. A word-count mismatch against
  the normalised reference raises `AlignmentError` rather than returning
  timings that would drift captions off the narration.
- Whisper splits some words and signals the continuation by omitting the
  leading space (`" a"` then `".m."` for "a.m."). Merging on that signal is
  what keeps counts equal; without it every abbreviation desyncs a beat. This
  was found by recording real output, not predicted.
- The fixture is recorded from the real model on real Kokoro audio rather than
  hand-written, so the fast suite tests behaviour that actually occurred.
- Overlapping or out-of-order timings raise instead of being clamped. Real
  output had no overlaps (adjacent words touch exactly), so a clamp would be
  silently repairing something that should not happen.
- `AlignConfig` defaults to `base`/`int8`/`cpu`. The model only has to be good
  enough to carry timings for text we already know; accuracy beyond that is
  rejected by the count check anyway.
- `mps` raises a `ValueError` naming CTranslate2 rather than being silently
  mapped to cpu — faster-whisper has no Metal backend at all.
- Audio duration is probed from the file rather than read from `Beat.duration`,
  so the bounds check is against the ground truth and not a field that could
  be stale.

Deviations from this plan:
- `AlignConfig` was added to `narrator/config.py`, which is not in this step's
  file list. CLAUDE.md states config lives there, one frozen dataclass per
  stage; defining it inside `align.py` would have fragmented config across
  modules for every later step. Additive only — no existing signature changed.

Known limitations:
- **Kokoro's synthesis is not reproducible across process state.** The same
  sentence produces a different wav (verified by sha256) depending on what ran
  earlier in the process, and whisper then hears "3 a.m." as either three
  tokens or one. Alignment of such text is therefore correct-or-loud, not
  always-correct: a story containing "3 a.m." may raise `AlignmentError` on
  one run and align cleanly on the next. The slow test asserts both branches;
  the deterministic behaviour is pinned in the fast suite by the fixture.
- `_merge_continuations` drops whitespace-only tokens silently. A real dropped
  word would still be caught by the count check.
- No caching. Alignment re-runs on every pipeline run; resuming is p7's job.


## p3b — align.py (tolerant) — 2026-09-21

Tests: 46 fast, 3 slow, all passing (116 fast / 8 slow across the project)
Files: `narrator/align.py`, `narrator/config.py`, `tests/test_align.py`,
`CLAUDE.md`

Decisions:
- The strict word-count rule is gone. Reference words and whisper tokens are
  matched with `difflib.SequenceMatcher` on normalised keys (letters and
  digits only, so "$40," lines up with "$40"), and only a match ratio below
  `AlignConfig.min_match_ratio` (0.85) raises. A model that writes "3am" for
  "3 a.m." no longer fails the beat it appears in.
- `Beat.words` carries the reference text, never whisper's. Captions must read
  as the story was written; a test asserts "3am" cannot leak through.
- Unmatched reference words are interpolated across the gap between their
  matched neighbours, divided evenly. The gap is wider than the token that
  replaced them — the difference is silence, and a caption there is harmless.
- When whisper drops a word whose neighbours touch, there is no gap to divide.
  Time is borrowed from a neighbour (never below 10ms) rather than emitting a
  zero-length word, because a caption with no duration never appears. If even
  borrowing cannot fit it, that raises rather than producing junk.
- Timings that come straight from whisper are still validated strictly: a
  backwards or overlapping matched timestamp raises, unchanged from p3.

Deviations from this plan: none. `narrator/speech.py` is listed for this step
but needed no change — see below.

Determinism investigation (step 3b c), all hashes of the same sentence:

| run | hash |
|---|---|
| unseeded, clean process | `e4557c14d75115d0` |
| unseeded, after pipeline reload + MPS | `3129d45dda12d2a0` |
| seeded + 1 thread, clean process | `fd54ffc6619ae8f7` |
| seeded + 1 thread, after reload + MPS | `2d691eb6cbe379b1` |
| seeded + 1 thread, clean process again | `fd54ffc6619ae8f7` |

`torch.manual_seed(0)` and `torch.set_num_threads(1)` do **not** make output
reproducible across process state — the two orderings still diverge. They do
change the bytes, so there is RNG in the pipeline, and two identical fresh
processes agree. Since seeding buys no reproducibility where it was needed,
it was not kept in `speech.py`: it would change every cached rendering and
invalidate existing caches for nothing. Recorded and moved on, per the plan.

Known limitations:
- Tolerance cuts both ways: two words swapped in the transcript are now
  absorbed (one matches, the other interpolates) instead of raising. Within
  the ratio threshold, word-order errors are no longer detected.
- Interpolated timings are evenly divided guesses, not measurements. For a
  long unmatched run they will drift against the audio.
- `_match` silently drops transcribed tokens that normalise to nothing
  (punctuation-only). They carry no text, so nothing is lost from captions.


## p4 — visuals/ — 2026-09-21

Tests: 59 fast, 0 slow, all passing (175 fast / 8 slow across the project)
Files: `narrator/visuals/__init__.py`, `query.py`, `pexels.py`, `stills.py`,
`tests/test_visuals_{query,stills,pexels,fetch}.py`, `narrator/config.py`,
`tests/ffprobe.py`, `CLAUDE.md`

Decisions:
- One interface, `fetch(beat, cfg, out_dir)`, and whatever the source the
  result is conformed to the configured resolution, fps and the beat's own
  duration. The assembler never has to know where a clip came from.
- `-stream_loop -1` plus `-t` covers both halves of "asset duration >= beat
  duration": a short clip loops, a long one is trimmed, in one command.
- Stock audio is stripped with `-an`. Narration is the only sound, and a
  stock clip's own track would otherwise appear under it.
- Generated backgrounds are procedural gradients via lavfi, so the fallback
  needs neither network nor bundled art. The variant is a hash of the beat's
  query, so a rerun looks identical, but it steps past whatever the previous
  beat used.
- `MissingApiKey` subclasses `PexelsError` so a caller can fall back on
  ordinary API failures while still treating an absent key as different.
- Every ffmpeg call passes `-nostdin` and runs under a timeout. See the
  limitation below: this was not defensive programming, it was a real hang.

Deviations from this plan:
- **The missing `PEXELS_API_KEY` did not stop the step.** The plan says to
  stop and ask; the standing instruction for this run was to continue through
  every step. The compromise: `VisualsConfig.require_pexels` defaults to False,
  so the pipeline still produces video offline, and every fallback logs a
  WARNING naming `PEXELS_API_KEY`. A test asserts that warning is emitted, so
  the skip cannot become silent. Setting `require_pexels=True` restores the
  plan's behaviour — a missing key is then fatal.
- **No Pexels code has ever run against the real API.** Every HTTP test is
  mocked, as the plan requires, so the request shape, the auth header and the
  payload parsing are all unverified against the live service.
- `tests/ffprobe.py` gained video helpers (resolution, fps, mean volume).

Known limitations:
- **"Paragraph-initial beats are a hint to change scene" is not implemented.**
  `segment.py` treats a paragraph break as a hard beat boundary but does not
  record which beats began one, and `Beat` has no field for it. Adding one
  would change the contract, which is a stop-and-ask condition. The
  consecutive-asset rule approximates it: a beat never reuses the previous
  beat's asset, and a repeated query gets a varying modifier appended.
- Keyword extraction is first-three-content-words. It has no idea which word
  is the *subject*; "the house had been empty" becomes "house been empty"
  minus stopwords, which is a serviceable stock query but not a good one.
- `require_pexels=True` is untested against a real key, per the deviation
  above.


## p5 — captions.py — 2026-09-21

Tests: 22 fast, 0 slow, all passing (197 fast / 8 slow across the project)
Files: `narrator/captions.py`, `tests/test_captions.py`, `narrator/config.py`

Decisions:
- The file is parsed back and inspected rather than string-compared. A
  formatting change that breaks nothing would fail a string comparison, and a
  timing bug that breaks everything would pass one.
- Karaoke uses ASS's own `\k` tags rather than one Dialogue line per word:
  libass then highlights word by word within a line, which is what makes the
  effect read as karaoke instead of flashing text.
- In an ASS karaoke style, PrimaryColour is the *sung* colour and
  SecondaryColour the not-yet-sung one. They are written in that order, which
  looks inverted in the source and is the opposite of what the names suggest.
- Grouping never crosses a beat boundary: a caption spanning two beats would
  sit over a visual cut.
- Escaping order is backslash first, then braces, or the escapes get escaped.
  A literal newline is converted to `\N`, since a real newline would end the
  Dialogue line and silently truncate the caption.
- `CaptionConfig.preset("vertical")` uses a bigger font, a much higher
  MarginV and a narrower wrap: a phone held close, in a narrower frame.

Deviations from this plan:
- **The stated precondition fails.** `ffmpeg -filters | grep -w ass` returns
  nothing on this machine: the Homebrew build has no libass. The plan says to
  stop. Under the standing instruction to continue, the step was done anyway,
  which is defensible because building the .ass file needs no filter --
  only burning it does, in p6. Recorded in CLAUDE.md gotchas.

Known limitations:
- **Nothing has rendered these captions.** Every assertion is structural:
  the file parses, the timings are right, the escaping holds. Whether libass
  draws them where intended is unverified and cannot be verified here.
- `wrap_width` is counted in characters, not measured in pixels, so a line of
  wide glyphs can still overflow a narrow frame.
- The font name is a string the renderer must resolve. If DejaVu Sans is
  absent, libass substitutes silently and the layout shifts.


## p6 — assemble.py — 2026-09-21

Tests: 21 fast, 0 slow, all passing (218 fast / 8 slow across the project)
Files: `narrator/assemble.py`, `tests/test_assemble.py`, `narrator/config.py`,
`tests/ffprobe.py`, `CLAUDE.md`

Decisions:
- One ffmpeg invocation: per-beat visuals concatenated, narration
  concatenated, music mixed under, captions applied, encoded. The command is a
  list, logged at DEBUG before it runs, and ffmpeg's stderr is carried into
  the exception rather than summarised.
- `amix=...:normalize=0` is load-bearing. The default normalises, which halves
  the narration when a music bed is added; a test compares the mixed narration
  against the unmixed one to keep that from regressing silently.
- `duration=first` stops a long music track outlasting the story, and
  `-stream_loop -1` covers a short one. Both directions are tested.
- Every clip is normalised (scale/crop/setsar/fps) before concat, which
  demands identical geometry. One odd asset would otherwise fail everything.
- The music level is measured, not assumed: narration is rendered silent and
  the bed's attenuation is measured against the source with volumedetect.

Deviations from this plan:
- **Captions are muxed, not burned.** This ffmpeg has no libass, so there is
  no `ass` filter. `assemble.py` detects that at runtime and falls back to a
  soft `mov_text` subtitle track, logging a WARNING naming libass;
  `AssembleConfig.require_burned_captions=True` makes it a hard failure
  instead. The burn path is written but has never executed here, and the test
  covering it asserts whichever branch this machine can actually reach.

Known limitations:
- **The burn path is untested in practice.** On a machine with libass the
  `ass` filter argument, its path escaping and the resulting overlay are all
  unverified.
- Soft subtitles lose the karaoke styling entirely: `mov_text` carries text
  and timing, not `\k` highlighting, colours or position. A viewer must also
  turn subtitles on. What p5 built is only fully realised once libass exists.
- Transitions are hard cuts. The plan mentioned crossfades in passing; none
  are implemented.


## p7 — pipeline.py + cli.py — 2026-09-21

Tests: 22 fast, 0 slow, all passing (240 fast / 8 slow across the project)
Files: `narrator/pipeline.py`, `narrator/cli.py`, `tests/test_pipeline.py`,
`tests/test_cli.py`, `tests/fixtures/tiny_story.txt`, `narrator/config.py`

Decisions:
- `PipelineConfig` derives the per-stage configs rather than taking them
  separately, so a preset cannot be applied to the visuals and forgotten for
  the captions.
- Resumption is keyed on content, not on a "stage completed" flag: narration
  is restored only if a wav exists for every beat, and word timings only if
  the cached text still matches the beat's text. Editing the story therefore
  invalidates exactly what changed.
- The voice is validated before any stage runs. A typo would otherwise cost a
  full segmentation and alignment before speech looked at it.
- The CLI catches the project's own exception types and exits 1 with one
  line. A traceback in a user's terminal is a bug report about us.
- `--dry-run` writes the manifest and returns before anything is synthesised,
  asserted by a spy rather than by checking for absent files.

Deviations from this plan: none.

Bugs this step found in earlier modules:
- `CaptionConfig.preset()` and two sibling `preset()` methods raised
  `TypeError: got multiple values for keyword argument 'width'` whenever a
  caller overrode width or height. Every preset now lets overrides win. This
  was invisible until something combined a preset with an override, which p7
  is the first code to do.

Known limitations:
- `tiny_story.txt`'s three sentences pack into a single beat at the default
  `max_chars=220`, so the golden test sets 60. The fixture exercises
  concatenation only because of that.
- Resumption trusts the run directory's filenames. A truncated wav from a
  killed run would be reused rather than detected; only the speech cache's own
  atomic write protects against that, and only for files it wrote.
- The CLI has one command. There is no way to run a single stage, which is
  what you actually want when debugging a bad render.


## p8 — verification — 2026-09-21

Tests: 240 fast, 8 slow, all passing. `ruff check` and `ruff format --check`
clean. Files: `tests/fixtures/lighthouse_story.txt` (the 300-word story used
below), `docs/BUILD_LOG.md`.

### Runs

| run | beats | expected | actual | drift |
|---|---|---|---|---|
| `tiny_story.txt` (dry run) | 1 | manifest only | manifest only, 0 files rendered | — |
| `tiny_story.txt` | 1 | — | 9.875s, 1920x1080 @30fps, v1/a1/s1 | +0.000s |
| `lighthouse_story.txt` (309 words) | 10 | 95.900s (sum of beats) | 95.933s | +0.033s |

The 300-word story took 40s wall clock to produce 96s of video, with real
Kokoro synthesis and real whisper alignment. Narration paced at ~3.2 words per
second. All 10 visuals were distinct, no consecutive repeats, no repeat
anywhere. Alignment matched the reference word count on every beat.

### What is wrong with the output, despite green tests

1. **Captions are stacked at the start of the video.** This is the serious
   one. `Beat.words` timings are relative to that beat's own audio, and
   `captions.build_ass` writes them unshifted, so all ten beats' captions
   begin at zero. The last caption in a 95.93s video ends at **4.80s**, and
   the file contains 9 backward jumps in time — one per beat boundary.
   Every test missed it: p5's fixtures were hand-built with words already in
   global time, and p7's golden fixture is a single beat at the default
   `max_chars`, so nothing ever concatenated two aligned beats.
2. **Captions lead the voice by ~0.30s.** Kokoro emits about 0.3s of silence
   before speaking; whisper reports the first word at 0.000. p6's
   "narration starts within 50ms of t=0" test passes because its lavfi sine
   fixture starts instantly, so the real leading silence is never exercised.
   Measured: 0.304s on the tiny story, 0.324s on the lighthouse story.
3. **Caption groups cross sentence boundaries.** One line reads
   "above. Mara" — the end of one sentence and the start of the next, held
   together on screen. Grouping respects beats but not sentences.
4. **Every visual is a generated gradient.** With no `PEXELS_API_KEY` the
   fallback is procedural colour, so the finished video is ten gradients with
   a slow push-in. It is honest output, not a bug, but it is not what the
   plan's "stock footage" describes and it looks like it.
5. **Captions are a soft subtitle track, off by default in most players.**
   Muxed `mov_text` also carries none of the karaoke highlighting p5 built.
6. **The installed `narrator` console script does not work.** The editable
   install uv writes (`_editable_impl_narrator.pth`) is not honoured, so the
   entry point raises `ModuleNotFoundError: No module named 'narrator'`.
   Copying the identical file under another name fixes it, which makes no
   obvious sense and was not worth chasing further. The CLI tests all pass
   because `CliRunner` imports in-process and never touches the script. Every
   run above used `python -c "from narrator.cli import main; main()"`.

### Where a failure could be swallowed silently

Listed, not fixed, as the plan asks.

- `pipeline._cached_audio` returns `None` on any mismatch, so a corrupt or
  truncated wav from a killed run is reused rather than detected. It only
  checks that a file exists.
- `pipeline._load_words` swallows `OSError` and `JSONDecodeError` and falls
  through to realignment. A corrupt cache is indistinguishable from a cold
  one — which is safe, but the operator is never told.
- `pipeline._visuals` reuses any `beat_NNN.mp4` in the run directory without
  checking its duration, resolution or fps. A stale asset from a run at a
  different preset would be silently concatenated.
- `visuals.fetch` catches every `PexelsError` and falls back to stills. It
  logs a WARNING, but a run with a working key that quietly degrades to
  gradients for half its beats still produces a video and exits 0.
- `align._merge_continuations` drops whitespace-only tokens, and `_match`
  drops transcribed tokens that normalise to nothing.
- Tolerant alignment absorbs word-order errors below the ratio threshold, and
  interpolated timings for unmatched words are guesses that nothing verifies.
- `speech._check_plausible` skips text under 20 characters, so a short beat
  that synthesised to silence passes unnoticed.
- `captions.build_ass` raises if a beat has no words, but nothing checks that
  the timings it receives are in the timeline it is writing into — which is
  exactly how finding 1 survived.
- `assemble` trusts `Beat.duration` for nothing and ffmpeg for everything: if
  a beat's asset is shorter than its narration, concat still succeeds and the
  audio runs past the picture.

### Recommended order for the next round

Finding 1 first — it makes the captions useless. Then 2, which is a single
offset. Then a test that concatenates two genuinely aligned beats, which is
what would have caught both.


## p9 — caption timing — 2026-09-21

Tests: 30 new fast (270 fast / 8 slow across the project), all passing
Files: `narrator/captions.py`, `narrator/speech.py`, `narrator/config.py`,
`tests/test_captions.py`, `tests/test_speech.py`, `CLAUDE.md`

Decisions:
- The offset is applied at caption-build time and `Beat.words` stay
  beat-relative, as the plan specifies. That keeps `align` independent of
  where a beat sits in the video, which is what lets it be tested from a
  fixture with no pipeline behind it.
- A beat with no `duration` now raises rather than being treated as zero.
  Guessing would silently shift every beat after it, which is the same class
  of bug as the one being fixed.
- Trimming is an amplitude threshold with a retained pad, not a silence
  filter in ffmpeg: the trim has to happen before the wav is written, because
  the wav is what whisper aligns against and what the cache stores.
- The trim settings are in the cache key. Audio trimmed at a different
  threshold is different audio.
- Sentence-aware grouping reuses segment.py's abbreviation asymmetry: a false
  split only shortens a caption, a missed one puts two sentences on screen.

Tests changed in an earlier module, and why:
- `tests/test_captions.py`'s `SIMPLE` fixture wrote beat 1's words in
  whole-video time (1.5-2.4). That is not what `align` produces, and it is
  exactly why p5, p7 and p8 all passed while the output was wrong. It now
  starts every beat at 0.0, like the real thing.

Deviations from this plan: none.

Verification of the specific p8 findings:
- Finding 1 (captions stacked at zero): `test_later_beats_are_offset_into_whole_video_time`,
  plus the three recorded-fixture concatenation tests the plan asked for.
- Finding 2 (captions lead the voice): leading silence is trimmed to the pad,
  asserted through ffprobe in `test_speech_onset_is_within_the_pad_of_zero`.
- Finding 3 (groups crossing sentences): `test_a_caption_never_holds_two_sentences`.

Known limitations:
- The offset is only as good as `Beat.duration`. If the assembler ever pads
  or crossfades between beats, the captions will drift by whatever it adds,
  and nothing would catch it.
- Trimming uses a fixed amplitude threshold. A voice that fades in gently
  loses its first moment; a recording with a noise floor above the threshold
  is not trimmed at all.
- The concatenation test uses durations derived from the recorded fixture
  (last word plus the pad) rather than a real trimmed wav, so it verifies the
  arithmetic rather than the end-to-end result. p12 is where that is checked.


## p10 — packaging — 2026-09-21

Tests: 5 new fast (275 fast / 8 slow across the project), all passing
Files: `tests/test_packaging.py`, `CLAUDE.md`

Decisions:
- The tests install the project into a throwaway venv and run the executable
  there, rather than testing the development venv's script. `uv run` re-syncs
  before every command and reverts the install to editable, so a test against
  this venv could never pass under `uv run pytest`. More importantly, what
  matters is that someone who installs the package gets a working command —
  which is what this now proves.
- A session-scoped fixture makes it one install for the whole run: 1.9s,
  because uv caches the wheel.
- `PYTHONPATH` is cleared and the working directory is elsewhere, so a passing
  test cannot be the source tree happening to be importable. One test asserts
  the installed `narrator.__file__` is *not* inside the repo.

Root cause, as far as it was worth chasing:
- uv's editable install writes `_editable_impl_narrator.pth` containing the
  repo path. This interpreter does not honour it — the path never reaches
  `sys.path` — while a `.pth` with byte-identical content under a different
  name is honoured, and `io.open_code` reads the file fine. Not permissions,
  not xattrs, not a missing trailing newline, not `.pth` processing being
  disabled: all four were tested and ruled out. `uv sync --no-editable`
  installs a real copy and the script works. That is documented in CLAUDE.md
  as the way to use the CLI from this venv.

Deviations from this plan: none. The plan asked for a test that runs the
installed executable as a subprocess; it does, just against a clean install
rather than this venv.

Known limitations:
- The development venv's console script is still broken after a default
  `uv sync`, and nothing in the fast suite now notices, because the tests
  moved to a clean install. The trade was deliberate: a test that fails
  depending on which flags the last sync used is worse than no test.
- The fresh-venv install is only exercised on this machine's Python. Nothing
  checks the package against another interpreter version.


## p11 — silent failures — 2026-09-22

Tests: 18 new fast (293 fast / 8 slow across the project), all passing
Files: `narrator/pipeline.py`, `narrator/visuals/__init__.py`,
`narrator/align.py`, `narrator/speech.py`, `narrator/captions.py`,
`narrator/assemble.py`, `narrator/config.py`, their tests, `CLAUDE.md`

The nine items from the p8 audit, in the logged order:

1. **Truncated cached wav reused.** Every cached file is now probed, and an
   unreadable or absurdly short one sends the stage back to synthesis with a
   warning. It is also deleted: `speech.synthesize` caches on the same
   filename, so leaving it would hand the same bad file straight back — the
   first version of this fix recovered into the identical failure, which the
   test caught.
2. **Corrupt word cache discarded in silence.** Realigning is still the
   recovery, but it now says which file it threw away and why.
3. **Stale visuals reused.** Each cached `beat_NNN.mp4` is checked for
   resolution, frame rate and length against the current config and beat; a
   mismatch re-fetches with a warning naming the beat.
4. **A failing Pexels key tolerated.** Falling back with no key set is how
   this project works offline and is logged as a summary. With a key set it
   means searches are failing, and more than `max_fallback_ratio` (0.5) of
   beats falling back now raises rather than producing mostly gradients.
5. **Dropped transcript tokens.** Tokens that normalise to nothing are logged
   at DEBUG with their text, rather than vanishing.
6. **Absorbed word-order damage and guessed timings.** A match ratio between
   the minimum and 0.99 warns that nearby timings are approximate, and any
   interpolated word warns with a count. A clean alignment stays silent, which
   is asserted.
7. **Short beats skipped the plausibility check.** `MIN_BEAT_SECONDS` (0.25s)
   applies regardless of character count, so "Yes." coming back as a click
   now raises.
8. **Captions never checked their timings belonged to the beat.** Words that
   run past the beat's duration, or backwards, raise `CaptionError`. This is
   the specific shape of the p8 finding: whole-video times in a beat-relative
   field.
9. **Visual shorter than its narration.** `assemble` probes each asset and
   refuses one that cannot cover its beat, and every clip is trimmed to its
   own beat so a long asset cannot stretch the video either.

Tests changed in an earlier module, and why:
- `test_silently_empty_synthesis_raises_rather_than_caching_garbage` asserted
  the message "implausible". Its fixture is now caught by the new duration
  floor first, which is an equally correct rejection, so the assertion accepts
  either.

Deviations from this plan: none.

Known limitations:
- Items 5 and 6 warn rather than raise. That is the intended behaviour of
  tolerant alignment, but it means a run with dozens of interpolated words
  still produces a video; nobody is forced to look at the log.
- The visual-geometry check trusts ffprobe's `avg_frame_rate`, which is an
  average: a variable-frame-rate asset could pass it and still stutter.
- `max_fallback_ratio` is a blunt instrument. Half a story on gradients
  passes; half plus one beat fails.


## Environment — ffmpeg with libass — 2026-09-22

Not a tagged step; a prerequisite for p12.

`brew uninstall ffmpeg`, `brew tap homebrew-ffmpeg/ffmpeg`,
`brew install homebrew-ffmpeg/ffmpeg/ffmpeg`. Built in 1m37s, mostly poured
from bottles. `ffmpeg -filters | grep -w ass` now prints a line and the build
config carries `--enable-libass`.

Full suite re-run afterwards, since ffmpeg changed underneath it: 292 passed,
1 skipped, 8 slow passed. The skip is
`test_requiring_burned_captions_fails_loudly_without_libass`, which is
unreachable now and says so.

The burn path executed for the first time. Verified beyond the filter list:
captions were rendered onto a black clip and the frame compared against the
same frame without them — 7550 bytes against 797, different hashes, so text
is genuinely drawn rather than the filter merely being accepted.

**p12 is blocked**: `PEXELS_API_KEY` is unset, which is one of this plan's
stop conditions, and the instruction for this round was not to work around it.
Nothing for p12 has been started or committed.
