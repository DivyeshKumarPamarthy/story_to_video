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
