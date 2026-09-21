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
- `_synthesize_audio` skips result chunks whose `audio` is `None` and only
  raises when every chunk is empty. A partial drop would shorten narration and
  is caught only if the plausibility band notices.
- The fast duration-band test measures a synthetic fixture generated at 15
  chars/sec, so it exercises the wav-writing and ffprobe path rather than the
  model's real pacing. The slow `test_real_synthesis_produces_narration` is
  what actually checks that.
- `p1`'s terminal-punctuation test was changed during this step: it asserted
  every beat ends on `.`/`!`/`?`, which is only true when the source text does.
  Now every beat except possibly the last. The `p1` tag was left in place.
