"""Story text -> beats.

Pure function: ``str`` in, ``list[Beat]`` out. No filesystem, no network, no
config, no logging.

Splitting happens on sentence boundaries only. Beats are then packed greedily
up to ``max_chars``; a single sentence longer than that is emitted whole rather
than cut, because a beat that stops mid-sentence sounds broken once narrated.
"""

from __future__ import annotations

import re

from narrator.beats import Beat

#: Words that take a trailing period without ending a sentence.
#:
#: Deliberately conservative. A missed split only makes a beat longer, but a
#: false split cuts a sentence in half mid-narration, so anything that is also
#: an ordinary word ("no", "am", "sat", "may") is left out even though it is a
#: real abbreviation. Dotted forms like "e.g.", "U.S." and "a.m." need no entry
#: here -- the single-letter rule in :func:`_is_boundary` covers them.
#:
#: Grouped by line: titles, latin/editorial, organisations and references,
#: months (may and mar omitted -- both are ordinary words).
_ABBREVIATIONS = frozenset(
    """
    mr mrs ms mx dr prof rev fr sr jr st mt capt sgt lt col gen gov hon messrs
    vs etc al cf ibid viz
    inc ltd dept fig vol ed eds pp approx
    jan feb apr jun jul aug sep sept oct nov dec
    """.split()
)

#: Terminal punctuation, optional closing quotes/brackets, then whitespace.
_SENTENCE_END = re.compile(
    r"""
    (?P<punct>[.!?…]+)          # . ! ? … possibly repeated: "?!", "..."
    (?P<closers>["'”’)\]»]*)   # closing quote or bracket
    (?P<gap>\s+)                     # a boundary must be followed by whitespace
    """,
    re.VERBOSE,
)

_PARAGRAPH_BREAK = re.compile(r"\n\s*\n")
_TRAILING_WORD = re.compile(r"([A-Za-z]+)$")
_WHITESPACE = re.compile(r"\s+")


def segment(text: str, max_chars: int = 220) -> list[Beat]:
    """Split ``text`` into narration beats.

    Beats never cut a sentence in half. Paragraph breaks always start a new
    beat. ``visual_query`` is left empty for a later stage to fill.
    """
    beats: list[Beat] = []
    for paragraph in _PARAGRAPH_BREAK.split(text):
        for chunk in _pack(_sentences(paragraph), max_chars):
            beats.append(Beat(index=len(beats), text=chunk, visual_query=""))
    return beats


def _sentences(paragraph: str) -> list[str]:
    """Split one paragraph into sentences, whitespace normalised."""
    sentences: list[str] = []
    start = 0
    for match in _SENTENCE_END.finditer(paragraph):
        if not _is_boundary(paragraph, match):
            continue
        sentences.append(paragraph[start : match.end("closers")])
        start = match.end()

    tail = paragraph[start:]
    if tail.strip():
        sentences.append(tail)

    return [normalised for s in sentences if (normalised := _normalise(s))]


def _is_boundary(paragraph: str, match: re.Match[str]) -> bool:
    """Is this punctuation a real sentence end, or an abbreviation or a
    dialogue tag?"""
    following = paragraph[match.end() : match.end() + 1]

    # "Is anyone there?" she called. -- a lowercase continuation is the same
    # sentence, whether it follows a closing quote or an abbreviation.
    if following and (following.islower() or following in ",;:"):
        return False

    # Only a lone period is ambiguous. "!" and "?!" and "..." always end one.
    if match.group("punct") == ".":
        word = _TRAILING_WORD.search(paragraph[: match.start("punct")])
        if word:
            token = word.group(1)
            # A single letter is an initial or part of a dotted acronym:
            # "J. R. R. Tolkien", "U.S.", "e.g.".
            if len(token) == 1:
                return False
            if token.lower() in _ABBREVIATIONS:
                return False

    return True


def _pack(sentences: list[str], max_chars: int) -> list[str]:
    """Greedily fill beats up to ``max_chars`` without splitting a sentence."""
    beats: list[str] = []
    current = ""
    for sentence in sentences:
        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current = f"{current} {sentence}"
        else:
            beats.append(current)
            current = sentence
    if current:
        beats.append(current)
    return beats


def _normalise(text: str) -> str:
    return _WHITESPACE.sub(" ", text).strip()
