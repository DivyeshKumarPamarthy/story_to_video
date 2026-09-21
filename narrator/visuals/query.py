"""Beat text -> a stock-footage search phrase. No model, just keywords."""

from __future__ import annotations

import re
from dataclasses import replace

from narrator.beats import Beat

#: Used when a beat has no content words of its own -- dialogue tags, short
#: connective sentences. Better a deliberate mood than an empty query.
FALLBACK_QUERY = "cinematic atmosphere"

#: Deliberately small. This is not linguistics; it is enough to stop "the" and
#: "had" from becoming the search term for a shot.
_STOPWORDS = frozenset(
    """
    a an the and or but so then than that this these those there here
    is was are were be been being am do does did done doing
    have has had having will would shall should can could may might must
    i you he she it we they him her them his hers its their our your my me
    of in on at to for with from by into onto over under about as if
    not no nor too very just only also even still yet again once
    what which who whom whose when where why how all any both each few more
    most other some such own same up down out off through during before after
    """.split()
)

_NON_WORD = re.compile(r"[^0-9a-z]+")


def build_query(text: str, max_words: int = 3) -> str:
    """The first few content words, in the order they appear."""
    seen: list[str] = []
    for raw in text.lower().split():
        token = _NON_WORD.sub("", raw)
        if not token or token in _STOPWORDS or len(token) < 3:
            continue
        if token not in seen:
            seen.append(token)
        if len(seen) == max_words:
            break
    return " ".join(seen) if seen else FALLBACK_QUERY


def add_queries(beats: list[Beat], max_words: int = 3) -> list[Beat]:
    """Fill ``visual_query`` on every beat. Input is not mutated.

    Consecutive beats are never given the same query: repeating it would fetch
    the same footage twice in a row, which is the tell of a generated video.
    Where the text gives nothing else, the beat index is appended so the
    downstream picker sees a different phrase.
    """
    queried: list[Beat] = []
    previous = ""
    for beat in beats:
        query = build_query(beat.text, max_words=max_words)
        if query == previous:
            query = _vary(query, beat.index)
        queried.append(replace(beat, visual_query=query))
        previous = query
    return queried


#: Mood words used only to break a repeat, so two beats in a row never search
#: for exactly the same thing.
_VARIATIONS = ("wide", "close", "dim", "drifting", "still", "distant")


def _vary(query: str, index: int) -> str:
    return f"{query} {_VARIATIONS[index % len(_VARIATIONS)]}"
