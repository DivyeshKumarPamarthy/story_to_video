"""Tests for narrator.visuals.query -- keyword extraction, no model."""

from __future__ import annotations

from narrator.beats import Beat
from narrator.visuals.query import FALLBACK_QUERY, add_queries, build_query


def beat(text: str, index: int = 0) -> Beat:
    return Beat(index=index, text=text, visual_query="")


def test_drops_stopwords_and_keeps_content_words():
    query = build_query("The house had been empty for nine years.")
    assert "the" not in query.split()
    assert "house" in query.split()


def test_is_lowercased_and_stripped_of_punctuation():
    query = build_query('"Is anyone there?" she called into the dark.')
    assert query == query.lower()
    assert all(token.isalnum() for token in query.split())


def test_is_capped_at_max_words():
    long_text = "Rain flooded the abandoned cathedral while lightning split the ancient sky above."
    assert len(build_query(long_text, max_words=3).split()) <= 3


def test_is_deterministic():
    text = "Mara stopped at the gate and counted the windows."
    assert build_query(text) == build_query(text)


def test_text_with_no_content_words_falls_back():
    assert build_query("And then, it was.") == FALLBACK_QUERY
    assert build_query("") == FALLBACK_QUERY
    assert build_query("   ") == FALLBACK_QUERY


def test_preserves_order_of_appearance():
    query = build_query("Lightning struck the cathedral.", max_words=3)
    assert query.split() == ["lightning", "struck", "cathedral"]


# --- as a pipeline stage ----------------------------------------------------


def test_add_queries_fills_visual_query_only():
    beats = [beat("The house had been empty for nine years.", 0)]
    out = add_queries(beats)

    assert out[0].visual_query != ""
    assert out[0].index == 0
    assert out[0].text == beats[0].text
    assert out[0].audio_path is None
    assert out[0].words == []
    assert out[0].asset_path is None


def test_add_queries_does_not_mutate_input():
    beats = [beat("Rain came in through the roof.")]
    add_queries(beats)
    assert beats[0].visual_query == ""


def test_add_queries_returns_empty_for_empty_input():
    assert add_queries([]) == []


def test_consecutive_beats_get_different_queries():
    # A repeated query would fetch the same footage twice in a row, which is
    # what makes a faceless video look generated.
    beats = [
        beat("The house stood empty.", 0),
        beat("The house stood empty.", 1),
        beat("The house stood empty.", 2),
    ]
    out = add_queries(beats)
    queries = [b.visual_query for b in out]

    assert queries[0] != queries[1]
    assert queries[1] != queries[2]


def test_distinct_text_keeps_its_own_keywords():
    beats = [beat("Lightning struck the cathedral.", 0), beat("The river froze solid.", 1)]
    out = add_queries(beats)

    assert "lightning" in out[0].visual_query
    assert "river" in out[1].visual_query
