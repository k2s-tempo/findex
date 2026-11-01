import pytest

from findex.index import build_index
from findex.snippets import (best_match, find_matches, make_snippet, prepare_text,
                             snippet_from_text)

FILLER = "lorem ipsum dolor sit amet " * 12      # ~320 символів без збігів


def test_find_matches_positions_and_case():
    text, lowered = prepare_text("The Soul is immortal. The soul!")
    ms = find_matches(lowered, {"soul"})
    assert [text[s:e] for s, e, _ in ms] == ["Soul", "soul"]   # регістр оригіналу збережено
    assert all(t == "soul" for _, _, t in ms)


def test_basic_highlight_and_ellipsis():
    raw = FILLER + "the Soul is immortal " + FILLER
    snip = snippet_from_text(raw, {"soul"})
    assert "**Soul**" in snip.text
    assert snip.text.startswith("...") and snip.text.endswith("...")
    assert snip.matched == ("soul",)


def test_window_is_bounded_by_radius():
    raw = FILLER + "soul" + FILLER
    snip = snippet_from_text(raw, {"soul"}, radius=80)
    inner = snip.text.strip(".").replace("**", "")
    assert len(inner) <= 80 + 4 + 80


def test_words_are_not_cut():
    raw = "alpha beta gamma " * 30 + "target " + "alpha beta gamma " * 30
    snip = snippet_from_text(raw, {"target"}, radius=40)
    body = snip.text.replace("...", "").replace("**", "")
    assert all(w in {"alpha", "beta", "gamma", "target"} for w in body.split())


def test_no_ellipsis_for_short_text():
    snip = snippet_from_text("only soul here", {"soul"})
    assert snip.text == "only **soul** here"


def test_newlines_collapsed():
    snip = snippet_from_text("the\nsoul\n\nis  here", {"soul"})
    assert snip.text == "the **soul** is here"


def test_prefers_window_with_more_distinct_terms():
    raw = ("justice " + FILLER) + ("justice " + FILLER) + "virtue and justice together " + FILLER
    snip = snippet_from_text(raw, {"justice", "virtue"})
    assert set(snip.matched) == {"justice", "virtue"}
    assert "**virtue**" in snip.text


def test_idf_weight_beats_frequent_stopword():
    # "the" трапляється частіше, але рідкий "andromache" має важити більше
    raw = ("the " + "x " * 60) * 4 + "an andromache appears " + "y " * 60
    snip = snippet_from_text(raw, {"the", "andromache"},
                             weights={"the": 0.1, "andromache": 3.0}, radius=40)
    assert "**andromache**" in snip.text


def test_best_match_ties_prefer_earliest():
    ms = [(0, 4, "a"), (1000, 1004, "a")]
    assert best_match(ms, {"a": 1.0}, 80) == 0


def test_custom_marks():
    snip = snippet_from_text("the soul", {"soul"}, marks=("[[", "]]"))
    assert snip.text == "the [[soul]]"


def test_hyphenated_and_numeric_tokens():
    snip = snippet_from_text("a self-driving car in 2026", {"self-driving", "2026"})
    assert "**self-driving**" in snip.text and "**2026**" in snip.text


def test_no_matches_shows_beginning():
    snip = snippet_from_text("start of the document " + FILLER, {"zzz"})
    assert snip.text.startswith("start of the document") and snip.matched == ()


def test_length_changing_lowercase_is_safe():
    snip = snippet_from_text("İstanbul soul", {"soul"})   # 'İ'.lower() довший за 'İ'
    assert "soul" in snip.text.lower() and "**" in snip.text


def test_make_snippet_from_file_and_missing_file(tmp_path):
    (tmp_path / "a.txt").write_text("intro " * 50 + "the soul is immortal " + "outro " * 50,
                                    encoding="utf-8")
    idx = build_index(tmp_path)
    snip = make_snippet(idx, 0, ["soul", "immortal"])
    assert "**soul**" in snip.text and "**immortal**" in snip.text
    (tmp_path / "a.txt").unlink()
    assert make_snippet(idx, 0, ["soul"]).text == "(файл документа недоступний)"