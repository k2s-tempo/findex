"""Швидкий find_matches (str.find + перевірка меж) дає те саме, що повний прохід TOKEN_RE."""

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.snippets import find_matches
from findex.tokenizer import TOKEN_RE


def reference_find_matches(lowered: str, terms: set[str]) -> list[tuple[int, int, str]]:
    return [(m.start(), m.end(), m.group()) for m in TOKEN_RE.finditer(lowered) if m.group() in terms]


CASES = [
    ("virtue and virtue.", {"virtue"}),
    ("x-virtue virtue-y virtue", {"virtue"}),
    ("don't t don", {"don", "t", "don't"}),
    ("o'brien's brien", {"brien", "o'brien's"}),
    ("3virtue virtue3 104 1045 a104", {"virtue", "104", "3"}),
    ("-virtue- 'virtue' --virtue", {"virtue"}),
    ("_virtue_ virtue_x", {"virtue"}),
    ("virtue² ²virtue virtue", {"virtue"}),
    ("école ecole", {"école"}),
    ("virtue", {"virtue"}),
    ("1 12 21 1", {"1"}),
    ("", {"a"}),
    ("abc", set()),
]


@pytest.mark.parametrize(("text", "terms"), CASES)
def test_matches_reference_on_known_cases(text: str, terms: set[str]) -> None:
    assert find_matches(text, terms) == reference_find_matches(text, terms)


def test_empty_term_is_ignored() -> None:
    assert find_matches("abc", {""}) == []


def test_results_are_sorted_by_position() -> None:
    assert find_matches("b a b a", {"a", "b"}) == [(0, 1, "b"), (2, 3, "a"), (4, 5, "b"), (6, 7, "a")]


TERMS = ["a", "b", "ab", "3", "1", "13", "a-b", "a'b", "é"]


@settings(max_examples=400, deadline=None)
@given(
    text=st.text(alphabet="abAB31 '-_.é²\n", max_size=80),
    terms=st.sets(st.sampled_from(TERMS)),
)
def test_matches_reference_property(text: str, terms: set[str]) -> None:
    lowered = text.lower()
    assert find_matches(lowered, terms) == reference_find_matches(lowered, terms)
