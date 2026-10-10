import pytest

from findex.index import build_index
from findex.search import merge_and, merge_diff, merge_or, search


@pytest.fixture
def index(tmp_path):
    (tmp_path / "a.txt").write_text("apple banana apple", encoding="utf-8")
    (tmp_path / "b.txt").write_text("banana cherry", encoding="utf-8")
    (tmp_path / "c.txt").write_text("cherry date", encoding="utf-8")
    return build_index(tmp_path)


@pytest.mark.parametrize("engine", ["merge", "set"])
@pytest.mark.parametrize("query, expected", [
    ("banana", [0, 1]),
    ("apple banana", [0]),            # неявний AND
    ("apple AND banana", [0]),
    ("apple OR cherry", [0, 1, 2]),
    ("banana NOT apple", [1]),
    ("NOT banana", [2]),
    ("apple cherry", []),
    ("Apple", [0]),                   # нормалізація регістру
    ("missing", []),
    ("apple OR date NOT banana", [0, 2]),
    ("", []),
])
def test_queries(index, engine, query, expected):
    assert search(index, query, engine) == expected


def test_engines_agree(index):
    for q in ["banana", "apple OR cherry", "NOT apple", "cherry NOT date"]:
        assert search(index, q, "merge") == search(index, q, "set")


def test_merge_primitives():
    assert merge_and([1, 3, 5], [3, 4, 5]) == [3, 5]
    assert merge_or([1, 3], [2, 3, 4]) == [1, 2, 3, 4]
    assert merge_diff([1, 2, 3, 4], [2, 4]) == [1, 3]


def test_bad_query(index):
    with pytest.raises(ValueError):
        search(index, "apple AND")
