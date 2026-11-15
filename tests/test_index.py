import dataclasses
from collections.abc import Mapping

import pytest

from findex.index import Posting, build_index


@pytest.fixture
def corpus(tmp_path):
    (tmp_path / "a.txt").write_text("apple banana apple", encoding="utf-8")
    (tmp_path / "b.txt").write_text("banana cherry", encoding="utf-8")
    return tmp_path


def test_postings_with_tf(corpus):
    idx = build_index(corpus)
    assert idx.postings["apple"] == [Posting(0, 2)]
    assert idx.postings["banana"] == [Posting(0, 1), Posting(1, 1)]
    assert idx.postings["cherry"] == [Posting(1, 1)]


def test_doc_meta(corpus):
    idx = build_index(corpus)
    assert [(d.name, d.length) for d in idx.docs] == [("a", 3), ("b", 2)]


def test_postings_sorted_by_doc_id(corpus):
    idx = build_index(corpus)
    for plist in idx.postings.values():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(ids)


def test_posting_is_frozen_and_hashable():
    p = Posting(1, 2)
    with pytest.raises(dataclasses.FrozenInstanceError):
        p.tf = 5
    assert hash(p) == hash(Posting(1, 2))
    assert not hasattr(p, "__dict__")


def test_positions(corpus):
    idx = build_index(corpus, with_positions=True)
    assert idx.positions["apple"] == {0: [0, 2]}
    assert idx.positions["banana"] == {0: [1], 1: [0]}


def test_missing_term_does_not_grow_index(corpus):
    idx = build_index(corpus)
    n = len(idx.postings)
    idx.postings.get("zzz")
    assert len(idx.postings) == n




def test_index_is_a_mapping(corpus):
    idx = build_index(corpus)
    assert isinstance(idx, Mapping)
    assert len(idx) == 3
    assert "apple" in idx and "zzz" not in idx
    assert idx["apple"] == [Posting(0, 2)]
    assert sorted(idx) == ["apple", "banana", "cherry"]
    assert dict(idx.items())["cherry"] == [Posting(1, 1)]


def test_missing_term_raises_keyerror(corpus):
    idx = build_index(corpus)
    with pytest.raises(KeyError):
        idx["zzz"]
    assert idx.get("zzz") is None  # get() приходить від Mapping


def test_df_and_num_docs(corpus):
    idx = build_index(corpus)
    assert idx.num_docs == 2
    assert idx.df("banana") == 2
    assert idx.df("apple") == 1
    assert idx.df("zzz") == 0


def test_avg_doc_length_is_cached(corpus):
    idx = build_index(corpus)
    assert "avg_doc_length" not in idx.__dict__   # ще не обчислено
    assert idx.avg_doc_length == 2.5              # (3 + 2) / 2
    assert "avg_doc_length" in idx.__dict__       # тепер лежить у кеші
    idx.docs = []                                 # зміна даних кеш не чіпає
    assert idx.avg_doc_length == 2.5


def test_repr(corpus):
    assert "2 док." in repr(build_index(corpus))
    assert "з позиціями" in repr(build_index(corpus, with_positions=True))


def test_close_releases_and_runs_callbacks(corpus):
    idx = build_index(corpus)
    called = []
    idx.on_close(lambda: called.append(1))
    idx.close()
    idx.close()  # повторне закриття безпечне
    assert idx.closed and called == [1]
    assert len(idx) == 0
    with pytest.raises(ValueError):
        idx["apple"]
