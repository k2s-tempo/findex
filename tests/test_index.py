import dataclasses

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
    assert not hasattr(p, "__dict__")  # slots=True


def test_positions(corpus):
    idx = build_index(corpus, with_positions=True)
    assert idx.positions["apple"] == {0: [0, 2]}
    assert idx.positions["banana"] == {0: [1], 1: [0]}


def test_missing_term_does_not_grow_index(corpus):
    idx = build_index(corpus)
    n = len(idx.postings)
    idx.postings.get("zzz")
    assert len(idx.postings) == n