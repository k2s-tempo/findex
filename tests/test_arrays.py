"""NumPy-ранжування (rank_fast) дає той самий порядок, що й еталонний цикл Python з лаби 3."""

import random
from pathlib import Path

import numpy as np
import pytest

from findex.arrays import rank_fast
from findex.index import DocMeta, Index, Posting, build_index
from findex.query import ranked_search
from findex.ranking import SearchResult, rank
from findex.scoring import BM25, TfIdf

SCORERS = [BM25(), BM25(k1=1.2, b=0.4), TfIdf(), TfIdf(norm=False)]
CORPUS = Path(__file__).resolve().parent.parent / "corpus"


def assert_same(fast: list[SearchResult], slow: list[SearchResult]) -> None:
    assert [(r.doc_id, r.name) for r in fast] == [(r.doc_id, r.name) for r in slow]
    assert [r.score for r in fast] == pytest.approx([r.score for r in slow], rel=1e-12)


def random_index(rng: random.Random) -> Index:
    n = rng.randint(1, 40)
    docs = [DocMeta(i, f"d{i}", f"d{i}.txt", rng.randint(1, 30)) for i in range(n)]
    postings: dict[str, list[Posting]] = {}
    for j in range(rng.randint(1, 6)):
        ids = sorted(rng.sample(range(n), rng.randint(0, n)))
        if ids:
            postings[f"t{j}"] = [Posting(i, rng.randint(1, 4)) for i in ids]
    return Index(postings, docs, None)


@pytest.mark.parametrize("scorer", SCORERS)
def test_matches_reference_on_random_indexes(scorer: BM25 | TfIdf) -> None:
    rng = random.Random(7)
    for _ in range(150):
        index = random_index(rng)
        terms = [
            f"t{rng.randint(0, 7)}" for _ in range(rng.randint(1, 4))
        ]  # t6, t7 і повтори можуть не існувати
        k = rng.choice([1, 2, 3, 5, 10, 100])
        candidates = None
        if rng.random() < 0.5:
            candidates = set(rng.sample(range(index.num_docs), rng.randint(0, index.num_docs)))
        assert_same(rank_fast(index, terms, scorer, k, candidates), rank(index, terms, scorer, k, candidates))


@pytest.fixture(scope="module")
def corpus_index() -> Index:
    if not CORPUS.is_dir():
        pytest.skip("немає каталогу corpus/")
    return build_index(CORPUS)


QUERIES = [["justice"], ["the"], ["justice", "virtue", "courage"], ["zzzz"], ["socrates", "the", "the"]]


@pytest.mark.parametrize("scorer", SCORERS)
@pytest.mark.parametrize("terms", QUERIES)
@pytest.mark.parametrize("k", [1, 10, 50])
def test_matches_reference_on_corpus(
    corpus_index: Index, scorer: BM25 | TfIdf, terms: list[str], k: int
) -> None:
    assert_same(rank_fast(corpus_index, terms, scorer, k), rank(corpus_index, terms, scorer, k))


def test_ranked_search_fast_flag(corpus_index: Index) -> None:
    query = "justice AND (virtue OR courage)"
    fast = ranked_search(corpus_index, query, BM25(), 10)
    slow = ranked_search(corpus_index, query, BM25(), 10, fast=False)
    assert fast[0] == slow[0]
    assert fast[1] == slow[1]
    assert_same(fast[2], slow[2])


def tiny() -> Index:
    docs = [DocMeta(i, f"d{i}", f"d{i}.txt", 10 + i) for i in range(4)]
    postings = {"a": [Posting(0, 2), Posting(2, 1), Posting(3, 5)], "b": [Posting(1, 1), Posting(2, 3)]}
    return Index(postings, docs, None)


def test_edge_cases() -> None:
    index = tiny()
    assert rank_fast(index, [], BM25(), 5) == []
    assert rank_fast(index, ["zzz"], BM25(), 5) == []
    assert rank_fast(index, ["a"], BM25(), 0) == []
    assert rank_fast(index, ["a"], BM25(), 5, candidates=set()) == []


def test_arrays_are_int32_cached_and_dropped_on_close() -> None:
    index = tiny()
    arrays = index.arrays
    assert arrays.lengths.dtype == np.int32
    assert arrays.lengths.tolist() == [10, 11, 12, 13]
    term = arrays.term("a")
    assert term is not None
    assert arrays.term("a") is term  # кеш
    assert term.doc_ids.dtype == np.int32
    assert term.tfs.dtype == np.int32
    assert term.doc_ids.tolist() == [0, 2, 3]
    assert term.tfs.tolist() == [2, 1, 5]
    assert arrays.term("nope") is None
    index.close()
    assert "arrays" not in index.__dict__


class Constant:
    def score(self, tf: int, df: int, doc_len: int, index: Index) -> float:
        return float(tf)


def test_unknown_scorer_falls_back_to_reference() -> None:
    index = tiny()
    assert_same(rank_fast(index, ["a", "b"], Constant(), 3), rank(index, ["a", "b"], Constant(), 3))


@pytest.mark.parametrize("lengths", [[0, 0, 0, 0], [0, 7, 0, 3]])
@pytest.mark.parametrize("scorer", SCORERS)
def test_zero_length_documents_match_reference(lengths: list[int], scorer: BM25 | TfIdf) -> None:
    docs = [DocMeta(i, f"d{i}", f"d{i}.txt", n) for i, n in enumerate(lengths)]
    index = Index({"a": [Posting(0, 2), Posting(1, 1), Posting(3, 5)]}, docs, None)
    assert_same(rank_fast(index, ["a"], scorer, 4), rank(index, ["a"], scorer, 4))
