import math

import pytest

from findex.index import build_index
from findex.ranking import SearchResult, rank
from findex.scoring import BM25, SCORERS, Scorer, TfIdf


@pytest.fixture
def idx(tmp_path):
    # 4 документи: "rare" лише в одному, "common" у всіх
    (tmp_path / "a.txt").write_text("rare common " + "filler " * 8, encoding="utf-8")
    (tmp_path / "b.txt").write_text("common " + "filler " * 9, encoding="utf-8")
    (tmp_path / "c.txt").write_text("common " + "filler " * 9, encoding="utf-8")
    (tmp_path / "d.txt").write_text("common " + "filler " * 9, encoding="utf-8")
    return build_index(tmp_path)


@pytest.mark.parametrize("scorer", [TfIdf(), TfIdf(norm=False), BM25()])
def test_scorers_satisfy_protocol(scorer):
    assert isinstance(scorer, Scorer)   # duck typing: жодного спільного базового класу
    assert Scorer not in type(scorer).__mro__


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_rare_term_beats_common(idx, scorer):
    rare = scorer.score(1, idx.df("rare"), 10, idx)
    common = scorer.score(1, idx.df("common"), 10, idx)
    assert rare > common > 0


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_repetition_saturates(idx, scorer):
    def s(tf):
        return scorer.score(tf, 1, 10, idx)
    assert s(2) > s(1)
    # 20-те повторення майже нічого не додає порівняно з другим
    assert s(20) - s(19) < (s(2) - s(1)) / 10
    assert s(100) < 10 * s(1)                     # не лінійне зростання (лінійне дало б 100)


@pytest.mark.parametrize("scorer", [TfIdf(), BM25()])
def test_short_doc_beats_long_doc(idx, scorer):
    short = scorer.score(1, 2, 20, idx)
    long_ = scorer.score(1, 2, 2000, idx)
    assert short > long_


def test_classic_tfidf_ignores_length(idx):
    t = TfIdf(norm=False)
    assert t.score(1, 2, 20, idx) == t.score(1, 2, 2000, idx)


def test_bm25_formula_by_hand(idx):
    # N=4, df=1, tf=1, doc_len = avg_len -> length_part = 1
    n, df, k1 = 4, 1, 1.5
    idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
    expected = idf * 1 * (k1 + 1) / (1 + k1)
    assert BM25().score(1, df, idx.avg_doc_length, idx) == pytest.approx(expected)


def test_zero_tf_gives_zero(idx):
    assert TfIdf().score(0, 1, 10, idx) == 0.0
    assert BM25().score(0, 1, 10, idx) == 0.0


def test_search_result_orders_by_score_only():
    a = SearchResult(1.0, doc_id=9, name="z")
    b = SearchResult(2.0, doc_id=1, name="a")
    assert a < b and max(a, b) is b
    assert SearchResult(1.0, 1, "x") == SearchResult(1.0, 2, "y")   # doc_id не порівнюється


def test_rank_orders_and_limits(idx):
    res = rank(idx, ["rare", "common"], BM25(), k=2)
    assert len(res) == 2
    assert res[0].name == "a"                    # єдиний із рідким терміном
    assert res[0].score >= res[1].score


def test_rank_ties_prefer_lower_doc_id(idx):
    res = rank(idx, ["common"], BM25(), k=3)     # b, c, d однакові
    ids = [r.doc_id for r in res if r.name != "a"]
    assert ids == sorted(ids)


def test_rank_candidates_filter(idx):
    res = rank(idx, ["common"], BM25(), k=10, candidates={2, 3})
    assert {r.doc_id for r in res} == {2, 3}


def test_rank_unknown_term_and_duplicates(idx):
    assert rank(idx, ["zzz"], BM25()) == []
    assert rank(idx, ["rare", "rare"], BM25()) == rank(idx, ["rare"], BM25())


def test_registry():
    assert set(SCORERS) == {"tfidf", "bm25"}
