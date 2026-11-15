import logging

import pytest

from findex.cache import cached_evaluator
from findex.index import build_index
from findex.query import parse, ranked_search
from findex.scoring import BM25


@pytest.fixture
def idx(tmp_path):
    (tmp_path / "d0.txt").write_text("alpha beta", encoding="utf-8")
    (tmp_path / "d1.txt").write_text("beta gamma", encoding="utf-8")
    (tmp_path / "d2.txt").write_text("alpha gamma", encoding="utf-8")
    return build_index(tmp_path, with_positions=True)


def test_miss_then_hit_in_log(idx, caplog):
    ev = cached_evaluator(idx)
    with caplog.at_level(logging.INFO, logger="findex"):
        first = ev(parse("alpha beta"))
        second = ev(parse("alpha beta"))
    assert first == second == (0,)
    messages = [r.getMessage() for r in caplog.records]
    assert messages[0].startswith("cache MISS") and messages[1].startswith("cache HIT")
    info = ev.cache_info()
    assert (info.hits, info.misses) == (1, 1)


def test_equal_trees_share_one_entry(idx):
    ev = cached_evaluator(idx)
    for q in ("alpha beta", "ALPHA  BETA", "alpha AND beta"):
        ev(parse(q))
    info = ev.cache_info()
    assert info.misses == 1 and info.hits == 2 and info.currsize == 1


def test_result_is_immutable_tuple(idx):
    result = cached_evaluator(idx)(parse("alpha"))
    assert isinstance(result, tuple) and result == (0, 2)


def test_maxsize_evicts_oldest(idx):
    ev = cached_evaluator(idx, maxsize=1)
    ev(parse("alpha"))
    ev(parse("beta"))
    ev(parse("alpha"))
    assert ev.cache_info().misses == 3 and ev.cache_info().hits == 0


def test_cache_cleared_on_index_close(idx):
    ev = cached_evaluator(idx)
    ev(parse("alpha"))
    assert ev.cache_info().currsize == 1
    idx.close()
    assert ev.cache_info().currsize == 0


def test_ranked_search_same_with_and_without_cache(idx):
    ev = cached_evaluator(idx)
    plain = ranked_search(idx, "alpha OR beta", BM25(), 5)
    cached = ranked_search(idx, "alpha OR beta", BM25(), 5, ev)
    again = ranked_search(idx, "alpha OR beta", BM25(), 5, ev)
    assert plain == cached == again
    assert ev.cache_info().hits == 1
