"""Бенчмарки (pytest-benchmark): search і build_index на corpus/ та захист від уповільнення.

Звичайний pytest ганяє їх по одному разу (у pyproject стоїть --benchmark-disable), тож вони лише
перевіряють, що код працює. Повні виміри:
    pytest tests/test_benchmarks.py --benchmark-enable --benchmark-only
"""

import random
import time
from collections.abc import Callable
from pathlib import Path

import pytest
from pytest_benchmark.fixture import BenchmarkFixture

from findex.arrays import rank_fast
from findex.index import DocMeta, Index, Posting, build_index
from findex.query import ranked_search
from findex.ranking import rank
from findex.scoring import BM25

CORPUS = Path(__file__).resolve().parent.parent / "corpus"
needs_corpus = pytest.mark.skipif(not CORPUS.is_dir(), reason="немає каталогу corpus/")


@pytest.fixture(scope="module")
def corpus_index() -> Index:
    return build_index(CORPUS)


@needs_corpus
def test_bench_build_index(benchmark: BenchmarkFixture) -> None:
    index = benchmark.pedantic(build_index, args=(CORPUS,), rounds=3, iterations=1, warmup_rounds=1)
    assert index.num_docs > 0


@needs_corpus
@pytest.mark.parametrize("query", ["justice", "justice virtue courage", "justice AND (virtue OR courage)"])
def test_bench_search(benchmark: BenchmarkFixture, corpus_index: Index, query: str) -> None:
    _, total, results = benchmark(ranked_search, corpus_index, query, BM25(), 10)
    assert total > 0
    assert results


def synthetic_index(n_docs: int) -> Index:
    rng = random.Random(1)
    docs = [DocMeta(i, f"d{i}", f"d{i}.txt", rng.randint(50, 5000)) for i in range(n_docs)]
    postings = {
        "common": [Posting(i, rng.randint(1, 20)) for i in range(n_docs)],
        "mid": [Posting(i, rng.randint(1, 5)) for i in range(0, n_docs, 10)],
        "rare": [Posting(i, 1) for i in range(0, n_docs, n_docs // 40)],
    }
    return Index(postings, docs, None)


def best_seconds(fn: Callable[[], object], repeat: int = 5) -> float:
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        fn()
        best = min(best, time.perf_counter() - start)
    return best


def test_numpy_ranking_stays_much_faster_than_python_loop() -> None:
    """Захист від уповільнення, що не залежить від машини: два рушії порівнюються між собою.

    На 20 000 документів NumPy швидший у десятки разів; поріг ×5 лишає величезний запас на шум
    повільного CI, а зламаний швидкий шлях (повернення до циклу по об'єктах) він ловить одразу.
    """
    index = synthetic_index(20_000)
    scorer = BM25()
    terms = ["common", "mid", "rare"]
    rank_fast(index, terms, scorer, 10)  # перший виклик конвертує постінги в масиви
    slow = best_seconds(lambda: rank(index, terms, scorer, 10))
    fast = best_seconds(lambda: rank_fast(index, terms, scorer, 10))
    assert fast * 5 < slow, f"NumPy {fast * 1000:.2f} мс проти циклу Python {slow * 1000:.2f} мс"
