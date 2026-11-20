"""Повільні інтеграційні тести (> 1 с): пропустити командою  pytest -m "not slow"."""

import random

import pytest

from findex.index import build_index
from findex.query import parse
from findex.scoring import BM25
from findex.storage import load, save

pytestmark = pytest.mark.slow


def test_large_synthetic_corpus_roundtrip_and_engines(make_corpus, tmp_path):
    rng = random.Random(7)
    vocab = [f"w{i}" for i in range(2500)]
    weights = [1 / (rank + 1) for rank in range(len(vocab))]  # розподіл, схожий на закон Ціпфа
    docs = {
        f"doc{i:03d}.txt": " ".join(rng.choices(vocab, weights, k=rng.randint(800, 1800))) for i in range(300)
    }
    index = build_index(make_corpus(docs), with_positions=True)
    assert index.num_docs == 300

    path = tmp_path / "big.bin"
    save(index, path, "array")
    loaded = load(path)
    assert loaded.postings == index.postings and loaded.positions == index.positions

    for _ in range(25):
        a, b, c = (f"w{rng.randint(0, 400)}" for _ in range(3))
        for query in (
            f"{a} {b}",
            f"{a} OR {b}",
            f"{a} NOT {b}",
            f'"{a} {b}" OR {c}',
            f"{a} AND ({b} OR {c})",
        ):
            tree = parse(query)
            assert tree.evaluate(loaded, "merge") == tree.evaluate(loaded, "set") == tree.evaluate(index)
    assert BM25().score(3, 10, 1000, loaded) > 0
