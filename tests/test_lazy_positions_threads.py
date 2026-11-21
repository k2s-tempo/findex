"""Ліниві позиції під багатьма потоками: веб-сервіс шукає в пулі потоків."""

import sys
import threading
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.lazy_positions import LazyPositions
from findex.storage import load_array, save_array

WORDS = ["alpha", "beta", "gamma", "delta", "epsilon", "zeta", "eta"]


@pytest.fixture
def small(tmp_path: Path) -> tuple[Index, Path]:
    for i in range(6):
        text = " ".join(WORDS[(i + j) % len(WORDS)] for j in range(40))
        (tmp_path / f"d{i}.txt").write_text(text, encoding="utf-8")
    index = build_index(tmp_path, with_positions=True)
    path = tmp_path / "idx.bin"
    save_array(index, path)
    return index, path


def test_concurrent_access_with_tiny_cache(small: tuple[Index, Path]) -> None:
    """Маленький кеш змушує потоки постійно витісняти записи один в одного; перемикання між потоками
    робимо максимально частим, щоб вікно гонки (вставка + видалення найстарішого) справді відкривалось."""
    index, path = small
    positions = load_array(path).positions
    assert isinstance(positions, LazyPositions)
    assert index.positions is not None
    positions._cache_limit = 2
    terms = list(positions)
    expected = {t: index.positions[t] for t in terms}
    errors: list[Exception] = []

    def work() -> None:
        try:
            for _ in range(200):
                for term in terms:
                    assert positions[term] == expected[term]
        except Exception as exc:
            errors.append(exc)

    old = sys.getswitchinterval()
    sys.setswitchinterval(1e-6)
    try:
        threads = [threading.Thread(target=work) for _ in range(8)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    finally:
        sys.setswitchinterval(old)
    assert not errors, errors[:1]
    assert len(positions._cache) <= 2
