"""
Спільні фікстури: маленький корпус, на якому будуються індекс і CLI-тести.

Тести можуть перевизначати фікстуру з тим самим іменем у своєму файлі (локальна має пріоритет).
"""
from collections.abc import Callable
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.storage import save

SAMPLE_DOCS: dict[str, str] = {
    "meno.txt": (
        "SOCRATES: Can virtue be taught? Is justice a part of virtue, and is the soul immortal?\n"
        "MENO: Virtue is the power of getting good with justice."
    ),
    "crito.txt": (
        "SOCRATES: My dear Crito, justice and the soul matter more than escape.\n"
        "CRITO: [Aside] Think of your sons! The [bold] opinion of the many."
    ),
    "ion.txt": "ION: Poetry and rhapsody; the poet is a winged thing.",
}


@pytest.fixture
def make_corpus(tmp_path: Path) -> Callable[[dict[str, str]], Path]:
    """Фабрика: {ім'я файлу: текст} -> каталог із такими файлами."""

    def factory(docs: dict[str, str]) -> Path:
        directory = tmp_path / "corpus"
        directory.mkdir(exist_ok=True)
        for name, text in docs.items():
            (directory / name).write_text(text, encoding="utf-8")
        return directory

    return factory


@pytest.fixture
def corpus(make_corpus: Callable[[dict[str, str]], Path]) -> Path:
    """Корпус із трьох коротких документів (meno, crito, ion)."""
    return make_corpus(SAMPLE_DOCS)


@pytest.fixture
def small_index(corpus: Path) -> Index:
    return build_index(corpus)


@pytest.fixture
def small_index_with_positions(corpus: Path) -> Index:
    return build_index(corpus, with_positions=True)


@pytest.fixture
def index_file(corpus: Path, tmp_path: Path) -> Path:
    """Збережений бінарний індекс (з позиціями) для CLI-тестів."""
    out = tmp_path / "index.bin"
    save(build_index(corpus, with_positions=True), out, "array")
    return out
