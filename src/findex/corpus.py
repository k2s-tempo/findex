"""
iter_documents(): генератор, що читає корпус документ за документом.

Правило: 1 файл Project Gutenberg (.txt) = 1 документ. З кожного файлу
відрізається стандартна преамбула/постамбула Gutenberg (ліцензійний текст).
"""

import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path

START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str


def strip_boilerplate(raw_text: str) -> str:
    """Прибирає ліцензійну преамбулу й постамбулу Gutenberg, лишає сам текст."""
    start_match = START_RE.search(raw_text)
    end_match = END_RE.search(raw_text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)
    return raw_text[start_idx:end_idx].strip()


_strip_boilerplate = strip_boilerplate  # старе ім'я: його імпортують dev-скрипти в scripts/


def list_documents(corpus_dir: Path) -> list[Path]:
    """Шляхи всіх .txt з corpus_dir (з підпапками) у відсортованому порядку.

    Позиція шляху в цьому списку це doc_id документа в індексі.
    """
    return sorted(corpus_dir.rglob("*.txt"))


def read_document(path: Path) -> Document:
    """Читає один файл і відрізає шапку та підвал Gutenberg."""
    raw_text = path.read_text(encoding="utf-8")
    return Document(doc_id=path.stem, path=path, text=strip_boilerplate(raw_text))


def iter_documents(corpus_dir: Path) -> Iterator[Document]:
    """Ліниво читає всі .txt з corpus_dir (з підпапками) по одному, у відсортованому порядку."""
    for path in list_documents(corpus_dir):
        yield read_document(path)
