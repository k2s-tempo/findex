"""
iter_documents(): генератор, що читає корпус документ за документом.

Правило: 1 файл Project Gutenberg (.txt) = 1 документ.
З кожного файлу відрізається стандартна преамбула/постамбула Gutenberg
(ліцензійний текст), залишається тільки основний текст твору.
"""
from pathlib import Path
from dataclasses import dataclass
import re

START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str


def _strip_boilerplate(raw_text: str) -> str:
    """Прибирає ліцензійну преамбулу й постамбулу Gutenberg, лишає сам текст."""
    start_match = START_RE.search(raw_text)
    end_match = END_RE.search(raw_text)

    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)

    return raw_text[start_idx:end_idx].strip()


def iter_documents(corpus_dir: Path):
    """
    Ліниво читає всі .txt файли з corpus_dir по одному.
    Не завантажує весь корпус у пам'ять одночасно —
    кожен наступний файл читається тільки коли до нього доходить черга.
    """
    for path in sorted(corpus_dir.rglob("*.txt")):
        raw_text = path.read_text(encoding="utf-8")
        clean_text = _strip_boilerplate(raw_text)
        yield Document(doc_id=path.stem, path=path, text=clean_text)