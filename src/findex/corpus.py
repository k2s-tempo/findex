import json
import re
from collections.abc import Iterator
from dataclasses import dataclass
from pathlib import Path
from typing import cast

START_RE = re.compile(r"\*\*\*\s*START OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)
END_RE = re.compile(r"\*\r*\*\s*END OF THE PROJECT GUTENBERG EBOOK.*?\*\*\*", re.IGNORECASE)


@dataclass
class Document:
    doc_id: str
    path: Path
    text: str


def strip_boilerplate(raw_text: str) -> str:
    start_match = START_RE.search(raw_text)
    end_match = END_RE.search(raw_text)
    start_idx = start_match.end() if start_match else 0
    end_idx = end_match.start() if end_match else len(raw_text)
    return raw_text[start_idx:end_idx].strip()


_strip_boilerplate = strip_boilerplate


def list_documents(corpus_dir: Path) -> list[Path]:
    return sorted(corpus_dir.rglob("*.txt"))


def read_document(path: Path) -> Document:
    raw_text = path.read_text(encoding="utf-8")
    return Document(doc_id=path.stem, path=path, text=strip_boilerplate(raw_text))


def iter_jsonl_documents(jsonl_path: Path) -> Iterator[Document]:
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line_no, raw_line in enumerate(f, start=1):
            line = raw_line.strip()
            if not line:
                continue
            raw_record: object = json.loads(line)
            if not isinstance(raw_record, dict):
                raise ValueError(f"{jsonl_path}:{line_no}: expected JSON object")
            record = cast(dict[str, object], raw_record)
            url = str(record.get("url", f"page-{line_no}"))
            title = str(record.get("title", "")).strip()
            body = str(record.get("text", "")).strip()
            full_text = f"{title}\n{body}".strip() if title else body
            yield Document(doc_id=url, path=jsonl_path, text=full_text)


def iter_documents(source: Path) -> Iterator[Document]:
    if source.is_file() and source.suffix.lower() == ".jsonl":
        yield from iter_jsonl_documents(source)
        return
    for path in list_documents(source):
        yield read_document(path)
