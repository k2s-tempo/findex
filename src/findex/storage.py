"""
Збереження й завантаження індексу.

  pickle - стандартна серіалізація Python (зберігає й позиції)
  array  - власний бінарний формат на array('I') + struct

Формат array FDX2 (little-endian):
  b"FDX2" | n_docs u32 | n_terms u32 | flags u32 (біт 0 = є позиції)
  n_docs  x [length u32 | name (u32 len + utf-8) | path (u32 len + utf-8)]
  n_terms x [term (u32 len + utf-8) | n u32 | doc_ids n*u32 | tfs n*u32
             | якщо є позиції: sum(tfs)*u32 позицій]
Позиції йдуть підряд у порядку постінгів; скільки їх у кожного постінга,
відомо з його tf, тому окремі лічильники не потрібні.
Старий формат FDX1 (без позицій і без flags) теж читається.
"""

import pickle
import struct
import sys
from array import array
from collections.abc import Generator, Iterable
from contextlib import contextmanager
from os import PathLike
from pathlib import Path
from typing import Literal

from findex.decorators import timed
from findex.index import DocMeta, Index, Posting
from findex.lazy_positions import LazyPositions

MAGIC_V1 = b"FDX1"
MAGIC_V2 = b"FDX2"
FLAG_POSITIONS = 1
BIG_ENDIAN = sys.byteorder == "big"
assert array("I").itemsize == 4, "очікується 4-байтовий array('I')"

PathArg = str | PathLike[str]
SaveFormat = Literal["array", "pickle"]
LoadFormat = Literal["auto", "array", "pickle"]


# ---------- pickle ----------


def save_pickle(index: Index, path: PathArg) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path: PathArg) -> Index:
    # УВАГА: pickle.load НЕБЕЗПЕЧНИЙ для чужих файлів. Pickle - це
    # програма для віртуальної машини pickle: файл може містити виклик
    # будь-якої функції (через __reduce__), наприклад os.system(...),
    # і вона виконається під час load() ще до того, як ми щось
    # перевіримо. Завантажуйте лише файли, які створили самі.
    with open(path, "rb") as f:
        obj: object = pickle.load(f)
    if not isinstance(obj, Index):
        raise ValueError("pickle-файл не містить індексу findex")
    return obj


# ---------- бінарний формат на array ----------


def _pack_str(s: str) -> bytes:
    data = s.encode("utf-8")
    return struct.pack("<I", len(data)) + data


def _u32_bytes(values: Iterable[int]) -> bytes:
    arr = array("I", values)
    if BIG_ENDIAN:
        arr.byteswap()
    return arr.tobytes()


def save_array(index: Index, path: PathArg) -> None:
    flags = FLAG_POSITIONS if index.positions is not None else 0
    with open(path, "wb") as f:
        f.write(MAGIC_V2)
        f.write(struct.pack("<III", len(index.docs), len(index.postings), flags))
        for d in index.docs:
            f.write(struct.pack("<I", d.length))
            f.write(_pack_str(d.name))
            f.write(_pack_str(d.path))
        for term, plist in index.postings.items():
            f.write(_pack_str(term))
            f.write(struct.pack("<I", len(plist)))
            f.write(_u32_bytes(p.doc_id for p in plist))
            f.write(_u32_bytes(p.tf for p in plist))
            if index.positions is not None:
                per_doc = index.positions[term]
                f.write(_u32_bytes(pos for p in plist for pos in per_doc[p.doc_id]))


class _Reader:
    def __init__(self, buf: bytes) -> None:
        self.buf = buf
        self.pos = 0

    def u32(self) -> int:
        value: int = struct.unpack_from("<I", self.buf, self.pos)[0]
        self.pos += 4
        return value

    def string(self) -> str:
        n = self.u32()
        s = self.buf[self.pos : self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def u32_array(self, n: int) -> "array[int]":
        arr: array[int] = array("I")
        arr.frombytes(self.buf[self.pos : self.pos + 4 * n])
        self.pos += 4 * n
        if BIG_ENDIAN:
            arr.byteswap()
        return arr


def load_array(path: PathArg) -> Index:
    reader = _Reader(Path(path).read_bytes())
    magic = reader.buf[:4]
    if magic not in (MAGIC_V1, MAGIC_V2):
        raise ValueError("це не файл індексу findex (невірна сигнатура)")
    reader.pos = 4
    n_docs = reader.u32()
    n_terms = reader.u32()
    flags = reader.u32() if magic == MAGIC_V2 else 0
    has_positions = bool(flags & FLAG_POSITIONS)

    docs: list[DocMeta] = []
    for doc_id in range(n_docs):
        length = reader.u32()
        name = reader.string()
        doc_path = reader.string()
        docs.append(DocMeta(doc_id, name, doc_path, length))

    postings: dict[str, list[Posting]] = {}
    entries: dict[str, tuple[array[int], array[int], int]] = {}
    for _ in range(n_terms):
        term = reader.string()
        n = reader.u32()
        ids = reader.u32_array(n)
        tfs = reader.u32_array(n)
        postings[term] = [Posting(d, t) for d, t in zip(ids, tfs, strict=True)]
        if has_positions:
            entries[term] = (ids, tfs, reader.pos)  # позиції лишаються в буфері: розгортаємо на вимогу
            reader.pos += 4 * sum(tfs)

    return Index(
        postings, docs, LazyPositions(reader.buf, entries, byteswap=BIG_ENDIAN) if has_positions else None
    )


# ---------- єдина точка входу ----------


def save(index: Index, path: PathArg, fmt: SaveFormat = "array") -> None:
    if fmt == "pickle":
        save_pickle(index, path)
    else:
        save_array(index, path)


@timed
def load(path: PathArg, fmt: LoadFormat = "auto") -> Index:
    if fmt == "auto":
        with open(path, "rb") as f:
            fmt = "array" if f.read(4) in (MAGIC_V1, MAGIC_V2) else "pickle"
    return load_array(path) if fmt == "array" else load_pickle(path)


@contextmanager
def open_index(path: PathArg, fmt: LoadFormat = "auto") -> Generator[Index, None, None]:
    """Контекстний менеджер: завантажує індекс і гарантовано закриває його.

        with open_index("index.bin") as idx:
            ...

    Закриття (idx.close()) відбувається і при винятку всередині with:
    звільняє пам'ять індексу й виконує зареєстровані on_close-дії (кеші).
    """
    index = load(path, fmt)
    try:
        yield index
    finally:
        index.close()
