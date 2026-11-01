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
from contextlib import contextmanager
from pathlib import Path

from findex.decorators import timed
from findex.index import DocMeta, Index, Posting

MAGIC_V1 = b"FDX1"
MAGIC_V2 = b"FDX2"
FLAG_POSITIONS = 1
BIG_ENDIAN = sys.byteorder == "big"
assert array("I").itemsize == 4, "очікується 4-байтовий array('I')"


# ---------- pickle ----------

def save_pickle(index: Index, path) -> None:
    with open(path, "wb") as f:
        pickle.dump(index, f, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path) -> Index:
    # УВАГА: pickle.load НЕБЕЗПЕЧНИЙ для чужих файлів. Pickle - це
    # програма для віртуальної машини pickle: файл може містити виклик
    # будь-якої функції (через __reduce__), наприклад os.system(...),
    # і вона виконається під час load() ще до того, як ми щось
    # перевіримо. Завантажуйте лише файли, які створили самі.
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------- бінарний формат на array ----------

def _pack_str(s: str) -> bytes:
    b = s.encode("utf-8")
    return struct.pack("<I", len(b)) + b


def _u32_bytes(values) -> bytes:
    a = array("I", values)
    if BIG_ENDIAN:
        a.byteswap()
    return a.tobytes()


def save_array(index: Index, path) -> None:
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
            if flags & FLAG_POSITIONS:
                per_doc = index.positions[term]
                f.write(_u32_bytes(pos for p in plist for pos in per_doc[p.doc_id]))


class _Reader:
    def __init__(self, buf: bytes):
        self.buf = buf
        self.pos = 0

    def u32(self) -> int:
        (v,) = struct.unpack_from("<I", self.buf, self.pos)
        self.pos += 4
        return v

    def string(self) -> str:
        n = self.u32()
        s = self.buf[self.pos:self.pos + n].decode("utf-8")
        self.pos += n
        return s

    def u32_array(self, n: int) -> array:
        a = array("I")
        a.frombytes(self.buf[self.pos:self.pos + 4 * n])
        self.pos += 4 * n
        if BIG_ENDIAN:
            a.byteswap()
        return a


def load_array(path) -> Index:
    r = _Reader(Path(path).read_bytes())
    magic = r.buf[:4]
    if magic not in (MAGIC_V1, MAGIC_V2):
        raise ValueError("це не файл індексу findex (невірна сигнатура)")
    r.pos = 4
    n_docs = r.u32()
    n_terms = r.u32()
    flags = r.u32() if magic == MAGIC_V2 else 0
    has_positions = bool(flags & FLAG_POSITIONS)

    docs = []
    for doc_id in range(n_docs):
        length = r.u32()
        name = r.string()
        doc_path = r.string()
        docs.append(DocMeta(doc_id, name, doc_path, length))

    postings = {}
    positions = {} if has_positions else None
    for _ in range(n_terms):
        term = r.string()
        n = r.u32()
        ids = r.u32_array(n)
        tfs = r.u32_array(n)
        postings[term] = [Posting(d, t) for d, t in zip(ids, tfs)]
        if has_positions:
            flat = r.u32_array(sum(tfs))
            per_doc, offset = {}, 0
            for d, t in zip(ids, tfs):
                per_doc[d] = flat[offset:offset + t].tolist()
                offset += t
            positions[term] = per_doc

    return Index(postings, docs, positions)


# ---------- єдина точка входу ----------

def save(index: Index, path, fmt: str = "array") -> None:
    {"pickle": save_pickle, "array": save_array}[fmt](index, path)


@timed
def load(path, fmt: str = "auto") -> Index:
    if fmt == "auto":
        with open(path, "rb") as f:
            fmt = "array" if f.read(4) in (MAGIC_V1, MAGIC_V2) else "pickle"
    return load_array(path) if fmt == "array" else load_pickle(path)


@contextmanager
def open_index(path, fmt: str = "auto"):
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
