"""
Збереження й завантаження індексу.

Два формати:
  pickle - стандартна серіалізація Python (зберігає й позиції)
  array  - власний бінарний формат на array('I') + struct (без позицій)

Формат array (little-endian):
  b"FDX1" | n_docs u32 | n_terms u32
  n_docs  x [length u32 | name (u32 len + utf-8) | path (u32 len + utf-8)]
  n_terms x [term (u32 len + utf-8) | n u32 | doc_ids n*u32 | tfs n*u32]
doc_id документа = його порядковий номер у блоці документів.
"""
import pickle
import struct
import sys
from array import array
from pathlib import Path

from findex.index import DocMeta, Index, Posting

MAGIC = b"FDX1"
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


def save_array(index: Index, path) -> None:
    with open(path, "wb") as f:
        f.write(MAGIC)
        f.write(struct.pack("<II", len(index.docs), len(index.postings)))
        for d in index.docs:
            f.write(struct.pack("<I", d.length))
            f.write(_pack_str(d.name))
            f.write(_pack_str(d.path))
        for term, plist in index.postings.items():
            ids = array("I", (p.doc_id for p in plist))
            tfs = array("I", (p.tf for p in plist))
            if BIG_ENDIAN:
                ids.byteswap()
                tfs.byteswap()
            f.write(_pack_str(term))
            f.write(struct.pack("<I", len(plist)))
            f.write(ids.tobytes())
            f.write(tfs.tobytes())


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
    if r.buf[:4] != MAGIC:
        raise ValueError("це не файл індексу findex (невірний magic)")
    r.pos = 4
    n_docs = r.u32()
    n_terms = r.u32()

    docs = []
    for doc_id in range(n_docs):
        length = r.u32()
        name = r.string()
        doc_path = r.string()
        docs.append(DocMeta(doc_id, name, doc_path, length))

    postings = {}
    for _ in range(n_terms):
        term = r.string()
        n = r.u32()
        ids = r.u32_array(n)
        tfs = r.u32_array(n)
        postings[term] = [Posting(d, t) for d, t in zip(ids, tfs)]

    return Index(postings, docs, None)


# ---------- єдина точка входу ----------

def save(index: Index, path, fmt: str = "array") -> None:
    {"pickle": save_pickle, "array": save_array}[fmt](index, path)


def load(path, fmt: str = "auto") -> Index:
    if fmt == "auto":
        with open(path, "rb") as f:
            fmt = "array" if f.read(4) == MAGIC else "pickle"
    return load_array(path) if fmt == "array" else load_pickle(path)