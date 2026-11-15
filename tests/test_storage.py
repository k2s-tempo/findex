import struct

import pytest

from findex.index import build_index
from findex.storage import load, open_index, save


@pytest.fixture
def index(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    (d / "a.txt").write_text("apple banana apple", encoding="utf-8")
    (d / "b.txt").write_text("banana cherry", encoding="utf-8")
    (d / "c.txt").write_text("яблуко banana", encoding="utf-8")
    return build_index(d, with_positions=True)


@pytest.fixture
def index_no_pos(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus2")
    (d / "a.txt").write_text("apple banana apple", encoding="utf-8")
    (d / "b.txt").write_text("banana cherry", encoding="utf-8")
    return build_index(d)


@pytest.mark.parametrize("fmt", ["pickle", "array"])
def test_roundtrip(tmp_path, index, fmt):
    out = tmp_path / "idx"
    save(index, out, fmt)
    loaded = load(out)
    assert loaded.postings == index.postings
    assert loaded.docs == index.docs
    assert loaded.positions == index.positions  # позиції тепер зберігають обидва формати


def test_array_without_positions(tmp_path, index_no_pos):
    save(index_no_pos, tmp_path / "i.bin", "array")
    loaded = load(tmp_path / "i.bin")
    assert loaded.positions is None
    assert loaded.postings == index_no_pos.postings


def test_array_reads_old_v1_format(tmp_path, index_no_pos):
    """Файли, збережені в лабі 2 (FDX1), досі читаються."""
    path = tmp_path / "old.bin"
    with open(path, "wb") as f:
        f.write(b"FDX1")
        f.write(struct.pack("<II", len(index_no_pos.docs), len(index_no_pos.postings)))
        for d in index_no_pos.docs:
            f.write(struct.pack("<I", d.length))
            for s in (d.name, d.path):
                b = s.encode()
                f.write(struct.pack("<I", len(b)) + b)
        for term, plist in index_no_pos.postings.items():
            b = term.encode()
            f.write(struct.pack("<I", len(b)) + b)
            f.write(struct.pack("<I", len(plist)))
            f.write(struct.pack(f"<{len(plist)}I", *[p.doc_id for p in plist]))
            f.write(struct.pack(f"<{len(plist)}I", *[p.tf for p in plist]))
    loaded = load(path)
    assert loaded.postings == index_no_pos.postings
    assert loaded.positions is None


def test_array_rejects_foreign_file(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not an index at all")
    with pytest.raises(ValueError):
        load(bad, "array")


def test_open_index_closes_on_exit(tmp_path, index):
    save(index, tmp_path / "i.bin", "array")
    with open_index(tmp_path / "i.bin") as idx:
        assert "banana" in idx
        assert not idx.closed
    assert idx.closed


def test_open_index_closes_on_exception(tmp_path, index):
    save(index, tmp_path / "i.bin", "array")
    holder = {}
    with pytest.raises(RuntimeError), open_index(tmp_path / "i.bin") as idx:
        holder["idx"] = idx
        raise RuntimeError("boom")
    assert holder["idx"].closed
