import pytest

from findex.index import build_index
from findex.storage import load, save


@pytest.fixture
def index(tmp_path_factory):
    d = tmp_path_factory.mktemp("corpus")
    (d / "a.txt").write_text("apple banana apple", encoding="utf-8")
    (d / "b.txt").write_text("banana cherry", encoding="utf-8")
    (d / "c.txt").write_text("яблуко banana", encoding="utf-8")
    return build_index(d, with_positions=True)


@pytest.mark.parametrize("fmt", ["pickle", "array"])
def test_roundtrip(tmp_path, index, fmt):
    out = tmp_path / "idx"
    save(index, out, fmt)
    loaded = load(out)  # формат визначається автоматично
    assert loaded.postings == index.postings
    assert loaded.docs == index.docs


def test_pickle_keeps_positions(tmp_path, index):
    save(index, tmp_path / "i.pkl", "pickle")
    assert load(tmp_path / "i.pkl").positions == index.positions


def test_array_drops_positions(tmp_path, index):
    save(index, tmp_path / "i.bin", "array")
    assert load(tmp_path / "i.bin").positions is None


def test_array_rejects_foreign_file(tmp_path):
    bad = tmp_path / "bad.bin"
    bad.write_bytes(b"not an index at all")
    with pytest.raises(ValueError):
        load(bad, "array")