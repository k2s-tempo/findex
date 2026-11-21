"""Позиції, що читаються з файлу ліниво (LazyPositions), поводяться як звичайний словник."""

import pickle
from pathlib import Path

import pytest

from findex.index import Index, build_index
from findex.lazy_positions import LazyPositions
from findex.query import ranked_search
from findex.scoring import BM25
from findex.storage import load_array, save_array

CORPUS = Path(__file__).resolve().parent.parent / "corpus"


@pytest.fixture
def small(tmp_path: Path) -> tuple[Index, Path]:
    (tmp_path / "a.txt").write_text("the soul of the city. the soul is one", encoding="utf-8")
    (tmp_path / "b.txt").write_text("a city and the soul", encoding="utf-8")
    index = build_index(tmp_path, with_positions=True)
    path = tmp_path / "idx.bin"
    save_array(index, path)
    return index, path


def test_loaded_positions_are_lazy_and_equal_to_dict(small: tuple[Index, Path]) -> None:
    index, path = small
    loaded = load_array(path)
    assert isinstance(loaded.positions, LazyPositions)
    assert loaded.positions == index.positions
    assert index.positions == loaded.positions
    assert loaded.positions is not None
    assert index.positions is not None
    assert loaded.positions["soul"] == index.positions["soul"]
    assert loaded.has_positions


def test_mapping_protocol(small: tuple[Index, Path]) -> None:
    index, path = small
    positions = load_array(path).positions
    assert positions is not None
    assert index.positions is not None
    assert len(positions) == len(index.positions)
    assert set(positions) == set(index.positions)
    assert "soul" in positions
    assert "nope" not in positions
    with pytest.raises(KeyError):
        positions["nope"]
    assert positions.get("nope") is None


def test_resave_of_loaded_index_is_byte_identical(small: tuple[Index, Path], tmp_path: Path) -> None:
    _, path = small
    again = tmp_path / "again.bin"
    save_array(load_array(path), again)
    assert again.read_bytes() == path.read_bytes()


def test_phrase_query_gives_same_result(small: tuple[Index, Path]) -> None:
    index, path = small
    loaded = load_array(path)
    for query in ['"the soul"', '"soul of the"', '"city and"', '"zzz yyy"']:
        built = ranked_search(index, query, BM25(), 10)
        lazy = ranked_search(loaded, query, BM25(), 10)
        assert built[1] == lazy[1]
        assert [r.doc_id for r in built[2]] == [r.doc_id for r in lazy[2]]


def test_cache_is_bounded_and_evicted_terms_are_recomputed(small: tuple[Index, Path]) -> None:
    index, path = small
    positions = load_array(path).positions
    assert isinstance(positions, LazyPositions)
    assert index.positions is not None
    positions._cache_limit = 1
    first = positions["soul"]
    assert positions["soul"] is first  # з кешу
    positions["city"]
    assert len(positions._cache) == 1
    assert positions["soul"] == index.positions["soul"]  # витіснений, але розгортається знову


def test_pickle_roundtrip_of_loaded_index(small: tuple[Index, Path]) -> None:
    index, path = small
    restored = pickle.loads(pickle.dumps(load_array(path)))
    assert restored.positions == index.positions


def test_close_drops_positions(small: tuple[Index, Path]) -> None:
    _, path = small
    loaded = load_array(path)
    loaded.close()
    assert loaded.positions is None


def test_index_without_positions_has_none(tmp_path: Path) -> None:
    (tmp_path / "a.txt").write_text("one two three", encoding="utf-8")
    path = tmp_path / "idx.bin"
    save_array(build_index(tmp_path, with_positions=False), path)
    assert load_array(path).positions is None


@pytest.mark.skipif(not CORPUS.is_dir(), reason="немає каталогу corpus/")
def test_phrase_search_on_corpus_same_as_in_memory(tmp_path: Path) -> None:
    index = build_index(CORPUS, with_positions=True)
    path = tmp_path / "corpus.bin"
    save_array(index, path)
    loaded = load_array(path)
    for query in ['"the soul"', '"justice and virtue"', '"of the"']:
        built = ranked_search(index, query, BM25(), 20)
        lazy = ranked_search(loaded, query, BM25(), 20)
        assert built[1] == lazy[1]
        assert [(r.doc_id, r.score) for r in built[2]] == [(r.doc_id, r.score) for r in lazy[2]]
