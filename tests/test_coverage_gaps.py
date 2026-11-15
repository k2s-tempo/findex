"""
Тести для місць, які звіт покриття показав як непокриті:
cache.py:45, cli.py:81-82 і 324->326, index.py:79, snippets.py:89->91 і 94->97, storage.py:56, 70, 114.
"""
import pickle
from importlib import metadata

import pytest
from typer.testing import CliRunner

from findex import storage
from findex.cache import cached_evaluator
from findex.cli import app
from findex.index import Index, build_index
from findex.query import parse
from findex.snippets import snippet_from_text
from findex.storage import load, load_pickle, save

runner = CliRunner()


# ---------- cache.py:45 ----------

def test_cache_clear_empties_cache(corpus):
    ev = cached_evaluator(build_index(corpus))
    ev(parse("virtue"))
    assert ev.cache_info().currsize == 1
    ev.cache_clear()
    assert ev.cache_info().currsize == 0


# ---------- index.py:79 ----------

def test_avg_doc_length_of_empty_index_is_zero():
    assert Index({}, []).avg_doc_length == 0.0


# ---------- snippets.py:89->91 і 94->97 ----------

def test_snippet_left_edge_inside_one_long_word():
    """Ліворуч від збігу немає пробілів: початок вікна лишається де є."""
    raw = "a" * 200 + ",soul"
    snip = snippet_from_text(raw, {"soul"}, radius=80)
    assert snip.text.startswith("...") and "**soul**" in snip.text


def test_snippet_right_edge_inside_one_long_word():
    """Праворуч від збігу немає пробілів: кінець вікна лишається де є."""
    raw = "soul," + "b" * 200
    snip = snippet_from_text(raw, {"soul"}, radius=80)
    assert snip.text.endswith("...") and "**soul**" in snip.text


# ---------- storage.py:56 ----------

def test_load_pickle_rejects_foreign_object(tmp_path):
    path = tmp_path / "not_an_index.pkl"
    with open(path, "wb") as f:
        pickle.dump({"це": "не індекс"}, f)
    with pytest.raises(ValueError, match="не містить індексу"):
        load_pickle(path)


# ---------- storage.py:70 і 114 ----------

def test_array_roundtrip_with_byteswap_branches(corpus, tmp_path, monkeypatch):
    """Вмикаємо гілки BIG_ENDIAN. Це перевіряє, що byteswap при записі й читанні
    взаємно обернені, але не заміняє прогін на справжній big-endian машині."""
    monkeypatch.setattr(storage, "BIG_ENDIAN", True)
    index = build_index(corpus, with_positions=True)
    out = tmp_path / "swapped.bin"
    save(index, out, "array")
    loaded = load(out)
    assert loaded.postings == index.postings
    assert loaded.docs == index.docs
    assert loaded.positions == index.positions


# ---------- cli.py:81-82 ----------

def test_version_when_package_not_installed(monkeypatch):
    def boom(name):
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(metadata, "version", boom)
    result = runner.invoke(app, ["--version"])
    assert result.exit_code == 0
    assert "версію не визначено" in result.stdout


# ---------- cli.py:324->326 ----------

def test_stats_of_index_without_documents(tmp_path):
    path = tmp_path / "empty.bin"
    save(Index({}, []), path, "array")
    result = runner.invoke(app, ["stats", str(path)])
    assert result.exit_code == 0
    assert "Документів" in result.stdout
    assert "Найкоротший" not in result.stdout
