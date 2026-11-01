import pytest

from findex.evaluation import LABELED, evaluate, precision_at_k
from findex.index import build_index
from findex.scoring import BM25


def test_precision_at_k_basic():
    assert precision_at_k(["a", "b", "c", "d", "e"], {"a", "c"}, 5) == 0.4
    assert precision_at_k(["A", "b"], {"a"}, 5) == 0.2          # регістр не важливий; ділимо на k
    assert precision_at_k([], {"a"}, 5) == 0.0
    assert precision_at_k(["a", "b", "c", "d", "e", "f"], {"f"}, 5) == 0.0   # лише топ-k


def test_labeled_set_is_well_formed():
    assert len(LABELED) == 10
    for query, relevant in LABELED:
        assert query and relevant
        assert all(n == n.lower() for n in relevant)             # імена малими літерами


def test_evaluate_on_small_corpus(tmp_path):
    (tmp_path / "doc_a.txt").write_text("wealth " * 5 + "filler " * 20, encoding="utf-8")
    (tmp_path / "doc_b.txt").write_text("wealth filler " + "other " * 30, encoding="utf-8")
    (tmp_path / "doc_c.txt").write_text("nothing here " * 10, encoding="utf-8")
    idx = build_index(tmp_path)
    rows = evaluate(idx, [("wealth", {"doc_a", "doc_b"})], BM25(), k=5)
    query, rel, names, p, total = rows[0]
    assert names[0] == "doc_a" and set(names) == {"doc_a", "doc_b"}
    assert p == pytest.approx(0.4) and total == 2


def test_evaluate_rejects_unknown_document(tmp_path):
    (tmp_path / "a.txt").write_text("x", encoding="utf-8")
    idx = build_index(tmp_path)
    with pytest.raises(ValueError, match="нема в індексі"):
        evaluate(idx, [("x", {"typo_doc"})], BM25())