"""Властивості (Hypothesis): інваріанти, що мають триматися для будь-яких вхідних даних."""

import tempfile
from pathlib import Path

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from findex.index import DocMeta, Index, Posting, build_index
from findex.merge import (
    difference,
    intersect,
    merge_and,
    merge_diff,
    merge_or,
    set_and,
    set_diff,
    set_or,
    union,
)
from findex.query import And, Node, Not, Or, Phrase, Term, parse
from findex.storage import load, save
from findex.tokenizer import tokenize

SETTINGS = settings(
    max_examples=100, deadline=None
)  # deadline=None: файловий ввід-вивід нестабільний за часом


# ---------- стратегії ----------

TERMS = st.text(alphabet=st.characters(categories=["Ll"]), min_size=1, max_size=8)


@st.composite
def indexes(draw: st.DrawFn, with_positions: bool | None = None) -> Index:
    """Випадковий коректний індекс: відсортовані постінги, узгоджені tf і позиції."""
    n_docs = draw(st.integers(1, 5))
    terms = draw(st.lists(TERMS, unique=True, max_size=6))
    positional = draw(st.booleans()) if with_positions is None else with_positions
    postings: dict[str, list[Posting]] = {}
    positions: dict[str, dict[int, list[int]]] = {}
    for term in terms:
        doc_ids = sorted(draw(st.lists(st.integers(0, n_docs - 1), unique=True, min_size=1)))
        postings[term] = []
        positions[term] = {}
        for doc_id in doc_ids:
            if positional:
                found = sorted(draw(st.lists(st.integers(0, 40), unique=True, min_size=1, max_size=4)))
                positions[term][doc_id] = found
                postings[term].append(Posting(doc_id, len(found)))
            else:
                postings[term].append(Posting(doc_id, draw(st.integers(1, 5))))
    docs = [DocMeta(i, f"doc{i}", f"p{i}.txt", draw(st.integers(0, 100))) for i in range(n_docs)]
    return Index(postings, docs, positions if positional else None)


LEAVES = st.one_of(
    st.builds(Term, st.text(alphabet="abcdef", min_size=1, max_size=4)),
    st.builds(
        Phrase,
        st.lists(st.text(alphabet="abcdef", min_size=1, max_size=3), min_size=2, max_size=3).map(tuple),
    ),
)
TREES = st.recursive(
    LEAVES,
    lambda children: st.one_of(
        st.builds(Not, children),
        st.builds(And, children, children),
        st.builds(Or, children, children),
    ),
    max_leaves=6,
)
SORTED_IDS = st.lists(st.integers(0, 60), unique=True).map(sorted)


# ---------- властивість 1: save/load повертає той самий індекс ----------


@pytest.mark.parametrize("fmt", ["array", "pickle"])
@SETTINGS
@given(index=indexes())
def test_load_of_save_gives_same_index(fmt, index):
    with tempfile.TemporaryDirectory() as tmp:
        path = Path(tmp) / "index.bin"
        save(index, path, fmt)
        loaded = load(path)
    assert loaded.postings == index.postings
    assert loaded.docs == index.docs
    assert loaded.positions == index.positions


# ---------- властивість 2: постінги завжди відсортовані й узгоджені ----------


@pytest.mark.parametrize("with_positions", [False, True])
@SETTINGS
@given(texts=st.lists(st.text(alphabet="abcde \n", max_size=60), min_size=1, max_size=5))
def test_built_postings_are_sorted_and_consistent(with_positions, texts):
    with tempfile.TemporaryDirectory() as tmp:
        for i, text in enumerate(texts):
            (Path(tmp) / f"doc{i}.txt").write_text(text, encoding="utf-8")
        index = build_index(Path(tmp), with_positions=with_positions)

    total_tf = [0] * len(index.docs)
    for term, plist in index.postings.items():
        ids = [p.doc_id for p in plist]
        assert ids == sorted(set(ids)), term  # строго зростають, без повторів
        for p in plist:
            assert p.tf >= 1
            total_tf[p.doc_id] += p.tf
            if with_positions:
                assert index.positions is not None
                found = index.positions[term][p.doc_id]
                assert found == sorted(found) and len(found) == p.tf
    assert total_tf == [d.length for d in index.docs]  # сума tf документа = його довжина


# ---------- властивість 3: merge збігається з операціями над set ----------


@SETTINGS
@given(a=SORTED_IDS, b=SORTED_IDS)
def test_merge_matches_set_operations(a, b):
    assert merge_and(a, b) == sorted(set(a) & set(b)) == set_and(a, b)
    assert merge_or(a, b) == sorted(set(a) | set(b)) == set_or(a, b)
    assert merge_diff(a, b) == sorted(set(a) - set(b)) == set_diff(a, b)
    for engine in ("merge", "set"):
        assert intersect(a, b, engine) == merge_and(a, b)
        assert union(a, b, engine) == merge_or(a, b)
        assert difference(a, b, engine) == merge_diff(a, b)


# ---------- властивість 4: токенізатор ----------


@settings(
    max_examples=50, deadline=None
)  # довільний Unicode дорогий: менше прикладів, щоб тест лишався швидким
@given(text=st.text())
def test_tokens_have_no_whitespace_and_are_not_empty(text):
    for token in tokenize(text):
        assert token and not any(ch.isspace() for ch in token)


@SETTINGS
@given(text=st.text(alphabet="abc XYZ'-_.,!0123456789 \n", max_size=80))
def test_tokenize_is_idempotent_on_its_output(text):
    tokens = list(tokenize(text))
    assert list(tokenize(" ".join(tokens))) == tokens


# ---------- властивість 5: parse(str(дерево)) == дерево ----------


@SETTINGS
@given(tree=TREES)
def test_parse_of_str_roundtrips(tree: Node):
    assert parse(str(tree)) == tree


# ---------- властивість 6: обидва рушії дають однакову відповідь на будь-яке дерево ----------


@pytest.fixture(scope="module")
def fixed_index(tmp_path_factory: pytest.TempPathFactory) -> Index:
    directory = tmp_path_factory.mktemp("props")
    texts = ["a b c d", "b c a f", "d e f a b", "c c c e", "a b a b", "f e d c b a"]
    for i, text in enumerate(texts):
        (directory / f"d{i}.txt").write_text(text, encoding="utf-8")
    return build_index(directory, with_positions=True)


@SETTINGS
@given(tree=TREES)
def test_engines_agree_on_any_tree(fixed_index: Index, tree: Node):
    assert tree.evaluate(fixed_index, "merge") == tree.evaluate(fixed_index, "set")
