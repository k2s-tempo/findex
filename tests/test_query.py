import pytest

from findex.index import build_index
from findex.query import And, Not, Or, Phrase, Term, parse, positive_terms, ranked_search
from findex.scoring import BM25

a, b, c, d = Term("a"), Term("b"), Term("c"), Term("d")


# ---------- тести парсера: порівнюємо ДЕРЕВА, а не рядки ----------


@pytest.mark.parametrize(
    "query, tree",
    [
        ("a", a),
        ("A", a),  # нормалізація регістру терміна
        ("a b", And(a, b)),  # неявний AND
        ("a AND b", And(a, b)),
        ("a OR b", Or(a, b)),
        ("NOT a", Not(a)),
        ("a b c", And(And(a, b), c)),  # AND лівоасоціативний
        ("a OR b OR c", Or(Or(a, b), c)),
        ("a OR b c", Or(a, And(b, c))),  # AND сильніший за OR
        ("a b OR c", Or(And(a, b), c)),
        ("(a OR b) c", And(Or(a, b), c)),  # дужки змінюють пріоритет
        ("a NOT b", And(a, Not(b))),  # NOT після терміна = AND NOT
        ("NOT a b", And(Not(a), b)),  # NOT сильніший за AND
        ("NOT NOT a", Not(Not(a))),
        ("NOT (a OR b)", Not(Or(a, b))),
        ("((a))", a),
        ('"b c"', Phrase(("b", "c"))),
        ('"B  c"', Phrase(("b", "c"))),  # нормалізація всередині фрази
        ('"a"', a),  # фраза з одного слова = термін
        ("and or", And(Term("and"), Term("or"))),  # малі літери - це звичайні терміни
        ("self-driving", Term("self-driving")),  # правила токенізатора діють і в запиті
    ],
)
def test_parse_trees(query, tree):
    assert parse(query) == tree


def test_parse_task_example():
    tree = parse('python AND (async OR await) NOT java "event loop"')
    expected = And(
        And(
            And(Term("python"), Or(Term("async"), Term("await"))),
            Not(Term("java")),
        ),
        Phrase(("event", "loop")),
    )
    assert tree == expected


@pytest.mark.parametrize(
    "query",
    [
        "",
        "   ",
        "a AND",
        "AND a",
        "a OR",
        "NOT",
        "(a",
        "a)",
        "()",
        "a (b",
        '"a b',
        '""',
        "a OR OR b",
    ],
)
def test_parse_errors(query):
    with pytest.raises(ValueError):
        parse(query)


# ---------- оператори Python та рядкове представлення ----------


def test_python_operators_build_same_tree():
    assert (a & ~b | c) == Or(And(a, Not(b)), c)
    assert (a | b & c) == Or(a, And(b, c))  # пріоритети Python збігаються з нашими
    assert parse("a NOT b OR c") == (a & ~b | c)


@pytest.mark.parametrize(
    "tree",
    [
        a,
        And(a, b),
        Or(a, And(b, c)),
        Not(Or(a, b)),
        And(Or(a, b), Not(c)),
        Phrase(("x", "y")),
        And(Phrase(("x", "y")), Not(d)),
    ],
)
def test_str_roundtrip(tree):
    assert parse(str(tree)) == tree


def test_nodes_are_hashable_and_frozen():
    assert hash(And(a, b)) == hash(And(a, b))
    assert len({And(a, b), And(a, b), Or(a, b)}) == 2
    with pytest.raises(AttributeError):
        a.text = "z"


def test_positive_terms_skip_not():
    assert positive_terms(parse('x (y OR z) NOT w "p q"')) == ["x", "y", "z", "p", "q"]


# ---------- виконання на індексі ----------


@pytest.fixture
def idx(tmp_path):
    (tmp_path / "d0.txt").write_text("the event loop runs forever", encoding="utf-8")  # 0
    (tmp_path / "d1.txt").write_text("loop the event never runs", encoding="utf-8")  # 1
    (tmp_path / "d2.txt").write_text("python async event loop, python await", encoding="utf-8")  # 2
    (tmp_path / "d3.txt").write_text("java event handling", encoding="utf-8")  # 3
    (tmp_path / "d4.txt").write_text("python java", encoding="utf-8")  # 4
    return build_index(tmp_path, with_positions=True)


@pytest.mark.parametrize(
    "query, expected",
    [
        ("event", [0, 1, 2, 3]),
        ("event loop", [0, 1, 2]),
        ("python java", [4]),
        ("python OR java", [2, 3, 4]),
        ("python NOT java", [2]),
        ("NOT event", [4]),
        ("python AND (async OR await)", [2]),
        ("(python OR java) NOT async", [3, 4]),
        ("zzz", []),
        ("event NOT zzz", [0, 1, 2, 3]),
    ],
)
def test_evaluate(idx, query, expected):
    assert parse(query).evaluate(idx) == expected


@pytest.mark.parametrize(
    "query, expected",
    [
        ('"event loop"', [0, 2]),  # d1 має ті самі слова, але в іншому порядку
        ('"loop the"', [1]),
        ('"the event"', [0, 1]),
        ('"event loop runs"', [0]),
        ('"loop event"', []),  # порядок слів важливий
        ('"event handling"', [3]),
        ('"python async"', [2]),
        ('"async event" NOT java', [2]),
        ('"event loop" python', [2]),
        ('"event zzz"', []),
    ],
)
def test_phrase(idx, query, expected):
    assert parse(query).evaluate(idx) == expected


def test_phrase_ignores_punctuation(idx):
    # у d2 "loop," з комою, пунктуація - не токен
    assert parse('"loop python"').evaluate(idx) == [2]


def test_phrase_needs_positions(tmp_path):
    (tmp_path / "x.txt").write_text("a b", encoding="utf-8")
    no_pos = build_index(tmp_path)
    with pytest.raises(ValueError, match="позиці"):
        parse('"a b"').evaluate(no_pos)
    assert parse("a b").evaluate(no_pos) == [0]  # без фраз усе працює


# ---------- пошук із рейтингом ----------


def test_ranked_search(idx):
    tree, total, results = ranked_search(idx, '"event loop" NOT java', BM25(), k=5)
    assert total == 2
    assert {r.name for r in results} == {"d0", "d2"}
    assert results[0].score >= results[1].score


def test_ranked_search_only_not(idx):
    tree, total, results = ranked_search(idx, "NOT event", BM25())
    assert total == 1 and [r.name for r in results] == ["d4"] and results[0].score == 0.0


def test_ranked_search_no_match(idx):
    _, total, results = ranked_search(idx, "zzz", BM25())
    assert total == 0 and results == []
