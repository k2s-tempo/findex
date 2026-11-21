"""
Мова запитів: лексер + рекурсивний спуск -> дерево вузлів Term/Phrase/And/Or/Not.

Синтаксис:
    python AND (async OR await) NOT java "event loop"

    * AND, OR, NOT - лише ВЕЛИКИМИ літерами (малі "and"/"or" це звичайні терміни);
    * сусідні операнди без оператора означають AND;
    * пріоритет: NOT > AND > OR; дужки змінюють порядок;
    * "..." - фраза: слова мають іти підряд (потрібен індекс із позиціями);
      пунктуація не є токеном, тому "event, loop" теж збігається з "event loop".

Граматика:
    or_expr  := and_expr ( OR and_expr )*
    and_expr := not_expr ( [AND] not_expr )*
    not_expr := NOT not_expr | atom
    atom     := TERM | PHRASE | "(" or_expr ")"

Вузли - frozen dataclass-и: порівнюються за значенням (a == b), хешуються, а також
підтримують оператори Python: Term("a") & ~Term("b") | Term("c").
Виконання (evaluate) працює у двох режимах: merge (два вказівники) і set (множини).
"""

import re
from abc import ABC, abstractmethod
from collections.abc import Callable, Iterator, Sequence
from dataclasses import dataclass
from typing import Literal

from findex.decorators import timed
from findex.index import Index
from findex.merge import Engine, difference, intersect, union
from findex.ranking import SearchResult, rank
from findex.scoring import Scorer
from findex.tokenizer import tokenize


class EmptyQueryError(ValueError):
    """Запит не містить жодного терміна або оператора."""


# ======================= вузли дерева =======================


class Node(ABC):
    """Спільна база: оператори й evaluate. Самі вузли - dataclass-и."""

    def __and__(self, other: "Node") -> "And":
        return And(self, other)

    def __or__(self, other: "Node") -> "Or":
        return Or(self, other)

    def __invert__(self) -> "Not":
        return Not(self)

    @abstractmethod
    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        """Відсортований список doc_id, що задовольняють вузол."""


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return [p.doc_id for p in index.postings.get(self.text, ())]

    def __str__(self) -> str:
        return self.text


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple[str, ...]

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        positions = index.positions
        if positions is None:
            raise ValueError("фрази потребують індексу з позиціями (побудуйте його з прапорцем --positions)")
        first, *rest = self.terms
        docs = Term(first).evaluate(index, engine)
        for t in rest:
            docs = intersect(docs, Term(t).evaluate(index, engine), engine)
        out: list[int] = []
        for doc_id in docs:
            starts = set(positions[first][doc_id])
            for offset, t in enumerate(rest, 1):
                there = set(positions[t][doc_id])
                starts = {s for s in starts if s + offset in there}
                if not starts:
                    break
            if starts:
                out.append(doc_id)
        return out

    def __str__(self) -> str:
        return '"' + " ".join(self.terms) + '"'


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return intersect(self.left.evaluate(index, engine), self.right.evaluate(index, engine), engine)

    def __str__(self) -> str:
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return union(self.left.evaluate(index, engine), self.right.evaluate(index, engine), engine)

    def __str__(self) -> str:
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index: Index, engine: Engine = "merge") -> list[int]:
        return difference(range(index.num_docs), self.child.evaluate(index, engine), engine)

    def __str__(self) -> str:
        return f"NOT {self.child}"


def positive_terms(node: Node) -> list[str]:
    """Терміни, що дають внесок у рейтинг: усе, що не під NOT."""
    if isinstance(node, Term):
        return [node.text]
    if isinstance(node, Phrase):
        return list(node.terms)
    if isinstance(node, (And, Or)):
        return positive_terms(node.left) + positive_terms(node.right)
    return []


# ======================= лексер =======================


@dataclass(frozen=True)
class _Tok:
    kind: Literal["LPAREN", "RPAREN", "TERM", "PHRASE", "OP"]
    text: str = ""
    terms: tuple[str, ...] = ()


_LEX_RE = re.compile(r'"[^"]*"|\(|\)|"|[^\s()"]+')
_OPS = {"AND", "OR", "NOT"}


def _lex(query: str) -> Iterator[_Tok]:
    for match in _LEX_RE.finditer(query):
        raw = match.group()
        if raw == '"':
            raise ValueError("незакриті лапки")
        if raw == "(":
            yield _Tok("LPAREN", raw)
        elif raw == ")":
            yield _Tok("RPAREN", raw)
        elif raw.startswith('"'):
            terms = tuple(tokenize(raw[1:-1]))
            if not terms:
                raise ValueError("порожня фраза")
            yield _Tok("TERM", terms[0]) if len(terms) == 1 else _Tok("PHRASE", terms=terms)
        elif raw in _OPS:
            yield _Tok("OP", raw)
        else:
            for tok in tokenize(raw):  # слово може дати 0, 1 або кілька токенів
                yield _Tok("TERM", tok)


# ======================= рекурсивний спуск =======================


class _Parser:
    def __init__(self, tokens: list[_Tok]) -> None:
        self.toks = tokens
        self.i = 0

    def peek(self) -> _Tok | None:
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self) -> _Tok | None:
        tok = self.peek()
        self.i += 1
        return tok

    def _peek_op(self, name: str) -> bool:
        tok = self.peek()
        return tok is not None and tok.kind == "OP" and tok.text == name

    def parse(self) -> Node:
        if not self.toks:
            raise EmptyQueryError("порожній запит")
        node = self.or_expr()
        leftover = self.peek()
        if leftover is not None:
            raise ValueError(
                "зайва закриваюча дужка"
                if leftover.kind == "RPAREN"
                else f"несподіваний елемент: {leftover.text}"
            )
        return node

    def or_expr(self) -> Node:
        node = self.and_expr()
        while self._peek_op("OR"):
            self.next()
            node = Or(node, self.and_expr())
        return node

    def _starts_operand(self, tok: _Tok | None) -> bool:
        return tok is not None and (
            tok.kind in ("TERM", "PHRASE", "LPAREN") or (tok.kind == "OP" and tok.text == "NOT")
        )

    def and_expr(self) -> Node:
        node = self.not_expr()
        while True:
            if self._peek_op("AND"):
                self.next()
                node = And(node, self.not_expr())
            elif self._starts_operand(self.peek()):  # неявний AND
                node = And(node, self.not_expr())
            else:
                return node

    def not_expr(self) -> Node:
        if self._peek_op("NOT"):
            self.next()
            return Not(self.not_expr())
        return self.atom()

    def atom(self) -> Node:
        tok = self.next()
        if tok is None:
            raise ValueError("запит закінчився несподівано")
        if tok.kind == "TERM":
            return Term(tok.text)
        if tok.kind == "PHRASE":
            return Phrase(tok.terms)
        if tok.kind == "LPAREN":
            node = self.or_expr()
            closing = self.next()
            if closing is None or closing.kind != "RPAREN":
                raise ValueError("незакрита дужка")
            return node
        raise ValueError(f"несподіваний елемент: {tok.text}")


def parse(query: str) -> Node:
    return _Parser(list(_lex(query))).parse()


# ======================= пошук + рейтинг =======================


@timed
def ranked_search(
    index: Index,
    query: str,
    scorer: Scorer,
    k: int = 10,
    evaluator: Callable[[Node], Sequence[int]] | None = None,
    engine: Engine = "merge",
    *,
    fast: bool = True,
) -> tuple[Node, int, list[SearchResult]]:
    """Boolean-фільтр визначає, ЩО знайдено; scorer - у якому порядку показати.

    evaluator - необов'язковий кешований обчислювач дерева (CachedEvaluator).
    Бал додають лише терміни поза NOT. Якщо таких немає (запит `NOT x`),
    документи повертаються з балом 0 за порядком doc_id.
    """
    tree = parse(query)
    ids = evaluator(tree) if evaluator is not None else tree.evaluate(index, engine)
    terms = positive_terms(tree)
    if not terms:
        return tree, len(ids), [SearchResult(0.0, d, index.docs[d].name) for d in ids[:k]]
    if fast:
        from findex.arrays import rank_fast  # NumPy вантажимо лише коли справді шукаємо

        return tree, len(ids), rank_fast(index, terms, scorer, k, candidates=ids)
    return tree, len(ids), rank(index, terms, scorer, k, candidates=set(ids))
