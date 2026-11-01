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
"""
import argparse
import logging
import re
import time
from dataclasses import dataclass
from pathlib import Path

from findex.cache import cached_evaluator
from findex.decorators import timed
from findex.ranking import SearchResult, rank
from findex.scoring import SCORERS
from findex.search import merge_and, merge_diff, merge_or
from findex.snippets import make_snippet
from findex.storage import open_index
from findex.tokenizer import tokenize



# ======================= вузли дерева =======================

class Node:
    """Спільна база: оператори й evaluate. Самі вузли - dataclass-и."""

    def __and__(self, other):
        return And(self, other)

    def __or__(self, other):
        return Or(self, other)

    def __invert__(self):
        return Not(self)

    def evaluate(self, index) -> list[int]:
        """Відсортований список doc_id, що задовольняють вузол."""
        raise NotImplementedError


@dataclass(frozen=True)
class Term(Node):
    text: str

    def evaluate(self, index):
        return [p.doc_id for p in index.postings.get(self.text, ())]

    def __str__(self):
        return self.text


@dataclass(frozen=True)
class Phrase(Node):
    terms: tuple

    def evaluate(self, index):
        if not index.has_positions:
            raise ValueError("фрази потребують індексу з позиціями "
                             "(побудуйте його з прапорцем --positions)")
        first, *rest = self.terms
        docs = Term(first).evaluate(index)
        for t in rest:
            docs = merge_and(docs, Term(t).evaluate(index))
        out = []
        for doc_id in docs:
            starts = set(index.positions[first][doc_id])
            for offset, t in enumerate(rest, 1):
                there = set(index.positions[t][doc_id])
                starts = {s for s in starts if s + offset in there}
                if not starts:
                    break
            if starts:
                out.append(doc_id)
        return out

    def __str__(self):
        return '"' + " ".join(self.terms) + '"'


@dataclass(frozen=True)
class And(Node):
    left: Node
    right: Node

    def evaluate(self, index):
        return merge_and(self.left.evaluate(index), self.right.evaluate(index))

    def __str__(self):
        return f"({self.left} AND {self.right})"


@dataclass(frozen=True)
class Or(Node):
    left: Node
    right: Node

    def evaluate(self, index):
        return merge_or(self.left.evaluate(index), self.right.evaluate(index))

    def __str__(self):
        return f"({self.left} OR {self.right})"


@dataclass(frozen=True)
class Not(Node):
    child: Node

    def evaluate(self, index):
        return merge_diff(range(index.num_docs), self.child.evaluate(index))

    def __str__(self):
        return f"NOT {self.child}"


def positive_terms(node) -> list[str]:
    """Терміни, що дають внесок у рейтинг: усе, що не під NOT."""
    if isinstance(node, Term):
        return [node.text]
    if isinstance(node, Phrase):
        return list(node.terms)
    if isinstance(node, (And, Or)):
        return positive_terms(node.left) + positive_terms(node.right)
    return []


# ======================= лексер =======================

_LEX_RE = re.compile(r'"[^"]*"|\(|\)|"|[^\s()"]+')
_OPS = {"AND", "OR", "NOT"}


def _lex(query: str):
    for raw in _LEX_RE.findall(query):
        if raw == '"':
            raise ValueError("незакриті лапки")
        if raw == "(":
            yield ("LPAREN", raw)
        elif raw == ")":
            yield ("RPAREN", raw)
        elif raw.startswith('"'):
            terms = tuple(tokenize(raw[1:-1]))
            if not terms:
                raise ValueError("порожня фраза")
            yield ("TERM", terms[0]) if len(terms) == 1 else ("PHRASE", terms)
        elif raw in _OPS:
            yield ("OP", raw)
        else:
            for tok in tokenize(raw):   # слово може дати 0, 1 або кілька токенів
                yield ("TERM", tok)


# ======================= рекурсивний спуск =======================

class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        tok = self.peek()
        self.i += 1
        return tok

    def parse(self):
        if not self.toks:
            raise ValueError("порожній запит")
        node = self.or_expr()
        if self.peek() is not None:
            raise ValueError("зайва закриваюча дужка" if self.peek()[0] == "RPAREN"
                             else f"несподіваний елемент: {self.peek()[1]}")
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.peek() == ("OP", "OR"):
            self.next()
            node = Or(node, self.and_expr())
        return node

    def _starts_operand(self, tok):
        return tok is not None and (
            tok[0] in ("TERM", "PHRASE", "LPAREN") or tok == ("OP", "NOT"))

    def and_expr(self):
        node = self.not_expr()
        while True:
            tok = self.peek()
            if tok == ("OP", "AND"):
                self.next()
                node = And(node, self.not_expr())
            elif self._starts_operand(tok):          # неявний AND
                node = And(node, self.not_expr())
            else:
                return node

    def not_expr(self):
        if self.peek() == ("OP", "NOT"):
            self.next()
            return Not(self.not_expr())
        return self.atom()

    def atom(self):
        tok = self.next()
        if tok is None:
            raise ValueError("запит закінчився несподівано")
        kind, value = tok
        if kind == "TERM":
            return Term(value)
        if kind == "PHRASE":
            return Phrase(value)
        if kind == "LPAREN":
            node = self.or_expr()
            if self.next() != ("RPAREN", ")"):
                raise ValueError("незакрита дужка")
            return node
        raise ValueError(f"несподіваний елемент: {value}")


def parse(query: str) -> Node:
    return _Parser(list(_lex(query))).parse()


# ======================= пошук + рейтинг =======================

@timed
def ranked_search(index, query: str, scorer, k: int = 10, evaluator=None):
    """Boolean-фільтр визначає, ЩО знайдено; scorer - у якому порядку показати.

    evaluator - необов'язковий кешований обчислювач дерева (cached_evaluator).
    Бал додають лише терміни поза NOT. Якщо таких немає (запит `NOT x`),
    документи повертаються з балом 0 за порядком doc_id.
    """
    tree = parse(query)
    ids = evaluator(tree) if evaluator else tree.evaluate(index)
    terms = positive_terms(tree)
    if not terms:
        return tree, len(ids), [SearchResult(0.0, d, index.docs[d].name) for d in ids[:k]]
    return tree, len(ids), rank(index, terms, scorer, k, candidates=set(ids))


def main():
    logging.basicConfig(level=logging.INFO, format="  [лог] %(message)s")
    parser = argparse.ArgumentParser(description="Ранжований Boolean-пошук")
    parser.add_argument("index", type=Path)
    parser.add_argument("queries", nargs="+",
                        help="один або кілька запитів; кеш спільний на всю сесію")
    parser.add_argument("--scorer", choices=SCORERS, default="bm25")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    scorer = SCORERS[args.scorer]()
    with open_index(args.index) as idx:
        evaluator = cached_evaluator(idx)
        for query in args.queries:
            print(f"\n> {query}")
            start = time.perf_counter()
            try:
                tree, total, results = ranked_search(idx, query, scorer, args.k, evaluator)
            except ValueError as e:
                print(f"Помилка в запиті: {e}")
                continue
            elapsed = time.perf_counter() - start
            print(f"Дерево: {tree}")
            print(f"Знайдено {total} док., показано {len(results)} ({scorer}, {elapsed * 1000:.3f} мс)")
            terms = positive_terms(tree)
            for i, r in enumerate(results, 1):
                print(f"{i:>2}. {r.score:7.3f}  {r.name}  ({idx.docs[r.doc_id].length} токенів)")
                print(f"      {make_snippet(idx, r.doc_id, terms).text}")
        print(f"\nКеш: {evaluator.cache_info()}")


if __name__ == "__main__":
    main()