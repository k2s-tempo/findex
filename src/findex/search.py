"""
Boolean-пошук по інвертованому індексу.

Граматика (NOT > AND > OR; сусідні терміни = неявний AND):
    expr     := and_expr (OR and_expr)*
    and_expr := not_expr ((AND)? not_expr)*
    not_expr := NOT not_expr | TERM
Терміни нормалізуються тим самим tokenize(), що й корпус.

Два рушії: merge (списки doc_id + два вказівники) і set (множини).
"""
import argparse
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable
from findex.stats import run_with_measurement
from findex.storage import load

from findex.index import Index, build_index
from findex.tokenizer import tokenize

OPS = {"AND", "OR", "NOT"}


# ---------- розбір запиту ----------

def _lex(query: str):
    for word in query.split():
        if word in OPS:
            yield ("OP", word)
        else:
            for token in tokenize(word):
                yield ("TERM", token)


class _Parser:
    def __init__(self, tokens):
        self.toks = tokens
        self.i = 0

    def peek(self):
        return self.toks[self.i] if self.i < len(self.toks) else None

    def next(self):
        t = self.peek()
        self.i += 1
        return t

    def parse(self):
        if not self.toks:
            return None
        node = self.or_expr()
        if self.peek() is not None:
            raise ValueError(f"зайвий елемент у запиті: {self.peek()[1]}")
        return node

    def or_expr(self):
        node = self.and_expr()
        while self.peek() == ("OP", "OR"):
            self.next()
            node = ("or", node, self.and_expr())
        return node

    def and_expr(self):
        node = self.not_expr()
        while True:
            t = self.peek()
            if t is None or t == ("OP", "OR"):
                break
            if t == ("OP", "AND"):
                self.next()
            node = ("and", node, self.not_expr())
        return node

    def not_expr(self):
        t = self.next()
        if t is None:
            raise ValueError("запит закінчився несподівано")
        if t == ("OP", "NOT"):
            return ("not", self.not_expr())
        if t[0] == "TERM":
            return ("term", t[1])
        raise ValueError(f"несподіваний оператор {t[1]}")


def parse(query: str):
    return _Parser(list(_lex(query))).parse()


# ---------- merge-рушій: відсортовані списки doc_id ----------

def merge_and(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            i += 1
        else:
            j += 1
    return out


def merge_or(a, b):
    out, i, j = [], 0, 0
    while i < len(a) and j < len(b):
        if a[i] == b[j]:
            out.append(a[i])
            i += 1
            j += 1
        elif a[i] < b[j]:
            out.append(a[i])
            i += 1
        else:
            out.append(b[j])
            j += 1
    out.extend(a[i:])
    out.extend(b[j:])
    return out


def merge_diff(a, b):
    """Елементи a, яких немає в b."""
    out, i, j = [], 0, 0
    while i < len(a):
        if j >= len(b) or a[i] < b[j]:
            out.append(a[i])
            i += 1
        elif a[i] == b[j]:
            i += 1
            j += 1
        else:
            j += 1
    return out


# ---------- вибір рушія ----------

@dataclass(frozen=True)
class Engine:
    from_postings: Callable
    and_: Callable
    or_: Callable
    not_: Callable  # (a, n_docs)
    to_list: Callable


MERGE = Engine(
    from_postings=lambda plist: [p.doc_id for p in plist],
    and_=merge_and,
    or_=merge_or,
    not_=lambda a, n: merge_diff(range(n), a),
    to_list=lambda a: a,
)

SET = Engine(
    from_postings=lambda plist: {p.doc_id for p in plist},
    and_=lambda a, b: a & b,
    or_=lambda a, b: a | b,
    not_=lambda a, n: set(range(n)) - a,
    to_list=sorted,
)

ENGINES = {"merge": MERGE, "set": SET}


def _eval(node, index: Index, eng: Engine):
    kind = node[0]
    if kind == "term":
        return eng.from_postings(index.postings.get(node[1], []))
    if kind == "not":
        return eng.not_(_eval(node[1], index, eng), len(index.docs))
    left = _eval(node[1], index, eng)
    right = _eval(node[2], index, eng)
    return eng.and_(left, right) if kind == "and" else eng.or_(left, right)


def search(index: Index, query: str, engine: str = "merge") -> list[int]:
    """Повертає відсортований список doc_id, що задовольняють запит."""
    ast = parse(query)
    if ast is None:
        return []
    eng = ENGINES[engine]
    return eng.to_list(_eval(ast, index, eng))


def main():
    parser = argparse.ArgumentParser(description="Boolean-пошук по індексу")
    parser.add_argument("index", type=Path,
                        help="файл індексу (напр. index.bin) або папка з корпусом")
    parser.add_argument("query")
    parser.add_argument("--engine", choices=ENGINES, default="merge")
    args = parser.parse_args()

    if args.index.is_dir():
        index, load_time, peak = run_with_measurement(args.index, build_index)
        print(f"Індекс побудовано з корпусу: {load_time:.3f} с, "
              f"пік {peak / 1024 / 1024:.2f} МБ")
    else:
        index, load_time, peak = run_with_measurement(args.index, load)
        print(f"Індекс завантажено: {load_time:.3f} с, "
              f"пік {peak / 1024 / 1024:.2f} МБ")

    start = time.perf_counter()
    hits = search(index, args.query, args.engine)
    elapsed = time.perf_counter() - start

    print(f"Знайдено документів: {len(hits)} ({args.engine}, {elapsed * 1000:.3f} мс)")
    for doc_id in hits[:20]:
        d = index.docs[doc_id]
        print(f"  {d.name}  ({d.length} токенів)")
    if len(hits) > 20:
        print(f"  ... і ще {len(hits) - 20}")


if __name__ == "__main__":
    main()