"""
Boolean-пошук без рейтингу (лаба 2): запит -> відсортований список doc_id.

Розбір запиту й вузли живуть у findex.query, операції над списками в findex.merge;
цей модуль зберігає зручну точку входу search() з вибором рушія (Literal) та
реекспортує merge-функції під їхніми старими іменами.
"""
from findex.index import Index
from findex.merge import Engine, merge_and, merge_diff, merge_or
from findex.query import EmptyQueryError, parse

__all__ = ["Engine", "merge_and", "merge_diff", "merge_or", "search"]


def search(index: Index, query: str, engine: Engine = "merge") -> list[int]:
    """Повертає відсортований список doc_id, що задовольняють запит (порожній запит -> [])."""
    try:
        tree = parse(query)
    except EmptyQueryError:
        return []
    return tree.evaluate(index, engine)
