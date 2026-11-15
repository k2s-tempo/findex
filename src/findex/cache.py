"""
Кеш відповідей: дерево запиту -> кортеж id документів.

Ключ кешу - САМЕ ДЕРЕВО, а не рядок запиту. Вузли - frozen dataclass-и, тобто
хешовані й порівнюються за значенням, тож "a b", "A  B" і "a AND b" дають
рівні дерева й один запис у кеші. Значення - кортеж, а не список: lru_cache
повертає той самий об'єкт, і змінюваний список можна було б зіпсувати ззовні.
"""

import logging
from functools import lru_cache

from findex.index import Index
from findex.query import Node

log = logging.getLogger("findex")


class CachedEvaluator:
    """evaluate(tree) -> tuple[int, ...] з lru_cache для конкретного індексу.

    Кеш прив'язаний до індексу (Index хешується за ідентичністю, тож класти його
    в ключ невигідно), а при index.close() очищується через on_close-дію.
    """

    def __init__(self, index: Index, maxsize: int = 128) -> None:
        self._index = index
        self._cached = lru_cache(maxsize=maxsize)(self._compute)
        index.on_close(self._cached.cache_clear)

    def _compute(self, tree: Node) -> tuple[int, ...]:
        log.info("cache MISS: %s", tree)  # тіло виконується лише при промаху
        return tuple(tree.evaluate(self._index))

    def __call__(self, tree: Node) -> tuple[int, ...]:
        hits_before = self._cached.cache_info().hits
        ids = self._cached(tree)
        if self._cached.cache_info().hits > hits_before:
            log.info("cache HIT:  %s", tree)
        return ids

    def cache_info(self):  # повертає functools._CacheInfo (приватний тип, тож без анотації)
        return self._cached.cache_info()

    def cache_clear(self) -> None:
        self._cached.cache_clear()


def cached_evaluator(index: Index, maxsize: int = 128) -> CachedEvaluator:
    return CachedEvaluator(index, maxsize)
