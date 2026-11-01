"""
Кеш відповідей: дерево запиту -> кортеж id документів.

Ключ кешу - САМЕ ДЕРЕВО, а не рядок запиту. Вузли - frozen dataclass-и, тобто
хешовані й порівнюються за значенням, тож "a b", "A  B" і "a AND b" дають
рівні дерева й один запис у кеші. Значення - кортеж, а не список: lru_cache
повертає той самий об'єкт, і змінюваний список можна було б зіпсувати ззовні.
"""
import logging
from functools import lru_cache

log = logging.getLogger("findex")


def cached_evaluator(index, maxsize: int = 128):
    """Повертає evaluate(tree) -> tuple[int, ...] з lru_cache для цього індексу.

    Кеш створюється в замиканні для конкретного індексу (Index хешується за
    ідентичністю, тож класти його в ключ невигідно), а при index.close()
    очищується через зареєстровану on_close-дію.
    """
    @lru_cache(maxsize=maxsize)
    def _cached(tree):
        log.info("cache MISS: %s", tree)        # тіло виконується лише при промаху
        return tuple(tree.evaluate(index))

    def evaluate(tree):
        hits_before = _cached.cache_info().hits
        ids = _cached(tree)
        if _cached.cache_info().hits > hits_before:
            log.info("cache HIT:  %s", tree)
        return ids

    evaluate.cache_info = _cached.cache_info
    evaluate.cache_clear = _cached.cache_clear
    index.on_close(_cached.cache_clear)
    return evaluate