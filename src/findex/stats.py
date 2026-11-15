"""
Статистика корпусу за ОДИН прохід (лаба 1) і допоміжний замір часу та пікової пам'яті.
"""
import time
import tracemalloc
from collections import Counter
from collections.abc import Callable
from pathlib import Path
from typing import TypedDict, TypeVar

from findex.corpus import iter_documents
from findex.tokenizer import tokenize

A = TypeVar("A")
R = TypeVar("R")


class CorpusStats(TypedDict):
    doc_count: int
    token_count: int
    vocab_size: int
    top_50: list[tuple[str, int]]


def compute_stats_lazy(corpus_dir: Path) -> CorpusStats:
    """Лінива версія: документи й токени обробляються по одному.

    Єдина структура, що росте, - Counter (результат задачі, а не копія вхідних даних).
    """
    doc_count = 0
    token_count = 0
    term_freq: Counter[str] = Counter()
    for doc in iter_documents(corpus_dir):     # генератор - 1 документ за раз
        doc_count += 1
        for token in tokenize(doc.text):        # генератор - 1 токен за раз
            token_count += 1
            term_freq[token] += 1
    return {
        "doc_count": doc_count,
        "token_count": token_count,
        "vocab_size": len(term_freq),
        "top_50": term_freq.most_common(50),
    }


def run_with_measurement(arg: A, compute_fn: Callable[[A], R]) -> tuple[R, float, int]:
    """Запускає compute_fn(arg) і повертає (результат, секунди, піковий tracemalloc у байтах)."""
    tracemalloc.start()
    start_time = time.perf_counter()
    result = compute_fn(arg)
    elapsed = time.perf_counter() - start_time
    _current, peak = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    return result, elapsed, peak
