"""Порівняння rank (цикл Python, лаба 3) і rank_fast (NumPy, лаба 8). Друкує таблицю в Markdown."""

import random
import sys
import time
from collections.abc import Callable

from findex.arrays import rank_fast
from findex.index import DocMeta, Index, Posting
from findex.ranking import rank
from findex.scoring import BM25, Scorer
from findex.storage import load


def best_ms(fn: Callable[[], object], repeat: int = 7) -> float:
    loops = 1
    while True:  # стільки викликів у серії, щоб вона тривала не менше 50 мс
        start = time.perf_counter()
        for _ in range(loops):
            fn()
        if time.perf_counter() - start >= 0.05:
            break
        loops *= 2
    best = float("inf")
    for _ in range(repeat):
        start = time.perf_counter()
        for _ in range(loops):
            fn()
        best = min(best, (time.perf_counter() - start) / loops)
    return best * 1000


def row(label: str, index: Index, terms: list[str], scorer: Scorer, k: int = 10) -> None:
    slow = rank(index, terms, scorer, k)
    cold = time.perf_counter()
    fast = rank_fast(index, terms, scorer, k)  # перший виклик: конвертація постінгів у масиви
    cold_ms = (time.perf_counter() - cold) * 1000
    assert [r.doc_id for r in fast] == [r.doc_id for r in slow], "порядок розійшовся"
    py = best_ms(lambda: rank(index, terms, scorer, k))
    nump = best_ms(lambda: rank_fast(index, terms, scorer, k))
    postings = sum(index.df(t) for t in terms)
    print(f"| {label} | {postings:,} | {py:.3f} | {nump:.3f} | {py / nump:.1f}x | {cold_ms:.2f} |".replace(",", " "))


def synthetic(n_docs: int) -> Index:
    rng = random.Random(1)
    docs = [DocMeta(i, f"d{i}", f"d{i}.txt", rng.randint(50, 5000)) for i in range(n_docs)]
    postings = {
        "common": [Posting(i, rng.randint(1, 20)) for i in range(n_docs)],
        "mid": [Posting(i, rng.randint(1, 5)) for i in range(0, n_docs, 10)],
        "rare": [Posting(i, 1) for i in range(0, n_docs, max(1, n_docs // 40))],
    }
    return Index(postings, docs, None)


def main() -> None:
    path = sys.argv[1] if len(sys.argv) > 1 else "index_big.bin"
    scorer = BM25()
    index = load(path, "array")
    by_df = sorted(index, key=index.df)
    rare, frequent = by_df[0], by_df[-1]
    print(f"Корпус {path}: {index.num_docs} док., {len(index)} термінів. Кращий із 7 серій, мс на виклик.\n")
    print("| Запит | Постінгів | Python, мс | NumPy, мс | Прискорення | Перший виклик NumPy, мс |")
    print("| --- | --- | --- | --- | --- | --- |")
    row(f"частий терм `{frequent}`", index, [frequent], scorer)
    row(f"рідкісний терм `{rare}`", index, [rare], scorer)
    row("три терми `justice virtue courage`", index, ["justice", "virtue", "courage"], scorer)
    for n in (20_000, 200_000):
        big = synthetic(n)
        row(f"синтетичний, {n:,} док.: `common`".replace(",", " "), big, ["common"], scorer)
        row(f"синтетичний, {n:,} док.: `common mid rare`".replace(",", " "), big, ["common", "mid", "rare"], scorer)


main()
