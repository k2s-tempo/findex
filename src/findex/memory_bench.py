"""
Скільки пам'яті займають постінги в трьох (+1 бонусному) варіантах
зберігання. Міряємо утримувану пам'ять (tracemalloc, current до/після).
"""
import argparse
import gc
import sys
import tracemalloc
from array import array
from dataclasses import dataclass
from pathlib import Path

from findex.index import Posting, build_index


@dataclass
class PlainPosting:
    """Звичайний dataclass: кожен екземпляр може мати __dict__."""
    doc_id: int
    tf: int


def make_source(corpus_dir: Path) -> dict:
    """term -> (array doc_ids, array tfs). Компактне джерело для всіх варіантів."""
    index = build_index(corpus_dir)
    return {
        term: (array("I", (p.doc_id for p in plist)),
               array("I", (p.tf for p in plist)))
        for term, plist in index.postings.items()
    }


def as_plain(src):
    return {t: [PlainPosting(d, f) for d, f in zip(ids, tfs)]
            for t, (ids, tfs) in src.items()}


def as_slots(src):
    return {t: [Posting(d, f) for d, f in zip(ids, tfs)]
            for t, (ids, tfs) in src.items()}


def as_arrays(src):
    return {t: (ids[:], tfs[:]) for t, (ids, tfs) in src.items()}


def as_flat(src):
    """Бонус: усі терміни в двох великих масивах + (початок, довжина) на термін."""
    ids_all, tfs_all, offsets = array("I"), array("I"), {}
    for t, (ids, tfs) in src.items():
        offsets[t] = (len(ids_all), len(ids))
        ids_all.extend(ids)
        tfs_all.extend(tfs)
    return offsets, ids_all, tfs_all


def measure(build, src) -> int:
    gc.collect()
    tracemalloc.start()
    base, _ = tracemalloc.get_traced_memory()
    obj = build(src)
    gc.collect()
    current, _ = tracemalloc.get_traced_memory()
    tracemalloc.stop()
    del obj
    return current - base


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("corpus", type=Path)
    args = parser.parse_args()

    src = make_source(args.corpus)
    n_terms = len(src)
    n_post = sum(len(ids) for ids, _ in src.values())
    single = sum(1 for ids, _ in src.values() if len(ids) == 1)
    print(f"Термінів: {n_terms}, постінгів: {n_post}, "
          f"термінів з одним постінгом: {single}\n")

    print("Розміри окремих об'єктів (байт, sys.getsizeof):")
    pl = PlainPosting(1, 2)
    print(f"  екземпляр PlainPosting: {sys.getsizeof(pl)}, "
          f"його __dict__: {sys.getsizeof(pl.__dict__)}")
    print(f"  екземпляр Posting (slots): {sys.getsizeof(Posting(1, 2))}")
    print(f"  array('I') з 1 елемента: {sys.getsizeof(array('I', [1]))}")
    print(f"  list з 1 елемента: {sys.getsizeof([0])}")
    print(f"  int 1000: {sys.getsizeof(1000)}\n")

    variants = [
        ("dataclass (без slots)", as_plain),
        ("dataclass slots=True", as_slots),
        ("пари array('I') на термін", as_arrays),
        ("бонус: плоскі array + зміщення", as_flat),
    ]
    print("| Варіант | Пам'ять, МБ | Байт на постінг |")
    print("|---|---|---|")
    for name, fn in variants:
        size = measure(fn, src)
        print(f"| {name} | {size / 1024 / 1024:.2f} | {size / n_post:.1f} |")


if __name__ == "__main__":
    main()