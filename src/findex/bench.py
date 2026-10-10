"""
Бенчмарк merge vs set: (1) на реальних термінах корпусу
(два найчастіші, два найрідкісніші), (2) на синтетичних списках,
щоб побачити масштабування, якого 23 документи не покажуть.
"""
import argparse
import random
import timeit
from pathlib import Path

from findex.index import build_index
from findex.search import ENGINES, _eval, merge_and, parse
from findex.storage import load


def pick_terms(index):
    tf_total = {t: sum(p.tf for p in pl) for t, pl in index.postings.items()}
    frequent = [t for t, _ in sorted(tf_total.items(),
                                     key=lambda kv: (-kv[1], kv[0]))[:2]]
    rare = sorted(t for t, pl in index.postings.items()
                  if len(pl) == 1 and tf_total[t] == 1)[:2]
    return frequent, rare


def time_query(index, query, engine, repeat=7, number=2000):
    ast = parse(query)
    eng = ENGINES[engine]
    best = min(timeit.repeat(lambda: eng.to_list(_eval(ast, index, eng)),
                             repeat=repeat, number=number))
    return best / number * 1e6  # мкс


def real_terms(index):
    (f1, f2), (r1, r2) = pick_terms(index)
    df = lambda t: len(index.postings[t])
    print(f"Найчастіші: {f1} (df={df(f1)}), {f2} (df={df(f2)})")
    print(f"Найрідкісніші: {r1} (df={df(r1)}), {r2} (df={df(r2)})\n")
    queries = [
        f"{f1} AND {f2}", f"{f1} OR {f2}", f"{f1} AND {r1}",
        f"{r1} OR {r2}", f"{r1} AND {r2}", f"{f1} NOT {r1}",
    ]
    print("| Запит | Знайдено | merge, мкс | set, мкс |")
    print("|---|---|---|---|")
    for q in queries:
        res_m = ENGINES["merge"].to_list(_eval(parse(q), index, ENGINES["merge"]))
        res_s = ENGINES["set"].to_list(_eval(parse(q), index, ENGINES["set"]))
        assert res_m == res_s, q
        print(f"| {q} | {len(res_m)} | {time_query(index, q, 'merge'):.2f} "
              f"| {time_query(index, q, 'set'):.2f} |")


def synthetic():
    random.seed(1)
    print("\nСинтетичні списки (AND двох відсортованих списків):")
    print("| Розмір кожного списку | merge, мс | set, мс |")
    print("|---|---|---|")
    for n in (1_000, 100_000, 1_000_000):
        a = sorted(random.sample(range(n * 3), n))
        b = sorted(random.sample(range(n * 3), n))
        number = max(1, 100_000 // n)
        tm = min(timeit.repeat(lambda: merge_and(a, b), repeat=3, number=number))
        ts = min(timeit.repeat(lambda: sorted(set(a) & set(b)), repeat=3, number=number))
        print(f"| {n:,} | {tm / number * 1000:.3f} | {ts / number * 1000:.3f} |")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="index.bin або папка з корпусом")
    args = parser.parse_args()
    index = build_index(args.source) if args.source.is_dir() else load(args.source)
    real_terms(index)
    synthetic()


if __name__ == "__main__":
    main()