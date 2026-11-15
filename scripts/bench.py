"""
Бенчмарк merge проти set: (1) на реальних термінах корпусу (два найчастіші, два найрідкісніші),
(2) на синтетичних відсортованих списках, щоб побачити масштабування, якого 23 документи не покажуть.

Запуск: uv run python scripts/bench.py index.bin
"""
import argparse
import random
import timeit
from pathlib import Path

from findex.index import Index, build_index
from findex.merge import Engine, merge_and, set_and
from findex.query import parse
from findex.storage import load


def pick_terms(index: Index) -> tuple[list[str], list[str]]:
    tf_total = {t: sum(p.tf for p in pl) for t, pl in index.postings.items()}
    frequent = [t for t, _ in sorted(tf_total.items(), key=lambda kv: (-kv[1], kv[0]))[:2]]
    rare = sorted(t for t, pl in index.postings.items() if len(pl) == 1 and tf_total[t] == 1 and t.isalpha())[:2]
    return frequent, rare


def time_query(index: Index, query: str, engine: Engine, repeat: int = 7, number: int = 2000) -> float:
    tree = parse(query)
    best = min(timeit.repeat(lambda: tree.evaluate(index, engine), repeat=repeat, number=number))
    return best / number * 1e6  # мкс


def real_terms(index: Index) -> None:
    (f1, f2), (r1, r2) = pick_terms(index)
    print(f"Найчастіші: {f1} (df={index.df(f1)}), {f2} (df={index.df(f2)})")
    print(f"Найрідкісніші: {r1} (df={index.df(r1)}), {r2} (df={index.df(r2)})\n")
    queries = [f"{f1} AND {f2}", f"{f1} OR {f2}", f"{f1} AND {r1}", f"{r1} OR {r2}", f"{r1} AND {r2}", f"{f1} NOT {r1}"]
    print("| Запит | Знайдено | merge, мкс | set, мкс |")
    print("|---|---|---|---|")
    for q in queries:
        found_merge = parse(q).evaluate(index, "merge")
        assert found_merge == parse(q).evaluate(index, "set"), q
        print(f"| {q} | {len(found_merge)} | {time_query(index, q, 'merge'):.2f} | {time_query(index, q, 'set'):.2f} |")


def synthetic() -> None:
    random.seed(1)
    print("\nСинтетичні списки (AND двох відсортованих списків):")
    print("| Розмір кожного списку | merge, мс | set, мс |")
    print("|---|---|---|")
    for n in (1_000, 100_000, 1_000_000):
        a = sorted(random.sample(range(n * 3), n))
        b = sorted(random.sample(range(n * 3), n))
        number = max(1, 100_000 // n)
        tm = min(timeit.repeat(lambda: merge_and(a, b), repeat=3, number=number))
        ts = min(timeit.repeat(lambda: set_and(a, b), repeat=3, number=number))
        print(f"| {n:,} | {tm / number * 1000:.3f} | {ts / number * 1000:.3f} |")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("source", type=Path, help="index.bin або папка з корпусом")
    args = parser.parse_args()
    index = build_index(args.source) if args.source.is_dir() else load(args.source)
    real_terms(index)
    synthetic()


if __name__ == "__main__":
    main()
