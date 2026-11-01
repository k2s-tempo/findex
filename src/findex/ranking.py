"""
Ранжований пошук: накопичуємо бали по документах і беремо топ-k.
"""
import argparse
import heapq
import time
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path

from findex.scoring import SCORERS, Scorer
from findex.storage import open_index
from findex.tokenizer import tokenize


@dataclass(frozen=True, order=True)
class SearchResult:
    """Порівнюється ЛИШЕ за score (order=True бере поля з compare=True).

    doc_id і name в порівнянні не беруть участі, тому max/sorted/nlargest
    впорядковують результати саме за балом.
    """
    score: float
    doc_id: int = field(compare=False)
    name: str = field(compare=False, default="")


def rank(index, terms, scorer: Scorer, k: int = 10, candidates=None):
    """Топ-k документів за сумою балів термінів запиту.

    terms      - уже нормалізовані терміни (дублікати ігноруються)
    candidates - множина doc_id, з якої дозволено брати результати
                 (Boolean-фільтр з кроку 3); None = усі документи
    """
    scores = defaultdict(float)
    for term in dict.fromkeys(terms):          # унікальні, у порядку появи
        plist = index.postings.get(term)
        if not plist:
            continue
        df = len(plist)
        for p in plist:
            if candidates is not None and p.doc_id not in candidates:
                continue
            scores[p.doc_id] += scorer.score(p.tf, df, index.docs[p.doc_id].length, index)

    # сортуємо за doc_id: nlargest стабільний, тож за однакового балу
    # першим іде документ із меншим doc_id (детермінований порядок)
    results = (
        SearchResult(score, doc_id, index.docs[doc_id].name)
        for doc_id, score in sorted(scores.items())
    )
    return heapq.nlargest(k, results)


def main():
    parser = argparse.ArgumentParser(description="Ранжований пошук")
    parser.add_argument("index", type=Path)
    parser.add_argument("query")
    parser.add_argument("--scorer", choices=SCORERS, default="bm25")
    parser.add_argument("-k", type=int, default=10)
    args = parser.parse_args()

    scorer = SCORERS[args.scorer]()
    with open_index(args.index) as idx:
        terms = list(tokenize(args.query))
        start = time.perf_counter()
        results = rank(idx, terms, scorer, args.k)
        elapsed = time.perf_counter() - start
        print(f"{scorer}  |  {len(results)} результатів, {elapsed * 1000:.3f} мс")
        for i, r in enumerate(results, 1):
            print(f"{i:>2}. {r.score:7.3f}  {r.name}  ({idx.docs[r.doc_id].length} токенів)")


if __name__ == "__main__":
    main()