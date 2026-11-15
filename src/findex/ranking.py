"""
Ранжований пошук: накопичуємо бали по документах і беремо топ-k.
"""
import heapq
from collections import defaultdict
from collections.abc import Collection, Iterable
from dataclasses import dataclass, field

from findex.index import Index
from findex.scoring import Scorer


@dataclass(frozen=True, order=True)
class SearchResult:
    """Порівнюється ЛИШЕ за score (order=True бере поля з compare=True).

    doc_id і name в порівнянні не беруть участі, тому max/sorted/nlargest
    впорядковують результати саме за балом.
    """

    score: float
    doc_id: int = field(compare=False)
    name: str = field(compare=False, default="")


def rank(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer,
    k: int = 10,
    candidates: Collection[int] | None = None,
) -> list[SearchResult]:
    """Топ-k документів за сумою балів термінів запиту.

    terms      - уже нормалізовані терміни (дублікати ігноруються)
    candidates - множина doc_id, з якої дозволено брати результати
                 (Boolean-фільтр); None = усі документи
    """
    scores: defaultdict[int, float] = defaultdict(float)
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
