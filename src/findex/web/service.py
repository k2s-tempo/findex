"""Пошук для веб-шару: звичайні dataclass-и, без Pydantic і без FastAPI."""

import math
import threading
import time
from dataclasses import dataclass

from findex.index import Index
from findex.query import positive_terms, ranked_search
from findex.scoring import ScorerName, make_scorer
from findex.snippets import make_snippet
from findex.web.deps import EvaluatorFn
from findex.web.settings import get_settings

API_MARKS = ("**", "**")  # підсвітка в JSON
PAGE_MARKS = ("\ue000", "\ue001")  # службові маркери: HTML-сторінка екранує текст і сама ставить <mark>

_gate: threading.BoundedSemaphore | None = None
_gate_lock = threading.Lock()


def _search_gate() -> threading.BoundedSemaphore:
    """Один на процес: обмежує, скільки пошуків (а отже і тимчасових копій текстів для сніпетів)
    живе в пам'яті одночасно. Решта запитів чекає тут, нічого не виділяючи."""
    global _gate
    with _gate_lock:
        if _gate is None:
            _gate = threading.BoundedSemaphore(get_settings().max_concurrent_searches)
        return _gate


@dataclass(frozen=True)
class Hit:
    rank: int
    doc_id: int
    name: str
    score: float
    tokens: int
    snippet: str


@dataclass(frozen=True)
class SearchPage:
    parsed: str
    total: int
    total_pages: int
    elapsed_ms: float
    hits: list[Hit]


def search(
    index: Index,
    evaluator: EvaluatorFn | None,
    query: str,
    scorer: ScorerName,
    k: int,
    page: int,
    marks: tuple[str, str] = API_MARKS,
) -> SearchPage:
    """Сторінка результатів. ValueError (поганий запит) іде нагору - маршрут зробить з нього 422."""
    with _search_gate():
        started = time.perf_counter()  # міряємо роботу, а не очікування в черзі
        offset = (page - 1) * k
        tree, total, results = ranked_search(index, query, make_scorer(scorer), offset + k, evaluator)
        terms = positive_terms(tree)
        hits = [
            Hit(
                rank=offset + n,
                doc_id=r.doc_id,
                name=r.name,
                score=round(r.score, 4),
                tokens=index.docs[r.doc_id].length,
                snippet=make_snippet(index, r.doc_id, terms, marks=marks).text,
            )
            for n, r in enumerate(results[offset : offset + k], 1)
        ]
        elapsed_ms = (time.perf_counter() - started) * 1000
    return SearchPage(str(tree), total, max(1, math.ceil(total / k)), elapsed_ms, hits)
