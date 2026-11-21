"""
NumPy-вигляд індексу для швидкого ранжування (лаба 8).

Постінги терміна лежать у двох масивах int32 (doc_id, tf), довжини документів в одному масиві
int32. Масиви будуються ліниво, лише для термінів, які справді шукали, і кешуються.
BM25 і TF-IDF рахуються виразами над масивами, топ-k береться через np.argpartition.

Результат має збігатися з еталонним ranking.rank (лаба 3): ті самі формули в тому самому порядку
операцій, логарифми tf беруться з таблиці math.log (а не np.log), бали додаються в тому самому
порядку термінів, а при рівних балах першим іде документ із меншим doc_id.
"""

import math
from collections.abc import Collection, Iterable
from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from findex.index import Index
from findex.ranking import SearchResult, rank
from findex.scoring import BM25, Scorer, TfIdf

IntArray = NDArray[np.int32]
FloatArray = NDArray[np.float64]


@dataclass(frozen=True, slots=True)
class TermArrays:
    doc_ids: IntArray
    tfs: IntArray


class ArrayIndex:
    """Масиви NumPy над готовим Index. Створюється через Index.arrays."""

    def __init__(self, index: Index) -> None:
        self._index = index
        self.num_docs = index.num_docs
        self.avg_len = index.avg_doc_length
        self.lengths: IntArray = np.fromiter(
            (d.length for d in index.docs), dtype=np.int32, count=len(index.docs)
        )
        self._terms: dict[str, TermArrays | None] = {}

    def term(self, term: str) -> TermArrays | None:
        """Масиви постінгів терміна (None, якщо його немає). Перший виклик конвертує, далі кеш."""
        if term in self._terms:
            return self._terms[term]
        plist = self._index.postings.get(term)
        found: TermArrays | None = None
        if plist:
            n = len(plist)
            found = TermArrays(
                np.fromiter((p.doc_id for p in plist), dtype=np.int32, count=n),
                np.fromiter((p.tf for p in plist), dtype=np.int32, count=n),
            )
        self._terms[term] = found
        return found


# _log_values[t] == math.log(t): таблиця росте за потреби. np.log міг би відрізнятися від math.log в
# останньому розряді, а тоді близькі бали міняли б місцями документи відносно еталона.
_log_values: FloatArray = np.zeros(1, dtype=np.float64)


def _log_table(max_tf: int) -> FloatArray:
    global _log_values
    table = _log_values
    if max_tf >= table.size:
        grown = np.zeros(max(max_tf + 1, 2 * table.size), dtype=np.float64)
        grown[: table.size] = table
        for t in range(max(table.size, 1), grown.size):
            grown[t] = math.log(t)
        _log_values = table = grown
    return table


def bm25_scores(scorer: BM25, tf: IntArray, df: int, dl: IntArray, n_docs: int, avg: float) -> FloatArray:
    idf = math.log(1 + (n_docs - df + 0.5) / (df + 0.5))
    avg = avg or 1.0
    tf_f = tf.astype(np.float64)
    length_part = 1 - scorer.b + scorer.b * dl / avg
    return np.asarray(idf * tf_f * (scorer.k1 + 1) / (tf_f + scorer.k1 * length_part), dtype=np.float64)


def tfidf_scores(scorer: TfIdf, tf: IntArray, df: int, dl: IntArray, n_docs: int, avg: float) -> FloatArray:
    idf = math.log(1 + n_docs / df)
    value = (1 + _log_table(int(tf.max()))[tf]) * idf
    if scorer.norm and avg:
        value = value / np.where(dl > 0, np.sqrt(dl / avg), 1.0)
    return np.asarray(value, dtype=np.float64)


def rank_fast(
    index: Index,
    terms: Iterable[str],
    scorer: Scorer,
    k: int = 10,
    candidates: Collection[int] | None = None,
) -> list[SearchResult]:
    """Те саме, що ranking.rank, але над масивами NumPy. Невідомий scorer: повертаємось до rank."""
    if not isinstance(scorer, (BM25, TfIdf)):
        return rank(index, terms, scorer, k, candidates)
    if k <= 0:
        return []

    arrays = index.arrays
    n = arrays.num_docs
    scores = np.zeros(n, dtype=np.float64)
    touched = np.zeros(n, dtype=np.bool_)
    allowed: NDArray[np.bool_] | None = None
    if candidates is not None:
        allowed = np.zeros(n, dtype=np.bool_)
        allowed[np.fromiter(candidates, dtype=np.int64, count=len(candidates))] = True

    for term in dict.fromkeys(terms):  # унікальні, у порядку появи (як в еталоні)
        ta = arrays.term(term)
        if ta is None:
            continue
        df = int(ta.doc_ids.size)
        ids, tf = ta.doc_ids, ta.tfs
        if allowed is not None:
            keep = allowed[ids]
            ids, tf = ids[keep], tf[keep]
        if ids.size == 0:
            continue
        lengths = arrays.lengths[ids]
        if isinstance(scorer, BM25):
            part = bm25_scores(scorer, tf, df, lengths, n, arrays.avg_len)
        else:
            part = tfidf_scores(scorer, tf, df, lengths, n, arrays.avg_len)
        scores[ids] += part  # doc_id в межах терміна унікальні, тож це безпечно
        touched[ids] = True

    cand = np.flatnonzero(touched)
    m = int(cand.size)
    if m == 0:
        return []
    vals = scores[cand]
    if k < m:
        # k-те за величиною значення; усе, що більше, точно в топі, а серед рівних беремо менші doc_id
        kth = vals[np.argpartition(vals, m - k)[m - k]]
        above = np.flatnonzero(vals > kth)
        tied = np.flatnonzero(vals == kth)[: k - above.size]
        sel = np.concatenate((above, tied))
    else:
        sel = np.arange(m)
    order = sel[np.lexsort((cand[sel], -vals[sel]))]  # бал за спаданням, потім doc_id за зростанням
    docs = index.docs
    return [SearchResult(float(vals[i]), int(cand[i]), docs[int(cand[i])].name) for i in order]
