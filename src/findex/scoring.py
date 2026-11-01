"""
Scorer: об'єкт, що перетворює (tf, df, довжина документа) на бал одного терміна.

Це typing.Protocol: TfIdf і BM25 НЕ наслідують спільний клас, вони просто мають
метод score з потрібною сигнатурою (duck typing). Будь-який такий об'єкт можна
підставити в rank() замість них.
"""
import math
from dataclasses import dataclass
from typing import Protocol, runtime_checkable


@runtime_checkable
class Scorer(Protocol):
    def score(self, tf: int, df: int, doc_len: int, index) -> float:
        """Бал одного терміна в одному документі.

        tf      - скільки разів термін у документі
        df      - у скількох документах корпусу є термін
        doc_len - довжина документа в токенах
        index   - індекс (з нього беремо num_docs і avg_doc_length)
        """
        ...


@dataclass(frozen=True)
class TfIdf:
    """(1 + ln tf) * idf, з м'якою нормалізацією за довжиною документа.

    idf = ln(1 + N/df): завжди > 0, рідкий термін важить більше за частий.
    1 + ln tf: повтори додають дедалі менше (логарифмічне насичення).
    norm=True ділить на sqrt(doc_len/avg_len), щоб довгі документи не вигравали
    просто за розмір. norm=False - класичний TF-IDF без урахування довжини.
    """
    norm: bool = True

    def score(self, tf: int, df: int, doc_len: int, index) -> float:
        if tf <= 0 or df <= 0:
            return 0.0
        idf = math.log(1 + index.num_docs / df)
        value = (1 + math.log(tf)) * idf
        if self.norm:
            avg = index.avg_doc_length
            value /= math.sqrt(doc_len / avg) if avg and doc_len else 1.0
        return value


@dataclass(frozen=True)
class BM25:
    """Okapi BM25.

    idf = ln(1 + (N - df + 0.5) / (df + 0.5))
    бал = idf * tf*(k1+1) / (tf + k1*(1 - b + b*doc_len/avg_len))

    k1 - наскільки швидко насичується внесок повторів (більше k1 - повільніше).
    b  - сила нормалізації за довжиною (0 - вимкнено, 1 - повна).
    """
    k1: float = 1.5
    b: float = 0.75

    def score(self, tf: int, df: int, doc_len: int, index) -> float:
        if tf <= 0 or df <= 0:
            return 0.0
        n = index.num_docs
        idf = math.log(1 + (n - df + 0.5) / (df + 0.5))
        avg = index.avg_doc_length or 1.0
        length_part = 1 - self.b + self.b * doc_len / avg
        return idf * tf * (self.k1 + 1) / (tf + self.k1 * length_part)


SCORERS = {"tfidf": TfIdf, "bm25": BM25}