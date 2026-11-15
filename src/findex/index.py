"""
Інвертований індекс: термін -> відсортований за doc_id список постінгів.

Index поводиться як read-only відображення (collections.abc.Mapping):
len(index) - кількість термінів, term in index, index[term], for term in index.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterator, Mapping
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path

from findex.corpus import iter_documents
from findex.decorators import timed
from findex.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # скільки разів термін зустрівся в документі


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str  # ім'я файлу без розширення, для показу результатів
    path: str
    length: int  # кількість токенів у документі


class Index(Mapping[str, list[Posting]]):
    """Інвертований індекс як об'єкт «як у Python».

    Не slots-клас: cached_property зберігає значення в __dict__ екземпляра.
    """

    def __init__(
        self,
        postings: dict[str, list[Posting]],
        docs: list[DocMeta],
        positions: dict[str, dict[int, list[int]]] | None = None,
    ) -> None:
        self.postings = postings  # term -> list[Posting]
        self.docs = docs  # індекс у списку = doc_id
        self.positions = positions  # term -> {doc_id: [позиції]} або None
        self.closed = False
        self._close_callbacks: list[Callable[[], object]] = []

    # --- мінімум для Mapping: решта (keys, items, get, ==) приходить безкоштовно ---
    def __getitem__(self, key: str) -> list[Posting]:
        self._check_open()
        return self.postings[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self.postings)

    def __len__(self) -> int:
        return len(self.postings)

    # --- швидший за стандартний (той викликає __getitem__ і ловить KeyError) ---
    def __contains__(self, key: object) -> bool:
        return key in self.postings

    def __repr__(self) -> str:
        pos = "з позиціями" if self.has_positions else "без позицій"
        state = ", закрито" if self.closed else ""
        return f"<Index: {len(self.docs)} док., {len(self.postings)} термінів, {pos}{state}>"

    # --- обчислювані властивості ---
    @property
    def num_docs(self) -> int:
        return len(self.docs)

    @cached_property
    def avg_doc_length(self) -> float:
        """Рахується один раз і кешується в __dict__ (повторне звернення безкоштовне)."""
        if not self.docs:
            return 0.0
        return sum(d.length for d in self.docs) / len(self.docs)

    @property
    def has_positions(self) -> bool:
        return self.positions is not None

    def df(self, term: str) -> int:
        """Document frequency: у скількох документах зустрічається термін."""
        return len(self.postings.get(term, ()))

    # --- життєвий цикл ---
    def on_close(self, callback: Callable[[], object]) -> None:
        """Зареєструвати дію, що виконається при close() (напр. cache_clear)."""
        self._close_callbacks.append(callback)

    def close(self) -> None:
        if self.closed:
            return
        self.closed = True
        for callback in self._close_callbacks:
            callback()
        self.postings = {}
        self.positions = None
        self.docs = []
        self.__dict__.pop("avg_doc_length", None)  # скинути кеш

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("індекс закрито")


@timed
def build_index(
    corpus_dir: Path,
    with_positions: bool = False,
    on_document: Callable[[DocMeta], None] | None = None,
) -> Index:
    """Будує індекс одним проходом по лінивому пайплайну з лаби 1."""
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    positions: defaultdict[str, dict[int, list[int]]] = defaultdict(dict)
    docs: list[DocMeta] = []

    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        if with_positions:
            local: defaultdict[str, list[int]] = defaultdict(list)
            for pos, token in enumerate(tokenize(doc.text)):
                local[token].append(pos)
            length = sum(len(plist) for plist in local.values())
            for term, plist in local.items():
                postings[term].append(Posting(doc_id, len(plist)))
                positions[term][doc_id] = plist
        else:
            counts = Counter(tokenize(doc.text))
            length = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id, tf))

        docs.append(DocMeta(doc_id, doc.doc_id, str(doc.path), length))
        if on_document is not None:
            on_document(docs[-1])

    # dict(...) прибирає поведінку defaultdict: пошук відсутнього терміна
    # не створюватиме порожній запис в індексі
    return Index(dict(postings), docs, dict(positions) if with_positions else None)
