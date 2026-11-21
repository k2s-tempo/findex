"""
Інвертований індекс: термін -> відсортований за doc_id список постінгів.

Index поводиться як read-only відображення (collections.abc.Mapping):
len(index) - кількість термінів, term in index, index[term], for term in index.
"""

from collections import Counter, defaultdict
from collections.abc import Callable, Iterable, Iterator, Mapping, Sequence
from dataclasses import dataclass
from functools import cached_property
from pathlib import Path
from typing import TYPE_CHECKING

from findex.corpus import Document, list_documents, read_document
from findex.decorators import timed
from findex.tokenizer import tokenize

if TYPE_CHECKING:
    from findex.arrays import ArrayIndex


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

    @cached_property
    def arrays(self) -> "ArrayIndex":
        """NumPy-вигляд постінгів для швидкого ранжування (будується ліниво, див. findex.arrays)."""
        from findex import arrays

        return arrays.ArrayIndex(self)

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
        self.__dict__.pop("arrays", None)

    def _check_open(self) -> None:
        if self.closed:
            raise ValueError("індекс закрито")


@dataclass(slots=True)
class PartialIndex:
    """Індекс шматка корпусу: документи з doc_id від start_id до start_id + len(docs) - 1.

    Саме це воркер повертає батьківському процесу (через pickle).
    """

    start_id: int
    docs: list[DocMeta]
    postings: dict[str, list[Posting]]
    positions: dict[str, dict[int, list[int]]] | None


def build_partial(
    doc_paths: Sequence[Path],
    start_id: int = 0,
    with_positions: bool = False,
    on_document: Callable[[DocMeta], None] | None = None,
) -> PartialIndex:
    """Індексує шматок корпусу. Функція рівня модуля: її можна віддати потоку чи процесу.

    Приймає шляхи, а не прочитані документи: файли читає сам воркер. doc_id шматка
    починаються зі start_id, тож діапазони різних шматків не перетинаються.
    """
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    positions: defaultdict[str, dict[int, list[int]]] = defaultdict(dict)
    docs: list[DocMeta] = []

    for doc_id, path in enumerate(doc_paths, start=start_id):
        doc = read_document(path)
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
    return PartialIndex(start_id, docs, dict(postings), dict(positions) if with_positions else None)


def merge(partials: Iterable[PartialIndex]) -> Index:
    """Зливає часткові індекси в один.

    doc_id мають утворювати суцільний діапазон 0..N-1 без дірок і перекриттів. Порядок
    partials не важливий (сортуємо за start_id). Діапазони різних шматків упорядковані,
    тож постінги кожного терміна просто дописуються один за одним і лишаються
    відсортованими. Споживає partials: списки постінгів переходять в Index без копіювання.
    """
    ordered = sorted(partials, key=lambda part: part.start_id)
    with_positions = bool(ordered) and ordered[0].positions is not None
    postings: dict[str, list[Posting]] = {}
    positions: dict[str, dict[int, list[int]]] = {}
    docs: list[DocMeta] = []

    for part in ordered:
        if (part.positions is not None) != with_positions:
            raise ValueError("часткові індекси змішують режими з позиціями й без")
        if part.start_id != len(docs):
            raise ValueError(
                f"doc_id мають бути суцільними: очікувався {len(docs)}, а шматок починається з {part.start_id}"
            )
        docs.extend(part.docs)
        for term, plist in part.postings.items():
            known = postings.get(term)
            if known is None:
                postings[term] = plist
            else:
                known.extend(plist)
        if part.positions is not None:
            for term, per_doc in part.positions.items():
                known_positions = positions.get(term)
                if known_positions is None:
                    positions[term] = per_doc
                else:
                    known_positions.update(per_doc)

    return Index(postings, docs, positions if with_positions else None)


@timed
def build_index(
    corpus_dir: Path,
    with_positions: bool = False,
    on_document: Callable[[DocMeta], None] | None = None,
) -> Index:
    """Послідовна збірка: один шматок на весь корпус, тобто merge([build_partial(усі шляхи)])."""
    paths = list_documents(corpus_dir)
    return merge([build_partial(paths, 0, with_positions, on_document)])


def build_index_from_documents(
    documents: Iterable[Document],
    with_positions: bool = False,
    on_document: Callable[[DocMeta], None] | None = None,
) -> Index:
    postings: defaultdict[str, list[Posting]] = defaultdict(list)
    positions: defaultdict[str, dict[int, list[int]]] = defaultdict(dict)
    docs: list[DocMeta] = []

    for doc_id, doc in enumerate(documents):
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

        meta = DocMeta(doc_id, doc.doc_id, str(doc.path), length)
        docs.append(meta)
        if on_document is not None:
            on_document(meta)

    return Index(dict(postings), docs, dict(positions) if with_positions else None)
