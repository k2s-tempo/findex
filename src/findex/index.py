"""
Інвертований індекс: термін -> відсортований за doc_id список постінгів.

Інваріант: doc_id призначається по зростанню в порядку обходу корпусу
(iter_documents віддає файли відсортованими), тому кожен список постінгів
автоматично відсортований за doc_id. Це потрібно для merge-пошуку.
"""
import argparse
from collections import Counter, defaultdict
from dataclasses import dataclass
from pathlib import Path

from findex.corpus import iter_documents
from findex.stats import run_with_measurement
from findex.tokenizer import tokenize


@dataclass(frozen=True, slots=True)
class Posting:
    doc_id: int
    tf: int  # скільки разів термін зустрівся в документі


@dataclass(frozen=True, slots=True)
class DocMeta:
    doc_id: int
    name: str    # ім'я файлу без розширення, для показу результатів
    path: str
    length: int  # кількість токенів у документі


@dataclass
class Index:
    postings: dict[str, list[Posting]]
    docs: list[DocMeta]
    # term -> {doc_id: [позиції токена]}; None, якщо індекс без позицій
    positions: dict[str, dict[int, list[int]]] | None = None


def build_index(corpus_dir: Path, with_positions: bool = False) -> Index:
    """Будує індекс одним проходом по лінивому пайплайну з лаби 1."""
    postings = defaultdict(list)
    positions = defaultdict(dict) if with_positions else None
    docs = []

    for doc_id, doc in enumerate(iter_documents(corpus_dir)):
        if with_positions:
            local = defaultdict(list)
            length = 0
            for pos, token in enumerate(tokenize(doc.text)):
                local[token].append(pos)
                length += 1
            for term, plist in local.items():
                postings[term].append(Posting(doc_id, len(plist)))
                positions[term][doc_id] = plist
        else:
            counts = Counter(tokenize(doc.text))
            length = sum(counts.values())
            for term, tf in counts.items():
                postings[term].append(Posting(doc_id, tf))

        docs.append(DocMeta(doc_id, doc.doc_id, str(doc.path), length))

    # dict(...) прибирає поведінку defaultdict: пошук відсутнього терміна
    # не створюватиме порожній запис в індексі
    return Index(
        dict(postings),
        docs,
        dict(positions) if positions is not None else None,
    )


def main():
    import time

    from findex.storage import save  # імпорт тут, щоб уникнути циклу index <-> storage

    parser = argparse.ArgumentParser(description="Побудова й збереження індексу")
    parser.add_argument("corpus", type=Path)
    parser.add_argument("--out", type=Path, help="куди зберегти індекс")
    parser.add_argument("--format", choices=["pickle", "array"], default="array")
    parser.add_argument("--positions", action="store_true",
                        help="зберігати позиції токенів")
    args = parser.parse_args()

    index, elapsed, peak = run_with_measurement(
        args.corpus, lambda d: build_index(d, args.positions)
    )

    total_postings = sum(len(p) for p in index.postings.values())
    print(f"Документів: {len(index.docs)}")
    print(f"Термінів у словнику: {len(index.postings)}")
    print(f"Постінгів: {total_postings}")
    print(f"Побудова: {elapsed:.3f} с, пік пам'яті {peak / 1024 / 1024:.2f} МБ")

    if args.out:
        t0 = time.perf_counter()
        save(index, args.out, args.format)
        save_time = time.perf_counter() - t0
        size = args.out.stat().st_size
        print(f"Збережено {args.out} ({args.format}): "
              f"{size / 1024 / 1024:.2f} МБ, {save_time:.3f} с")


if __name__ == "__main__":
    # Запускаємо main із модуля findex.index, а не з __main__. Інакше класи
    # Posting/DocMeta матимуть ім'я __main__.Posting, pickle запише саме його,
    # і в іншому процесі (findex.search) load не знайде такий клас.
    from findex.index import main as _main

    _main()