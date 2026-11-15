"""
Ціна передачі часткових індексів між процесами (лаба 5): розмір pickle і час dumps/loads.

Воркер pickle-ить свій PartialIndex, батько читає його назад (pickle.loads). Цей скрипт
робить те саме в одному процесі й міряє складові окремо від самої індексації.

    uv run python scripts\\pickle_cost.py data\\big
"""

import pickle
import sys
import time
from pathlib import Path

from findex.corpus import list_documents
from findex.index import build_partial
from findex.parallel import plan_chunks


def main(corpus: Path, n_chunks: int = 16) -> None:
    paths = list_documents(corpus)
    for with_positions in (True, False):
        started = time.perf_counter()
        partials = [build_partial(chunk, start, with_positions) for start, chunk in plan_chunks(paths, n_chunks)]
        build = time.perf_counter() - started

        started = time.perf_counter()
        blobs = [pickle.dumps(part) for part in partials]  # те, що робить воркер
        dump = time.perf_counter() - started

        started = time.perf_counter()
        for blob in blobs:
            pickle.loads(blob)  # те, що робить батько
        load = time.perf_counter() - started

        size_mb = sum(len(blob) for blob in blobs) / 1024 / 1024
        print(
            f"позиції: {'так' if with_positions else 'ні':3} | індексація {build:6.2f} с | "
            f"dumps {dump:5.2f} с | loads {load:5.2f} с | pickle {size_mb:7.1f} МБ"
        )


if __name__ == "__main__":
    if len(sys.argv) != 2:
        sys.exit("Використання: python scripts/pickle_cost.py КАТАЛОГ_КОРПУСУ")
    main(Path(sys.argv[1]))
