"""
Паралельна збірка індексу: розбиття на шматки (map) і злиття (reduce).

План: список шляхів ділиться на суцільні шматки, кожен шматок індексується
функцією build_partial (вона живе на рівні модуля, тож її можна віддати потоку
або процесу), а merge склеює часткові індекси. Воркери отримують ШЛЯХИ, а не
прочитані документи: файли читає сам воркер, тому через pickle їде мало даних.

Три виконавці з однаковим кодом: serial (звичайний цикл), threads
(ThreadPoolExecutor) і processes (ProcessPoolExecutor). Процеси завжди
запускаються методом spawn: він однаковий на Windows, macOS і Linux, тож
код безпечний для spawn, а не випадково працює лише з fork.
"""

import multiprocessing
import os
import time
from collections.abc import Callable, MutableMapping, Sequence
from concurrent.futures import Executor, ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Literal

from findex.corpus import list_documents
from findex.index import Index, PartialIndex, build_partial, merge

ExecutorName = Literal["serial", "threads", "processes"]
ExecutorChoice = Literal["auto", "serial", "threads", "processes"]  # що можна вказати в CLI
EXECUTORS: tuple[ExecutorName, ...] = ("serial", "threads", "processes")

# Пороги вибрані за бенчмарком (AMD Ryzen 7 5800X, 8 ядер / 16 потоків, Windows, Python 3.11):
# 4 процеси дали 1,03× на 6 МБ, 1,55× на 19 МБ і 1,82× на 63 МБ, а 8 і 16 воркерів були
# повільнішими за 4, бо батько послідовно розпаковує результати воркерів. Для іншої машини
# числа інші: перемірте скриптом scripts/bench_parallel.py.
AUTO_MIN_BYTES = 8 * 1024 * 1024  # менший корпус: запуск процесів не окупається
MAX_DEFAULT_WORKERS = 4

# Шматків більше, ніж воркерів: файли дуже різні за розміром (одні діалоги в десятки
# разів довші за інші), і воркер, що швидко закінчив, бере наступний шматок.
CHUNKS_PER_WORKER = 4

Chunk = tuple[int, list[Path]]  # (doc_id першого документа, шляхи шматка)


def default_workers() -> int:
    """Типова кількість воркерів: число ядер, але не більше MAX_DEFAULT_WORKERS."""
    return max(1, min(os.cpu_count() or 1, MAX_DEFAULT_WORKERS))


def resolve_executor(
    choice: ExecutorChoice, workers: int | None, corpus_bytes: int
) -> tuple[ExecutorName, int]:
    """Перетворює вибір користувача на конкретного виконавця й число воркерів.

    auto: процеси для корпусу від AUTO_MIN_BYTES, інакше serial. Для serial воркерів завжди 1.
    """
    if choice == "auto":
        resolved: ExecutorName = "processes" if corpus_bytes >= AUTO_MIN_BYTES else "serial"
    else:
        resolved = choice
    if resolved == "serial":
        return "serial", 1
    return resolved, workers or default_workers()


def plan_chunks(paths: Sequence[Path], n_chunks: int) -> list[Chunk]:
    """Ділить шляхи на суцільні шматки, приблизно рівні за сумарним розміром файлів.

    Шматок суцільний (шляхи йдуть підряд), тож doc_id діапазону відомі наперед:
    start_id шматка дорівнює позиції його першого файлу у відсортованому списку.
    Шматків може вийти менше, ніж n_chunks (дуже великий файл займає кілька «слотів»).
    """
    if n_chunks < 1:
        raise ValueError("n_chunks має бути не менше 1")
    if not paths:
        return []
    sizes = [path.stat().st_size for path in paths]
    total = sum(sizes)
    slots = min(n_chunks, len(paths))

    chunks: list[Chunk] = []
    last_slot = -1
    done = 0  # сумарний розмір файлів, уже розкладених по шматках
    for doc_id, (path, size) in enumerate(zip(paths, sizes, strict=True)):
        # слот за серединою файлу; якщо всі файли порожні, ділимо за кількістю
        slot = int((done + size / 2) / total * slots) if total > 0 else doc_id * slots // len(paths)
        slot = min(slot, slots - 1)
        if slot != last_slot:
            chunks.append((doc_id, []))
            last_slot = slot
        chunks[-1][1].append(path)
        done += size
    return chunks


def _timed_partial(paths: list[Path], start_id: int, with_positions: bool) -> tuple[PartialIndex, float]:
    """build_partial + CPU-час потоку, що його виконав (для колонки CPU у бенчмарку).

    thread_time() рахує лише поточний потік, тож сума по шматках правильна і для потоків,
    і для процесів. Функція рівня модуля, тому її теж можна віддати процесу.
    """
    started = time.thread_time()
    part = build_partial(paths, start_id, with_positions)
    return part, time.thread_time() - started


def _make_pool(executor: ExecutorName, workers: int) -> Executor:
    if executor == "threads":
        return ThreadPoolExecutor(max_workers=workers, thread_name_prefix="findex")
    return ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn"))


def _map_chunks(
    chunks: Sequence[Chunk],
    with_positions: bool,
    executor: ExecutorName,
    workers: int,
    on_chunk: Callable[[int], None] | None,
) -> tuple[list[PartialIndex], float]:
    """Повертає часткові індекси й сумарний CPU-час воркерів (секунди)."""
    partials: list[PartialIndex] = []
    worker_cpu = 0.0
    if executor == "serial":
        for start_id, paths in chunks:
            part, cpu = _timed_partial(paths, start_id, with_positions)
            partials.append(part)
            worker_cpu += cpu
            if on_chunk is not None:
                on_chunk(len(part.docs))
        return partials, worker_cpu

    with _make_pool(executor, workers) as pool:
        futures = [pool.submit(_timed_partial, paths, start_id, with_positions) for start_id, paths in chunks]
        try:
            for future in as_completed(futures):
                part, cpu = future.result()  # виняток воркера піднімається тут, зі стеком воркера
                partials.append(part)
                worker_cpu += cpu
                if on_chunk is not None:
                    on_chunk(len(part.docs))
        except BaseException:
            for future in futures:
                future.cancel()  # шматки, що ще не стартували, не запускаємо
            raise
    return partials, worker_cpu


def build_parallel(
    corpus_dir: Path,
    with_positions: bool = False,
    *,
    executor: ExecutorName = "serial",
    workers: int = 1,
    on_chunk: Callable[[int], None] | None = None,
    timings: MutableMapping[str, float] | None = None,
) -> Index:
    """Будує індекс корпусу вибраним виконавцем.

    on_chunk(n) викликається з головного потоку, коли готовий шматок із n документів.
    timings (якщо передано) отримує "map" і "merge" (скільки тривали паралельна частина
    і послідовне злиття: саме воно задає межу прискорення за законом Амдала) та
    "worker_cpu" (сумарний CPU-час воркерів; запуск інтерпретатора воркера не входить).
    Якщо воркер упав, збірка падає з його винятком: ні зависання, ні тихого пропуску.
    """
    if executor not in EXECUTORS:
        raise ValueError(f"невідомий виконавець {executor!r}, очікується один із {EXECUTORS}")
    if workers < 1:
        raise ValueError("workers має бути не менше 1")

    paths = list_documents(corpus_dir)
    n_chunks = 1 if executor == "serial" else workers * CHUNKS_PER_WORKER
    chunks = plan_chunks(paths, n_chunks)

    started = time.perf_counter()
    partials, worker_cpu = _map_chunks(chunks, with_positions, executor, workers, on_chunk)
    mapped = time.perf_counter()
    index = merge(partials)
    finished = time.perf_counter()

    if timings is not None:
        timings["map"] = mapped - started
        timings["merge"] = finished - mapped
        timings["worker_cpu"] = worker_cpu
    return index
