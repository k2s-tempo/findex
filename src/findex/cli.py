"""
Командний рядок findex: index / search / stats.

Вивід розділено за призначенням:
  stdout - результат (таблиця або JSON), його можна перенаправляти у файл;
  stderr - діагностика (logging), смуга прогресу та повідомлення про помилки.
Очікувана помилка (немає файлу, зламаний запит) дає один рядок у stderr і код 1,
без стека викликів; стек видно лише з -vv.
"""

import json
import logging
import os
import pickle
import struct
import sys
import time
from collections import Counter
from concurrent.futures import BrokenExecutor
from dataclasses import asdict, dataclass
from importlib import metadata
from pathlib import Path
from typing import Annotated, NoReturn

import typer
from rich.console import Console
from rich.markup import escape
from rich.progress import BarColumn, MofNCompleteColumn, Progress, TextColumn, TimeElapsedColumn
from rich.table import Table

from findex.cache import cached_evaluator
from findex.merge import Engine
from findex.parallel import ExecutorName, build_parallel
from findex.query import Node, positive_terms, ranked_search
from findex.scoring import ScorerName, make_scorer
from findex.snippets import make_snippet
from findex.stats import compute_stats_lazy, run_with_measurement
from findex.storage import LoadFormat, SaveFormat, open_index, save

log = logging.getLogger("findex")

app = typer.Typer(
    help="findex: пошукова система по корпусу текстів (інвертований індекс, BM25/TF-IDF).",
    no_args_is_help=True,
    add_completion=False,
)

_HL_OPEN, _HL_CLOSE = "\ue000", "\ue001"  # службові маркери підсвітки (не трапляються в тексті)
_LOAD_ERRORS = (ValueError, struct.error, EOFError, pickle.UnpicklingError)
_PICKLE_HELP = (
    "Дозволити читання pickle-індексу. НЕБЕЗПЕЧНО для чужих файлів: "
    "pickle може виконати довільний код під час завантаження."
)


# ---------- спільні допоміжні функції ----------


def fail(message: str) -> NoReturn:
    """Очікувана помилка: один зрозумілий рядок у stderr і ненульовий код виходу."""
    typer.echo(f"Помилка: {message}", err=True)
    raise typer.Exit(code=1)


def _setup_logging(verbosity: int, ctx: typer.Context) -> None:
    """Єдине місце налаштування логера: обробник на stderr, рівень залежить від -v."""
    level = {0: logging.WARNING, 1: logging.INFO}.get(verbosity, logging.DEBUG)
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("%(levelname)s %(name)s: %(message)s"))
    log.addHandler(handler)
    log.setLevel(level)

    def cleanup() -> None:
        # після завершення команди прибираємо обробник (важливо для повторних викликів у тестах)
        log.removeHandler(handler)
        log.setLevel(logging.NOTSET)

    ctx.call_on_close(cleanup)


def _version_callback(value: bool) -> None:
    if value:
        try:
            typer.echo(f"findex {metadata.version('findex')}")
        except metadata.PackageNotFoundError:
            typer.echo("findex (версію не визначено: пакет не встановлено)")
        raise typer.Exit()


@app.callback()
def main(
    ctx: typer.Context,
    verbose: Annotated[
        int,
        typer.Option("--verbose", "-v", count=True, help="-v: INFO (час, кеш), -vv: DEBUG і стек помилок."),
    ] = 0,
    version: Annotated[
        bool,
        typer.Option("--version", callback=_version_callback, is_eager=True, help="Показати версію й вийти."),
    ] = False,
) -> None:
    _setup_logging(verbose, ctx)


def _rich_text(snippet: str) -> str:
    """Сніпет зі службовими маркерами -> розмітка rich (решту тексту екрануємо)."""
    return escape(snippet).replace(_HL_OPEN, "[bold yellow]").replace(_HL_CLOSE, "[/bold yellow]")


def _load_format(allow_pickle: bool) -> LoadFormat:
    """Без --allow-pickle читаємо лише власний бінарний формат: чужий файл не виконається."""
    return "auto" if allow_pickle else "array"


def _bad_index(path: Path) -> NoReturn:
    fail(
        f"файл {path} не схожий на індекс findex або пошкоджений "
        "(pickle-індекси читаються лише з --allow-pickle)"
    )


def _json_ascii_only() -> bool:
    """Якщо stdout не UTF-8 (напр. cp1251 при перенаправленні), JSON пишемо ASCII-escape."""
    return (sys.stdout.encoding or "").lower().replace("-", "") != "utf8"


# ---------- findex index ----------


@app.command("index")
def index_cmd(
    corpus: Annotated[Path, typer.Argument(help="Каталог із .txt-файлами корпусу.")],
    out: Annotated[Path, typer.Option("--out", "-o", help="Куди зберегти індекс.")] = Path("index.bin"),
    positions: Annotated[
        bool,
        typer.Option(
            "--positions/--no-positions", help="Зберігати позиції токенів (потрібні для фраз у лапках)."
        ),
    ] = True,
    fmt: Annotated[SaveFormat, typer.Option("--format", "-f", help="Формат файлу індексу.")] = "array",
    executor: Annotated[
        ExecutorName,
        typer.Option("--executor", "-x", help="Як розпаралелити збірку: serial, threads або processes."),
    ] = "serial",
    workers: Annotated[
        int | None,
        typer.Option(
            "--workers", "-w", min=1, help="Кількість воркерів (типово: число ядер; для serial ігнорується)."
        ),
    ] = None,
) -> None:
    """Побудувати індекс корпусу й зберегти його у файл."""
    if not corpus.is_dir():
        fail(f"каталог корпусу не знайдено: {corpus}")
    total = sum(1 for _ in corpus.rglob("*.txt"))
    if total == 0:
        fail(f"у каталозі {corpus} немає жодного .txt-файлу")

    n_workers = 1 if executor == "serial" else (workers or os.cpu_count() or 1)
    progress = Progress(
        TextColumn("{task.description}"),
        BarColumn(),
        MofNCompleteColumn(),
        TimeElapsedColumn(),
        console=Console(stderr=True),
    )
    started = time.perf_counter()
    try:
        with progress:
            task = progress.add_task(f"Індексація ({executor}, воркерів: {n_workers})", total=total)
            index = build_parallel(
                corpus,
                positions,
                executor=executor,
                workers=n_workers,
                on_chunk=lambda n_docs: progress.advance(task, n_docs),
            )
    except (OSError, UnicodeDecodeError) as e:
        log.debug("збірка індексу не вдалась", exc_info=True)  # стек воркера видно з -vv
        fail(f"не вдалося прочитати корпус: {e}")
    except BrokenExecutor:
        log.debug("воркер завершився аварійно", exc_info=True)
        fail("воркер завершився аварійно (перегляньте стек із -vv)")
    log.info(
        "індекс збудовано за %.2f с (%s, воркерів: %d)", time.perf_counter() - started, executor, n_workers
    )

    try:
        save(index, out, fmt)
    except OSError as e:
        fail(f"не вдалося записати {out}: {e.strerror or e}")
    size_mb = out.stat().st_size / 1024 / 1024
    typer.echo(
        f"Збережено {out}: {len(index.docs)} док., {len(index)} термінів, "
        f"{'з позиціями' if positions else 'без позицій'}, {size_mb:.2f} МБ"
    )


# ---------- findex search ----------


@dataclass(frozen=True)
class ResultRow:
    rank: int
    score: float
    doc: str
    doc_id: int
    tokens: int
    snippet: str


@dataclass(frozen=True)
class SearchOutcome:
    tree: Node
    total: int
    rows: list[ResultRow]
    elapsed_ms: float


def _run_search(
    index_path: Path,
    query: str,
    scorer: ScorerName,
    k: int,
    engine: Engine,
    with_snippets: bool,
    marks: tuple[str, str],
    allow_pickle: bool,
) -> SearchOutcome:
    """Завантажує індекс, виконує запит і збирає рядки результату; помилки -> fail()."""
    try:
        with open_index(index_path, _load_format(allow_pickle)) as idx:
            evaluator = cached_evaluator(idx)
            start = time.perf_counter()
            try:
                tree, total, results = ranked_search(idx, query, make_scorer(scorer), k, evaluator, engine)
            except ValueError as e:
                log.debug("помилка запиту", exc_info=True)
                fail(f"некоректний запит: {e}")
            elapsed_ms = (time.perf_counter() - start) * 1000
            terms = positive_terms(tree)
            rows = [
                ResultRow(
                    rank=rank_no,
                    score=round(r.score, 4),
                    doc=r.name,
                    doc_id=r.doc_id,
                    tokens=idx.docs[r.doc_id].length,
                    snippet=make_snippet(idx, r.doc_id, terms, marks=marks).text if with_snippets else "",
                )
                for rank_no, r in enumerate(results, 1)
            ]
    except _LOAD_ERRORS:
        log.debug("не вдалося завантажити індекс", exc_info=True)
        _bad_index(index_path)
    return SearchOutcome(tree, total, rows, elapsed_ms)


@app.command("search")
def search_cmd(
    index_path: Annotated[
        Path, typer.Argument(metavar="INDEX", help="Файл індексу (створений findex index).")
    ],
    query: Annotated[
        str,
        typer.Argument(help="Запит, напр. 'justice AND (courage OR temperance) NOT slave \"the soul\"'."),
    ],
    scorer: Annotated[ScorerName, typer.Option("--scorer", "-s", help="Модель ранжування.")] = "bm25",
    engine: Annotated[Engine, typer.Option("--engine", "-e", help="Як виконувати AND/OR/NOT.")] = "merge",
    k: Annotated[int, typer.Option("-k", min=1, help="Скільки результатів показати.")] = 10,
    as_json: Annotated[bool, typer.Option("--json", help="Вивести результат як JSON у stdout.")] = False,
    snippets: Annotated[bool, typer.Option("--snippets/--no-snippets", help="Показувати сніпети.")] = True,
    allow_pickle: Annotated[bool, typer.Option("--allow-pickle", help=_PICKLE_HELP)] = False,
) -> None:
    """Знайти документи за запитом і показати ранжований список зі сніпетами."""
    if not index_path.is_file():
        fail(f"файл індексу не знайдено: {index_path}")
    marks = ("**", "**") if as_json else (_HL_OPEN, _HL_CLOSE)
    outcome = _run_search(index_path, query, scorer, k, engine, snippets, marks, allow_pickle)

    if as_json:
        payload = {
            "query": query,
            "parsed": str(outcome.tree),
            "scorer": scorer,
            "engine": engine,
            "total": outcome.total,
            "shown": len(outcome.rows),
            "results": [asdict(row) for row in outcome.rows],
        }
        typer.echo(json.dumps(payload, ensure_ascii=_json_ascii_only(), indent=2))
        return

    if not outcome.rows:
        typer.echo("Нічого не знайдено.")
        return
    table = Table(
        title=f"Запит: {outcome.tree}",
        caption=f"Знайдено {outcome.total} док., показано {len(outcome.rows)} "
        f"({scorer}, {engine}, {outcome.elapsed_ms:.2f} мс)",
        show_lines=snippets,
    )
    table.add_column("#", justify="right")
    table.add_column("Бал", justify="right")
    table.add_column("Документ")
    table.add_column("Токенів", justify="right")
    if snippets:
        table.add_column("Сніпет", overflow="fold")
    for row in outcome.rows:
        cells = [str(row.rank), f"{row.score:.3f}", escape(row.doc), str(row.tokens)]
        if snippets:
            cells.append(_rich_text(row.snippet))
        table.add_row(*cells)
    Console().print(table)


# ---------- findex stats ----------


def _print_corpus_stats(console: Console, path: Path, top: int) -> None:
    result, elapsed, peak = run_with_measurement(path, compute_stats_lazy)
    if result["doc_count"] == 0:
        fail(f"у каталозі {path} немає жодного .txt-файлу")
    summary = Table(
        title=f"Корпус {path}", caption=f"{elapsed:.2f} с, пік пам'яті {peak / 1024 / 1024:.2f} МБ"
    )
    summary.add_column("Метрика")
    summary.add_column("Значення", justify="right")
    summary.add_row("Документів", f"{result['doc_count']:,}")
    summary.add_row("Токенів", f"{result['token_count']:,}")
    summary.add_row("Унікальних термінів", f"{result['vocab_size']:,}")
    console.print(summary)
    top_table = Table(title=f"Топ-{top} термінів")
    top_table.add_column("Термін")
    top_table.add_column("Частота", justify="right")
    for term, freq in result["top_50"][:top]:
        top_table.add_row(escape(term), f"{freq:,}")
    console.print(top_table)


def _print_index_stats(console: Console, path: Path, top: int, allow_pickle: bool) -> None:
    try:
        with open_index(path, _load_format(allow_pickle)) as idx:
            postings = sum(len(p) for p in idx.postings.values())
            lengths = [d.length for d in idx.docs]
            totals = Counter({t: sum(p.tf for p in pl) for t, pl in idx.postings.items()})
            top_terms = [(t, f, idx.df(t)) for t, f in totals.most_common(top)]
            has_positions = idx.has_positions
            avg = idx.avg_doc_length
            n_docs, n_terms = idx.num_docs, len(idx)
    except _LOAD_ERRORS:
        log.debug("не вдалося завантажити індекс", exc_info=True)
        _bad_index(path)

    summary = Table(title=f"Індекс {path}")
    summary.add_column("Метрика")
    summary.add_column("Значення", justify="right")
    summary.add_row("Документів", f"{n_docs:,}")
    summary.add_row("Термінів у словнику", f"{n_terms:,}")
    summary.add_row("Постінгів", f"{postings:,}")
    summary.add_row("Позиції токенів", "так" if has_positions else "ні")
    summary.add_row("Середня довжина документа", f"{avg:,.0f}")
    if lengths:
        summary.add_row("Найкоротший / найдовший", f"{min(lengths):,} / {max(lengths):,}")
    console.print(summary)
    top_table = Table(title=f"Топ-{top} термінів за сумарною частотою")
    top_table.add_column("Термін")
    top_table.add_column("Частота", justify="right")
    top_table.add_column("df", justify="right")
    for term, freq, df in top_terms:
        top_table.add_row(escape(term), f"{freq:,}", str(df))
    console.print(top_table)


@app.command("stats")
def stats_cmd(
    path: Annotated[Path, typer.Argument(help="Файл індексу або каталог корпусу.")],
    top: Annotated[
        int, typer.Option("--top", "-n", min=1, help="Скільки найчастіших термінів показати.")
    ] = 10,
    allow_pickle: Annotated[bool, typer.Option("--allow-pickle", help=_PICKLE_HELP)] = False,
) -> None:
    """Статистика індексу (файл) або корпусу (каталог, один лінивий прохід)."""
    console = Console()
    if path.is_dir():
        _print_corpus_stats(console, path, top)
    elif path.is_file():
        _print_index_stats(console, path, top, allow_pickle)
    else:
        fail(f"не знайдено ні файлу індексу, ні каталогу корпусу: {path}")


if __name__ == "__main__":
    app()
