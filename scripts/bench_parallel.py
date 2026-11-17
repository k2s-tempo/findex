"""
Бенчмарк паралельної індексації (лаба 5, M3).

Кожен замір це окремий процес Python: так пік пам'яті стосується лише цього запуску,
а запуск «холодний», як у реальній команді findex index. Головний процес лише
запускає заміри, рахує медіани й малює графік.

    uv run python scripts\\bench_parallel.py data\\raw --repeat 3
    uv run python scripts\\bench_parallel.py data\\raw --executors threads --tag py313t

Колонки: wall (годинник), CPU (CPU-час батька + воркерів-процесів; запуск інтерпретатора
воркера не входить), пік RSS батьківського процесу, час merge, прискорення відносно serial.
Файл має лишатися безпечним для spawn: уся робота під if __name__ == "__main__".
"""

import argparse
import json
import math
import os
import platform
import statistics
import subprocess
import sys
import time
from pathlib import Path
from typing import Any

FIELDS = ("wall", "cpu", "rss_mb", "map", "merge")


def peak_rss_mb() -> float:
    """Пік пам'яті поточного процесу, МБ (Windows: peak working set; Linux/macOS: ru_maxrss).

    Без psutil на Windows повертає nan: у таблиці буде n/a, решта бенчмарку працює.
    """
    try:
        import resource

        peak = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
        return peak / (1024 * 1024 if sys.platform == "darwin" else 1024)
    except ImportError:  # Windows
        pass
    try:
        import psutil
    except ImportError:
        return math.nan
    return psutil.Process().memory_info().peak_wset / 1024 / 1024  # type: ignore[attr-defined]  # peak_wset є лише у psutil на Windows


def run_one(corpus: Path, executor: str, workers: int, positions: bool) -> dict[str, Any]:
    """Один замір (виконується в дочірньому процесі)."""
    from findex.parallel import build_parallel

    timings: dict[str, float] = {}
    cpu_before, wall_before = time.process_time(), time.perf_counter()
    build_parallel(corpus, positions, executor=executor, workers=workers, timings=timings)  # type: ignore[arg-type]  # виконавець приходить рядком з командного рядка
    wall = time.perf_counter() - wall_before
    cpu = time.process_time() - cpu_before
    gil = getattr(sys, "_is_gil_enabled", None)
    gil_enabled = None if gil is None else bool(gil())  # дивимось ДО імпорту psutil: розширення може увімкнути GIL
    if executor == "processes":
        cpu += timings["worker_cpu"]  # CPU воркерів-процесів у батьківський process_time не потрапляє
    return {
        "wall": wall,
        "cpu": cpu,
        "rss_mb": peak_rss_mb(),
        "map": timings["map"],
        "merge": timings["merge"],
        "gil_enabled": gil_enabled,
    }


def measure(corpus: Path, executor: str, workers: int, positions: bool) -> dict[str, Any]:
    command = [sys.executable, __file__, "--one", str(corpus), executor, str(workers)]
    if not positions:
        command.append("--no-positions")
    done = subprocess.run(command, capture_output=True, text=True, check=False)
    if done.returncode != 0:
        raise SystemExit(f"замір {executor}×{workers} впав:\n{done.stderr}")
    return json.loads(done.stdout.strip().splitlines()[-1])


def gil_state() -> str:
    check = getattr(sys, "_is_gil_enabled", None)  # з'явилось в 3.13
    return "n/a (< 3.13)" if check is None else ("увімкнено" if check() else "ВИМКНЕНО")


def machine_info() -> dict[str, Any]:
    return {
        "python": platform.python_version(),
        "implementation": platform.python_implementation(),
        "gil": gil_state(),
        "platform": platform.platform(),
        "processor": platform.processor(),
        "cpu_count": os.cpu_count(),
    }


def median_rows(runs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key in dict.fromkeys((r["executor"], r["workers"]) for r in runs):
        group = [r for r in runs if (r["executor"], r["workers"]) == key]
        row: dict[str, Any] = {"executor": key[0], "workers": key[1], "runs": len(group)}
        for field in FIELDS:
            row[field] = statistics.median(r[field] for r in group)
        rows.append(row)
    base = next(r["wall"] for r in rows if r["executor"] == "serial")
    for row in rows:
        row["speedup"] = base / row["wall"]
    return rows


def rss_text(value: float) -> str:
    return "n/a" if math.isnan(value) else f"{value:.0f}"


def markdown_table(rows: list[dict[str, Any]]) -> str:
    lines = [
        "| Executor | Workers | Wall (s, median) | CPU (s) | Peak RSS (MB) | Merge (s) | Speedup |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for r in rows:
        lines.append(
            f"| {r['executor']} | {r['workers']} | {r['wall']:.2f} | {r['cpu']:.2f} | {rss_text(r['rss_mb'])} "
            f"| {r['merge']:.3f} | {r['speedup']:.2f}× |"
        )
    return "\n".join(lines)


def plot(rows: list[dict[str, Any]], png: Path, title: str) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(7, 4.5))
    top = max(r["workers"] for r in rows)
    ax.plot([1, top], [1, top], "k--", label="ідеальний лінійний ріст")
    for executor, marker in (("threads", "s"), ("processes", "o")):
        pts = sorted((r["workers"], r["speedup"]) for r in rows if r["executor"] == executor)
        if pts:
            ax.plot(*zip(*pts, strict=True), marker=marker, label=executor)
    ax.axhline(1.0, color="gray", lw=0.8)
    ax.set_xlabel("воркерів")
    ax.set_ylabel("прискорення відносно serial")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)
    png.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(png, dpi=150)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("corpus", type=Path, nargs="?")
    parser.add_argument("--repeat", type=int, default=3, help="замірів на клітинку (не менше 3)")
    parser.add_argument("--workers", type=int, nargs="+", help="типово: 1 2 4 8 і число ядер")
    parser.add_argument("--executors", nargs="+", default=["threads", "processes"], choices=["threads", "processes"])
    parser.add_argument("--no-positions", action="store_true")
    parser.add_argument("--tag", default="", help="суфікс імен вихідних файлів, напр. py313t")
    parser.add_argument("--out-dir", type=Path, default=Path("bench"))
    parser.add_argument("--png", type=Path, default=None, help="куди зберегти графік (типово bench/speedup[-tag].png)")
    parser.add_argument("--one", nargs=3, metavar=("CORPUS", "EXECUTOR", "WORKERS"), help=argparse.SUPPRESS)
    args = parser.parse_args()

    positions = not args.no_positions
    if args.one:
        corpus, executor, workers = args.one
        print(json.dumps(run_one(Path(corpus), executor, int(workers), positions)))
        return

    if args.corpus is None:
        parser.error("потрібен каталог корпусу")
    info = machine_info()
    print(json.dumps(info, ensure_ascii=False, indent=2))
    workers_list = sorted(set(args.workers or [1, 2, 4, 8, os.cpu_count() or 1]))

    for path in args.corpus.rglob("*.txt"):  # прогрів файлового кешу ОС
        path.read_bytes()
    measure(args.corpus, "serial", 1, positions)  # прогрів: імпорти, .pyc

    plan = [("serial", 1)] + [(e, w) for e in args.executors for w in workers_list]
    runs: list[dict[str, Any]] = []
    for executor, workers in plan:
        for attempt in range(args.repeat):
            result = measure(args.corpus, executor, workers, positions)
            runs.append({"executor": executor, "workers": workers, **result})
            print(f"{executor:9} x{workers:<2} run {attempt + 1}/{args.repeat}: wall {result['wall']:.2f} s", flush=True)

    rows = median_rows(runs)
    suffix = f"-{args.tag}" if args.tag else ""
    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / f"results{suffix}.json").write_text(
        json.dumps(
            {"machine": info, "positions": positions, "runs": runs, "median": rows}, ensure_ascii=False, indent=2
        ),
        encoding="utf-8",
    )
    table = markdown_table(rows)
    (args.out_dir / f"table{suffix}.md").write_text(table + "\n", encoding="utf-8")
    gil_seen = sorted({str(r["gil_enabled"]) for r in runs})
    print(f"\nGIL під час замірів (за sys._is_gil_enabled()): {', '.join(gil_seen)}")
    print("\n" + table)
    try:
        plot(rows, args.png or args.out_dir / f"speedup{suffix}.png", f"Прискорення індексації (Python {info['python']})")
    except ImportError:
        print("\nmatplotlib не встановлено: графік пропущено (таблиця й JSON збережені).")


if __name__ == "__main__":
    main()
