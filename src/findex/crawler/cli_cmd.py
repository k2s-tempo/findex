from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Annotated

import typer
from rich.console import Console
from rich.live import Live
from rich.table import Table

from findex.crawler.crawl import CrawlStats, crawl
from findex.crawler.fetch import FetchResult


def register_crawl(app: typer.Typer) -> None:
    @app.command("crawl")
    def crawl_cmd(
        seed_url: Annotated[str, typer.Argument(help="Seed URL.")],
        max_pages: Annotated[int, typer.Option("--max-pages", "-m", min=1)] = 500,
        concurrency: Annotated[int, typer.Option("--concurrency", "-c", min=1)] = 10,
        per_host: Annotated[int, typer.Option("--per-host", min=1)] = 2,
        delay: Annotated[float, typer.Option("--delay", min=0.0)] = 0.25,
        out: Annotated[Path, typer.Option("--out", "-o")] = Path("data/crawl.jsonl"),
        log_path: Annotated[Path, typer.Option("--log")] = Path("crawl.log"),
        debug_loop: Annotated[bool, typer.Option("--debug-loop")] = False,
    ) -> None:
        out.parent.mkdir(parents=True, exist_ok=True)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        stats = CrawlStats()

        def render() -> Table:
            t = Table(title="findex crawl")
            t.add_column("Pages", justify="right")
            t.add_column("Pages/s", justify="right")
            t.add_column("In-flight", justify="right")
            t.add_column("Errors", justify="right")
            t.add_column("Queue", justify="right")
            t.add_row(
                f"{stats.fetched}/{max_pages}",
                f"{stats.pages_per_sec:.2f}",
                str(stats.in_flight),
                str(stats.errors),
                str(stats.queue_depth),
            )
            return t

        async def _run() -> None:
            with (
                out.open("w", encoding="utf-8") as of,
                log_path.open("w", encoding="utf-8") as lf,
                Live(render(), console=Console(stderr=True)) as live,
            ):

                def on_fetch(res: FetchResult | None, url: str, err: str | None) -> None:
                    if res is not None:
                        lf.write(
                            f"{res.status}\t{res.num_bytes}B\t {res.elapsed:.3f}s\tattempts={res.attempts}\t{url}\n"
                        )
                    else:
                        lf.write(f"ERR\t0B\t0.000s\t{err or 'unknown'}\t{url}\n")
                    lf.flush()
                    live.update(render())

                async for page in crawl(
                    [seed_url],
                    max_pages=max_pages,
                    concurrency=concurrency,
                    per_host=per_host,
                    per_host_delay=delay,
                    stats=stats,
                    on_fetch=on_fetch,
                ):
                    of.write(
                        json.dumps(
                            {
                                "url": page.url,
                                "title": page.title,
                                "text": page.text,
                                "fetched_at": page.fetched_at,
                            },
                            ensure_ascii=False,
                        )
                        + "\n"
                    )
                    of.flush()
                    live.update(render())

        asyncio.run(_run(), debug=debug_loop)
        typer.echo(
            f"Done: {stats.fetched} pages ({stats.pages_per_sec:.2f} pages/s), errors: {stats.errors}. Out: {out}"
        )
