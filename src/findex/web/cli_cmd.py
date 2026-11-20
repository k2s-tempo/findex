"""Команда findex serve: піднімає uvicorn (вхід у програму лишається через CLI)."""

import os
from pathlib import Path
from typing import Annotated

import typer

from findex.web.settings import Settings


def register_serve(app: typer.Typer) -> None:
    @app.command("serve")
    def serve_cmd(
        index: Annotated[
            Path | None, typer.Option("--index", "-i", help="Файл індексу (інакше змінна INDEX_PATH).")
        ] = None,
        host: Annotated[str, typer.Option("--host", envvar="HOST", help="Адреса прослуховування.")] = "127.0.0.1",
        port: Annotated[int, typer.Option("--port", "-p", envvar="PORT", min=1, max=65535)] = 8000,
        workers: Annotated[int, typer.Option("--workers", "-w", min=1, help="Кількість процесів.")] = 1,
    ) -> None:
        """Запустити веб-сервіс пошуку (HTTP API + сторінка)."""
        import uvicorn  # імпорт тут, щоб решта команд CLI не платила за його завантаження

        if index is not None:
            os.environ["INDEX_PATH"] = str(index)  # воркери-процеси успадковують оточення
        settings = Settings()
        if not settings.index_path.is_file():
            typer.echo(f"Помилка: файл індексу не знайдено: {settings.index_path}", err=True)
            raise typer.Exit(code=1)
        uvicorn.run(
            "findex.web.app:app",
            host=host,
            port=port,
            workers=workers,
            log_level=settings.log_level.lower(),
        )
