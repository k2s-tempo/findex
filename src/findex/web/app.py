"""FastAPI-застосунок findex: lifespan завантажує індекс один раз, middleware веде id запитів."""

import logging
import struct
import sys
import time
import uuid
from collections.abc import AsyncGenerator, Awaitable, Callable
from contextlib import asynccontextmanager
from importlib import metadata
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from findex.cache import cached_evaluator
from findex.index import Index
from findex.storage import load
from findex.web.pages import make_pages_router
from findex.web.routes import make_router
from findex.web.settings import Settings, get_settings

log = logging.getLogger("findex.web")

_LOAD_ERRORS = (OSError, ValueError, struct.error, EOFError)


def configure_logging(level: str) -> None:
    """Логи в stderr (їх бачить панель платформи). Якщо CLI вже поставив обробник - не дублюємо."""
    root = logging.getLogger("findex")
    if not root.handlers:
        handler = logging.StreamHandler(sys.stderr)
        handler.setFormatter(logging.Formatter("%(asctime)s %(levelname)s %(name)s: %(message)s"))
        root.addHandler(handler)
    root.setLevel(level.upper())


def _try_load(settings: Settings) -> Index | None:
    try:
        index = load(settings.index_path, "array")  # лише власний бінарний формат: pickle тут небезпечний
    except _LOAD_ERRORS:
        log.exception("не вдалося завантажити індекс з %s", settings.index_path)
        return None
    log.info("індекс завантажено: %d док., %d термінів", index.num_docs, len(index))
    return index


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    settings = get_settings()
    configure_logging(settings.log_level)
    index = _try_load(settings)
    app.state.index = index
    app.state.evaluator = cached_evaluator(index) if index is not None else None
    try:
        yield
    finally:
        if index is not None:
            index.close()


async def request_context(request: Request, call_next: Callable[[Request], Awaitable[Response]]) -> Response:
    """id запиту; неочікуваний виняток -> у лог зі стеком і id, назовні 500 без стека."""
    request_id = uuid.uuid4().hex[:12]
    request.state.request_id = request_id
    started = time.perf_counter()
    try:
        response = await call_next(request)
    except Exception:
        log.exception(
            "необроблений виняток, request_id=%s, %s %s", request_id, request.method, request.url.path
        )
        response = JSONResponse(
            {"detail": "Internal Server Error", "request_id": request_id}, status_code=500
        )
    response.headers["X-Request-ID"] = request_id
    log.info(
        "%s %s -> %d (%.1f мс) request_id=%s",
        request.method,
        request.url.path,
        response.status_code,
        (time.perf_counter() - started) * 1000,
        request_id,
    )
    return response


def _version() -> str:
    try:
        return metadata.version("findex")
    except metadata.PackageNotFoundError:
        return "0+unknown"


def create_app(search_mode: str | None = None) -> FastAPI:
    mode = search_mode or get_settings().search_mode
    app = FastAPI(
        title="findex",
        version=_version(),
        lifespan=lifespan,
        docs_url="/api/docs",  # /docs/{doc_id} зайнятий під документи
        redoc_url=None,
    )
    app.middleware("http")(request_context)
    app.include_router(make_router(mode))
    app.include_router(make_pages_router())
    app.mount("/static", StaticFiles(directory=Path(__file__).parent / "static"), name="static")
    return app


app = create_app()
