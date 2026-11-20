"""Обмеження одночасних пошуків; /health не залежить від пулу потоків."""

import inspect
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from findex.index import DocMeta, Index, Posting
from findex.web import service
from findex.web.app import create_app
from findex.web.deps import get_index
from findex.web.routes import make_router
from findex.web.settings import Settings, get_settings


def make_index(tmp_path: Path) -> Index:
    (tmp_path / "plato.txt").write_text("justice and virtue are one.\n", encoding="utf-8")
    docs = [DocMeta(0, "plato.txt", str(tmp_path / "plato.txt"), 5)]
    return Index({"justice": [Posting(0, 1)], "virtue": [Posting(0, 1)]}, docs, None)


def test_settings_concurrency_default_and_env(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("MAX_CONCURRENT_SEARCHES", raising=False)
    assert Settings().max_concurrent_searches == 4
    monkeypatch.setenv("MAX_CONCURRENT_SEARCHES", "2")
    assert Settings().max_concurrent_searches == 2
    monkeypatch.setenv("MAX_CONCURRENT_SEARCHES", "0")
    with pytest.raises(ValidationError):
        Settings()


def test_gate_is_built_from_settings(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(service, "_gate", None)
    monkeypatch.setenv("MAX_CONCURRENT_SEARCHES", "3")
    get_settings.cache_clear()
    try:
        gate = service._search_gate()
        assert [gate.acquire(blocking=False) for _ in range(4)] == [True, True, True, False]
    finally:
        get_settings.cache_clear()


def test_concurrent_searches_are_bounded(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    index = make_index(tmp_path)
    monkeypatch.setattr(service, "_gate", threading.BoundedSemaphore(2))
    state = {"active": 0, "peak": 0}
    lock = threading.Lock()
    real = service.ranked_search

    def slow(*args, **kwargs):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
        try:
            time.sleep(0.05)
            return real(*args, **kwargs)
        finally:
            with lock:
                state["active"] -= 1

    monkeypatch.setattr(service, "ranked_search", slow)
    app = create_app("sync")
    app.dependency_overrides[get_index] = lambda: index

    def call(_: int) -> int:
        return TestClient(app).get("/search", params={"q": "virtue"}).status_code

    with ThreadPoolExecutor(8) as pool:
        codes = list(pool.map(call, range(8)))
    assert codes == [200] * 8
    assert 1 <= state["peak"] <= 2


def test_health_does_not_use_thread_pool() -> None:
    endpoint = next(r.endpoint for r in make_router("sync").routes if getattr(r, "path", "") == "/health")
    assert inspect.iscoroutinefunction(endpoint)
