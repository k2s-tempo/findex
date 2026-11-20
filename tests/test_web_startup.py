"""lifespan завантажує індекс зі Settings.index_path; findex serve запускає uvicorn."""

import logging
from collections.abc import Iterator
from importlib import metadata
from pathlib import Path

import pytest
from fastapi.testclient import TestClient
from typer.testing import CliRunner

from findex.cli import app as cli_app
from findex.index import DocMeta, Index, Posting
from findex.storage import save
from findex.web import app as web_app
from findex.web.app import create_app
from findex.web.settings import get_settings


@pytest.fixture(autouse=True)
def _isolated_state() -> Iterator[None]:
    """Settings кешуються, а configure_logging чіпає логер findex: усе повертаємо як було."""
    logger = logging.getLogger("findex")
    handlers, level = list(logger.handlers), logger.level
    get_settings.cache_clear()
    yield
    get_settings.cache_clear()
    logger.handlers[:] = handlers
    logger.setLevel(level)


def write_index(tmp_path: Path) -> Path:
    (tmp_path / "plato.txt").write_text("justice and virtue are one.\n", encoding="utf-8")
    index = Index(
        {"justice": [Posting(0, 1)], "virtue": [Posting(0, 1)]},
        [DocMeta(0, "plato.txt", str(tmp_path / "plato.txt"), 5)],
        None,
    )
    path = tmp_path / "index.bin"
    save(index, path, "array")
    return path


def test_lifespan_loads_index_once(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("INDEX_PATH", str(write_index(tmp_path)))
    with TestClient(create_app("sync")) as client:
        assert client.get("/health").json() == {"status": "ok", "docs": 1}
        body = client.get("/search", params={"q": "justice"}).json()
        assert body["total"] == 1
        assert "**justice**" in body["results"][0]["snippet"]


def _truncated(tmp_path: Path) -> Path:
    path = tmp_path / "cut.bin"
    path.write_bytes(write_index(tmp_path).read_bytes()[:20])
    return path


def _garbage(tmp_path: Path) -> Path:
    path = tmp_path / "garbage.bin"
    path.write_bytes(b"this is not an index")
    return path


@pytest.mark.parametrize("make_path", [lambda p: p / "missing.bin", _garbage, _truncated])
def test_bad_index_file_gives_503_and_logs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, caplog: pytest.LogCaptureFixture, make_path
) -> None:
    monkeypatch.setenv("INDEX_PATH", str(make_path(tmp_path)))
    with caplog.at_level(logging.ERROR, logger="findex.web"), TestClient(create_app("sync")) as client:
        assert client.get("/health").status_code == 503
        assert client.get("/stats").status_code == 503
    assert "не вдалося завантажити індекс" in caplog.text


def test_version_fallback_when_package_not_installed(monkeypatch: pytest.MonkeyPatch) -> None:
    def boom(name: str) -> str:
        raise metadata.PackageNotFoundError(name)

    monkeypatch.setattr(web_app.metadata, "version", boom)
    assert web_app._version() == "0+unknown"


def _fake_uvicorn(monkeypatch: pytest.MonkeyPatch) -> dict[str, object]:
    calls: dict[str, object] = {}

    def fake_run(target: str, **kwargs: object) -> None:
        calls["target"] = target
        calls.update(kwargs)

    monkeypatch.setattr("uvicorn.run", fake_run)
    for name in ("HOST", "PORT", "LOG_LEVEL"):
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv("INDEX_PATH", "placeholder")  # serve пише INDEX_PATH в оточення: monkeypatch поверне як було
    return calls


def test_serve_starts_uvicorn(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_uvicorn(monkeypatch)
    index = write_index(tmp_path)
    result = CliRunner().invoke(
        cli_app, ["serve", "--index", str(index), "--host", "0.0.0.0", "--port", "9001", "--workers", "2"]
    )
    assert result.exit_code == 0
    assert calls == {
        "target": "findex.web.app:app",
        "host": "0.0.0.0",
        "port": 9001,
        "workers": 2,
        "log_level": "info",
    }


def test_serve_defaults_use_index_path_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_uvicorn(monkeypatch)
    monkeypatch.setenv("INDEX_PATH", str(write_index(tmp_path)))
    result = CliRunner().invoke(cli_app, ["serve"])
    assert result.exit_code == 0
    assert (calls["host"], calls["port"], calls["workers"]) == ("127.0.0.1", 8000, 1)


def test_serve_reads_port_and_host_from_env(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_uvicorn(monkeypatch)
    monkeypatch.setenv("INDEX_PATH", str(write_index(tmp_path)))
    monkeypatch.setenv("PORT", "10000")
    monkeypatch.setenv("HOST", "0.0.0.0")
    assert CliRunner().invoke(cli_app, ["serve"]).exit_code == 0
    assert (calls["host"], calls["port"]) == ("0.0.0.0", 10000)


def test_serve_fails_without_index_file(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _fake_uvicorn(monkeypatch)
    result = CliRunner().invoke(cli_app, ["serve", "--index", str(tmp_path / "nope.bin")])
    assert result.exit_code == 1
    assert "не знайдено" in result.output
    assert calls == {}
