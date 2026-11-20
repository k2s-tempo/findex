"""Тести веб-шару: TestClient + dependency_overrides, без справжнього index.bin."""

import logging
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.deps import get_index


def make_index(tmp_path: Path) -> Index:
    (tmp_path / "plato.txt").write_text("justice and virtue are one. virtue is knowledge.\n", encoding="utf-8")
    (tmp_path / "crito.txt").write_text("crito speaks of virtue and of the laws.\n", encoding="utf-8")
    docs = [
        DocMeta(0, "plato.txt", str(tmp_path / "plato.txt"), 9),
        DocMeta(1, "crito.txt", str(tmp_path / "crito.txt"), 8),
    ]
    postings = {
        "justice": [Posting(0, 1)],
        "virtue": [Posting(0, 2), Posting(1, 1)],
        "crito": [Posting(1, 1)],
    }
    return Index(postings, docs, None)


@pytest.fixture(params=["sync", "async"])
def client(request: pytest.FixtureRequest, tmp_path: Path) -> TestClient:
    index = make_index(tmp_path)
    app = create_app(request.param)
    app.dependency_overrides[get_index] = lambda: index
    return TestClient(app)


def test_search_ok(client: TestClient) -> None:
    r = client.get("/search", params={"q": "virtue", "k": 1})
    assert r.status_code == 200
    body = r.json()
    assert body["total"] == 2
    assert body["total_pages"] == 2
    assert len(body["results"]) == 1
    assert set(body["results"][0]) == {"rank", "doc_id", "name", "score", "tokens", "snippet"}
    assert "**virtue**" in body["results"][0]["snippet"]


def test_search_pagination(client: TestClient) -> None:
    first = client.get("/search", params={"q": "virtue", "k": 1, "page": 1}).json()
    second = client.get("/search", params={"q": "virtue", "k": 1, "page": 2}).json()
    assert first["results"][0]["doc_id"] != second["results"][0]["doc_id"]
    assert [first["results"][0]["rank"], second["results"][0]["rank"]] == [1, 2]
    assert client.get("/search", params={"q": "virtue", "k": 1, "page": 3}).json()["results"] == []


def test_search_boolean(client: TestClient) -> None:
    body = client.get("/search", params={"q": "justice AND virtue"}).json()
    assert body["total"] == 1
    assert body["results"][0]["name"] == "plato.txt"


@pytest.mark.parametrize(
    "params",
    [
        {},
        {"q": ""},
        {"q": "virtue", "k": 0},
        {"q": "virtue", "k": 51},
        {"q": "virtue", "page": 0},
        {"q": "virtue", "scorer": "nope"},
        {"q": "virtue", "extra": "1"},
        {"q": "(virtue"},
        {"q": "   "},
    ],
)
def test_search_422(client: TestClient, params: dict[str, str | int]) -> None:
    assert client.get("/search", params=params).status_code == 422


def test_doc_found_and_404(client: TestClient) -> None:
    ok = client.get("/docs/0")
    assert ok.status_code == 200
    assert ok.json()["name"] == "plato.txt"
    assert client.get("/docs/99").status_code == 404


def test_doc_file_missing_is_404(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    (tmp_path / "plato.txt").unlink()
    app = create_app("sync")
    app.dependency_overrides[get_index] = lambda: index
    assert TestClient(app).get("/docs/0").status_code == 404


def test_stats(client: TestClient) -> None:
    body = client.get("/stats").json()
    assert body["docs"] == 2
    assert body["terms"] == 3
    assert body["postings"] == 4
    assert body["has_positions"] is False


def test_health_ok(client: TestClient) -> None:
    assert client.get("/health").json() == {"status": "ok", "docs": 2}


def test_health_503_when_index_not_loaded() -> None:
    # без with TestClient(...) lifespan не запускається, тож індекс не завантажений
    r = TestClient(create_app("sync")).get("/health")
    assert r.status_code == 503


def test_unexpected_error_is_500_without_stack(caplog: pytest.LogCaptureFixture) -> None:
    def boom() -> Index:
        raise RuntimeError("boom-secret")

    app = create_app("sync")
    app.dependency_overrides[get_index] = boom
    with caplog.at_level(logging.ERROR, logger="findex.web"):
        r = TestClient(app).get("/stats")
    assert r.status_code == 500
    body = r.json()
    assert body["detail"] == "Internal Server Error"
    assert "boom-secret" not in r.text
    assert "Traceback" not in r.text
    assert r.headers["X-Request-ID"] == body["request_id"]
    assert body["request_id"] in caplog.text
    assert "boom-secret" in caplog.text  # у лозі стек є
