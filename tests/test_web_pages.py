"""Тести HTML-сторінок: головна, пошук, підсвітка, екранування, пагінація, перегляд документа."""

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from findex.index import DocMeta, Index, Posting
from findex.web.app import create_app
from findex.web.deps import get_index


def make_index(tmp_path: Path) -> Index:
    (tmp_path / "plato.txt").write_text(
        "justice and virtue are one.\n\nvirtue is <b>knowledge</b> & more.\n", encoding="utf-8"
    )
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


@pytest.fixture
def client(tmp_path: Path) -> TestClient:
    index = make_index(tmp_path)
    app = create_app("sync")
    app.dependency_overrides[get_index] = lambda: index
    return TestClient(app)


def test_home_empty_shows_examples(client: TestClient) -> None:
    r = client.get("/")
    assert r.status_code == 200
    assert "text/html" in r.headers["content-type"]
    assert "justice AND virtue" in r.text
    assert "2 документів" in r.text


def test_search_page_has_highlight_and_scores(client: TestClient) -> None:
    r = client.get("/", params={"q": "virtue"})
    assert r.status_code == 200
    assert "<mark>virtue</mark>" in r.text
    assert "plato.txt" in r.text
    assert 'class="score"' in r.text


def test_search_page_no_results(client: TestClient) -> None:
    r = client.get("/", params={"q": "zzzz"})
    assert r.status_code == 200
    assert "Нічого не знайдено" in r.text


def test_search_page_bad_query_is_422_with_message(client: TestClient) -> None:
    r = client.get("/", params={"q": "(virtue"})
    assert r.status_code == 422
    assert "Некоректний запит" in r.text


def test_query_is_escaped(client: TestClient) -> None:
    r = client.get("/", params={"q": "<script>alert(1)</script>"})
    assert "<script>alert(1)</script>" not in r.text


def test_pagination_links(client: TestClient, monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr("findex.web.pages.PAGE_SIZE", 1)
    first = client.get("/", params={"q": "virtue", "page": 1}).text
    assert "page=2" in first
    assert "Попередня" not in first
    second = client.get("/", params={"q": "virtue", "page": 2}).text
    assert "Попередня" in second
    assert "Наступна" not in second


def test_doc_view_escapes_text_and_links_back(client: TestClient) -> None:
    r = client.get("/d/0", params={"q": "virtue"})
    assert r.status_code == 200
    assert "&lt;b&gt;knowledge&lt;/b&gt;" in r.text
    assert "<b>knowledge</b>" not in r.text
    assert "q=virtue" in r.text


def test_doc_view_404(client: TestClient) -> None:
    assert client.get("/d/99").status_code == 404


def test_doc_view_missing_file_404(tmp_path: Path) -> None:
    index = make_index(tmp_path)
    (tmp_path / "plato.txt").unlink()
    app = create_app("sync")
    app.dependency_overrides[get_index] = lambda: index
    assert TestClient(app).get("/d/0").status_code == 404


def test_home_503_when_index_not_loaded() -> None:
    assert TestClient(create_app("sync")).get("/").status_code == 503


def test_static_css(client: TestClient) -> None:
    r = client.get("/static/style.css")
    assert r.status_code == 200
    assert "--accent" in r.text
