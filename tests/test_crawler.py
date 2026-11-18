from __future__ import annotations

import asyncio
import json
from pathlib import Path

import httpx

from findex.corpus import iter_documents
from findex.crawler.crawl import canonicalize_url, crawl
from findex.crawler.fetch import fetch
from findex.crawler.robots import RobotsCache


def test_canonicalize_url() -> None:
    u1 = canonicalize_url("HTTP://Example.COM:http/?")
    u2 = canonicalize_url("HTTP://Example.COM:80/docs/page/?b=2&a=1#top")
    u3 = canonicalize_url("http://example.com/docs/page?a=1&b=2")
    assert u2 == u3
    assert u1.startswith("http://")


def test_fetch_retries_429_and_respects_retry_after() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        if calls == 1:
            return httpx.Response(429, headers={"Retry-After": "0"}, text="rate limited")
        return httpx.Response(200, text="ok")

    async def run() -> int:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "https://example.org/a", base_backoff=0.01)
            assert res.attempts == 2
            assert res.text == "ok"
            return res.status

    assert asyncio.run(run()) == 200
    assert calls == 2


def test_fetch_does_not_retry_404() -> None:
    calls = 0

    def handler(request: httpx.Request) -> httpx.Response:
        nonlocal calls
        calls += 1
        return httpx.Response(404, text="not found")

    async def run() -> int:
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            res = await fetch(client, "https://example.org/missing", base_backoff=0.01)
            assert res.attempts == 1
            return res.status

    assert asyncio.run(run()) == 404
    assert calls == 1


def test_robots_cache() -> None:
    def handler(request: httpx.Request) -> httpx.Response:
        if request.url.path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /private/\n")
        return httpx.Response(200, text="ok")

    async def run() -> tuple[bool, bool]:
        cache = RobotsCache()
        transport = httpx.MockTransport(handler)
        async with httpx.AsyncClient(transport=transport) as client:
            ok_public = await cache.allowed(client, "https://docs.test/public/page")
            ok_private = await cache.allowed(client, "https://docs.test/private/secret")
            return ok_public, ok_private

    assert asyncio.run(run()) == (True, False)


def test_crawl_respects_max_pages_and_dedup(tmp_path: Path) -> None:
    pages_html = {
        "/": "<title>Home</title><a href='/p1'>1</a><a href='/p1#frag'>1dup</a><a href='/p2'>2</a><a href='/blocked'>no</a>",
        "/p1": "<title>Page One</title><p>async coroutine event loop</p><a href='/p3'>3</a>",
        "/p2": "<title>Page Two</title><p>semaphore queue frontier /p><a href='/p3'>3dup</a>",
        "/p3": "<title>Page Three</title><p>structured concurrency taskgroup</p>",
        "/blocked": "<title>Blocked</title><p>secret</p>",
    }
    fetched_paths: list[str] = []

    def handler(request: httpx.Request) -> httpx.Response:
        path = request.url.path
        if path == "/robots.txt":
            return httpx.Response(200, text="User-agent: *\nDisallow: /blocked\n")
        fetched_paths.append(path)
        html = pages_html.get(path)
        if html is None:
            return httpx.Response(404)
        return httpx.Response(200, text=html)

    async def run() -> list[str]:
        transport = httpx.MockTransport(handler)
        collected: list[str] = []
        jsonl_file = tmp_path / "crawl.jsonl"
        with jsonl_file.open("w", encoding="utf-8") as f:
            async for page in crawl(
                ["https://site.local/"],
                max_pages=3,
                concurrency=4,
                per_host=2,
                per_host_delay=0.0,
                transport=transport,
            ):
                collected.append(page.url)
                f.write(json.dumps({"url": page.url, "title": page.title, "text": page.text}) + "\n")
        return collected

    urls = asyncio.run(run())
    assert len(urls) == 3
    assert len(set(urls)) == 3
    assert "/blocked" not in fetched_paths

    docs = list(iter_documents(tmp_path / "crawl.jsonl"))
    assert len(docs) == 3
