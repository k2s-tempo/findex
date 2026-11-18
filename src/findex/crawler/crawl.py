from __future__ import annotations

import asyncio
import contextlib
import time
from collections.abc import AsyncIterator, Callable, Iterable
from dataclasses import dataclass, field
from datetime import UTC, datetime
from html.parser import HTMLParser
from urllib.parse import parse_qsl, urlencode, urljoin, urlsplit, urlunsplit

import httpx

from findex.crawler.fetch import USER_AGENT, FetchResult, fetch
from findex.crawler.robots import RobotsCache


@dataclass(frozen=True, slots=True)
class Page:
    url: str
    title: str
    text: str
    fetched_at: str


@dataclass(slots=True)
class CrawlStats:
    fetched: int = 0
    errors: int = 0
    in_flight: int = 0
    queue_depth: int = 0
    started_at: float = field(default_factory=time.perf_counter)

    @property
    def pages_per_sec(self) -> float:
        dt = max(1e-6, time.perf_counter() - self.started_at)
        return self.fetched / dt


def canonicalize_url(url: str) -> str:
    parts = urlsplit(url.strip())
    scheme = parts.scheme.lower()
    netloc = parts.netloc.lower()
    if scheme == "http" and netloc.endswith(":80"):
        netloc = netloc[:-3]
    elif scheme == "https" and netloc.endswith(":443"):
        netloc = netloc[:-4]

    path = parts.path or "/"
    if len(path) > 1 and path.endswith("/"):
        path = path.rstrip("/")

    sorted_query = urlencode(sorted(parse_qsl(parts.query, keep_blank_values=True)))
    return urlunsplit((scheme, netloc, path, sorted_query, ""))


class _HTMLExtractor(HTMLParser):
    _SKIP_TAGS = frozenset({"script", "style", "noscript", "svg"})

    def __init__(self) -> None:
        super().__init__()
        self._skip_depth = 0
        self._in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []
        self.links: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS:
            self._skip_depth += 1
        elif t == "title":
            self._in_title = True
        elif t == "a":
            for k, v in attrs:
                if k.lower() == "href" and v is not None and v.strip():
                    self.links.append(v.strip())

    def handle_endtag(self, tag: str) -> None:
        t = tag.lower()
        if t in self._SKIP_TAGS and self._skip_depth > 0:
            self._skip_depth -= 1
        elif t == "title":
            self._in_title = False

    def handle_data(self, data: str) -> None:
        if self._skip_depth > 0:
            return
        stripped = data.strip()
        if not stripped:
            return
        if self._in_title:
            self.title_parts.append(stripped)
        self.text_parts.append(stripped)


def extract_page_and_links(base_url: str, html: str) -> tuple[str, str, list[str]]:
    parser = _HTMLExtractor()
    parser.feed(html)
    title = " ".join(parser.title_parts).strip()
    text = " ".join(parser.text_parts).strip()
    links: list[str] = []
    for raw_href in parser.links:
        joined = urljoin(base_url, raw_href)
        parts = urlsplit(joined)
        if parts.scheme.lower() in ("http", "https") and parts.netloc:
            links.append(canonicalize_url(joined))
    return title, text, links


async def crawl(
    seeds: Iterable[str],
    *,
    max_pages: int = 200,
    concurrency: int = 10,
    per_host: int = 2,
    per_host_delay: float = 0.25,
    allowed_domains: set[str] | None = None,
    timeout: float = 10.0,
    transport: httpx.AsyncBaseTransport | None = None,
    stats: CrawlStats | None = None,
    on_fetch: Callable[[FetchResult | None, str, str | None], None] | None = None,
) -> AsyncIterator[Page]:
    if max_pages < 1:
        return

    queue: asyncio.Queue[str] = asyncio.Queue()
    out_queue: asyncio.Queue[Page | None] = asyncio.Queue()
    seen: set[str] = set()
    active_stats = stats if stats is not None else CrawlStats()

    seed_list = [canonicalize_url(s) for s in seeds]
    domains = (
        {d.lower() for d in allowed_domains}
        if allowed_domains is not None
        else {urlsplit(s).netloc.lower() for s in seed_list}
        | {f"www.{urlsplit(s).netloc.lower()}" for s in seed_list}
    )

    for url in seed_list:
        if url not in seen and urlsplit(url).netloc.lower() in domains:
            seen.add(url)
            queue.put_nowait(url)
    active_stats.queue_depth = queue.qsize()

    global_sem = asyncio.Semaphore(concurrency)
    host_sems: dict[str, asyncio.Semaphore] = {}
    host_rate_locks: dict[str, asyncio.Lock] = {}
    host_last_request: dict[str, float] = {}
    robots = RobotsCache()

    stop_event = asyncio.Event()
    accepted_pages = 0
    accept_lock = asyncio.Lock()

    async def wait_host_politeness(host: str) -> None:
        if per_host_delay <= 0:
            return
        lock = host_rate_locks.setdefault(host, asyncio.Lock())
        async with lock:
            now = time.perf_counter()
            last = host_last_request.get(host)
            if last is not None:
                wait_for = per_host_delay - (now - last)
                if wait_for > 0:
                    await asyncio.sleep(wait_for)
            host_last_request[host] = time.perf_counter()

    async def worker(client: httpx.AsyncClient) -> None:
        nonlocal accepted_pages
        while not stop_event.is_set():
            url = await queue.get()
            active_stats.queue_depth = queue.qsize()
            host = urlsplit(url).netloc.lower()
            host_sem = host_sems.setdefault(host, asyncio.Semaphore(per_host))

            try:
                if stop_event.is_set():
                    continue

                if not await robots.allowed(client, url):
                    if on_fetch is not None:
                        on_fetch(None, url, "disallowed by robots.txt")
                    continue

                async with global_sem, host_sem:
                    if stop_event.is_set():
                        continue
                    await wait_host_politeness(host)
                    active_stats.in_flight += 1
                    try:
                        res = await fetch(client, url, timeout=timeout)
                    finally:
                        active_stats.in_flight -= 1

                if on_fetch is not None:
                    on_fetch(res, url, None)

                if res.status != 200:
                    active_stats.errors += 1
                    continue

                title, text, links = await asyncio.to_thread(extract_page_and_links, res.final_url, res.text)
                page = Page(
                    url=canonicalize_url(res.final_url),
                    title=title,
                    text=text,
                    fetched_at=datetime.now(UTC).isoformat(),
                )

                should_emit = False
                async with accept_lock:
                    if accepted_pages < max_pages:
                        accepted_pages += 1
                        active_stats.fetched = accepted_pages
                        should_emit = True
                        if accepted_pages >= max_pages:
                            stop_event.set()

                if should_emit:
                    await out_queue.put(page)

                if not stop_event.is_set():
                    for link in links:
                        if link in seen:
                            continue
                        if urlsplit(link).netloc.lower() in domains:
                            seen.add(link)
                            queue.put_nowait(link)
                    active_stats.queue_depth = queue.qsize()

            except asyncio.CancelledError:
                raise
            except Exception as exc:
                active_stats.errors += 1
                if on_fetch is not None:
                    on_fetch(None, url, str(exc))
            finally:
                queue.task_done()
                active_stats.queue_depth = queue.qsize()

    async def runner() -> None:
        headers = {"User-Agent": USER_AGENT}
        async with httpx.AsyncClient(
            headers=headers,
            follow_redirects=True,
            max_redirects=5,
            transport=transport,
        ) as client:
            try:
                async with asyncio.TaskGroup() as tg:
                    tasks = [tg.create_task(worker(client)) for _ in range(concurrency)]
                    join_task = tg.create_task(queue.join())
                    stop_task = tg.create_task(stop_event.wait())
                    _, pending = await asyncio.wait(
                        {join_task, stop_task}, return_when=asyncio.FIRST_COMPLETED
                    )
                    stop_event.set()
                    for p in pending:
                        p.cancel()
                    for t in tasks:
                        t.cancel()
            except* asyncio.CancelledError:
                pass
            finally:
                await out_queue.put(None)

    runner_task = asyncio.create_task(runner())
    try:
        while True:
            item = await out_queue.get()
            if item is None:
                break
            yield item
    finally:
        stop_event.set()
        if not runner_task.done():
            runner_task.cancel()
            with contextlib.suppress(asyncio.CancelledError):
                await runner_task
