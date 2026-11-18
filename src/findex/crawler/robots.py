from __future__ import annotations

import asyncio
from urllib.parse import urlsplit
from urllib.robotparser import RobotFileParser

import httpx

from findex.crawler.fetch import USER_AGENT


class RobotsCache:
    def __init__(self, user_agent: str = USER_AGENT, timeout: float = 5.0) -> None:
        self.user_agent = user_agent
        self.timeout = timeout
        self._parsers: dict[str, RobotFileParser | None] = {}
        self._disallowed_hosts: set[str] = set()
        self._locks: dict[str, asyncio.Lock] = {}

    async def allowed(self, client: httpx.AsyncClient, url: str) -> bool:
        parts = urlsplit(url)
        host_key = f"{parts.scheme}://{parts.netloc}"

        lock = self._locks.setdefault(host_key, asyncio.Lock())
        async with lock:
            if host_key not in self._parsers:
                robots_url = f"{host_key}/robots.txt"
                parser = RobotFileParser()
                parser.set_url(robots_url)
                try:
                    async with asyncio.timeout(self.timeout):
                        resp = await client.get(robots_url)
                    if resp.status_code == 200:
                        await asyncio.to_thread(parser.parse, resp.text.splitlines())
                        self._parsers[host_key] = parser
                    elif resp.status_code in (401, 403):
                        self._disallowed_hosts.add(host_key)
                        self._parsers[host_key] = None
                    else:
                        self._parsers[host_key] = None
                except (TimeoutError, httpx.HTTPError):
                    self._parsers[host_key] = None

        if host_key in self._disallowed_hosts:
            return False
        cached = self._parsers[host_key]
        if cached is None:
            return True
        return cached.can_fetch(self.user_agent, url)
