from __future__ import annotations

import asyncio
import email.utils
import random
import time
from dataclasses import dataclass
from datetime import UTC, datetime

import httpx

USER_AGENT = "findex-crawler/0.6 (+https://github.com/k2s-tempo/findex; educational search engine)"


@dataclass(frozen=True, slots=True)
class FetchResult:
    url: str
    final_url: str
    status: int
    text: str
    num_bytes: int
    elapsed: float
    attempts: int


def parse_retry_after(header_value: str | None) -> float | None:
    if not header_value:
        return None
    val = header_value.strip()
    if val.isdigit():
        return max(0.0, float(val))
    try:
        dt = email.utils.parsedate_to_datetime(val)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=UTC)
        delta = (dt - datetime.now(UTC)).total_seconds()
        return max(0.0, delta)
    except (TypeError, ValueError):
        return None


def compute_backoff(attempt: int, base: float = 0.5, cap: float = 10.0) -> float:
    exp = min(cap, base * (2**attempt))
    return exp + random.uniform(0.0, 0.5)


async def fetch(
    client: httpx.AsyncClient,
    url: str,
    *,
    timeout: float = 10.0,
    max_retries: int = 3,
    base_backoff: float = 0.5,
) -> FetchResult:
    started = time.perf_counter()
    last_exc: Exception | None = None

    for attempt in range(max_retries + 1):
        try:
            async with asyncio.timeout(timeout):
                resp = await client.get(url)

            status = resp.status_code
            body = resp.text
            num_bytes = len(resp.content)
            elapsed = time.perf_counter() - started

            if 400 <= status < 500 and status != 429:
                return FetchResult(
                    url=url,
                    final_url=str(resp.url),
                    status=status,
                    text="",
                    num_bytes=num_bytes,
                    elapsed=elapsed,
                    attempts=attempt + 1,
                )

            if status == 429 or status >= 500:
                if attempt >= max_retries:
                    return FetchResult(
                        url=url,
                        final_url=str(resp.url),
                        status=status,
                        text="",
                        num_bytes=num_bytes,
                        elapsed=elapsed,
                        attempts=attempt + 1,
                    )
                retry_after = parse_retry_after(resp.headers.get("Retry-After")) if status == 429 else None
                delay = (
                    retry_after if retry_after is not None else compute_backoff(attempt, base=base_backoff)
                )
                await asyncio.sleep(delay)
                continue

            return FetchResult(
                url=url,
                final_url=str(resp.url),
                status=status,
                text=body,
                num_bytes=num_bytes,
                elapsed=elapsed,
                attempts=attempt + 1,
            )

        except (TimeoutError, httpx.TimeoutException, httpx.TransportError) as exc:
            last_exc = exc
            if attempt >= max_retries:
                break
            await asyncio.sleep(compute_backoff(attempt, base=base_backoff))

    elapsed = time.perf_counter() - started
    raise TimeoutError(f"Failed to fetch {url} after {max_retries + 1} attempts: {last_exc}") from last_exc
