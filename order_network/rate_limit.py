"""HTTP transport that waits and retry when the LLM endpoint is rate limited (429) or overloaded (503)."""

import asyncio
import logging
import re

import httpx2

log = logging.getLogger("rate-limit")

RETRY_HINT = re.compile(r"retry in ([\d.]+)s", re.IGNORECASE)
RETRYABLE = {429, 503}


def should_retry(response: httpx2.Response) -> bool:
    """Retry per-minute limits and temporary overload, but not an exhausted daily quota."""
    return response.status_code in RETRYABLE and "PerDay" not in response.text


def retry_delay(response: httpx2.Response, attempt: int) -> float:
    header = response.headers.get("retry-after")
    if header and header.replace(".", "", 1).isdigit():
        return float(header)
    match = RETRY_HINT.search(response.text)
    return float(match.group(1)) + 1 if match else min(10 * 2 ** attempt, 60)


class AsyncRateLimitTransport(httpx2.AsyncHTTPTransport):
    def __init__(self, max_attempts: int = 6, **kwargs):
        super().__init__(**kwargs)
        self.max_attempts = max_attempts

    async def handle_async_request(self, request: httpx2.Request) -> httpx2.Response:
        for attempt in range(self.max_attempts):
            response = await super().handle_async_request(request)
            if response.status_code not in RETRYABLE or attempt == self.max_attempts - 1:
                return response
            await response.aread()
            if not should_retry(response):
                return response
            delay = retry_delay(response, attempt)
            await response.aclose()
            log.warning("LLM endpoint returned %d, retrying in %.0fs (attempt %d/%d)", response.status_code, delay, attempt + 1, self.max_attempts)
            await asyncio.sleep(delay)
        return response

