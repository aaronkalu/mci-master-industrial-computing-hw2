import httpx2

from order_network.rate_limit import AsyncRateLimitTransport, retry_delay


def test_retry_delay_sources():
    request = httpx2.Request("POST", "http://llm")
    assert retry_delay(httpx2.Response(429, headers={"retry-after": "7"}, request=request), 0) == 7
    assert retry_delay(httpx2.Response(429, text="Please retry in 38.9s.", request=request), 0) == 39.9
    assert retry_delay(httpx2.Response(429, text="slow down", request=request), 1) == 20


async def test_async_transport_retries_after_429(monkeypatch):
    responses = iter([httpx2.Response(429, text="retry in 0s"), httpx2.Response(200, json={"ok": True})])

    async def fake_handle(self, request):
        return next(responses)

    async def no_sleep(_):
        pass

    monkeypatch.setattr(httpx2.AsyncHTTPTransport, "handle_async_request", fake_handle)
    monkeypatch.setattr("order_network.rate_limit.asyncio.sleep", no_sleep)
    async with httpx2.AsyncClient(transport=AsyncRateLimitTransport()) as client:
        response = await client.post("http://llm/v1/chat/completions")
    assert response.status_code == 200


def test_daily_quota_is_not_retried():
    from order_network.rate_limit import should_retry

    request = httpx2.Request("POST", "http://llm")
    assert not should_retry(httpx2.Response(429, text="GenerateRequestsPerDayPerProjectPerModel-FreeTier", request=request))
    assert should_retry(httpx2.Response(429, text="GenerateRequestsPerMinutePerProjectPerModel-FreeTier", request=request))
    assert should_retry(httpx2.Response(503, text="high demand", request=request))
