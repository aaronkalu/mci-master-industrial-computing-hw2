import time

import httpx
import pytest
from fastapi.testclient import TestClient

from order_network import config
from order_network.registry import create_app


def card(name: str, skill: str) -> dict:
    return {
        "name": name, "description": f"{name} agent", "version": "1.0.0",
        "supportedInterfaces": [{"protocolBinding": "JSONRPC", "url": "http://x", "protocolVersion": "1.0"}],
        "capabilities": {}, "defaultInputModes": ["application/json"], "defaultOutputModes": ["application/json"],
        "skills": [{"id": skill, "name": skill, "description": f"does {skill}", "tags": ["demo", "stock"],
                    "inputModes": ["text/plain"], "outputModes": ["application/json"]}],
    }


CARDS = {"http://inv:1": card("Inventory Agent", "stock-check"), "http://prod:2": card("Production Agent", "production-scheduling")}


def handler(request: httpx.Request) -> httpx.Response:
    base = f"{request.url.scheme}://{request.url.host}:{request.url.port}"
    if base == "http://bad:3":
        return httpx.Response(200, json={"name": "missing fields"})
    if base in CARDS and request.url.path == "/.well-known/agent-card.json":
        return httpx.Response(200, json=CARDS[base])
    return httpx.Response(404)


@pytest.fixture
def client():
    app = create_app(http_client=httpx.AsyncClient(transport=httpx.MockTransport(handler)), ttl_seconds=0.5)
    with TestClient(app) as c:
        yield c


def test_register_and_discover_by_skill(client):
    assert client.post("/agents", json={"url": "http://inv:1"}).json()["name"] == "inventory-agent"
    client.post("/agents", json={"url": "http://prod:2/"})
    assert len(client.get("/agents").json()) == 2
    found = client.get("/agents", params={"skill": "stock-check"}).json()
    assert [a["url"] for a in found] == ["http://inv:1"]
    assert client.get("/agents", params={"skill": "stock"}).json()  # tag match
    assert client.get("/agents", params={"q": "production"}).json()[0]["name"] == "production-agent"


def test_rejects_unreachable_and_invalid_cards(client):
    assert client.post("/agents", json={"url": "http://unknown:9"}).status_code == 502
    assert client.post("/agents", json={"url": "http://bad:3"}).status_code == 422
    assert client.post("/agents", json={"url": "not-a-url"}).status_code == 422


def test_entries_expire_without_heartbeat(client):
    client.post("/agents", json={"url": "http://inv:1"})
    time.sleep(0.6)
    assert client.get("/agents").json() == []
    assert client.get("/agents/inventory-agent").status_code == 404


def test_heartbeat_keeps_registration_date(client):
    first = client.post("/agents", json={"url": "http://inv:1"}).json()
    second = client.post("/agents", json={"url": "http://inv:1"}).json()
    assert second["registered_at"] == first["registered_at"] and second["last_seen"] >= first["last_seen"]


def test_deregister(client):
    client.post("/agents", json={"url": "http://inv:1"})
    assert client.delete("/agents/inventory-agent").status_code == 204
    assert client.get("/agents").json() == []


def test_token_protects_writes(client, monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_TOKEN", "secret")
    assert client.post("/agents", json={"url": "http://inv:1"}).status_code == 401
    ok = client.post("/agents", json={"url": "http://inv:1"}, headers={"Authorization": "Bearer secret"})
    assert ok.status_code == 200
    assert client.get("/agents").status_code == 200
