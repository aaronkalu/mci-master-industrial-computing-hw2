import uuid

import httpx
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, ToolCallPart
from pydantic_ai.models.function import FunctionModel

from order_network import a2a_client, config
from order_network.a2a_compat import rfc3339, strip_kinds, tag_task
from order_network.agents import production
from order_network.runtime import create_a2a_app

REQUEST = '[{"material_number": "252654", "quantity": 12}]'


def production_app():
    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("schedule_production", {
            "lines": [{"material_number": "252654", "quantity": 12}]})])

    return create_a2a_app(production.create_agent(FunctionModel(model)), name=production.NAME,
                          description=production.DESCRIPTION, skills=production.SKILLS,
                          url="http://host.docker.internal:8103", register=False)


def test_card_advertises_a2a_03_for_litellm():
    with TestClient(production_app()) as client:
        card = client.get("/.well-known/agent-card.json").json()
    assert card["url"] == "http://host.docker.internal:8103" and card["protocolVersion"] == "0.3"
    assert card["supportedInterfaces"][0]["protocolVersion"] == "0.3"


def test_blocking_legacy_request_returns_finished_kind_tagged_task():
    """The request shape a2a-sdk (LiteLLM's A2A gateway) sends in 0.3 mode."""
    payload = {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
                    "parts": [{"kind": "text", "text": REQUEST}]},
        "configuration": {"blocking": True},
    }}
    with TestClient(production_app()) as client:
        result = client.post("/", json=payload).json()["result"]

    assert result["kind"] == "task" and result["status"]["state"] == "completed"
    assert result["status"]["timestamp"].endswith("Z")
    part = result["artifacts"][0]["parts"][0]
    assert part["kind"] == "data" and part["data"]["result"]["jobs"][0]["quantity"] == 12
    assert all(m["kind"] == "message" for m in result["history"])


def test_v1_requests_are_unchanged():
    payload = {"jsonrpc": "2.0", "id": "1", "method": "message/send", "params": {
        "message": {"role": "user", "messageId": uuid.uuid4().hex, "parts": [{"text": REQUEST}]}}}
    with TestClient(production_app()) as client:
        result = client.post("/", json=payload).json()["result"]
    assert "task" in result and "kind" not in result["task"]


def test_helpers():
    assert strip_kinds({"kind": "message", "parts": [{"kind": "text", "text": "x"}]}) == {"parts": [{"text": "x"}]}
    assert rfc3339("2026-09-26T12:00:00+02:00") == "2026-09-26T10:00:00Z"
    assert rfc3339("not a date") == "not a date"
    assert tag_task({"id": "t", "status": {"state": "working"}})["kind"] == "task"


async def test_litellm_discovery_maps_agents(monkeypatch):
    monkeypatch.setattr(config, "REGISTRY_BACKEND", "litellm")
    monkeypatch.setattr(config, "REGISTRY_URL", "http://proxy:4000")
    monkeypatch.setattr(config, "REGISTRY_TOKEN", "sk-test")
    agents = [
        {"agent_id": "a1", "agent_name": "inventory-agent", "agent_card_params": {
            "description": "stock", "skills": [{"id": "stock-check", "name": "Stock", "tags": ["inventory"]}]}},
        {"agent_id": "a2", "agent_name": "production-planning-agent", "agent_card_params": {
            "description": "plans", "skills": [{"id": "production-scheduling", "name": "Plan", "tags": []}]}},
    ]

    def handler(request: httpx.Request) -> httpx.Response:
        assert request.headers["authorization"] == "Bearer sk-test" and request.url.path == "/v1/agents"
        return httpx.Response(200, json=agents)

    real_client = httpx.AsyncClient
    monkeypatch.setattr(a2a_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    entry = await a2a_client.resolve("stock-check")
    assert entry["url"] == "http://proxy:4000/v1/a2a/a1/message/send"
    assert [e["name"] for e in await a2a_client.discover(query="plans")] == ["production-planning-agent"]


async def test_send_task_accepts_bare_task_result(monkeypatch):
    """LiteLLM returns A2A 0.3 results: the task itself, kind-tagged, already completed (blocking)."""
    task = {"kind": "task", "id": "t1", "contextId": "c", "status": {"state": "completed"},
            "artifacts": [{"artifactId": "a", "parts": [{"kind": "data", "data": {"result": {"ok": True}}}]}]}

    def handler(request: httpx.Request) -> httpx.Response:
        return httpx.Response(200, json={"jsonrpc": "2.0", "id": "1", "result": task})

    real_client = httpx.AsyncClient
    monkeypatch.setattr(a2a_client.httpx, "AsyncClient",
                        lambda **kw: real_client(transport=httpx.MockTransport(handler), **kw))
    result = await a2a_client.send_task("http://proxy:4000/v1/a2a/x/message/send", "hi", agent_name="x")
    assert result.data == {"ok": True} and result.task_id == "t1"
