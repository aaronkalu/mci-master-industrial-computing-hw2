import json
import time
import uuid

import pytest
from fastapi.testclient import TestClient
from pydantic_ai.messages import ModelResponse, RetryPromptPart, TextPart, ToolCallPart, ToolReturnPart
from pydantic_ai.models.function import FunctionModel

from order_network import a2a_client
from order_network.agents import inventory, orchestrator, order_intake, production
from order_network.agents.common import as_pairs, parse_order_lines
from order_network.runtime import create_a2a_app

EMAIL = "Anna from Alpine Robotics: 50 x 149449 and twelve 252654, ref 9938812, by 2026-10-30."


def parts_of(messages, kind):
    return [p for m in messages for p in getattr(m, "parts", []) if isinstance(p, kind)]


def rpc(client: TestClient, method: str, params: dict) -> dict:
    response = client.post("/", json={"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params})
    response.raise_for_status()
    return response.json()


def test_inventory_agent_over_a2a(monkeypatch):
    async def fake_production(skill, text, **kwargs):
        assert skill == "production-scheduling" and json.loads(text) == [{"material_number": "149449", "quantity": 20}]
        plan = {"planned_on": "2026-09-28", "latest_finish_date": "2026-10-08", "jobs": [{
            "material_number": "149449", "description": "Hydraulic pump housing", "quantity": 20,
            "production_line": "CASTING-1", "start_date": "2026-10-02", "finish_date": "2026-10-08",
            "lead_time_business_days": 8}]}
        return a2a_client.AgentResult(agent="production", state="completed", text="", data=plan, task_id="p" * 8)

    monkeypatch.setattr(a2a_client, "call_skill", fake_production)

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("check_stock", {"lines": [
            {"material_number": "149449", "quantity": 50}, {"material_number": "255565", "quantity": 100}]})])

    app = create_a2a_app(inventory.create_agent(FunctionModel(model)), name=inventory.NAME,
                         description=inventory.DESCRIPTION, skills=inventory.SKILLS, url="http://testserver",
                         register=False)
    with TestClient(app) as client:
        card = client.get("/.well-known/agent-card.json").json()
        assert card["name"] == "Inventory Agent" and card["skills"][0]["id"] == "stock-check"

        sent = rpc(client, "message/send", {"message": {
            "role": "user", "messageId": uuid.uuid4().hex, "parts": [{"text": "50 x 149449, 100 x 255565"}]}})
        task_id = sent["result"]["task"]["id"]
        for _ in range(50):
            task = rpc(client, "tasks/get", {"id": task_id})["result"]
            if task["status"]["state"] in a2a_client.TERMINAL_STATES:
                break
            time.sleep(0.1)

    assert task["status"]["state"] == "completed"
    text, data = a2a_client._extract(task)
    assert [line["shortfall"] for line in data["lines"]] == [20, 0]
    assert data["fully_available"] is False
    assert data["production_plan"]["latest_finish_date"] == "2026-10-08"


async def test_order_intake_rejects_order_references_and_retries():
    def model(messages, info):
        retries = parts_of(messages, RetryPromptPart)
        if not parts_of(messages, ToolReturnPart):
            return ModelResponse(parts=[ToolCallPart("find_material_numbers", {"text": EMAIL})])
        if not retries:
            return ModelResponse(parts=[ToolCallPart("submit_order_request", {"lines": [
                {"material_number": "149449", "quantity": 50}, {"material_number": "9938812", "quantity": 1}]})])
        return ModelResponse(parts=[ToolCallPart("submit_order_request", {
            "customer": "Alpine Robotics", "requested_delivery": "2026-10-30", "lines": [
                {"material_number": "149449", "quantity": 50}, {"material_number": "252654", "quantity": 12}]})])

    result = await order_intake.create_agent(FunctionModel(model)).run(EMAIL)
    order = result.output
    assert [(l.material_number, l.quantity) for l in order.lines] == [("149449", 50), ("252654", 12)]
    assert [i.value for i in order.ignored_numbers] == ["9938812"]
    assert "not valid material numbers" in parts_of(result.all_messages(), RetryPromptPart)[0].model_response()


async def test_orchestrator_enforces_delegation_workflow(monkeypatch):
    calls = []
    extracted = {"lines": [{"material_number": "149449", "quantity": 50}], "ignored_numbers": []}

    async def fake_call_skill(skill, text, **kwargs):
        calls.append((skill, text))
        data = extracted if skill == "order-extraction" else {"lines": [], "fully_available": False}
        return a2a_client.AgentResult(agent=skill, state="completed", text="", data=data, task_id="t" * 8)

    monkeypatch.setattr(a2a_client, "call_skill", fake_call_skill)
    steps = iter([
        ("order-extraction", "paraphrased: 50 x 149449"),
        None,  # answers too early -> retry
        ("stock-check", '[{"material_number": "149449", "quantity": 5}]'),  # wrong quantity -> retry
        None,
        ("stock-check", [{"material_number": "149449", "quantity": 50}]),  # structured lines instead of text
        None,
    ])

    def model(messages, info):
        step = next(steps)
        if step is None:
            return ModelResponse(parts=[TextPart("Confirmation: 30 from stock, 20 produced by 2026-10-08.")])
        key = "lines" if isinstance(step[1], list) else "request"
        return ModelResponse(parts=[ToolCallPart("delegate", {"skill": step[0], key: step[1]})])

    result = await orchestrator.create_agent(FunctionModel(model)).run(EMAIL)
    assert [skill for skill, _ in calls] == ["order-extraction", "stock-check", "stock-check"]
    assert json.loads(calls[2][1]) == [{"material_number": "149449", "quantity": 50}]
    assert calls[0][1] == EMAIL  # original email is forwarded, not the paraphrase
    retries = [p.model_response() for p in parts_of(result.all_messages(), RetryPromptPart)]
    assert "Workflow incomplete" in retries[0]
    assert '"quantity": 50' in retries[1]
    assert result.output.startswith("Confirmation")


async def test_orchestrator_reports_failed_delegation(monkeypatch):
    async def failing(skill, text, **kwargs):
        raise a2a_client.A2AError(f"No registered agent offers skill '{skill}'")

    monkeypatch.setattr(a2a_client, "call_skill", failing)

    def model(messages, info):
        if not parts_of(messages, ToolReturnPart):
            return ModelResponse(parts=[ToolCallPart("delegate", {"skill": "order-extraction", "request": EMAIL})])
        if len(parts_of(messages, ToolReturnPart)) == 1:
            return ModelResponse(parts=[ToolCallPart("delegate", {"skill": "stock-check", "request": "[]"})])
        return ModelResponse(parts=[TextPart("Order extraction agent unavailable.")])

    result = await orchestrator.create_agent(FunctionModel(model)).run(EMAIL)
    returns = parts_of(result.all_messages(), ToolReturnPart)
    assert all(str(r.content).startswith("ERROR") for r in returns)
    assert result.output == "Order extraction agent unavailable."


def test_parse_order_lines():
    assert as_pairs(parse_order_lines('Lines: [{"material_number": "1", "quantity": 2}]')) == [("1", 2)]
    assert parse_order_lines('{"lines": []}') == []
    assert parse_order_lines("50 x 149449") is None


async def test_production_agent_must_use_requested_lines():
    request = '[{"material_number": "252654", "quantity": 12}]'

    def model(messages, info):
        quantity = 12 if parts_of(messages, RetryPromptPart) else 50
        return ModelResponse(parts=[ToolCallPart("schedule_production", {
            "lines": [{"material_number": "252654", "quantity": quantity}]})])

    result = await production.create_agent(FunctionModel(model)).run(request)
    assert [job.quantity for job in result.output.jobs] == [12]


async def test_inventory_reports_unavailable_production(monkeypatch):
    async def unavailable(skill, text, **kwargs):
        raise a2a_client.A2AError("No registered agent offers skill 'production-scheduling'")

    monkeypatch.setattr(a2a_client, "call_skill", unavailable)

    def model(messages, info):
        return ModelResponse(parts=[ToolCallPart("check_stock", {"lines": [{"material_number": "252654", "quantity": 3}]})])

    report = (await inventory.create_agent(FunctionModel(model)).run("3 x 252654")).output
    assert report.production_plan is None and "unavailable" in report.production_note


def test_failed_task_reports_error():
    def model(messages, info):
        raise RuntimeError("model endpoint returned 404")

    app = create_a2a_app(inventory.create_agent(FunctionModel(model)), name=inventory.NAME,
                         description=inventory.DESCRIPTION, skills=inventory.SKILLS, url="http://testserver",
                         register=False)
    with TestClient(app) as client:
        sent = rpc(client, "message/send", {"message": {
            "role": "user", "messageId": uuid.uuid4().hex, "parts": [{"text": "5 x 149449"}]}})
        task_id = sent["result"]["task"]["id"]
        for _ in range(50):
            task = rpc(client, "tasks/get", {"id": task_id})["result"]
            if task["status"]["state"] in a2a_client.TERMINAL_STATES:
                break
            time.sleep(0.1)

    assert task["status"]["state"] == "failed"
    text, _ = a2a_client._extract(task)
    assert "model endpoint returned 404" in text
