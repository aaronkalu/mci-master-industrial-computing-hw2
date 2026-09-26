"""Agent discovery and A2A calls.

Two registry backends:
- "local":   this project's registry service (GET /agents?skill=...), agents are called directly.
- "litellm": the LiteLLM agent gateway used in the lecture (GET /v1/agents), agents are called
             through the proxy at /v1/a2a/{agent_id}/message/send - as in course Sample 6.
"""

import asyncio
import json
import logging
import time
import uuid
from dataclasses import dataclass

import httpx

from order_network import config
from order_network.registry import skill_matches, text_matches

log = logging.getLogger("a2a")

TERMINAL_STATES = {"completed", "failed", "canceled", "rejected"}
INPUT_STATES = {"input-required", "auth-required"}


class A2AError(RuntimeError):
    pass


@dataclass
class AgentResult:
    agent: str
    state: str
    text: str
    data: object | None
    task_id: str

    def as_tool_output(self) -> str:
        if self.data is not None:
            return json.dumps(self.data, ensure_ascii=False, default=str)
        return self.text


def registry_headers() -> dict:
    return {"Authorization": f"Bearer {config.REGISTRY_TOKEN}"} if config.REGISTRY_TOKEN else {}


def litellm_entry(agent: dict) -> dict:
    return {
        "name": agent["agent_name"],
        "url": f"{config.REGISTRY_URL}/v1/a2a/{agent['agent_id']}/message/send",
        "card": agent.get("agent_card_params") or {},
        "agent_id": agent["agent_id"],
    }


async def discover(skill: str | None = None, query: str | None = None) -> list[dict]:
    async with httpx.AsyncClient(timeout=10.0, headers=registry_headers()) as client:
        if config.REGISTRY_BACKEND == "litellm":
            response = await client.get(f"{config.REGISTRY_URL}/v1/agents")
            response.raise_for_status()
            entries = sorted((litellm_entry(a) for a in response.json()), key=lambda e: e["name"])
            if skill:
                entries = [e for e in entries if skill_matches(e["card"], skill)]
            if query:
                entries = [e for e in entries if text_matches(e["name"], e["card"], query)]
            return entries
        params = {k: v for k, v in {"skill": skill, "q": query}.items() if v}
        response = await client.get(f"{config.REGISTRY_URL}/agents", params=params)
        response.raise_for_status()
        return response.json()


async def resolve(skill: str) -> dict:
    matches = await discover(skill=skill)
    if not matches:
        raise A2AError(f"No registered agent offers skill '{skill}'")
    return matches[0]


def _extract(task: dict) -> tuple[str, object | None]:
    texts, data = [], None
    for artifact in task.get("artifacts", []):
        for part in artifact.get("parts", []):
            if "data" in part:
                payload = part["data"]
                data = payload.get("result", payload) if isinstance(payload, dict) else payload
            elif "text" in part:
                texts.append(part["text"])
    if not texts and data is None:
        agent_messages = [m for m in task.get("history", []) if m.get("role") == "agent"]
        message = task.get("status", {}).get("message") or (agent_messages[-1] if agent_messages else {})
        texts = [p["text"] for p in message.get("parts", []) if "text" in p]
    return "".join(texts).strip(), data


def _task_from_result(result: dict) -> dict | None:
    if "task" in result:
        return result["task"]
    return result if "status" in result and "id" in result else None


async def _rpc(http: httpx.AsyncClient, url: str, method: str, params: dict, agent: str) -> dict:
    response = await http.post(url, json={"jsonrpc": "2.0", "id": uuid.uuid4().hex, "method": method, "params": params})
    if response.status_code >= 400 and "application/json" not in response.headers.get("content-type", ""):
        raise A2AError(f"{agent}: HTTP {response.status_code} for {method}: {response.text[:300]}")
    body = response.json()
    if "error" in body:
        raise A2AError(f"{agent}: {method} failed: {body['error']}")
    return body["result"]


async def send_task(agent_url: str, text: str, agent_name: str = "", timeout: float | None = None,
                    poll_interval: float = 0.5, on_state=None) -> AgentResult:
    """Send a message via A2A `message/send` and poll `tasks/get` until the task reaches a terminal state."""
    timeout = timeout or config.A2A_TIMEOUT_SECONDS
    agent = agent_name or agent_url
    headers = registry_headers() if agent_url.startswith(config.REGISTRY_URL) else {}
    message = {
        "kind": "message", "role": "user", "messageId": uuid.uuid4().hex,
        "parts": [{"kind": "text", "text": text}],
    }
    async with httpx.AsyncClient(timeout=httpx.Timeout(timeout, connect=5.0), headers=headers) as http:
        result = await _rpc(http, agent_url, "message/send", {
            "message": message,
            "configuration": {"acceptedOutputModes": ["application/json", "text/plain"], "blocking": False},
        }, agent)
        task = _task_from_result(result)
        if task is None:
            reply = result.get("message", result)
            text = "".join(p.get("text", "") for p in reply.get("parts", [])).strip()
            return AgentResult(agent=agent_name, state="completed", text=text, data=None, task_id="")
        task_id = task["id"]

        deadline = time.monotonic() + timeout
        last_state = None
        while True:
            state = task["status"]["state"]
            if state != last_state:
                log.info("%s task %s -> %s", agent, task_id[:8], state)
                if on_state:
                    on_state(agent_name, state)
                last_state = state
            if state in TERMINAL_STATES or state in INPUT_STATES:
                break
            if time.monotonic() > deadline:
                raise A2AError(f"{agent} did not finish task {task_id} within {timeout:.0f}s")
            await asyncio.sleep(poll_interval)
            task = _task_from_result(await _rpc(http, agent_url, "tasks/get", {"id": task_id}, agent)) or task

    text, data = _extract(task)
    if state != "completed":
        raise A2AError(f"{agent} ended task {task_id} in state '{state}': {text or 'no details'}")
    return AgentResult(agent=agent_name, state=state, text=text, data=data, task_id=task_id)


async def call_skill(skill: str, text: str, **kwargs) -> AgentResult:
    entry = await resolve(skill)
    return await send_task(entry["url"], text, agent_name=entry["name"], **kwargs)
