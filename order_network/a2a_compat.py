"""A2A 0.3 compatibility for fasta2a agents.

fasta2a speaks the A2A method names (`message/send`, `tasks/get`) with 1.0-style payloads. Clients built on the
official a2a-sdk in 0.3 mode - for example the LiteLLM agent gateway used in the lecture - send `kind`-tagged
payloads, omit `acceptedOutputModes`, request `blocking` execution and expect `kind`-tagged responses.
This adapter translates between both dialects.
"""

import asyncio
import json
import time
from datetime import datetime, timezone
from typing import Any

from fasta2a import FastA2A
from fasta2a.schema import a2a_response_ta
from starlette.requests import Request
from starlette.responses import Response

from order_network import config

TERMINAL_STATES = {"completed", "failed", "canceled", "rejected", "input-required", "auth-required"}
DEFAULT_OUTPUT_MODES = ["application/json", "text/plain"]


def strip_kinds(message: dict) -> dict:
    message = {k: v for k, v in message.items() if k != "kind"}
    message["parts"] = [{k: v for k, v in part.items() if k != "kind"} for part in message.get("parts", [])]
    return message


def tag_parts(parts: list[dict]) -> list[dict]:
    tagged = []
    for part in parts:
        kind = "data" if "data" in part else "file" if ("url" in part or "raw" in part) else "text"
        tagged.append({"kind": kind, **part})
    return tagged


def tag_message(message: dict) -> dict:
    return {"kind": "message", **message, "parts": tag_parts(message.get("parts", []))}


def rfc3339(timestamp: str) -> str:
    """fasta2a writes naive local timestamps; protobuf-based clients require RFC 3339 with a timezone."""
    try:
        parsed = datetime.fromisoformat(timestamp)
    except ValueError:
        return timestamp
    return parsed.astimezone(timezone.utc).isoformat().replace("+00:00", "Z")


def tag_task(task: dict) -> dict:
    task = {"kind": "task", **task}
    status = dict(task.get("status", {}))
    if "timestamp" in status:
        status["timestamp"] = rfc3339(status["timestamp"])
    if "message" in status:
        status["message"] = tag_message(status["message"])
    task["status"] = status
    task["history"] = [tag_message(m) for m in task.get("history", [])]
    task["artifacts"] = [{**a, "parts": tag_parts(a.get("parts", []))} for a in task.get("artifacts", [])]
    return task


class CompatA2A(FastA2A):
    """FastA2A app that also serves A2A 0.3 clients (e.g. LiteLLM's agent gateway)."""

    async def _agent_card_endpoint(self, request: Request) -> Response:
        card = json.loads((await super()._agent_card_endpoint(request)).body)
        card.update(url=self.url, protocolVersion="0.3", preferredTransport="JSONRPC")
        for interface in card.get("supportedInterfaces", []):
            interface["protocolVersion"] = "0.3"
        return Response(json.dumps(card), media_type="application/json")

    async def _agent_run_endpoint(self, request: Request) -> Response:
        payload = json.loads(await request.body())
        params = payload.get("params") or {}
        message = params.get("message") or {}
        legacy = "kind" in message
        blocking = False
        if payload.get("method") == "message/send":
            if legacy:
                params["message"] = strip_kinds(message)
            configuration = params.get("configuration")
            if isinstance(configuration, dict):
                configuration.setdefault("acceptedOutputModes", DEFAULT_OUTPUT_MODES)
                blocking = bool(configuration.get("blocking"))

        response = await super()._agent_run_endpoint(_with_body(request, json.dumps(payload).encode()))
        if not (legacy or blocking) or response.media_type != "application/json":
            return response

        body = json.loads(response.body)
        result = body.get("result")
        if isinstance(result, dict):
            task = result.get("task", result if "status" in result else None)
            if task is not None:
                if blocking:
                    task = await self._wait_for(task)
                body["result"] = tag_task(task) if legacy else {"task": task}
        return Response(json.dumps(body), media_type="application/json", headers=_extra_headers(response))

    async def _wait_for(self, task: dict[str, Any]) -> dict[str, Any]:
        deadline = time.monotonic() + config.A2A_TIMEOUT_SECONDS
        while task["status"]["state"] not in TERMINAL_STATES and time.monotonic() < deadline:
            await asyncio.sleep(0.5)
            polled = await self.task_manager.get_task(
                {"jsonrpc": "2.0", "id": "blocking-wait", "method": "tasks/get", "params": {"id": task["id"]}}
            )
            task = json.loads(a2a_response_ta.dump_json(polled, by_alias=True)).get("result", task)
        return task


def _with_body(request: Request, body: bytes) -> Request:
    async def receive():
        return {"type": "http.request", "body": body, "more_body": False}

    return Request(request.scope, receive)


def _extra_headers(response: Response) -> dict[str, str]:
    return {k: v for k, v in response.headers.items() if k.lower() not in {"content-length", "content-type"}}
