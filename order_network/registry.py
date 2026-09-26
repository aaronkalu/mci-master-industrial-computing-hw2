"""Agent registry: agents register their A2A endpoint, the registry fetches and validates the agent card.

Entries expire when an agent stops sending heartbeats (re-registrations) within the TTL.
"""

import logging
import re
import time
from contextlib import asynccontextmanager
from typing import Annotated

import httpx
from fasta2a.schema import agent_card_ta
from fastapi import Depends, FastAPI, Header, HTTPException, Query
from pydantic import BaseModel, HttpUrl, ValidationError

from order_network import config

log = logging.getLogger("registry")

AGENT_CARD_PATH = "/.well-known/agent-card.json"


class RegisterRequest(BaseModel):
    url: HttpUrl


class AgentEntry(BaseModel):
    name: str
    url: str
    card: dict
    registered_at: float
    last_seen: float


def slugify(name: str) -> str:
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-")


def skill_matches(card: dict, skill: str) -> bool:
    skill = skill.lower()
    return any(s["id"].lower() == skill or skill in (t.lower() for t in s.get("tags", [])) for s in card.get("skills", []))


def text_matches(name: str, card: dict, query: str) -> bool:
    haystack = " ".join(
        [name, card.get("description", "")]
        + [f"{s['name']} {s.get('description', '')} {' '.join(s.get('tags', []))}" for s in card.get("skills", [])]
    ).lower()
    return all(term in haystack for term in query.lower().split())


def create_app(http_client: httpx.AsyncClient | None = None, ttl_seconds: float | None = None) -> FastAPI:
    ttl = ttl_seconds if ttl_seconds is not None else config.REGISTRY_TTL_SECONDS
    agents: dict[str, AgentEntry] = {}

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.http = http_client or httpx.AsyncClient(timeout=5.0)
        yield
        if http_client is None:
            await app.state.http.aclose()

    app = FastAPI(title="A2A Agent Registry", version="1.0.0", lifespan=lifespan)

    def require_token(authorization: Annotated[str | None, Header()] = None) -> None:
        if config.REGISTRY_TOKEN and authorization != f"Bearer {config.REGISTRY_TOKEN}":
            raise HTTPException(status_code=401, detail="Invalid or missing registry token")

    def alive() -> list[AgentEntry]:
        now = time.time()
        for name in [n for n, e in agents.items() if now - e.last_seen > ttl]:
            log.info("Agent '%s' expired (no heartbeat for %.0fs)", name, ttl)
            del agents[name]
        return sorted(agents.values(), key=lambda e: e.name)

    @app.get("/health")
    async def health() -> dict:
        return {"status": "ok", "agents": len(alive())}

    @app.post("/agents", response_model=AgentEntry, dependencies=[Depends(require_token)])
    async def register(request: RegisterRequest) -> AgentEntry:
        base_url = str(request.url).rstrip("/")
        try:
            response = await app.state.http.get(base_url + AGENT_CARD_PATH)
            response.raise_for_status()
            card = agent_card_ta.validate_python(response.json())
        except httpx.HTTPError as e:
            raise HTTPException(status_code=502, detail=f"Agent card not reachable at {base_url}: {e}") from e
        except (ValueError, ValidationError) as e:
            raise HTTPException(status_code=422, detail=f"Invalid agent card: {e}") from e

        name = slugify(card["name"])
        now = time.time()
        existing = agents.get(name)
        if existing is None:
            log.info("Registered agent '%s' at %s with skills %s", name, base_url, [s["id"] for s in card["skills"]])
        entry = AgentEntry(
            name=name, url=base_url, card=dict(card),
            registered_at=existing.registered_at if existing else now, last_seen=now,
        )
        agents[name] = entry
        return entry

    @app.get("/agents", response_model=list[AgentEntry])
    async def list_agents(
        skill: Annotated[str | None, Query(description="Skill id or tag")] = None,
        q: Annotated[str | None, Query(description="Free-text search in name, description and skills")] = None,
    ) -> list[AgentEntry]:
        result = alive()
        if skill:
            result = [e for e in result if skill_matches(e.card, skill)]
        if q:
            result = [e for e in result if text_matches(e.name, e.card, q)]
        return result

    @app.get("/agents/{name}", response_model=AgentEntry)
    async def get_agent(name: str) -> AgentEntry:
        alive()
        if name not in agents:
            raise HTTPException(status_code=404, detail=f"Agent '{name}' not registered")
        return agents[name]

    @app.delete("/agents/{name}", status_code=204, dependencies=[Depends(require_token)])
    async def deregister(name: str) -> None:
        if agents.pop(name, None):
            log.info("Deregistered agent '%s'", name)

    return app
