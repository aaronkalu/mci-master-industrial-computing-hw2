import asyncio
import logging
import os
import uuid
from contextlib import asynccontextmanager

import httpx
import httpx2
import uvicorn
from fasta2a import FastA2A
from fasta2a.broker import InMemoryBroker
from fasta2a.pydantic_ai import AgentWorker, worker_lifespan
from fasta2a.schema import AgentProvider, Message, Part, Skill
from fasta2a.storage import InMemoryStorage
from pydantic_ai import Agent
from pydantic_ai.models import Model
from pydantic_ai.models.google import GoogleModel
from pydantic_ai.models.openai import OpenAIChatModel
from pydantic_ai.settings import ModelSettings
from pydantic_ai.providers.google import GoogleProvider
from pydantic_ai.providers.openai import OpenAIProvider

from order_network import config
from order_network.a2a_client import registry_headers
from order_network.a2a_compat import CompatA2A
from order_network.rate_limit import AsyncRateLimitTransport
from order_network.registry import slugify

log = logging.getLogger("runtime")

PROVIDER = AgentProvider(organization="MCI Industrial Computing", url="https://www.mci.edu")


def configure_logging() -> None:
    os.environ.setdefault("PYDANTIC_AI_NO_BANNER", "1")
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")
    logging.getLogger("httpx").setLevel(logging.WARNING)


def rate_limited_client() -> httpx2.AsyncClient:
    """Retries per-minute rate limits (429) and overload (503); an exhausted daily quota fails immediately."""
    return httpx2.AsyncClient(transport=AsyncRateLimitTransport(), timeout=600)


def build_model() -> Model:
    settings = ModelSettings(max_tokens=config.LLM_MAX_TOKENS, temperature=0)
    if config.LLM_PROVIDER == "google":
        # Native Gemini API: needed for Gemini 3 thinking models, which require thought signatures on tool calls.
        return GoogleModel(
            config.LLM_MODEL,
            provider=GoogleProvider(api_key=config.LLM_API_KEY, http_client=rate_limited_client()),
            settings=settings,
        )
    return OpenAIChatModel(
        config.LLM_MODEL,
        provider=OpenAIProvider(
            base_url=config.LLM_BASE_URL, api_key=config.LLM_API_KEY,
            http_client=rate_limited_client(),
        ),
        settings=settings,
    )


def public_url(port: int) -> str:
    return os.getenv("AGENT_PUBLIC_URL", f"http://localhost:{port}").rstrip("/")


async def _local_registration(client: httpx.AsyncClient, url: str, name: str, card_url: str) -> None:
    response = await client.post(f"{config.REGISTRY_URL}/agents", json={"url": url})
    response.raise_for_status()


async def _litellm_registration(client: httpx.AsyncClient, url: str, name: str, card_url: str) -> None:
    """Create the agent in LiteLLM's agent registry if missing, update it if its URL changed."""
    agents = (await client.get(f"{config.REGISTRY_URL}/v1/agents")).raise_for_status().json()
    existing = next((a for a in agents if a["agent_name"] == name), None)
    if existing and (existing.get("agent_card_params") or {}).get("url") == url:
        return
    card = (await client.get(card_url)).raise_for_status().json()
    if existing:
        response = await client.patch(f"{config.REGISTRY_URL}/v1/agents/{existing['agent_id']}",
                                      json={"agent_card_params": card})
    else:
        response = await client.post(f"{config.REGISTRY_URL}/v1/agents", json={
            "agent_name": name, "agent_card_params": card, "litellm_params": {"make_public": True},
        })
    response.raise_for_status()


async def _registration_loop(url: str, name: str, card_url: str) -> None:
    register = _litellm_registration if config.REGISTRY_BACKEND == "litellm" else _local_registration
    registered = False
    delay = 1.0
    async with httpx.AsyncClient(timeout=10.0, headers=registry_headers()) as client:
        while True:
            try:
                await register(client, url, name, card_url)
                if not registered:
                    log.info("'%s' registered at %s (%s) as %s", name, config.REGISTRY_URL, config.REGISTRY_BACKEND, url)
                registered, delay = True, config.HEARTBEAT_SECONDS
            except httpx.HTTPError as e:
                if registered or delay == 1.0:
                    log.warning("Registration of '%s' failed (%s), retrying", name, e)
                registered, delay = False, min(delay * 2, config.HEARTBEAT_SECONDS)
            await asyncio.sleep(delay)


async def _deregister(name: str) -> None:
    try:
        async with httpx.AsyncClient(timeout=3.0, headers=registry_headers()) as client:
            if config.REGISTRY_BACKEND == "litellm":
                agents = (await client.get(f"{config.REGISTRY_URL}/v1/agents")).json()
                for agent in (a for a in agents if a["agent_name"] == name):
                    await client.delete(f"{config.REGISTRY_URL}/v1/agents/{agent['agent_id']}")
            else:
                await client.delete(f"{config.REGISTRY_URL}/agents/{name}")
    except (httpx.HTTPError, ValueError):
        pass


class ReportingAgentWorker(AgentWorker):
    """Logs failures and attaches the error to the task history so callers see why a task failed."""

    async def run_task(self, params) -> None:
        try:
            await super().run_task(params)
        except Exception as e:
            log.exception("Task %s failed", params["id"])
            error = Message(role="agent", parts=[Part(text=f"{type(e).__name__}: {e}")], message_id=uuid.uuid4().hex)
            await self.storage.update_task(params["id"], state="failed", new_messages=[error])
            raise


def create_a2a_app(agent: Agent, *, name: str, description: str, skills: list[Skill], url: str,
                   register: bool = True, local_url: str | None = None) -> FastA2A:
    """Expose a pydantic-ai agent via A2A and keep it registered in the agent registry while it runs."""
    storage, broker = InMemoryStorage(), InMemoryBroker()
    worker = ReportingAgentWorker(agent=agent, broker=broker, storage=storage)
    registry_name = slugify(name)

    @asynccontextmanager
    async def lifespan(app: FastA2A):
        async with worker_lifespan(app, worker, agent):
            card_url = f"{(local_url or url).rstrip('/')}/.well-known/agent-card.json"
            heartbeat = asyncio.create_task(_registration_loop(url, registry_name, card_url)) if register else None
            try:
                yield
            finally:
                if heartbeat:
                    heartbeat.cancel()
                    await _deregister(registry_name)

    return CompatA2A(
        storage=storage, broker=broker, name=name, url=url, version="1.0.0",
        description=description, provider=PROVIDER, skills=skills, lifespan=lifespan,
    )


def serve(app, port: int) -> None:
    uvicorn.run(app, host=os.getenv("AGENT_HOST", "0.0.0.0"), port=port, log_level="warning")
