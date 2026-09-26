# MCI Industrial Computing – HW2: Distributed Remote A2A System with Dynamic Registry Lookup

A multi-agent order fulfilment network for an industrial parts manufacturer:
- **Agents:** four independent A2A agents, each in its own process on its own port, or each in its own Docker container.
- **Registry:** the agents register with an agent registry at startup and find each other there at runtime by skill. Nobody has hardcoded URLs. Two registries are supported:
  - **LiteLLM's agent gateway**, as in the lecture and course Sample 6
  - **this project's own registry service**, which runs without LiteLLM

A customer order email goes in; an order confirmation comes out. It covers stock availability, production dates and the delivery date. The agents work together to produce it.

## Based on the course repository

This project extends the base code in [chmaurer/mci_industrial_computing](https://github.com/chmaurer/mci_industrial_computing):

| Course sample | What was reused | What was extended |
|---------------|-----------------|-------------------|
| `Sample 6 a2a remote` | `fasta2a` + `pydantic-ai` agents (`OpenAIChatModel`/`OpenAIProvider` on LiteLLM, uvicorn); the extraction rule (numbers starting with 9 are order references); the material numbers and lead-time domain; registry lookup via LiteLLM `GET /v1/agents`; calls via `/v1/a2a/{agent_id}/message/send`; routing by agent description/skill (`main_dynamic_lookup.py`) | a 3rd and 4th agent (inventory, coordinator); the inventory agent calls the production agent itself over A2A; self-registration in LiteLLM instead of manual registration; own registry backend; typed results; input validation; failure reporting |
| `Sample 5 a2a localhost` | a root agent that delegates to specialised sub-agents | the delegation runs across process and network boundaries (remote A2A), not in one ADK process |

## Components

| Component | Port | A2A skill | Role |
|-----------|------|-----------|------|
| **Agent Registry** | 8000 | – | FastAPI service. Agents register their URL; the registry fetches and validates the A2A agent card and serves lookups by skill, tag or text. Entries expire after `REGISTRY_TTL_SECONDS` without a heartbeat. |
| **Order Fulfilment Coordinator** (root agent) | 8100 | `order-fulfilment` | LLM agent that finds the specialists in the registry at runtime and delegates to them over A2A, then writes the confirmation |
| **Order Intake Agent** | 8101 | `order-extraction` | Email → structured order. Finds valid 6-digit material numbers; numbers starting with 9 are customer references and are ignored (same rule as sample 6). |
| **Inventory Agent** | 8102 | `stock-check` | Own SQLite stock DB: available stock, quantity reservable now, shortfall, warehouse. For shortfalls it asks the production agent itself, found through the registry. |
| **Production Planning Agent** | 8103 | `production-scheduling` | Schedules the shortfall quantities on production lines (backlog + capacity + setup): finish dates and lead times in business days |

## How the agents collaborate

```
client.py ──(1) GET /agents?skill=order-fulfilment──▶ Registry :8000 ◀──(0) POST /agents + heartbeat── all agents
    │                                                  ▲         ▲
    └─(2) A2A message/send + tasks/get ──▶ Coordinator :8100     │
                                              │ (3) list_available_agents / resolve skill
                                              ├─(4) order-extraction ──▶ Order Intake :8101
                                              └─(5) stock-check      ──▶ Inventory    :8102
                                                                            │ (6) resolve production-scheduling
                                                                            └─ A2A ──▶ Production :8103 (shortfalls only)
                                              (7) order confirmation ──▶ client
```

0. **Self-registration.** On startup, each agent posts its public URL to the registry. The registry fetches `/.well-known/agent-card.json`, validates it against the A2A `AgentCard` schema and stores it. Agents re-register every `HEARTBEAT_SECONDS`, and deregister when they shut down.
1. **Finding the entry point.** The client looks up the agent that offers the `order-fulfilment` skill. It needs only the registry URL.
2. **Sending the order.** The client sends the order with A2A `message/send` and polls `tasks/get` until the task completes.
3. **Discovering specialists.** The coordinator lists the available agents from the registry, then routes each step by **skill id**, not by URL.
4. **Extraction.** The Order Intake agent turns the email into structured order lines.
5. **Stock check.** The Inventory agent checks those lines against its stock.
6. **Production.** For any shortfall, the Inventory agent looks up `production-scheduling` in the registry itself and asks the Production agent over A2A for a plan. The plan comes back inside the stock report.
7. **Confirmation.** The coordinator combines extraction, stock and production results into the confirmation.

If an agent stops or crashes, its registry entry expires. The caller then reports which step failed: for example, the stock report says that production planning is unavailable. When the agent comes back it registers again automatically, and the next request finds it.

### Robustness

- **Deterministic numbers:** all numbers come from deterministic code, not from the LLM. The specialists finish with an *output function* (`pydantic-ai` `ToolOutput`): the LLM only chooses the arguments, and stock and schedule results come back as a typed A2A `DataPart`.
- **Agent-to-agent handoff:** the inventory agent builds the production request (shortfalls only) in code and sends it straight to the production agent. No LLM copies numbers between agents.
- **Input checks:** every specialist checks its arguments against its input and asks the model to retry (`ModelRetry`) on a mismatch:
  - order intake: the material numbers found in the email
  - inventory and production: the JSON order lines they received
- **Coordinator workflow:** an output validator makes the coordinator delegate extraction and stock check before answering, and checks that it passed the extracted lines on unchanged. An agent that fails is reported as a failure instead of making the coordinator retry forever.
- **Failure details:** failed A2A tasks carry the error message in the task history, so callers see why a step failed.
- **Registry checks:** the registry rejects cards it can't reach (502) or that are invalid (422). An optional `REGISTRY_TOKEN` protects register and deregister.

## Setup

Requirements: Python ≥ 3.11 and [uv](https://docs.astral.sh/uv/) (or pip), plus an OpenAI-compatible LLM endpoint: a LiteLLM proxy as in the lecture, or a local Ollama.

```bash
git clone <this-repo> && cd mci-a2a-order-network
uv sync                     # or: python -m venv .venv && source .venv/bin/activate && pip install -r requirements.txt
cp .env.example .env        # set LITELLM_API, LITELLM_KEY, LITELLM_DEFAULT_MODEL
```

| Variable | Description | Default |
|----------|-------------|---------|
| `LITELLM_API` / `LITELLM_KEY` / `LITELLM_DEFAULT_MODEL` | LLM endpoint used by all agents | `http://localhost:4000`, –, `gpt-4o-mini` |
| `LLM_PROVIDER` | `openai` (OpenAI-compatible) or `google` (native Gemini) | auto-detected from `LITELLM_API` |
| `REGISTRY_BACKEND` | `local` (own registry) or `litellm` (LiteLLM agent registry) | `local` |
| `REGISTRY_URL` | registry URL: own registry, or the LiteLLM proxy for `litellm` | `http://localhost:8000` |
| `REGISTRY_TOKEN` | own registry: optional shared secret for (de)registration; LiteLLM: API key | empty / `LITELLM_KEY` |
| `REGISTRY_TTL_SECONDS` / `HEARTBEAT_SECONDS` | expiry and heartbeat interval | 30 / 10 |
| `AGENT_PUBLIC_URL` | URL an agent registers under (set per agent by the launchers) | `http://localhost:<port>` |
| `A2A_TIMEOUT_SECONDS` | max wait for a delegated task | 300 |
| `LLM_MAX_TOKENS` | completion token cap per LLM call | 1024 |

## Run

### Option A – separate processes on separate ports

```bash
uv run python run_local.py            # starts registry + 4 agents, waits until all are registered
uv run python client.py               # in a second terminal: sends the demo order
```

`uv run python run_local.py --demo` does both: it starts the network, runs the demo client once and shuts down. All component output is prefixed and color-coded, and saved to `logs/network_<timestamp>.log`.

You can also start each component by hand, one terminal each:

```bash
uv run python -m order_network registry
uv run python -m order_network order-intake
uv run python -m order_network inventory
uv run python -m order_network production
uv run python -m order_network orchestrator
```

### Option B – LiteLLM agent registry (lecture setup, Sample 6)

With `REGISTRY_BACKEND=litellm`, no own registry is started:
- **Registration:** each agent registers itself in LiteLLM's agent registry (`POST /v1/agents`) and removes itself when it shuts down.
- **Discovery and calls:** the client and agents look each other up via `GET /v1/agents`, and all A2A calls go through the proxy (`/v1/a2a/{agent_id}/message/send`).

```bash
REGISTRY_BACKEND=litellm REGISTRY_URL=http://localhost:4000 \
LITELLM_API=http://localhost:4000 LITELLM_KEY=<litellm key> LITELLM_DEFAULT_MODEL=gpt-4o-mini \
uv run python run_local.py --demo
```

- **Which LiteLLM:** use the lecture's proxy, or the local one in [`litellm-local`](litellm-local/README.md) (LiteLLM + Postgres in Docker).
- **Agent address:** agents register under `http://host.docker.internal:<port>`, so a LiteLLM running in Docker can reach them. For a remote LiteLLM, use `--agent-host <your-ip>`, and allow that host in LiteLLM's `user_url_allowed_hosts`.
- **Protocol:** LiteLLM's A2A gateway uses the official `a2a-sdk` in A2A 0.3 mode. `order_network/a2a_compat.py` adapts the fasta2a agents to it: a 0.3 agent card, `kind`-tagged payloads, `blocking` requests and RFC 3339 timestamps.

### Option C – one Docker container per component (own registry)

```bash
docker compose up --build -d                  # registry + 4 agents, each in its own container
docker compose run --rm client                # demo order from inside the compose network
docker compose run --rm client "Please ship 30 x 418870 and 5 x 310022"
docker compose down
```

Inside Docker, agents register under their service names, e.g. `http://inventory:8102`. For a local Ollama on the host, set `LITELLM_API=http://host.docker.internal:11434/v1` in `.env`.

### Client options

```bash
python client.py --list                                        # show registered agents and skills
python client.py "Hi, we need 25 couplings 252653 by 2026-10-20"
python client.py --skill stock-check '[{"material_number": "149449", "quantity": 50}]'   # call a specialist directly
curl "http://localhost:8000/agents?skill=production-scheduling"    # registry API; docs at http://localhost:8000/docs
```

## Tests

```bash
uv run pytest
```

The tests cover:
- **Domain logic:** number classification, stock checks, business-day scheduling.
- **Registry:** registration, card validation, lookup by skill/tag/text, TTL expiry, token protection.
- **A2A roundtrip:** a full exchange with a real A2A app (agent card, `message/send`, `tasks/get`, DataPart result).
- **Agent behaviour:** intake retries on invalid numbers, the coordinator's workflow checks, and how it handles failed agents. These use `FunctionModel`, so no LLM is needed.

## Example run

Real logs from `run_local.py --demo` against the Gemini API (`gemini-3.5-flash-lite`, free tier):
- [`docs/example_network.log`](docs/example_network.log) – every process: registrations, delegations, the inventory → production A2A call, task states.
- [`docs/example_client.log`](docs/example_client.log) – the client's view: registry lookup, task progress, final confirmation.

Through the local LiteLLM (Option B), with LiteLLM as agent registry, A2A gateway and LLM proxy (`gpt-4o-mini` routed to Gemini):
- [`docs/example_litellm_network.log`](docs/example_litellm_network.log)
- [`docs/example_litellm_client.log`](docs/example_litellm_client.log)

### LLM endpoints

- **OpenAI-compatible (default):** any endpoint works, for example the lecture's LiteLLM proxy or a local Ollama.
- **Gemini API:** set `LITELLM_API` to the Gemini URL and the agents switch to pydantic-ai's native Gemini client automatically. Gemini 3 models need "thought signatures" passed back with tool calls, and the native client handles that. Override with `LLM_PROVIDER=openai|google`.
- **Free-tier limits:** rate limits (429) and temporary overload (503) are retried with backoff. The free tier allows only a few requests per minute and about 20 per day per model; if one model's daily quota runs out, switch `LITELLM_DEFAULT_MODEL` to another Flash model.

## Project layout

```
order_network/
  __main__.py          python -m order_network <registry|orchestrator|order-intake|inventory|production>
  registry.py          agent registry service (FastAPI)
  runtime.py           A2A app factory with self-registration (own registry or LiteLLM), LLM model factory
  a2a_compat.py        A2A 0.3 compatibility for LiteLLM's A2A gateway (a2a-sdk)
  rate_limit.py        retry transport for 429/503 from OpenAI-compatible endpoints
  a2a_client.py        registry discovery + A2A message/send + tasks/get polling
  models.py            shared pydantic data contracts
  agents/              coordinator + 3 specialist agents (cards, skills, instructions, output functions)
  domain/              deterministic business logic (extraction, inventory DB, production planning)
client.py              entry point for users
run_local.py           multi-process launcher
docker-compose.yml     one container per component
tests/
```
