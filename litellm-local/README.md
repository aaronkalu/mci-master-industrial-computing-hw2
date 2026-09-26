# Local LiteLLM proxy (lecture setup at home)

The course samples use the lecturer's LiteLLM proxy (`http://192.168.1.60:4000`), which is only reachable in the lecture network. This folder runs the same kind of proxy locally in Docker: LiteLLM + Postgres. LiteLLM needs Postgres to store MCP servers and agents.

It provides:
- **LLM gateway:** `http://localhost:4000` (OpenAI-compatible). `gpt-4o-mini` is routed to the Gemini free tier, with automatic fallbacks to other Flash models when a daily quota runs out.
- **MCP server registry and gateway:** `/v1/mcp/server` and `/mcp`. Used by HW1.
- **A2A agent registry and gateway:** `/v1/agents` and `/v1/a2a/{agent_id}/message/send`. Used by HW2.
- **Admin UI:** http://localhost:4000/ui. Log in with the master key.

## Setup

```bash
cp .env.example .env      # set GEMINI_API_KEY (AI Studio) and choose a LITELLM_MASTER_KEY
docker compose up -d
curl localhost:4000/health/liveliness
```

Use the master key as `LITELLM_KEY` in the homework projects:

| Project | Settings |
|---------|----------|
| HW1 | `LITELLM_API=http://localhost:4000`, `MCP_SERVER_URL=http://localhost:4000/mcp`, `MCP_API_KEY=<master key>` |
| HW2 | `LITELLM_API=http://localhost:4000`, `REGISTRY_BACKEND=litellm`, `REGISTRY_URL=http://localhost:4000` |

LiteLLM runs in Docker and reaches services on your machine via `host.docker.internal`. That host is allowed in `config.yaml` (`user_url_allowed_hosts`).

To stop it: `docker compose stop` keeps the database; `docker compose down -v` removes everything.
