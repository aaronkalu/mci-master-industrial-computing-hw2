import json
import logging
from datetime import date

from fasta2a.schema import Skill
from pydantic_ai import Agent, ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, ModelResponse, ToolCallPart, ToolReturnPart

from order_network import a2a_client
from order_network.agents.common import as_pairs, latest_user_text, lines_json, parse_order_lines
from order_network.models import OrderLine
from order_network.registry import slugify
from order_network.runtime import build_model

log = logging.getLogger("orchestrator")

NAME = "Order Fulfilment Coordinator"
DESCRIPTION = ("Root agent: handles a complete customer order by discovering specialised agents in the registry "
               "and delegating extraction, stock check and production planning to them via A2A.")
SKILLS = [
    Skill(
        id="order-fulfilment",
        name="Order fulfilment",
        description="Takes a customer order email and returns an order confirmation with availability, "
                    "production dates and the expected delivery date.",
        tags=["orders", "fulfilment", "coordination", "root-agent"],
        examples=["Please deliver 50 units of 149449 and twelve 252654 couplings. Our ref: 9938812."],
        input_modes=["text/plain"],
        output_modes=["text/plain"],
    )
]

INSTRUCTIONS = """You coordinate a network of specialised agents to fulfil customer orders.
You cannot look anything up yourself; you must delegate. Workflow:
1. Call list_available_agents to see which skills are currently available.
2. delegate(skill="order-extraction") to extract the order from the customer email.
3. delegate(skill="stock-check") with the "lines" list from step 2 as JSON. The stock report also contains the
   production plan (finish dates) for every shortfall - the inventory agent obtains it from the production agent.
4. Write the order confirmation for the customer:
   - one bullet per material: description, ordered quantity, quantity shipped from stock (warehouse),
     quantity produced and its production finish date
   - numbers that were ignored and why
   - expected complete delivery date (latest production finish date, or 'immediately' if all in stock)
   - if the customer requested a delivery date, state whether it can be met
Use only numbers returned by the agents. If an agent fails, say which step failed."""

EXTRACTION_SKILL = "order-extraction"
REQUIRED_SKILLS = (EXTRACTION_SKILL, "stock-check")


async def list_available_agents() -> list[dict]:
    """List all agents currently registered in the agent registry with their skills."""
    agents = await a2a_client.discover()
    return [
        {
            "agent": a["name"],
            "description": a["card"].get("description"),
            "skills": [{"id": s["id"], "description": s["description"]} for s in a["card"].get("skills", [])],
        }
        for a in agents
        if a["name"] != slugify(NAME)
    ]


async def delegate(ctx: RunContext, skill: str, request: str = "", lines: list[OrderLine] | None = None) -> str:
    """Send a request to the agent that offers the given skill (looked up in the registry) and return its result.
    For 'order-extraction' the original customer message is always forwarded unchanged.

    Args:
        skill: Skill id, e.g. 'order-extraction' or 'stock-check'.
        request: The message for the agent (plain text or JSON).
        lines: Order lines to send instead of a text request (used for 'stock-check').
    """
    if skill == EXTRACTION_SKILL:
        request = latest_user_text(ctx) or request
    elif lines:
        request = lines_json(lines)
    if not request.strip():
        return "ERROR: provide either a request text or order lines."
    try:
        result = await a2a_client.call_skill(skill, request)
    except (a2a_client.A2AError, OSError) as e:
        log.warning("Delegation to '%s' failed: %s", skill, e)
        return f"ERROR: {e}"
    except Exception as e:
        log.exception("Delegation to '%s' failed", skill)
        return f"ERROR: {type(e).__name__}: {e}"
    log.info("Delegated '%s' to %s (task %s): %s", skill, result.agent, result.task_id[:8], request[:200])
    return result.as_tool_output()


def delegation_results(ctx: RunContext) -> tuple[dict[str, tuple[str, str]], set[str]]:
    """Return {skill: (request, result)} of the last successful delegation per skill and the skills that failed."""
    calls: dict[str, tuple[str, str]] = {}
    for message in ctx.messages:
        if isinstance(message, ModelResponse):
            for part in message.parts:
                if isinstance(part, ToolCallPart) and part.tool_name == "delegate":
                    args = part.args_as_dict()
                    request = json.dumps(args["lines"]) if args.get("lines") else str(args.get("request", ""))
                    calls[part.tool_call_id] = (args.get("skill", ""), request)
    succeeded: dict[str, tuple[str, str]] = {}
    failed: set[str] = set()
    for message in ctx.messages:
        if isinstance(message, ModelRequest):
            for part in message.parts:
                if isinstance(part, ToolReturnPart) and part.tool_call_id in calls:
                    skill, request = calls[part.tool_call_id]
                    content = str(part.content)
                    if content.startswith("ERROR"):
                        failed.add(skill)
                    else:
                        succeeded[skill] = (request, content)
    return succeeded, failed


def create_agent(model=None) -> Agent:
    agent = Agent(
        model or build_model(),
        name="orchestrator",
        instructions=INSTRUCTIONS,
        tools=[list_available_agents, delegate],
        retries=3,
    )

    @agent.instructions
    def current_date() -> str:
        return f"Today is {date.today().isoformat()}."

    @agent.output_validator
    def enforce_workflow(ctx: RunContext, output: str) -> str:
        done, failed = delegation_results(ctx)
        missing = [s for s in REQUIRED_SKILLS if s not in done and s not in failed]
        if missing:
            raise ModelRetry(f"Workflow incomplete: you still have to delegate {missing} before answering.")
        if "order-extraction" in done and "stock-check" in done:
            extracted = parse_order_lines(done["order-extraction"][1])
            requested = parse_order_lines(done["stock-check"][0])
            if extracted is not None and (requested is None or as_pairs(requested) != as_pairs(extracted)):
                raise ModelRetry(
                    "The stock check must use exactly the extracted order lines. "
                    f"Delegate stock-check again with: {lines_json(extracted)}"
                )
        return output

    return agent
