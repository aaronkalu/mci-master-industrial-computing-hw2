import logging

import httpx
from fasta2a.schema import Skill
from pydantic import ValidationError
from pydantic_ai import Agent, RunContext, ToolOutput

from order_network import a2a_client, config
from order_network.agents.common import ensure_lines_match_request, lines_json
from order_network.domain import inventory
from order_network.models import OrderLine, ProductionPlan, StockReport
from order_network.runtime import build_model

log = logging.getLogger("inventory")

NAME = "Inventory Agent"
DESCRIPTION = ("Checks warehouse stock for material numbers. For missing quantities it asks the production "
               "planning agent (found via the registry) for a production plan.")
SKILLS = [
    Skill(
        id="stock-check",
        name="Stock check",
        description="For each material number and quantity: available stock, quantity that can be reserved now, "
                    "shortfall, warehouse location and - for shortfalls - the production plan with finish dates.",
        tags=["inventory", "stock", "warehouse", "availability"],
        examples=['[{"material_number": "149449", "quantity": 50}]'],
        input_modes=["text/plain", "application/json"],
        output_modes=["application/json"],
    )
]

DB_PATH = config.DATA_DIR / "inventory.db"
PRODUCTION_SKILL = "production-scheduling"

INSTRUCTIONS = """You are the inventory system of an industrial parts manufacturer.
The request contains material numbers with requested quantities (as JSON or text).
Call check_stock exactly once with ALL requested lines. Do not change material numbers or quantities."""


async def request_production(lines: list[OrderLine]) -> tuple[ProductionPlan | None, str | None]:
    try:
        result = await a2a_client.call_skill(PRODUCTION_SKILL, lines_json(lines))
        plan = ProductionPlan.model_validate(result.data)
    except (a2a_client.A2AError, httpx.HTTPError, ValidationError) as e:
        log.warning("Production planning failed: %s", e)
        return None, f"Production planning unavailable: {e}"
    log.info("Production plan received from %s (task %s)", result.agent, result.task_id[:8])
    return plan, None


async def check_stock(ctx: RunContext, lines: list[OrderLine]) -> StockReport:
    """Check stock for all requested order lines and return the stock report."""
    ensure_lines_match_request(ctx, lines)
    report = inventory.check_stock(DB_PATH, lines)
    if report.production_request:
        report.production_plan, report.production_note = await request_production(report.production_request)
    return report


def create_agent(model=None) -> Agent:
    inventory.init_db(DB_PATH)
    return Agent(
        model or build_model(),
        name="inventory",
        instructions=INSTRUCTIONS,
        output_type=ToolOutput(check_stock, name="check_stock"),
        retries=3,
    )
