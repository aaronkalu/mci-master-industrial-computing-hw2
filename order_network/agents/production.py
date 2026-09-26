from fasta2a.schema import Skill
from pydantic_ai import Agent, RunContext, ToolOutput

from order_network.agents.common import ensure_lines_match_request
from order_network.domain.planning import plan_production
from order_network.models import OrderLine, ProductionPlan
from order_network.runtime import build_model

NAME = "Production Planning Agent"
DESCRIPTION = "Schedules production jobs for missing quantities and returns start/finish dates and lead times."
SKILLS = [
    Skill(
        id="production-scheduling",
        name="Production scheduling",
        description="Plans production for material numbers and quantities on the right production line, "
                    "considering line backlog and capacity. Returns finish dates and lead times in business days.",
        tags=["production", "planning", "lead-time", "scheduling"],
        examples=['[{"material_number": "252654", "quantity": 20}]'],
        input_modes=["text/plain", "application/json"],
        output_modes=["application/json"],
    )
]

INSTRUCTIONS = """You are the production planning system of an industrial parts manufacturer.
The request contains material numbers with quantities that must be produced (as JSON or text).
Call schedule_production exactly once with ALL lines. Do not change material numbers or quantities."""


def schedule_production(ctx: RunContext, lines: list[OrderLine]) -> ProductionPlan:
    """Schedule production for all given lines and return the production plan."""
    ensure_lines_match_request(ctx, lines)
    return plan_production(lines)


def create_agent(model=None) -> Agent:
    return Agent(
        model or build_model(),
        name="production",
        instructions=INSTRUCTIONS,
        output_type=ToolOutput(schedule_production, name="schedule_production"),
        retries=3,
    )
