from datetime import date

from fasta2a.schema import Skill
from pydantic_ai import Agent, ModelRetry, RunContext, ToolOutput

from order_network.agents.common import latest_user_text
from order_network.domain.extraction import classify_numbers
from order_network.models import OrderLine, OrderRequest
from order_network.runtime import build_model

NAME = "Order Intake Agent"
DESCRIPTION = "Turns free-text customer emails into structured order requests (material numbers + quantities)."
SKILLS = [
    Skill(
        id="order-extraction",
        name="Order extraction",
        description="Extracts customer, material numbers, quantities and requested delivery date from an order email. "
                    "Filters out customer order references (numbers starting with 9).",
        tags=["orders", "extraction", "email", "intake"],
        examples=["Hi, please send 12 pcs of 252653 and 5 of 149449 by 2026-10-15."],
        input_modes=["text/plain"],
        output_modes=["application/json"],
    )
]

INSTRUCTIONS = """You extract order data from customer emails.
1. Call find_material_numbers with the complete customer text.
2. Match every valid material number to the quantity the customer asked for (convert words like 'twelve' to 12;
   if no quantity is given use 1).
3. Finish by calling submit_order_request with the customer name (if any), the order lines and the requested
   delivery date (ISO format, only if the customer mentioned one).
Never include numbers the tool marked as ignored."""


def find_material_numbers(text: str) -> dict:
    """Find valid material numbers in the customer text and list the numbers that must be ignored (with reason).

    Args:
        text: The complete, unmodified customer message.
    """
    materials, ignored = classify_numbers(text)
    return {"material_numbers": materials, "ignored": [i.model_dump() for i in ignored]}


def submit_order_request(
    ctx: RunContext, lines: list[OrderLine], customer: str | None = None, requested_delivery: date | None = None
) -> OrderRequest:
    """Submit the final structured order request."""
    materials, ignored = classify_numbers(latest_user_text(ctx))
    submitted = [line.material_number for line in lines]
    invalid = [m for m in submitted if m not in materials]
    missing = [m for m in materials if m not in submitted]
    if invalid:
        raise ModelRetry(f"These are not valid material numbers from the email: {invalid}. Valid ones: {materials}.")
    if missing:
        raise ModelRetry(f"You missed these material numbers from the email: {missing}.")
    if len(set(submitted)) != len(submitted):
        raise ModelRetry("Each material number must appear only once; add up the quantities instead.")
    return OrderRequest(customer=customer, lines=lines, requested_delivery=requested_delivery, ignored_numbers=ignored)


def create_agent(model=None) -> Agent:
    agent = Agent(
        model or build_model(),
        name="order-intake",
        instructions=INSTRUCTIONS,
        tools=[find_material_numbers],
        output_type=ToolOutput(submit_order_request, name="submit_order_request"),
        retries=3,
    )

    @agent.instructions
    def current_date() -> str:
        return f"Today is {date.today().isoformat()}."

    return agent
