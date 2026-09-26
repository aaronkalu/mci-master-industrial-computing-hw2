import json
import re

from pydantic import ValidationError
from pydantic_ai import ModelRetry, RunContext
from pydantic_ai.messages import ModelRequest, UserPromptPart

from order_network.models import OrderLine

JSON_PATTERN = re.compile(r"[\[{].*[\]}]", re.DOTALL)


def latest_user_text(ctx: RunContext) -> str:
    for message in reversed(ctx.messages):
        if isinstance(message, ModelRequest):
            texts = [p.content for p in message.parts if isinstance(p, UserPromptPart) and isinstance(p.content, str)]
            if texts:
                return "\n".join(texts)
    return ""


def parse_order_lines(text: str, key: str = "lines") -> list[OrderLine] | None:
    """Order lines from a JSON list (or an object holding them under `key`) inside `text`; None if there is none."""
    match = JSON_PATTERN.search(text)
    if not match:
        return None
    try:
        data = json.loads(match.group())
        items = data.get(key) if isinstance(data, dict) else data
        return [OrderLine.model_validate(item) for item in items] if isinstance(items, list) else None
    except (json.JSONDecodeError, ValidationError, TypeError):
        return None


def as_pairs(lines: list[OrderLine]) -> list[tuple[str, int]]:
    return sorted((line.material_number, line.quantity) for line in lines)


def lines_json(lines: list[OrderLine]) -> str:
    return json.dumps([line.model_dump() for line in lines])


def ensure_lines_match_request(ctx: RunContext, lines: list[OrderLine]) -> None:
    """If the request contained JSON order lines, the model must pass exactly those lines on."""
    expected = parse_order_lines(latest_user_text(ctx))
    if expected is not None and as_pairs(expected) != as_pairs(lines):
        raise ModelRetry(f"Use exactly the order lines from the request: {lines_json(expected)}")
