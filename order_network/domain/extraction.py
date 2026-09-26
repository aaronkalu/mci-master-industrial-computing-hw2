import re

from order_network.models import IgnoredNumber

NUMBER_PATTERN = re.compile(r"(?<![\d-])(?<!\d[.,])\d{5,10}(?!\d)(?![.,-]\d)")


def classify_numbers(text: str) -> tuple[list[str], list[IgnoredNumber]]:
    """Material numbers are exactly 6 digits and never start with 9 (9xxx are customer order references)."""
    materials: list[str] = []
    ignored: list[IgnoredNumber] = []
    for value in NUMBER_PATTERN.findall(text):
        if value.startswith("9"):
            reason = "starts with 9: customer order reference, not a material number"
        elif len(value) != 6:
            reason = f"{len(value)} digits: material numbers have exactly 6 digits"
        else:
            if value not in materials:
                materials.append(value)
            continue
        if all(i.value != value for i in ignored):
            ignored.append(IgnoredNumber(value=value, reason=reason))
    return materials, ignored
