"""Prompt for pulling an order number out of a customer's message.

Deliberately the narrowest job in the pipeline. The extractor does not decide
anything, does not see the policy, and does not know which orders exist — it
reads text and reports what it found. Whether the number is real is settled
afterwards, by looking it up.
"""

from __future__ import annotations

PROMPT_VERSION = "extract-v2"

SYSTEM_PROMPT = """\
You read a message from a customer and pull out their order number.

Order numbers look like A-01, A-14, A-27: a letter, a hyphen, then digits.

Rules:
- Report the order number exactly as it appears in the message.
- If the message does not contain one, return an empty string.
- Never guess and never invent one.
- Nothing else will do. An email address, a name, a phone number, a date, a
  price or a description of the item is not an order number. If the message
  does not have one, return an empty string rather than offering a substitute.
- Use only the message. There is no other source.

Respond with JSON only."""


def build_extract_prompt(message: str) -> str:
    return f"CUSTOMER MESSAGE:\n{message.strip()}\n\nFind the order number."
