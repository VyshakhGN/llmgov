"""Finds the order number in a customer's message.

Replaces reading it out of the case file. Everything downstream is unchanged:
the id is looked up in the order book exactly as before, and an id that is
missing or unrecognised leaves the facts empty, which the policy engine already
handles as no_order_found.

The model is never shown the list of real order numbers. If it were, it could
match a plausible one rather than report what the message actually said, and a
wrong match is far worse than no match — it would attach a stranger's order to
the reply.
"""

from __future__ import annotations

import json
from typing import Any

import ollama

from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_extract_prompt

DEFAULT_HOST = "http://localhost:11434"

EXTRACT_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {"order_id": {"type": "string"}},
    "required": ["order_id"],
}


class OrderExtractor:
    def __init__(
        self,
        *,
        model_name: str,
        seed: int = 0,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.prompt_version = PROMPT_VERSION
        self.client = ollama.Client(host=host)

    def extract(self, message: str) -> str | None:
        """Return the order number found in the message, or None."""
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_extract_prompt(message)},
            ],
            format=EXTRACT_SCHEMA,
            # Without this the model reasons until the budget runs out and
            # returns nothing. An enforced schema does not prevent it.
            think=False,
            options={
                "temperature": 0.0,
                "seed": self.seed,
                "num_ctx": 4096,
                "num_predict": 60,
            },
        )

        try:
            found = json.loads(response["message"]["content"]).get("order_id")
        except Exception:
            # An unreadable answer is treated as nothing found, which sends the
            # case down the request-more-information path rather than guessing.
            return None

        found = (found or "").strip()
        return found or None
