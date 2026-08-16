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
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_extract_prompt(message)},
            ],
            format=EXTRACT_SCHEMA,
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
            return None

        found = (found or "").strip()
        return found or None
