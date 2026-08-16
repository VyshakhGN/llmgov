"""Writes the customer reply, given a decision that has already been made.

Runs on masked text, so it never sees a real IBAN or card number and therefore
cannot put one in a reply. The refund amount and the return window are passed in
rather than worked out, so the model has no opportunity to invent a figure — an
earlier version was told neither and quoted a plausible-sounding 30 days.
"""

from __future__ import annotations

import ollama

from ..policy import PolicyEngine
from ..risk.masking import mask_pii
from ..schemas import Case
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_draft_prompt

DEFAULT_HOST = "http://localhost:11434"


class Drafter:
    def __init__(
        self,
        *,
        model_name: str,
        policy: PolicyEngine,
        seed: int = 0,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.model_name = model_name
        self.policy = policy
        self.seed = seed
        self.prompt_version = PROMPT_VERSION
        self.client = ollama.Client(host=host)

    def write(self, case: Case) -> str:
        """Return a reply for this case. The case's own draft is ignored."""
        masked_message, _ = mask_pii(case.user_message)
        masked_case = case.model_copy(update={"user_message": masked_message})

        outcome = self.policy.decide(case.system_facts)
        amount = self.policy.refund_amount(case.system_facts, outcome.decision)
        window = self.policy.window_for(case.system_facts)

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": build_draft_prompt(
                        masked_case,
                        outcome.because,
                        amount,
                        window,
                        self.policy.refund_processing_days,
                    ),
                },
            ],
            # Reasoning models otherwise spend the whole budget thinking and
            # return an empty reply. A JSON schema does not prevent this; the
            # router survives only because its larger budget leaves room to
            # think and still answer.
            think=False,
            options={
                "temperature": 0.0,
                "seed": self.seed,
                "num_ctx": 8192,
                "num_predict": 400,
            },
        )
        return (response.message.content or "").strip()
