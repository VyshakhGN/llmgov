from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..schemas import PolicyDecision, ReturnCondition, SystemFacts


@dataclass(frozen=True)
class PolicyOutcome:
    decision: PolicyDecision
    rule: str
    because: str


class PolicyEngine:
    def __init__(
        self,
        rules: list[dict[str, str]],
        return_windows: dict[str, int],
        damaged_return_deduction_pct: float = 0.0,
        refund_processing_days: int | None = None,
    ) -> None:
        self.rules = rules
        self.return_windows = return_windows
        self.damaged_return_deduction_pct = damaged_return_deduction_pct
        self.refund_processing_days = refund_processing_days
        self.checks: dict[str, Callable[[SystemFacts], bool]] = {
            "no_order_found": lambda f: f.order_id is None,
            "delivery_date_unknown": lambda f: f.delivered_days_ago is None,
            "category_unknown": lambda f: f.category is None,
            "item_faulty": lambda f: f.is_faulty,
            "return_damaged": lambda f: f.return_condition is ReturnCondition.DAMAGED,
            "within_return_window": lambda f: f.delivered_days_ago <= self._window(f),
            "outside_return_window": lambda f: f.delivered_days_ago > self._window(f),
        }

    def refund_amount(self, facts: SystemFacts, decision: PolicyDecision) -> float | None:
        if facts.order_value_eur is None:
            return None
        if decision is PolicyDecision.APPROVE_REFUND:
            return facts.order_value_eur
        if decision is PolicyDecision.PARTIAL_REFUND:
            kept = 1 - self.damaged_return_deduction_pct / 100
            return round(facts.order_value_eur * kept, 2)
        return None

    def window_for(self, facts: SystemFacts) -> int | None:
        """The return window that applies, or None if the category is unknown."""
        if facts.category is None:
            return None
        return self.return_windows.get(facts.category.value)

    def _window(self, facts: SystemFacts) -> int:
        window = self.window_for(facts)
        if window is None:
            raise ValueError(f"no return window configured for {facts.category}")
        return window

    def decide(self, facts: SystemFacts) -> PolicyOutcome:
        for rule in self.rules:
            name = rule["when"]
            check = self.checks.get(name)
            if check is None:
                raise ValueError(f"unknown policy check: {name}")
            if check(facts):
                return PolicyOutcome(
                    decision=PolicyDecision(rule["decision"]),
                    rule=name,
                    because=rule.get("because", ""),
                )

        return PolicyOutcome(
            decision=PolicyDecision.REQUEST_INFO,
            rule="no_rule_matched",
            because="no policy rule applied to these facts",
        )


def load_policy(path: str | Path) -> PolicyEngine:
    with open(path, encoding="utf-8") as fh:
        doc: dict[str, Any] = yaml.safe_load(fh)
    return PolicyEngine(
        rules=doc["rules"],
        return_windows=doc["return_windows"],
        damaged_return_deduction_pct=doc.get("damaged_return_deduction_pct", 0.0),
        refund_processing_days=doc.get("refund_processing_days"),
    )
