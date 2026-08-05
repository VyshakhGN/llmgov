from __future__ import annotations
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import yaml

from ..schemas import PolicyDecision, SystemFacts


@dataclass(frozen=True)
class PolicyOutcome:
    decision: PolicyDecision
    rule: str
    because: str


class PolicyEngine:
    def __init__(self, rules: list[dict[str, str]]) -> None:
        self.rules = rules
        self.checks: dict[str, Callable[[SystemFacts], bool]] = {
            "no_order_found": lambda f: f.order_id is None,
            "return_window_unknown": lambda f: f.return_window_days_remaining is None,
            "within_return_window": lambda f: (f.return_window_days_remaining or 0) >= 0,
            "outside_return_window": lambda f: (f.return_window_days_remaining or 0) < 0,
        }

    def decide(self, facts: SystemFacts) -> PolicyOutcome:
        """First matching rule wins."""
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
    return PolicyEngine(rules=doc["rules"])
