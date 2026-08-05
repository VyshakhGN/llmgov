from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import Any
import yaml
from .policy import PolicyEngine
from .schemas import Case, Guideline, SystemFacts


@dataclass(frozen=True)
class GuidelineCorpus:
    guidelines: list[Guideline]
    high_order_value_eur: float


def load_guidelines(path: str | Path) -> GuidelineCorpus:
    doc = _read_yaml(path)
    return GuidelineCorpus(
        guidelines=[Guideline(**g) for g in doc["guidelines"]],
        high_order_value_eur=doc["high_order_value_eur"],
    )


def load_orders(path: str | Path) -> dict[str, SystemFacts]:
    doc = _read_yaml(path)
    return {oid: SystemFacts(order_id=oid, **facts) for oid, facts in doc["orders"].items()}


def load_cases(
    path: str | Path,
    orders: dict[str, SystemFacts],
    policy: PolicyEngine,
) -> list[Case]:
    doc = _read_yaml(path)
    cases = []
    for raw in doc["cases"]:
        order_id = raw.pop("order_id", None)

        if order_id is None:
            facts = SystemFacts()
        elif order_id in orders:
            facts = orders[order_id]
        else:
            raise ValueError(
                f"{raw.get('case_id', '?')}: order_id {order_id!r} not found in orders file"
            )

        if "policy_decision" in raw:
            raise ValueError(
                f"{raw.get('case_id', '?')}: policy_decision comes from the policy "
                "engine, remove it from the case file"
            )
        raw["policy_decision"] = policy.decide(facts).decision

        cases.append(Case(system_facts=facts, **raw))

    ids = [c.case_id for c in cases]
    if len(set(ids)) != len(ids):
        raise ValueError(f"duplicate case_id in {path}")
    return cases


def _read_yaml(path: str | Path) -> dict[str, Any]:
    with open(path, encoding="utf-8") as fh:
        return yaml.safe_load(fh)
