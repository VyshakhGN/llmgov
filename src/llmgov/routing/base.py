from __future__ import annotations
from typing import Protocol, runtime_checkable
from ..schemas import Action, Case, Mode, RiskCategory, RiskSignals, RoutingDecision


@runtime_checkable
class Router(Protocol):

    mode: Mode
    guideline_corpus_version: str

    def route(self, case: Case, risk_signals: RiskSignals) -> RoutingDecision:
        ...


def fail_safe(reason: str) -> RoutingDecision:
    return RoutingDecision(
        action=Action.NEEDS_REVIEW,
        risk_category=RiskCategory.NONE,
        justification=f"Router failure, failed closed to review: {reason}",
        parse_error=reason,
    )
