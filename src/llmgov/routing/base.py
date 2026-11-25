from __future__ import annotations
from typing import Protocol, runtime_checkable
from ..risk.masking import mask_pii
from ..schemas import Action, Case, Mode, RiskCategory, RiskSignals, RoutingDecision


@runtime_checkable
class Router(Protocol):

    mode: Mode
    guideline_corpus_version: str

    def route(self, case: Case, risk_signals: RiskSignals) -> RoutingDecision:
        ...


def fail_safe(reason: str, *, stage: str | None = None) -> RoutingDecision:
    source = f"{stage.capitalize()} stage" if stage else "Router"
    return RoutingDecision(
        action=Action.NEEDS_REVIEW,
        risk_category=RiskCategory.NONE,
        justification=f"{source} failure, failed closed to review: {reason}",
        parse_error=reason,
        failure_stage=stage,
    )


def stage_failure(stage: str, exc: Exception) -> RoutingDecision:
    reason, _ = mask_pii(f"{type(exc).__name__}: {exc}")
    return fail_safe(reason, stage=stage)
