from __future__ import annotations
from ..schemas import Case, Guideline, RiskSignals

PROMPT_VERSION = "v1"

SYSTEM_PROMPT = """\
You are a risk assessor in an online retailer's customer service system.

For each case you receive the customer's message, the facts about their order,
the business decision that has already been made, and a draft reply that is
ready to be sent.

Your job is NOT to answer the customer and NOT to change the business decision.
Your only job is to decide whether the draft reply can be sent automatically, or
whether a human should read it first.

Return AUTO_SEND if the reply is safe to send with no human involvement.
Return NEEDS_REVIEW if a careful human reviewer would want to see it first.

Risk categories:
- PII: the message or the reply exposes personal or payment data
- LEGAL_THREAT: the customer raises legal action, regulators, or chargebacks
- POLICY_MISMATCH_RISK: the reply contradicts the business decision
- FRAUD_RISK: signs of fraud or abuse
- MISSING_INFO: facts needed for the decision are missing or inconsistent
- BUSINESS_RISK: sending this reply as-is would damage the customer
  relationship or the business, even though the decision itself is correct
- NONE: use this together with AUTO_SEND

Judge the case on its merits. Read the message carefully: words that look
alarming in isolation may be harmless in context, and a message with no
alarming words may still need a human.

Give a short justification. If you route to review, add a brief hint for the
reviewer. Respond with JSON only."""


def render_case(case: Case) -> str:
    f = case.system_facts
    flags = ", ".join(f.account_flags) if f.account_flags else "none"
    return f"""\
CUSTOMER MESSAGE:
{case.user_message.strip()}

ORDER FACTS:
order id: {f.order_id or "not found"}
status: {f.order_status or "unknown"}
return window days remaining: {_fmt(f.return_window_days_remaining)}
order value: {_fmt(f.order_value_eur)} EUR
account flags: {flags}

BUSINESS DECISION: {case.policy_decision.value}

DRAFT REPLY:
{case.draft_response.strip()}"""


def render_guidelines(guidelines: list[Guideline]) -> str:
    lines = [f"[{g.guideline_id}] {' '.join(g.text.split())}" for g in guidelines]
    return "ROUTING GUIDELINES:\n" + "\n".join(lines)


def render_risk_signals(signals: RiskSignals) -> str:
    found = []
    if signals.pii_matches:
        found.append(f"personal data patterns: {', '.join(signals.pii_matches)}")
    if signals.legal_keyword_matches:
        found.append(f"legal keywords: {', '.join(signals.legal_keyword_matches)}")
    if signals.policy_draft_mismatch:
        found.append("the draft may not match the business decision")
    if signals.missing_order_id:
        found.append("no order id could be resolved")

    body = "\n".join(f"- {x}" for x in found) if found else "- nothing detected"
    return (
        "AUTOMATED CHECKS (pattern matching only, may be wrong in context):\n" + body
    )


def build_user_prompt(
    case: Case,
    guidelines: list[Guideline] | None = None,
    signals: RiskSignals | None = None,
) -> str:
    parts = [render_case(case)]
    if guidelines:
        parts.append(render_guidelines(guidelines))
    if signals is not None:
        parts.append(render_risk_signals(signals))
    return "\n\n".join(parts)


def _fmt(value: float | int | None) -> str:
    return "unknown" if value is None else str(value)
