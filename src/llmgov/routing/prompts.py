from __future__ import annotations
from ..schemas import Case, Guideline

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
category: {f.category.value if f.category else "unknown"}
delivered: {_fmt(f.delivered_days_ago)} days ago
faulty on arrival: {"yes" if f.is_faulty else "no"}
condition on return: {f.return_condition.value if f.return_condition else "not yet returned"}
order value: {_fmt(f.order_value_eur)} EUR

CUSTOMER:
account age: {f"{f.customer_since_months} months" if f.customer_since_months is not None else "unknown"}
orders placed: {_fmt(f.total_orders)}
refunds in the last 12 months: {_fmt(f.refunds_last_12m)}
account flags: {flags}

BUSINESS DECISION: {case.policy_decision.value}

DRAFT REPLY:
{case.draft_response.strip()}"""


def render_guidelines(guidelines: list[Guideline]) -> str:
    lines = [f"[{g.guideline_id}] {' '.join(g.text.split())}" for g in guidelines]
    return "ROUTING GUIDELINES:\n" + "\n".join(lines)


def build_user_prompt(
    case: Case,
    guidelines: list[Guideline] | None = None,
) -> str:
    parts = [render_case(case)]
    if guidelines:
        parts.append(render_guidelines(guidelines))
    return "\n\n".join(parts)


def _fmt(value: float | int | None) -> str:
    return "unknown" if value is None else str(value)
