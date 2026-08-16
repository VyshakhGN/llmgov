"""Prompt for the reply drafter.

The drafter never sees the routing guidelines. If it did, it would write replies
that already comply with them, and the router would be judging output shaped by
its own rulebook.
"""

from __future__ import annotations

from ..schemas import Case, PolicyDecision

PROMPT_VERSION = "draft-v4"

SYSTEM_PROMPT = """\
You write replies to customers for an online retailer's returns team.

You are given the customer's message, the facts about their order, and the
decision that has already been made about their refund. Write the reply that
tells them the outcome.

Rules:
- The decision is already made. Communicate it; do not change it or argue with it.
- Use only the amount, return window, and payment time you are given. Never
  state a figure or a timescale from memory.
- Write as "we", never "I". You are writing on behalf of the company.
- Any account, card, email or phone details in the message are already partly
  hidden, like DE89****00. Copy them exactly as they appear if you need them.
  Never invent one: if the customer gave no account details, write "your
  original payment method".
- Be brief and plain. Two or three sentences is usually enough.
- Be warm but not effusive. No marketing language.
- Write only the reply itself, with no subject line, greeting placeholders, or
  sign-off name."""

_DECISION_BRIEF = {
    PolicyDecision.APPROVE_REFUND: "The refund is approved.",
    PolicyDecision.PARTIAL_REFUND: (
        "A reduced refund is approved because the item came back damaged. "
        "Explain the reduction."
    ),
    PolicyDecision.DENY_REFUND: "The refund is refused.",
    PolicyDecision.REQUEST_INFO: (
        "No decision yet — ask the customer for what is missing."
    ),
}


def _item_is_back(order_status: str | None) -> bool:
    return order_status == "RETURN_RECEIVED"


def _return_state(order_status: str | None) -> str:
    if order_status is None:
        return "unknown"
    if _item_is_back(order_status):
        return "yes, we have it back and have checked it"
    return "no, the item is still with the customer"


def _return_brief(decision: PolicyDecision, item_is_back: bool) -> str:
    """What to say about the return itself.

    Listing every branch and leaving the model to pick went badly: it applied the
    not-yet-received wording to items already in the warehouse. The branch is
    chosen here instead, from facts the system already holds.
    """
    if decision is PolicyDecision.REQUEST_INFO:
        return ""
    if decision is PolicyDecision.DENY_REFUND:
        return (
            "There is no refund and there will not be one. Do not invite the "
            "customer to send the item back, and do not suggest a payment might "
            "follow later."
        )
    if item_is_back:
        return "The item is back with us and has been checked, so the refund is being issued now."
    return (
        "We do not have the item yet. The refund is issued once it reaches us and "
        "has been checked. Do not write anything implying we already have it."
    )


def build_draft_prompt(
    case: Case,
    reason: str,
    refund_amount: float | None,
    return_window_days: int | None,
    refund_processing_days: int | None = None,
) -> str:
    f = case.system_facts
    amount = (
        f"{refund_amount:.2f} EUR" if refund_amount is not None else "not applicable"
    )
    # "from delivery" matters: the model otherwise describes the window as
    # running from the purchase date, which is both wrong and a policy claim.
    window = (
        f"{return_window_days} days from delivery"
        if return_window_days is not None
        else "unknown"
    )
    # Only meaningful when money is actually moving.
    payment_time = (
        f"\nOnce issued, the refund reaches the customer within "
        f"{refund_processing_days} working days."
        if refund_amount is not None and refund_processing_days is not None
        else ""
    )
    returned = _return_brief(case.policy_decision, _item_is_back(f.order_status))
    if returned:
        returned = f"\n{returned}"
    return f"""\
CUSTOMER MESSAGE:
{case.user_message.strip()}

ORDER FACTS:
order id: {f.order_id or "not found"}
category: {f.category.value if f.category else "unknown"}
delivered: {f.delivered_days_ago if f.delivered_days_ago is not None else "unknown"} days ago
return window for this category: {window}
item returned to us: {_return_state(f.order_status)}
faulty on arrival: {"yes" if f.is_faulty else "no"}
order value: {f.order_value_eur if f.order_value_eur is not None else "unknown"} EUR

DECISION: {case.policy_decision.value}
{_DECISION_BRIEF[case.policy_decision]}
Reason: {reason.strip()}{returned}
Amount to refund: {amount}{payment_time}

Write the reply."""
