from __future__ import annotations
import re
from ..schemas import Case, PolicyDecision, RiskSignals

IBAN_RE = re.compile(r"\b[A-Z]{2}\d{2}(?:[ ]?[A-Z0-9]{2,4}){2,8}\b")

# Separators sit between digits only, so a trailing space is not swallowed.
CARD_RE = re.compile(r"\b\d(?:[ -]?\d){12,18}\b")

EMAIL_RE = re.compile(r"\b[\w.+-]+@[\w-]+\.[\w.-]+\b")

# Deliberately loose; short matches are discarded by the digit-count check in
# masking, so groupings like "12 34" are left alone.
PHONE_RE = re.compile(r"(?:\+\d{1,3}[ -]?)?\d{2,4}(?:[ -]?\d{2,4}){1,4}\b")

LEGAL_KEYWORDS = (
    "lawyer",
    "solicitor",
    "attorney",
    "legal action",
    "legal advice",
    "take legal",
    "sue",
    "suing",
    "court",
    "small claims",
    "regulator",
    "consumer protection",
    "chargeback",
    "charge back",
    "dispute the payment",
    "file a complaint",
)

APPROVAL_MARKERS = (
    "approved",
    "we have approved",
    "refund of",
    "back on your card",
    "back to your account",
    "refund will be processed",
)

DENIAL_MARKERS = (
    "unable to",
    "not able to",
    "can't accept",
    "cannot accept",
    "can't help",
    "cannot help",
    "outside the",
    "falls outside",
    "no longer eligible",
)

def _luhn_ok(digits: str) -> bool:
    total = 0
    for i, ch in enumerate(reversed(digits)):
        n = int(ch)
        if i % 2 == 1:
            n *= 2
            if n > 9:
                n -= 9
        total += n
    return total % 10 == 0


def _has_card(text: str) -> bool:
    for m in CARD_RE.finditer(text):
        digits = re.sub(r"[ -]", "", m.group())
        if 13 <= len(digits) <= 19 and _luhn_ok(digits):
            return True
    return False


def _draft_stance(draft: str) -> PolicyDecision | None:
    low = draft.lower()
    approves = any(m in low for m in APPROVAL_MARKERS)
    denies = any(m in low for m in DENIAL_MARKERS)

    if approves and not denies:
        return PolicyDecision.APPROVE_REFUND
    if denies and not approves:
        return PolicyDecision.DENY_REFUND
    return None


def detect_risk_signals(case: Case) -> RiskSignals:
    message = case.user_message
    draft = case.draft_response
    both = f"{message}\n{draft}"

    has_iban = bool(IBAN_RE.search(both))
    has_card = _has_card(both)
    has_email = bool(EMAIL_RE.search(both))
    has_phone = bool(PHONE_RE.search(both))

    pii: list[str] = []
    if has_iban:
        pii.append("iban")
    if has_card:
        pii.append("card_number")
    if has_email:
        pii.append("email")
    if has_phone:
        pii.append("phone")
    low_message = message.lower()
    legal = [kw for kw in LEGAL_KEYWORDS if kw in low_message]

    stance = _draft_stance(draft)
    if stance is None:
        mismatch = False
    elif case.policy_decision is PolicyDecision.REQUEST_INFO:
        mismatch = True
    else:
        mismatch = stance is not case.policy_decision

    return RiskSignals(
        pii_matches=pii,
        legal_keyword_matches=legal,
        has_iban_like=has_iban,
        has_card_like=has_card,
        has_email=has_email,
        has_phone=has_phone,
        policy_draft_mismatch=mismatch,
        missing_order_id=case.system_facts.order_id is None,
    )
