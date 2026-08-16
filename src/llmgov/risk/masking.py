"""Partially masks personal data before text reaches the model.

Keeps just enough for a customer to recognise their own detail — the shape
retailers and banks actually use — while removing the part that identifies them.
The model gets the same masked text the customer will, so nothing has to be
substituted back in afterwards.

    DE89 3704 0044 0532 0130 00  ->  DE89****00
    4111 1111 1111 1111          ->  ****1111
    m.rashford@example.com       ->  m.r***@example.com
    +49 170 1234567              ->  ***4567

Order matters. The phone pattern overlaps with IBANs and card numbers, so those
are masked first and their digits are gone before the phone pass runs.
"""

from __future__ import annotations

import re

from .detectors import CARD_RE, EMAIL_RE, IBAN_RE, PHONE_RE, _luhn_ok


def _mask_iban(match: re.Match[str]) -> str:
    """Keep the country code and check digits, and the last two characters."""
    flat = re.sub(r"\s", "", match.group())
    if len(flat) < 8:
        return match.group()
    return f"{flat[:4]}****{flat[-2:]}"


def _mask_card(match: re.Match[str]) -> str:
    """Last four only, as PCI-DSS permits. Runs the checksum first so order
    numbers and tracking IDs are left alone."""
    digits = "".join(c for c in match.group() if c.isdigit())
    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
        return match.group()
    return f"****{digits[-4:]}"


def _mask_email(match: re.Match[str]) -> str:
    """Keep the first three characters and the whole domain."""
    local, _, domain = match.group().partition("@")
    return f"{local[:3]}***@{domain}"


def _mask_phone(match: re.Match[str]) -> str:
    digits = "".join(c for c in match.group() if c.isdigit())
    if len(digits) < 6:
        return match.group()
    return f"***{digits[-4:]}"


# Applied in this order; earlier ones win.
_MASKS = (
    ("iban", IBAN_RE, _mask_iban),
    ("card", CARD_RE, _mask_card),
    ("email", EMAIL_RE, _mask_email),
    ("phone", PHONE_RE, _mask_phone),
)


def mask_pii(text: str) -> tuple[str, list[str]]:
    """Return the masked text and which kinds of data were found.

    The kinds are recorded, never the values — traces must not accumulate the
    personal data the system exists to protect.
    """
    found: list[str] = []

    for kind, pattern, replace in _MASKS:
        new_text = pattern.sub(replace, text)
        if new_text != text:
            found.append(kind)
            text = new_text

    return text, found
