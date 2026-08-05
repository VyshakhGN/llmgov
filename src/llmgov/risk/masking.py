from __future__ import annotations
import re
from .detectors import CARD_RE, EMAIL_RE, IBAN_RE, PHONE_RE, _luhn_ok


def _mask_card(match: re.Match[str]) -> str:
    digits = "".join(c for c in match.group() if c.isdigit())
    if 13 <= len(digits) <= 19 and _luhn_ok(digits):
        return "<CARD>"
    return match.group()


def mask_pii(text: str) -> tuple[str, list[str]]:
    applied: list[str] = []

    for placeholder, pattern, replacement in (
        ("<IBAN>", IBAN_RE, "<IBAN>"),
        ("<CARD>", CARD_RE, _mask_card),
        ("<EMAIL>", EMAIL_RE, "<EMAIL>"),
        ("<PHONE>", PHONE_RE, "<PHONE>"),
    ):
        new_text = pattern.sub(replacement, text)
        if new_text != text:
            applied.append(placeholder)
            text = new_text

    return text, applied
