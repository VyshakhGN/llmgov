from __future__ import annotations

import re

from .detectors import CARD_RE, EMAIL_RE, IBAN_RE, PHONE_RE, _luhn_ok


def _mask_iban(match: re.Match[str]) -> str:
    flat = re.sub(r"\s", "", match.group())
    if len(flat) < 8:
        return match.group()
    return f"{flat[:4]}****{flat[-2:]}"


def _mask_card(match: re.Match[str]) -> str:
    digits = "".join(c for c in match.group() if c.isdigit())
    if not (13 <= len(digits) <= 19 and _luhn_ok(digits)):
        return match.group()
    return f"****{digits[-4:]}"


def _mask_email(match: re.Match[str]) -> str:
    local, _, domain = match.group().partition("@")
    return f"{local[:3]}***@{domain}"


def _mask_phone(match: re.Match[str]) -> str:
    digits = "".join(c for c in match.group() if c.isdigit())
    if len(digits) < 6:
        return match.group()
    return f"***{digits[-4:]}"


_MASKS = (
    ("iban", IBAN_RE, _mask_iban),
    ("card", CARD_RE, _mask_card),
    ("email", EMAIL_RE, _mask_email),
    ("phone", PHONE_RE, _mask_phone),
)


def mask_pii(text: str) -> tuple[str, list[str]]:
    found: list[str] = []

    for kind, pattern, replace in _MASKS:
        new_text = pattern.sub(replace, text)
        if new_text != text:
            found.append(kind)
            text = new_text

    return text, found
