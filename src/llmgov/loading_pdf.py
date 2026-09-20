from __future__ import annotations

import re
from pathlib import Path

import pdfplumber

from .schemas import Guideline, GuidelineCorpus

CLAUSE_HEAD = re.compile(
    r"^\d+\.\d+\s+(?P<id>G-[A-Z]{3,4}-\d{3})\s*[·|]\s*(?P<category>[A-Z_]+)\s*$"
)
BASIS = re.compile(r"^Basis:\s*(?P<source>.+)$")
SECTION_HEAD = re.compile(r"^\d+\.\s+\S")
RUNNING_FURNITURE = re.compile(r"^(?:\d+\s+of\s+\d+\b|file:///|Page\s+\d+\b)")


def read_pdf_text(path: str | Path) -> str:
    with pdfplumber.open(path) as pdf:
        return "\n".join(page.extract_text() or "" for page in pdf.pages)


def _usable_lines(text: str) -> list[str]:
    out = []
    for raw in text.splitlines():
        line = raw.strip()
        if not line or RUNNING_FURNITURE.match(line):
            continue
        out.append(line)
    return out


def parse_policy_document(text: str) -> list[Guideline]:
    """Pull clauses out of a Shape A policy document.

    A clause is a numbered heading carrying an id and a risk category, one or more
    lines of rule text, then a Basis line. Anything outside that shape - the title,
    section headings, section intros, running headers - is ignored.
    """
    guidelines: list[Guideline] = []
    pending: dict[str, str] | None = None
    body: list[str] = []

    def close(source: str) -> None:
        if pending is None:
            raise ValueError(f"Basis line with no clause above it: {source!r}")
        if not body:
            raise ValueError(f"clause {pending['id']} has no text")
        guidelines.append(
            Guideline(
                guideline_id=pending["id"],
                risk_category=pending["category"],
                text=" ".join(body),
                source=source,
            )
        )

    for line in _usable_lines(text):
        head = CLAUSE_HEAD.match(line)
        if head:
            if pending is not None:
                raise ValueError(f"clause {pending['id']} is missing its Basis line")
            pending, body = head.groupdict(), []
            continue

        if pending is None:
            continue

        basis = BASIS.match(line)
        if basis:
            close(basis.group("source").strip())
            pending, body = None, []
            continue

        if SECTION_HEAD.match(line):
            raise ValueError(f"clause {pending['id']} is missing its Basis line")

        body.append(line)

    if pending is not None:
        raise ValueError(f"clause {pending['id']} is missing its Basis line")
    if not guidelines:
        raise ValueError("no clauses found - is this a Shape A policy document?")

    seen = {g.guideline_id for g in guidelines}
    if len(seen) != len(guidelines):
        raise ValueError("the document repeats a guideline id")

    return guidelines


def load_guidelines_pdf(path: str | Path) -> GuidelineCorpus:
    return GuidelineCorpus(guidelines=parse_policy_document(read_pdf_text(path)))
