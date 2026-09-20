from __future__ import annotations

import html as htmllib
import re
import sys
from pathlib import Path

from llmgov.loading_pdf import load_guidelines_pdf

HTML = Path("data/guidelines/v2/returns-escalation-policy.html")
PDF = Path("data/guidelines/v2/returns-escalation-policy.pdf")

CLAUSE = re.compile(
    r'<div class="clause">'
    r'<p class="head">[\d.]+\s*&nbsp;\s*(?P<id>G-[A-Z]{3,4}-\d{3})\s*'
    r'<span class="cat">&middot;\s*(?P<category>[A-Z_]+)</span></p>'
    r'<p class="body">(?P<text>.*?)</p>'
    r'<p class="basis">Basis:\s*(?P<source>.*?)</p>'
    r"</div>",
    re.S,
)


def flat(value: object) -> str:
    text = getattr(value, "value", value)
    return " ".join(htmllib.unescape(str(text)).split())


def read_source(path: Path) -> dict[str, dict[str, str]]:
    """The authored document, before it was printed."""
    clauses = {
        m.group("id"): {k: flat(v) for k, v in m.groupdict().items()}
        for m in CLAUSE.finditer(path.read_text(encoding="utf-8"))
    }
    if not clauses:
        raise SystemExit(f"no clauses found in {path} - has the markup changed?")
    return clauses


def check(html_path: Path, pdf_path: Path) -> int:
    expected = read_source(html_path)
    got = {
        g.guideline_id: {
            "category": flat(g.risk_category),
            "text": flat(g.text),
            "source": flat(g.source),
        }
        for g in load_guidelines_pdf(pdf_path).guidelines
    }

    missing = sorted(set(expected) - set(got))
    extra = sorted(set(got) - set(expected))
    mismatches = [
        (gid, field, expected[gid][field], got[gid][field])
        for gid in sorted(set(expected) & set(got))
        for field in ("category", "text", "source")
        if expected[gid][field] != got[gid][field]
    ]

    print(f"authored   {html_path}  {len(expected)} clauses")
    print(f"extracted  {pdf_path}  {len(got)} clauses")
    print()
    print(f"  lost in printing or extraction : {', '.join(missing) if missing else 'none'}")
    print(f"  not in the authored document   : {', '.join(extra) if extra else 'none'}")
    print(f"  field mismatches               : {len(mismatches)}")

    for gid, field, a, b in mismatches:
        print(f"\n  {gid} · {field}\n    authored : {a}\n    extracted: {b}")

    print()
    print(
        f"fidelity: {len(expected) - len(missing)}/{len(expected)} clauses recovered, "
        f"{len(mismatches)} field differences"
    )
    return len(missing) + len(extra) + len(mismatches)


if __name__ == "__main__":
    sys.exit(1 if check(HTML, PDF) else 0)
