from __future__ import annotations

import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import ollama

PROMPT_VERSION = "judge-v1"
DEFAULT_HOST = "http://localhost:11434"

SYSTEM_PROMPT = """\
You check whether a written explanation is supported by the material it was based on.

A risk assessor was shown a customer-service case and decided whether the reply could
be sent automatically. It then wrote a short justification. You are given both.

Say whether every factual claim in the justification is supported by the material.

A claim is unsupported if the material does not contain it, contradicts it, or if the
justification attributes something to a guideline that the guideline does not say -
for example inventing a numeric threshold where the guideline states none.

Do not judge whether the decision was correct. Only whether the justification's claims
are backed by what the assessor was given.

Respond with JSON only."""

_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "supported": {"type": "boolean"},
        "unsupported_claim": {"type": "string"},
    },
    "required": ["supported", "unsupported_claim"],
}


def build_judge_prompt(material: str, justification: str) -> str:
    return f"""\
MATERIAL THE ASSESSOR WAS GIVEN:
{material}

ITS JUSTIFICATION:
{justification.strip()}

Is every claim in the justification supported by the material? If not, quote the first
claim that is not. Leave the quote empty if everything is supported."""


class GroundednessJudge:
    def __init__(
        self,
        *,
        model_name: str,
        seed: int = 0,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.model_name = model_name
        self.seed = seed
        self.prompt_version = PROMPT_VERSION
        self.client = ollama.Client(host=host)

    def judge(self, trace: dict[str, Any]) -> dict[str, Any] | None:
        material = (trace["stages"] or {}).get("router_prompt") or ""
        justification = (trace["decision"] or {}).get("justification") or ""
        if not material or not justification:
            return None

        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": build_judge_prompt(material, justification)},
            ],
            think=False,
            format=_SCHEMA,
            options={
                "temperature": 0.0,
                "seed": self.seed,
                "num_ctx": 8192,
                "num_predict": 300,
            },
        )
        raw = (response.message.content or "").strip()
        try:
            verdict = json.loads(raw)
        except json.JSONDecodeError:
            return {"case_id": trace["case_id"], "supported": None, "parse_error": raw[:400]}

        return {
            "case_id": trace["case_id"],
            "supported": bool(verdict.get("supported")),
            "unsupported_claim": str(verdict.get("unsupported_claim", "")).strip(),
        }


def judge_run(
    run_dir: Path,
    *,
    model_name: str,
    report: Callable[[str], None] | None = None,
) -> float | None:
    """Score justification groundedness for a finished run; write groundedness.jsonl."""
    emit = report or (lambda text: print(text, end="", flush=True))
    traces = [
        json.loads(line)
        for line in (run_dir / "traces.jsonl").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    judge = GroundednessJudge(model_name=model_name)

    rows = []
    for i, trace in enumerate(traces, start=1):
        row = judge.judge(trace)
        if row is None:
            continue
        rows.append(row)
        if row["supported"] is None:
            emit(f"[{i}/{len(traces)}] {row['case_id']}  unparseable\n")
        elif row["supported"]:
            emit(f"[{i}/{len(traces)}] {row['case_id']}  supported\n")
        else:
            emit(
                f"[{i}/{len(traces)}] {row['case_id']}  NOT supported  "
                f"{row['unsupported_claim'][:90]}\n"
            )

    if not rows:
        emit("nothing to judge in this run\n")
        return None

    with open(run_dir / "groundedness.jsonl", "w", encoding="utf-8") as fh:
        for row in rows:
            fh.write(json.dumps(row) + "\n")

    return read_groundedness(run_dir)


def read_groundedness(run_dir: Path) -> float | None:
    path = run_dir / "groundedness.jsonl"
    if not path.exists():
        return None
    verdicts = [
        json.loads(line)["supported"]
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    usable = [v for v in verdicts if v is not None]
    return sum(usable) / len(usable) if usable else None
