from __future__ import annotations
import json
import time
import uuid
from collections.abc import Iterable
from pathlib import Path

from ..risk import detect_risk_signals
from ..routing.base import Router, fail_safe
from ..schemas import Case, TraceRecord


def new_run_id(mode: str, runs_dir: str | Path) -> str:
    """Sequential: 1-rag, 2-prompt_only, ... The exact time is on every trace."""
    runs_dir = Path(runs_dir)
    numbers = [
        int(p.name.split("-", 1)[0])
        for p in runs_dir.glob("*-*")
        if p.is_dir() and p.name.split("-", 1)[0].isdigit()
    ]
    return f"{max(numbers, default=0) + 1}-{mode}"


def run_cases(
    cases: Iterable[Case], router: Router, run_id: str, progress: bool = True
) -> list[TraceRecord]:
    cases = list(cases)
    total = len(cases)
    traces = []

    for i, case in enumerate(cases, start=1):
        signals = detect_risk_signals(case)

        if progress:
            gold = case.gold.action.value if case.gold else "?"
            print(f"[{i}/{total}] {case.case_id}  gold={gold:13} ", end="", flush=True)

        started = time.perf_counter()
        usage: dict[str, int] = {}
        try:
            decision = router.route(case, signals)
            usage = dict(getattr(router, "last_usage", {}))
        except Exception as exc:
            decision = fail_safe(f"{type(exc).__name__}: {exc}")
        latency_ms = (time.perf_counter() - started) * 1000

        if progress:
            hit = "  " if case.gold is None else ("ok" if decision.action is case.gold.action else "XX")
            note = "" if decision.risk_category.value == "NONE" else decision.risk_category.value
            print(f"-> {decision.action.value:13} {note:22} {latency_ms / 1000:5.1f}s  {hit}")

            # Temporary: show what the model actually received.
            masked_text = getattr(router, "last_masked_text", "")
            if masked_text:
                print(f"        {masked_text}")

        traces.append(
            TraceRecord(
                trace_id=str(uuid.uuid4()),
                case_id=case.case_id,
                run_id=run_id,
                mode=router.mode,
                guideline_corpus_version=router.guideline_corpus_version,
                # Defaulted, so a non-LLM router would still record cleanly.
                prompt_version=getattr(router, "prompt_version", "n/a"),
                model_name=getattr(router, "model_name", "n/a"),
                model_params=getattr(router, "model_params", {}),
                seed=getattr(router, "seed", None),
                risk_signals=signals,
                decision=decision,
                enforced_action=decision.action,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                gold=case.gold,
            )
        )
    return traces


def write_traces(traces: Iterable[TraceRecord], path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as fh:
        for t in traces:
            fh.write(t.model_dump_json() + "\n")
    return path


def read_traces(path: str | Path) -> list[TraceRecord]:
    with open(path, encoding="utf-8") as fh:
        return [TraceRecord(**json.loads(line)) for line in fh if line.strip()]
