from __future__ import annotations
import json
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path

from ..drafting import Drafter
from ..extraction import OrderExtractor
from ..policy import PolicyEngine
from ..risk import detect_risk_signals
from ..routing.base import Router, fail_safe
from ..schemas import Case, PipelineStages, SystemFacts, TraceRecord


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
    cases: Iterable[Case],
    router: Router,
    run_id: str,
    progress: bool = True,
    policy: PolicyEngine | None = None,
    drafter: Drafter | None = None,
    extractor: OrderExtractor | None = None,
    orders: dict[str, SystemFacts] | None = None,
    report: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
) -> list[TraceRecord]:
    cases = list(cases)
    total = len(cases)
    traces = []
    emit = report or (lambda text: print(text, end="", flush=True))

    for i, case in enumerate(cases, start=1):
        if should_stop is not None and should_stop():
            emit(f"stopped after {i - 1} of {total} cases\n")
            break

        if progress:
            gold = case.gold.action.value if case.gold else "?"
            emit(f"[{i}/{total}] {case.case_id}  gold={gold:13} ")

        extracted_order_id = None
        if extractor is not None:
            if progress:
                emit("id... ")
            extracted_order_id = extractor.extract(case.user_message)
            if progress:
                mark = "" if extracted_order_id == case.order_id else "!"
                emit(f"{extracted_order_id or '-'}{mark:1}  ")
            facts = (orders or {}).get(extracted_order_id or "", SystemFacts())
            case = case.model_copy(
                update={
                    "system_facts": facts,
                    "policy_decision": policy.decide(facts).decision,
                }
            )

        generated_draft = None
        if drafter is not None:
            if progress:
                emit("drafting... ")
            generated_draft = drafter.write(case)
            case = case.model_copy(update={"draft_response": generated_draft})

        signals = detect_risk_signals(case)

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
            emit(
                f"router: {decision.action.value:13} {note:22} "
                f"{latency_ms / 1000:5.1f}s  {hit}\n"
            )

            if generated_draft:
                emit(f"        draft: {' '.join(generated_draft.split())}\n")
            masked_text = getattr(router, "last_masked_text", "")
            if masked_text:
                emit(f"        masked: {masked_text}\n")

        outcome = policy.decide(case.system_facts) if policy else None
        stages = PipelineStages(
            policy_rule=outcome.rule if outcome else "",
            policy_reason=outcome.because.strip() if outcome else "",
            refund_amount_eur=(
                policy.refund_amount(case.system_facts, outcome.decision)
                if policy and outcome
                else None
            ),
            masked=list(getattr(router, "last_masked", [])),
            masked_message=getattr(router, "last_masked_text", ""),
            router_prompt=getattr(router, "last_prompt", ""),
            generated_draft=generated_draft,
            draft_prompt_version=drafter.prompt_version if drafter else None,
            extracted_order_id=extracted_order_id,
            expected_order_id=case.order_id,
            extract_prompt_version=extractor.prompt_version if extractor else None,
        )

        traces.append(
            TraceRecord(
                trace_id=str(uuid.uuid4()),
                case_id=case.case_id,
                run_id=run_id,
                mode=router.mode,
                guideline_corpus_version=router.guideline_corpus_version,
                prompt_version=getattr(router, "prompt_version", "n/a"),
                model_name=getattr(router, "model_name", "n/a"),
                model_params=getattr(router, "model_params", {}),
                seed=getattr(router, "seed", None),
                stages=stages,
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
