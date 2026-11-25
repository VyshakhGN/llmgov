from __future__ import annotations
import json
import time
import uuid
from collections.abc import Callable, Iterable
from pathlib import Path
from typing import Any

from ..drafting import Drafter
from ..extraction import OrderExtractor
from ..policy import PolicyEngine
from ..risk import detect_risk_signals
from ..routing.base import Router, fail_safe, stage_failure
from ..environment import describe
from ..schemas import Case, PipelineStages, StageCost, SystemFacts, TraceRecord


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
    frozen_drafts: dict[str, str] | None = None,
    extractor: OrderExtractor | None = None,
    orders: dict[str, SystemFacts] | None = None,
    report: Callable[[str], None] | None = None,
    should_stop: Callable[[], bool] | None = None,
    frozen_draft_metadata: dict[str, Any] | None = None,
) -> list[TraceRecord]:
    cases = list(cases)
    total = len(cases)
    traces = []
    env = describe(
        *{
            m
            for m in (
                getattr(router, "model_name", None),
                getattr(drafter, "model_name", None),
                getattr(extractor, "model_name", None),
                getattr(getattr(router, "index", None), "model_name", None),
            )
            if m
        }
    )
    emit = report or (lambda text: print(text, end="", flush=True))
    architecture = getattr(router, "architecture", "baseline")

    for i, case in enumerate(cases, start=1):
        if should_stop is not None and should_stop():
            emit(f"stopped after {i - 1} of {total} cases\n")
            break

        if progress:
            gold = case.gold.action.value if case.gold else "?"
            emit(f"[{i}/{total}] {case.case_id}  gold={gold:13} ")

        extracted_order_id = None
        preparation_failure = None
        if extractor is not None:
            if progress:
                emit("id... ")
            started = time.perf_counter()
            try:
                extracted_order_id = extractor.extract(case.user_message)
                if architecture == "option-b" and extractor.last_parse_error:
                    raise ValueError("Extractor returned an invalid order ID response")
            except Exception as exc:
                if architecture != "option-b":
                    raise
                preparation_failure = stage_failure("extract", exc)
                if extractor.last_cost.ms is None:
                    extractor.last_cost.ms = (time.perf_counter() - started) * 1000
            else:
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
        drafted = False
        pinned = (
            preparation_failure is None
            and frozen_drafts is not None
            and case.case_id in frozen_drafts
        )
        if pinned:
            generated_draft = frozen_drafts[case.case_id]
            case = case.model_copy(update={"draft_response": generated_draft})
            if progress:
                emit("pinned draft... ")
        elif drafter is not None and preparation_failure is None:
            if progress:
                emit("drafting... ")
            drafted = True
            started = time.perf_counter()
            try:
                generated_draft = drafter.write(case)
                if architecture == "option-b" and not generated_draft.strip():
                    raise ValueError("Drafter returned an empty reply")
            except Exception as exc:
                if architecture != "option-b":
                    raise
                preparation_failure = stage_failure("draft", exc)
                if drafter.last_cost.ms is None:
                    drafter.last_cost.ms = (time.perf_counter() - started) * 1000
            else:
                case = case.model_copy(update={"draft_response": generated_draft})

        signals = detect_risk_signals(case)

        usage: dict[str, int] = {}
        latency_ms = None
        routed = preparation_failure is None
        if preparation_failure is not None:
            decision = preparation_failure
        else:
            started = time.perf_counter()
            try:
                decision = router.route(case, signals)
                usage = dict(getattr(router, "last_usage", {}))
            except Exception as exc:
                decision = (
                    stage_failure("route", exc)
                    if architecture == "option-b"
                    else fail_safe(f"{type(exc).__name__}: {exc}")
                )
            latency_ms = (time.perf_counter() - started) * 1000
        masked_text = getattr(router, "last_masked_text", "") if routed else ""

        if progress:
            hit = "  " if case.gold is None else ("ok" if decision.action is case.gold.action else "XX")
            note = "" if decision.risk_category.value == "NONE" else decision.risk_category.value
            emit(
                f"router: {decision.action.value:13} {note:22} "
                f"{(latency_ms or 0) / 1000:5.1f}s  {hit}\n"
            )

            if generated_draft:
                emit(f"        draft: {' '.join(generated_draft.split())}\n")
            if masked_text:
                emit(f"        masked: {masked_text}\n")

        facts_used = case.system_facts if routed or drafted else None
        outcome = policy.decide(facts_used) if policy and facts_used is not None else None
        agent_traces = getattr(router, "last_agent_traces", {}) if routed else {}
        stages = PipelineStages(
            policy_rule=outcome.rule if outcome else "",
            policy_reason=outcome.because.strip() if outcome else "",
            refund_amount_eur=(
                policy.refund_amount(facts_used, outcome.decision)
                if policy and outcome and facts_used is not None
                else None
            ),
            masked=list(getattr(router, "last_masked", [])) if routed else [],
            masked_message=masked_text,
            router_prompt=getattr(router, "last_prompt", "") if routed else "",
            generated_draft=generated_draft,
            draft_prompt_version=(
                "pinned" if pinned else (drafter.prompt_version if drafted else None)
            ),
            extracted_order_id=extracted_order_id,
            expected_order_id=case.order_id,
            extract_prompt_version=extractor.prompt_version if extractor else None,
            system_facts=facts_used,
            frozen_draft_metadata=frozen_draft_metadata if pinned else None,
            agents=agent_traces,
        )

        stage_costs: dict[str, StageCost] = {
            name: trace.cost for name, trace in agent_traces.items()
        }
        if extractor is not None:
            stage_costs["extract"] = extractor.last_cost
        if drafted:
            stage_costs["draft"] = drafter.last_cost
        retrieval_ms = getattr(router, "last_retrieval_ms", None) if routed else None
        if retrieval_ms is not None:
            stage_costs["retrieve"] = StageCost(ms=retrieval_ms)
        route_ms = getattr(router, "last_route_ms", None) if routed else None
        if routed and (route_ms is not None or not agent_traces):
            stage_costs["route"] = StageCost(
                ms=route_ms if route_ms is not None else latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
            )

        traces.append(
            TraceRecord(
                trace_id=str(uuid.uuid4()),
                case_id=case.case_id,
                run_id=run_id,
                mode=router.mode,
                architecture=architecture,
                guideline_corpus_version=router.guideline_corpus_version,
                prompt_version=getattr(router, "prompt_version", "n/a"),
                model_name=getattr(router, "model_name", "n/a"),
                model_params=getattr(router, "model_params", {}),
                retriever_config=getattr(router, "retriever_config", {}),
                environment=env,
                seed=getattr(router, "seed", None),
                stages=stages,
                risk_signals=signals,
                decision=decision,
                enforced_action=decision.action,
                latency_ms=latency_ms,
                prompt_tokens=usage.get("prompt_tokens"),
                completion_tokens=usage.get("completion_tokens"),
                stage_costs=stage_costs,
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
