from __future__ import annotations
import math
from collections.abc import Mapping, Sequence
from dataclasses import asdict, dataclass, field
from statistics import mean
from typing import Any

from ..schemas import Action, RetrievedGuideline, StageCost, TraceRecord

COST_RATIOS = (1, 2, 3, 5, 10, 20)
STAGES = ("extract", "draft", "retrieve", "route")
SEARCH_GROUPS = {"all": "all cases", "needs_review": "needing a human", "auto_send": "fine to send"}


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    coverage: float
    selective_risk: float
    aurc: float | None
    precision: float
    recall: float
    f1: float
    accuracy: float
    review_rate: float
    always_send_correct: int
    always_review_correct: int
    cost_by_ratio: dict[str, int] = field(default_factory=dict)
    router_failures: int = 0
    mean_latency_ms: float = 0.0
    stage_costs: dict[str, dict[str, Any]] = field(default_factory=dict)
    costs_per_stage: bool = False
    retrieval: dict[str, dict[str, Any]] = field(default_factory=dict)
    retrieval_k: int | None = None

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def _aurc(ranked_review: Sequence[bool]) -> float:
    missed = 0
    total = 0.0
    for sent, needs_review in enumerate(ranked_review, start=1):
        missed += needs_review
        total += missed / sent
    return total / len(ranked_review)


def _risk_coverage(scored: Sequence[TraceRecord]) -> float | None:
    probs = [t.decision.auto_send_probability for t in scored]
    if any(p is None for p in probs):
        return None
    gold_review = [t.gold.action is Action.NEEDS_REVIEW for t in scored]
    return _aurc([g for _, g in sorted(zip(probs, gold_review), key=lambda x: -x[0])])


def _p95(values: list[float]) -> float:
    ordered = sorted(values)
    return ordered[max(0, math.ceil(0.95 * len(ordered)) - 1)]


def _summarise(costs: Sequence[StageCost]) -> dict[str, Any]:
    ms = [c.ms for c in costs if c.ms is not None]
    prompt = [c.prompt_tokens for c in costs if c.prompt_tokens is not None]
    completion = [c.completion_tokens for c in costs if c.completion_tokens is not None]
    return {
        "n": len(costs),
        "mean_ms": round(mean(ms), 1) if ms else None,
        "p95_ms": round(_p95(ms), 1) if ms else None,
        "mean_prompt_tokens": round(mean(prompt)) if prompt else None,
        "mean_completion_tokens": round(mean(completion)) if completion else None,
    }


def _stage_costs(scored: Sequence[TraceRecord]) -> tuple[dict[str, dict[str, Any]], bool]:
    measured = [t for t in scored if t.stage_costs]
    if not measured:
        router_only = [
            StageCost(ms=t.latency_ms, prompt_tokens=t.prompt_tokens, completion_tokens=t.completion_tokens)
            for t in scored
        ]
        return {"route": _summarise(router_only)}, False

    out = {}
    for stage in STAGES:
        costs = [t.stage_costs[stage] for t in measured if stage in t.stage_costs]
        if costs:
            out[stage] = _summarise(costs)
    out["total"] = _summarise([
        StageCost(
            ms=sum(c.ms or 0.0 for c in t.stage_costs.values()),
            prompt_tokens=sum(c.prompt_tokens or 0 for c in t.stage_costs.values()),
            completion_tokens=sum(c.completion_tokens or 0 for c in t.stage_costs.values()),
        )
        for t in measured
    ])
    return out, True


def gold_rank(retrieved: Sequence[RetrievedGuideline], wanted: Sequence[str]) -> int | None:
    ranks = [g.rank for g in retrieved if g.guideline_id in wanted and g.rank is not None]
    return min(ranks) if ranks else None


def _retrieval(
    scored: Sequence[TraceRecord], relevant: Mapping[str, Sequence[str]] | None
) -> tuple[dict[str, dict[str, Any]], int | None]:
    groups: dict[str, list[int | None]] = {name: [] for name in SEARCH_GROUPS}
    k = None
    for t in scored:
        found = t.decision.retrieved_guidelines
        wanted = relevant.get(t.case_id, []) if relevant is not None else t.gold.guidelines
        if not found or not wanted:
            continue
        k = k or t.retriever_config.get("top_k") or len(found)
        rank = gold_rank(found, wanted)
        groups["all"].append(rank)
        groups["needs_review" if t.gold.action is Action.NEEDS_REVIEW else "auto_send"].append(rank)

    out = {}
    for name, ranks in groups.items():
        if ranks:
            out[name] = {
                "n": len(ranks),
                "hits": sum(r is not None for r in ranks),
                "mrr": round(mean(1 / r if r else 0.0 for r in ranks), 3),
            }
    return out, k


def score(
    traces: Sequence[TraceRecord], relevant: Mapping[str, Sequence[str]] | None = None
) -> Metrics:
    scored = [t for t in traces if t.gold is not None]
    if not scored:
        raise ValueError("no traces carry a gold label")

    tp = fp = tn = fn = 0

    for t in scored:
        gold_review = t.gold.action is Action.NEEDS_REVIEW
        pred_review = t.enforced_action is Action.NEEDS_REVIEW

        if gold_review and pred_review:
            tp += 1
        elif gold_review and not pred_review:
            fn += 1
        elif not gold_review and pred_review:
            fp += 1
        else:
            tn += 1

    n = len(scored)
    sent = tn + fn
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)
    aurc = _risk_coverage(scored)
    stage_costs, per_stage = _stage_costs(scored)
    retrieval, retrieval_k = _retrieval(scored, relevant)

    return Metrics(
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        coverage=_ratio(sent, n),
        selective_risk=_ratio(fn, sent),
        aurc=aurc,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0,
        accuracy=_ratio(tp + tn, n),
        review_rate=_ratio(tp + fp, n),
        always_send_correct=tn + fp,
        always_review_correct=tp + fn,
        cost_by_ratio={str(w): w * fn + fp for w in COST_RATIOS},
        router_failures=sum(1 for t in scored if t.decision.parse_error),
        mean_latency_ms=sum(t.latency_ms or 0.0 for t in scored) / n,
        stage_costs=stage_costs,
        costs_per_stage=per_stage,
        retrieval=retrieval,
        retrieval_k=retrieval_k,
    )


def _pct(value: float | None) -> str:
    return f"{value:.1%}" if value is not None else "n/a"


def _secs(ms: float | None) -> str:
    return f"{ms / 1000:.1f}s" if ms is not None else "-"


def _count(value: float | None) -> str:
    return f"{value:.0f}" if value is not None else "-"


def _retrieval_lines(m: Metrics) -> list[str]:
    if not m.retrieval:
        return []
    lines = ["", f"  deciding guideline in the top {m.retrieval_k}    found    MRR"]
    for group, row in m.retrieval.items():
        found = f"{row['hits']}/{row['n']}"
        lines.append(f"    {SEARCH_GROUPS[group]:30} {found:>6}   {row['mrr']:.2f}")
    return lines


def _cost_lines(m: Metrics) -> list[str]:
    if not m.stage_costs:
        return []
    lines = ["", "  per case                  mean      p95   tokens in    out"]
    for stage, row in m.stage_costs.items():
        if stage == "total":
            label = "whole pipeline"
        elif stage == "route" and not m.costs_per_stage:
            label = "route, incl. search"
        else:
            label = stage
        lines.append(
            f"    {label:20} {_secs(row['mean_ms']):>7}  {_secs(row['p95_ms']):>7}"
            f"   {_count(row['mean_prompt_tokens']):>9} {_count(row['mean_completion_tokens']):>6}"
        )
    absent = [s for s in STAGES if s not in m.stage_costs]
    if m.costs_per_stage and absent:
        lines.append(f"    not part of this run: {', '.join(absent)}")
    if not m.costs_per_stage:
        lines.append("    older run: only the router call was timed")
    return lines


def format_report(m: Metrics, mode: str) -> str:
    lines = [
        f"mode: {mode}    cases: {m.n}",
        "",
        f"  coverage             {m.coverage:.1%}     sent without a human ({m.tn + m.fn} of {m.n})",
        f"  selective risk       {m.selective_risk:.1%}     of those, wrong ({m.fn})",
        f"  AURC                 {_pct(m.aurc)}     selective risk averaged over every cut-off",
        "",
        f"  unsafe auto-sends    {m.fn}     of {m.tp + m.fn} that needed a human",
        f"  unnecessary reviews  {m.fp}     of {m.tn + m.fp} that were fine",
        "",
        "  cost at n:1 weighting of a miss against a false alarm",
        "    " + "  ".join(f"{w}:1 {c}" for w, c in m.cost_by_ratio.items()),
        "",
        f"  baselines            always-send {m.always_send_correct}/{m.n}"
        f"    always-review {m.always_review_correct}/{m.n}",
        "",
        f"  [precision {m.precision:.3f}  recall {m.recall:.3f}"
        f"  f1 {m.f1:.3f}  accuracy {m.accuracy:.3f}]",
    ]
    lines += _retrieval_lines(m)
    lines += _cost_lines(m)
    if m.router_failures:
        lines.append(f"  router failures      {m.router_failures}")
    return "\n".join(lines)
