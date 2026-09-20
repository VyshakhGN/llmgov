from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..schemas import Action, TraceRecord

COST_RATIOS = (1, 2, 3, 5, 10, 20)


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    coverage: float
    selective_risk: float
    groundedness: float | None
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

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(num: float, den: float) -> float:
    return num / den if den else 0.0


def score(
    traces: Sequence[TraceRecord],
    groundedness: float | None = None,
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

    return Metrics(
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        coverage=_ratio(sent, n),
        selective_risk=_ratio(fn, sent),
        groundedness=groundedness,
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
    )


def format_report(m: Metrics, mode: str) -> str:
    grounded = (
        f"{m.groundedness:.1%}" if m.groundedness is not None else "not judged"
    )
    lines = [
        f"mode: {mode}    cases: {m.n}",
        "",
        f"  coverage             {m.coverage:.1%}     sent without a human ({m.tn + m.fn} of {m.n})",
        f"  selective risk       {m.selective_risk:.1%}     of those, wrong ({m.fn})",
        f"  groundedness         {grounded}     of its reasons, fully supported",
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
    if m.router_failures:
        lines.append(f"  router failures      {m.router_failures}")
    return "\n".join(lines)
