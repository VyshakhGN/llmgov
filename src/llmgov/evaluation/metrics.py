from __future__ import annotations
from collections.abc import Sequence
from dataclasses import asdict, dataclass
from typing import Any

from ..schemas import Action, TraceRecord


@dataclass
class Metrics:
    n: int
    tp: int
    fp: int
    tn: int
    fn: int
    precision: float
    recall: float
    f1: float
    accuracy: float
    review_rate: float
    unnecessary_review_rate: float
    unsafe_auto_send_rate: float
    router_failures: int
    mean_latency_ms: float

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def score(traces: Sequence[TraceRecord]) -> Metrics:
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
    precision = _ratio(tp, tp + fp)
    recall = _ratio(tp, tp + fn)

    return Metrics(
        n=n,
        tp=tp,
        fp=fp,
        tn=tn,
        fn=fn,
        precision=precision,
        recall=recall,
        f1=_ratio(2 * precision * recall, precision + recall) if (precision + recall) else 0.0,
        accuracy=_ratio(tp + tn, n),
        review_rate=_ratio(tp + fp, n),
        unnecessary_review_rate=_ratio(fp, tp + fp),
        unsafe_auto_send_rate=_ratio(fn, n),
        router_failures=sum(1 for t in scored if t.decision.parse_error),
        mean_latency_ms=sum(t.latency_ms or 0.0 for t in scored) / n,
    )


def format_report(m: Metrics, mode: str) -> str:
    lines = [
        f"mode: {mode}    cases: {m.n}",
        "",
        f"  precision            {m.precision:.3f}",
        f"  recall               {m.recall:.3f}",
        f"  f1 score             {m.f1:.3f}",
        f"  accuracy             {m.accuracy:.3f}",
        "",
        f"  false negatives      {m.fn}     (unsafe auto-sends)",
        f"  false positives      {m.fp}     (unnecessary reviews)",
        "",
        f"  review rate          {m.review_rate:.1%}",
    ]
    if m.router_failures:
        lines.append(f"  router failures      {m.router_failures}")
    return "\n".join(lines)
