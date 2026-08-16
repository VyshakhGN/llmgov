from __future__ import annotations
from collections import defaultdict
from collections.abc import Sequence
from dataclasses import asdict, dataclass, field
from typing import Any

from ..schemas import Action, TraceRecord


@dataclass
class CategoryResult:
    total: int = 0
    caught: int = 0
    correct_category: int = 0

    @property
    def recall(self) -> float:
        return self.caught / self.total if self.total else 0.0


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
    by_category: dict[str, dict[str, Any]] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _ratio(num: int, den: int) -> float:
    return num / den if den else 0.0


def score(traces: Sequence[TraceRecord]) -> Metrics:
    scored = [t for t in traces if t.gold is not None]
    if not scored:
        raise ValueError("no traces carry a gold label")

    tp = fp = tn = fn = 0
    cats: dict[str, CategoryResult] = defaultdict(CategoryResult)

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
            
        if gold_review:
            c = cats[t.gold.risk_category.value]
            c.total += 1
            if pred_review:
                c.caught += 1
                if t.decision.risk_category is t.gold.risk_category:
                    c.correct_category += 1

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
        by_category={
            k: {
                "total": v.total,
                "caught": v.caught,
                "recall": round(v.recall, 3),
                "correct_category": v.correct_category,
            }
            for k, v in sorted(cats.items())
        },
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
    if m.by_category:
        lines += ["", "  recall by risk category:"]
        for cat, v in m.by_category.items():
            lines.append(
                f"    {cat:22} {v['caught']}/{v['total']}"
                f"   category correct: {v['correct_category']}"
            )
    return "\n".join(lines)
