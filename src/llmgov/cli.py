from __future__ import annotations
import argparse
import json
from pathlib import Path
from .drafting import Drafter
from .extraction import OrderExtractor
from .evaluation import format_report, new_run_id, run_cases, score, write_traces
from .loading import load_cases, load_customers, load_guidelines, load_orders
from .policy import load_policy
from .routing import LlmRouter, check_available
from .routing.base import Router
from .routing.retrieval import DEFAULT_EMBED_MODEL, DEFAULT_K, GuidelineIndex
from .schemas import Mode

DEFAULT_GUIDELINES = Path("data/guidelines/v1/guidelines.yaml")
DEFAULT_CASES = Path("data/cases/v0.yaml")
DEFAULT_ORDERS = Path("data/orders/v0.yaml")
DEFAULT_CUSTOMERS = Path("data/customers/v0.yaml")
DEFAULT_POLICY = Path("data/policy/v1/rules.yaml")
DEFAULT_MODEL = "qwen3.5:4b"
RUNS_DIR = Path("runs")


def build_router(
    mode: str, guidelines_path: Path, model: str, think: bool, top_k: int = DEFAULT_K
) -> Router:
    corpus = load_guidelines(guidelines_path)

    if mode == "prompt_only":
        check_available(model)
        return LlmRouter(
            mode=Mode.PROMPT_ONLY,
            model_name=model,
            guideline_corpus_version="none",
            think=think,
        )

    if mode == "full_context":
        check_available(model)
        return LlmRouter(
            mode=Mode.FULL_CONTEXT,
            model_name=model,
            guideline_corpus_version=guidelines_path.parent.name,
            guidelines=corpus.guidelines,
            think=think,
        )

    if mode == "rag":
        check_available(model)
        check_available(DEFAULT_EMBED_MODEL)
        print(f"indexing {len(corpus.guidelines)} guidelines...")
        return LlmRouter(
            mode=Mode.RAG,
            model_name=model,
            guideline_corpus_version=guidelines_path.parent.name,
            index=GuidelineIndex(corpus.guidelines),
            top_k=top_k,
            think=think,
        )

    raise SystemExit(f"mode not implemented yet: {mode}")


def main() -> None:
    parser = argparse.ArgumentParser(prog="llmgov")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="route a case set and score the result")
    run.add_argument("--mode", default="rag")
    run.add_argument("--model", default=DEFAULT_MODEL, help="ollama model, LLM modes only")
    run.add_argument(
        "--think",
        action="store_true",
        help="let a reasoning model think before answering (slower)",
    )
    run.add_argument(
        "--top-k", type=int, default=DEFAULT_K, help="guidelines retrieved, rag mode"
    )
    run.add_argument("--cases", type=Path, default=DEFAULT_CASES)
    run.add_argument("--orders", type=Path, default=DEFAULT_ORDERS)
    run.add_argument("--customers", type=Path, default=DEFAULT_CUSTOMERS)
    run.add_argument(
        "--no-extract",
        action="store_true",
        help="take the order id from the case file instead of reading it from "
        "the customer's message",
    )
    run.add_argument("--policy", type=Path, default=DEFAULT_POLICY)
    run.add_argument("--guidelines", type=Path, default=DEFAULT_GUIDELINES)
    run.add_argument("--runs-dir", type=Path, default=RUNS_DIR)

    args = parser.parse_args()
    if args.command != "run":
        parser.error("unknown command")

    orders = load_orders(args.orders, load_customers(args.customers))
    policy = load_policy(args.policy)
    cases = load_cases(args.cases, orders, policy)
    router = build_router(args.mode, args.guidelines, args.model, args.think, args.top_k)
    run_id = new_run_id(args.mode, args.runs_dir)

    drafter = Drafter(model_name=args.model, policy=policy)
    extractor = None if args.no_extract else OrderExtractor(model_name=args.model)
    traces = run_cases(
        cases,
        router,
        run_id,
        policy=policy,
        drafter=drafter,
        extractor=extractor,
        orders=orders,
    )
    metrics = score(traces)

    out = args.runs_dir / run_id
    write_traces(traces, out / "traces.jsonl")
    (out / "metrics.json").write_text(
        json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
    )

    print(format_report(metrics, args.mode))
    print(f"\nwritten to {out}")


if __name__ == "__main__":
    main()
