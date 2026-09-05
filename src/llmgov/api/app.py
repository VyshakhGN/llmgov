from __future__ import annotations

import json
from datetime import datetime, timezone
import queue
import threading
from pathlib import Path
from typing import Any

import yaml
from fastapi import FastAPI, Form, Request
from fastapi.responses import HTMLResponse, RedirectResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates

from ..cli import (
    DEFAULT_CASES,
    DEFAULT_CUSTOMERS,
    DEFAULT_GUIDELINES,
    DEFAULT_MODEL,
    DEFAULT_ORDERS,
    DEFAULT_POLICY,
    RUNS_DIR,
    build_router,
)
from ..drafting import Drafter
from ..evaluation import new_run_id, run_cases, score, write_traces
from ..extraction import OrderExtractor
from ..loading import load_cases, load_customers, load_guidelines, load_orders
from ..policy import load_policy
from ..schemas import HumanReview
from ..routing.retrieval import DEFAULT_K

app = FastAPI(title="llmgov")
STATIC = Path(__file__).parent / "static"
templates = Jinja2Templates(directory=str(Path(__file__).parent / "templates"))
app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def _logo() -> str | None:
    for name in ("logo.svg", "logo.png", "logo.webp", "logo.jpg"):
        if (STATIC / name).exists():
            return f"/static/{name}"
    return None


@app.middleware("http")
async def add_logo(request: Request, call_next):
    request.state.logo = _logo()
    return await call_next(request)

_run_lock = threading.Lock()
_stop = threading.Event()


def _dataset() -> dict[str, Any]:
    policy = load_policy(DEFAULT_POLICY)
    orders = load_orders(DEFAULT_ORDERS, load_customers(DEFAULT_CUSTOMERS))
    return {
        "policy": policy,
        "orders": orders,
        "cases": load_cases(DEFAULT_CASES, orders, policy),
        "guidelines": load_guidelines(DEFAULT_GUIDELINES).guidelines,
        "rules": yaml.safe_load(Path(DEFAULT_POLICY).read_text(encoding="utf-8")),
    }


def _run_dirs() -> list[Path]:
    if not RUNS_DIR.exists():
        return []
    dirs = [d for d in RUNS_DIR.iterdir() if d.is_dir() and (d / "metrics.json").exists()]
    return sorted(dirs, key=lambda d: int(d.name.split("-", 1)[0]), reverse=True)


def _reviews(run: Path) -> dict[str, dict[str, Any]]:
    path = run / "reviews.jsonl"
    if not path.exists():
        return {}
    out: dict[str, dict[str, Any]] = {}
    for line in path.read_text(encoding="utf-8").splitlines():
        if line.strip():
            row = json.loads(line)
            out[row["case_id"]] = row
    return out


def _review_summary(run: Path, flagged: int) -> dict[str, int]:
    done = _reviews(run).values()
    return {
        "flagged": flagged,
        "reviewed": len(done),
        "unchanged": sum(1 for r in done if r["outcome"] == "APPROVED_UNCHANGED"),
        "edited": sum(1 for r in done if r["outcome"] == "APPROVED_EDITED"),
        "rejected": sum(1 for r in done if r["outcome"] == "REJECTED"),
    }


def _traces(run: Path) -> list[dict[str, Any]]:
    path = run / "traces.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


@app.get("/", response_class=HTMLResponse)
def runs_index(request: Request):
    rows = []
    for d in _run_dirs():
        m = json.loads((d / "metrics.json").read_text(encoding="utf-8"))
        rows.append({"id": d.name, "mode": d.name.split("-", 1)[1], "m": m})
    return templates.TemplateResponse(request, "runs.html", {"runs": rows})


@app.get("/runs/{run_id}", response_class=HTMLResponse)
def run_detail(request: Request, run_id: str):
    run = RUNS_DIR / run_id
    metrics = json.loads((run / "metrics.json").read_text(encoding="utf-8"))
    rows = []
    for t in _traces(run):
        gold = t.get("gold") or {}
        rows.append(
            {
                "case_id": t["case_id"],
                "gold": gold.get("action"),
                "gold_cat": gold.get("risk_category"),
                "action": t["decision"]["action"],
                "category": t["decision"]["risk_category"],
                "hit": gold.get("action") == t["decision"]["action"] if gold else None,
                "rule": t["stages"]["policy_rule"],
                "latency": round((t.get("latency_ms") or 0) / 1000, 1),
            }
        )
    flagged = sum(1 for r in rows if r["action"] == "NEEDS_REVIEW")
    return templates.TemplateResponse(
        request,
        "run.html",
        {
            "run_id": run_id,
            "metrics": metrics,
            "rows": rows,
            "summary": _review_summary(run, flagged),
        },
    )


@app.get("/runs/{run_id}/review", response_class=HTMLResponse)
def review_queue(request: Request, run_id: str):
    run = RUNS_DIR / run_id
    cases = {c.case_id: c for c in _dataset()["cases"]}
    reviews = _reviews(run)
    flagged = [
        {"t": t, "case": cases.get(t["case_id"]), "review": reviews.get(t["case_id"])}
        for t in _traces(run)
        if t["decision"]["action"] == "NEEDS_REVIEW"
    ]
    return templates.TemplateResponse(
        request,
        "review.html",
        {
            "run_id": run_id,
            "flagged": flagged,
            "summary": _review_summary(run, len(flagged)),
        },
    )


@app.post("/runs/{run_id}/review/{case_id}")
def review_submit(
    run_id: str,
    case_id: str,
    outcome: str = Form(...),
    reviewer_id: str = Form("reviewer"),
    edited_response: str = Form(""),
    notes: str = Form(""),
):
    run = RUNS_DIR / run_id
    review = HumanReview(
        reviewer_id=reviewer_id.strip() or "reviewer",
        outcome=outcome,
        edited_response=(
            edited_response.strip() or None
            if outcome == "APPROVED_EDITED"
            else None
        ),
        notes=notes.strip(),
        reviewed_at=datetime.now(timezone.utc),
    )
    row = {"case_id": case_id, **json.loads(review.model_dump_json())}
    with open(run / "reviews.jsonl", "a", encoding="utf-8") as fh:
        fh.write(json.dumps(row) + "\n")
    return RedirectResponse(f"/runs/{run_id}/review#{case_id}", status_code=303)


@app.get("/runs/{run_id}/cases/{case_id}", response_class=HTMLResponse)
def case_detail(request: Request, run_id: str, case_id: str):
    run = RUNS_DIR / run_id
    trace = next((t for t in _traces(run) if t["case_id"] == case_id), None)
    case = next((c for c in _dataset()["cases"] if c.case_id == case_id), None)
    return templates.TemplateResponse(
        request,
        "case.html",
        {"run_id": run_id, "case_id": case_id, "t": trace, "case": case},
    )


@app.get("/rules", response_class=HTMLResponse)
def rules_page(request: Request):
    return templates.TemplateResponse(request, "rules.html", {"rules": _dataset()["rules"]})


@app.get("/guidelines", response_class=HTMLResponse)
def guidelines_page(request: Request):
    gs = _dataset()["guidelines"]
    sourced = sum(1 for g in gs if "Company policy" not in g.source)
    return templates.TemplateResponse(
        request, "guidelines.html", {"guidelines": gs, "sourced": sourced}
    )


@app.get("/cases", response_class=HTMLResponse)
def cases_page(request: Request):
    cases = _dataset()["cases"]
    auto = sum(1 for c in cases if c.gold and c.gold.action.value == "AUTO_SEND")
    return templates.TemplateResponse(
        request,
        "cases.html",
        {"cases": cases, "auto": auto, "review": len(cases) - auto},
    )


@app.get("/orders", response_class=HTMLResponse)
def orders_page(request: Request):
    d = _dataset()
    return templates.TemplateResponse(
        request, "orders.html", {"orders": d["orders"], "policy": d["policy"]}
    )


@app.get("/customers", response_class=HTMLResponse)
def customers_page(request: Request):
    orders = _dataset()["orders"]
    owned: dict[str, list[str]] = {}
    for oid, f in orders.items():
        if f.customer_id:
            owned.setdefault(f.customer_id, []).append(oid)
    rows = []
    for cid, raw in sorted(load_customers(DEFAULT_CUSTOMERS).items()):
        total = raw.get("total_orders") or 0
        refunds = raw.get("refunds_last_12m") or 0
        rows.append(
            {
                "id": cid,
                **raw,
                "rate": round(refunds / total, 2) if total else None,
                "owns": owned.get(cid, []),
            }
        )
    return templates.TemplateResponse(request, "customers.html", {"customers": rows})


@app.get("/run", response_class=HTMLResponse)
def run_form(request: Request):
    return templates.TemplateResponse(
        request, "new.html", {"cases": _dataset()["cases"], "default_k": DEFAULT_K}
    )


@app.post("/run")
def run_start(
    mode: str = Form("rag"),
    top_k: int = Form(DEFAULT_K),
    case_ids: list[str] = Form(default=[]),
):
    if not _run_lock.acquire(blocking=False):
        return StreamingResponse(
            iter(["a run is already in progress"]), media_type="text/plain"
        )

    _stop.clear()
    channel: queue.Queue[str | None] = queue.Queue()

    def work():
        try:
            d = _dataset()
            selected = [c for c in d["cases"] if not case_ids or c.case_id in case_ids]
            channel.put(f"indexing guidelines for {mode}...")
            router = build_router(mode, DEFAULT_GUIDELINES, DEFAULT_MODEL, False, top_k)
            run_id = new_run_id(mode, RUNS_DIR)
            channel.put(f"run {run_id} - {len(selected)} of {len(d['cases'])} cases")

            traces = run_cases(
                selected,
                router,
                run_id,
                policy=d["policy"],
                drafter=Drafter(model_name=DEFAULT_MODEL, policy=d["policy"]),
                extractor=OrderExtractor(model_name=DEFAULT_MODEL),
                orders=d["orders"],
                report=channel.put,
                should_stop=_stop.is_set,
            )

            metrics = score(traces)
            out = RUNS_DIR / run_id
            write_traces(traces, out / "traces.jsonl")
            (out / "metrics.json").write_text(
                json.dumps(metrics.to_dict(), indent=2), encoding="utf-8"
            )
            channel.put(f"DONE /runs/{run_id}")
        except Exception as exc:
            channel.put(f"ERROR {type(exc).__name__}: {exc}")
        finally:
            channel.put(None)
            _run_lock.release()

    threading.Thread(target=work, daemon=True).start()

    def stream():
        while True:
            item = channel.get()
            if item is None:
                return
            yield item

    return StreamingResponse(stream(), media_type="text/plain")


@app.post("/run/stop")
def run_stop():
    _stop.set()
    return {"stopping": True}
