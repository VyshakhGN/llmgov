from __future__ import annotations
import json
import math
import re
import time
from typing import Any
import ollama

from ..risk.masking import mask_pii as mask_text
from ..schemas import (
    Action,
    Case,
    Guideline,
    Mode,
    RiskCategory,
    RiskSignals,
    RoutingDecision,
)
from .prompts import PROMPT_VERSION, SYSTEM_PROMPT, build_user_prompt
from .retrieval import DEFAULT_K, GuidelineIndex, case_probes

DEFAULT_HOST = "http://localhost:11434"
TOP_LOGPROBS = 10
ACTION_VALUE = re.compile(r'"action"\s*:\s*"')

ROUTER_SCHEMA: dict[str, Any] = {
    "type": "object",
    "properties": {
        "action": {"type": "string", "enum": [a.value for a in Action]},
        "risk_category": {"type": "string", "enum": [r.value for r in RiskCategory]},
        "confidence": {"type": "number", "minimum": 0, "maximum": 1},
        "justification": {"type": "string"},
    },
    "required": ["action", "risk_category", "confidence", "justification"],
}


class LlmRouter:
    def __init__(
        self,
        *,
        mode: Mode,
        model_name: str,
        guideline_corpus_version: str,
        guidelines: list[Guideline] | None = None,
        index: GuidelineIndex | None = None,
        top_k: int = DEFAULT_K,
        mask_pii: bool = True,
        seed: int = 0,
        think: bool | None = None,
        host: str = DEFAULT_HOST,
    ) -> None:
        self.mode = mode
        self.model_name = model_name
        self.guideline_corpus_version = guideline_corpus_version
        self.guidelines = guidelines
        self.index = index
        self.top_k = top_k
        self.retriever_config = (
            {"top_k": top_k, **index.config} if index is not None else {}
        )
        self.mask_pii = mask_pii
        self.seed = seed
        self.think = think
        self.prompt_version = PROMPT_VERSION
        self.client = ollama.Client(host=host)
        self.model_params = {"temperature": 0.0, "seed": seed, "num_ctx": 8192}
        if think is not None:
            self.model_params["think"] = think
        # Read by the runner after route() to fill in the trace.
        self.last_usage: dict[str, int] = {}
        self.last_masked: list[str] = []
        self.last_masked_text: str = ""
        self.last_prompt: str = ""
        self.last_retrieval_ms: float | None = None
        self.last_route_ms: float | None = None

    def route(self, case: Case, risk_signals: RiskSignals) -> RoutingDecision:
        case = self._mask(case)
        self.last_retrieval_ms = None
        self.last_route_ms = None
        retrieved = []
        guidelines = self.guidelines
        if self.index is not None:
            started = time.perf_counter()
            retrieved = self.index.search_many(case_probes(case), k=self.top_k)
            self.last_retrieval_ms = (time.perf_counter() - started) * 1000
            shown = {r.guideline_id for r in retrieved}
            guidelines = [g for g in self.index.guidelines if g.guideline_id in shown]

        prompt = build_user_prompt(case, guidelines=guidelines)
        self.last_prompt = prompt

        extra = {} if self.think is None else {"think": self.think}
        started = time.perf_counter()
        response = self.client.chat(
            model=self.model_name,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
            format=ROUTER_SCHEMA,
            options={
                "temperature": 0.0,
                "seed": self.seed,
                "num_ctx": 8192,
                "num_predict": 400,
            },
            logprobs=True,
            top_logprobs=TOP_LOGPROBS,
            **extra,
        )
        self.last_route_ms = (time.perf_counter() - started) * 1000

        self.last_usage = {
            "prompt_tokens": response.get("prompt_eval_count") or 0,
            "completion_tokens": response.get("eval_count") or 0,
        }

        decision = _parse(response["message"]["content"])
        decision.retrieved_guidelines = retrieved
        decision.auto_send_probability = (
            0.0
            if decision.parse_error
            else _auto_send_probability(response["message"]["content"], response.logprobs)
        )
        return decision

    def _mask(self, case: Case) -> Case:
        self.last_masked, self.last_masked_text = [], ""
        if not self.mask_pii:
            return case

        message, applied_message = mask_text(case.user_message)
        draft, applied_draft = mask_text(case.draft_response)
        if not (applied_message or applied_draft):
            return case

        self.last_masked = sorted(set(applied_message + applied_draft))
        self.last_masked_text = " ".join(message.split())
        return case.model_copy(
            update={"user_message": message, "draft_response": draft}
        )


def _parse(raw: str) -> RoutingDecision:
    try:
        data = json.loads(raw)
        action = Action(data["action"])
        category = RiskCategory(data["risk_category"])
    except Exception as exc:
        return RoutingDecision(
            action=Action.NEEDS_REVIEW,
            risk_category=RiskCategory.NONE,
            justification="Unparseable model output; failed closed to review.",
            raw_model_output=raw,
            parse_error=f"{type(exc).__name__}: {exc}",
        )


    if action is Action.AUTO_SEND:
        category = RiskCategory.NONE

    confidence = data.get("confidence")
    if not isinstance(confidence, (int, float)) or not 0.0 <= confidence <= 1.0:
        confidence = None

    return RoutingDecision(
        action=action,
        risk_category=category,
        confidence=confidence,
        justification=str(data.get("justification", "")).strip(),
        raw_model_output=raw,
    )


def _auto_send_probability(content: str, logprobs: list[Any] | None) -> float | None:
    match = ACTION_VALUE.search(content)
    if match is None or not logprobs:
        return None

    start = match.end()
    offset = 0
    for token in logprobs:
        end = offset + len(token.token)
        if end > start:
            lead = token.token[: start - offset]
            mass = {Action.AUTO_SEND: 0.0, Action.NEEDS_REVIEW: 0.0}
            for alt in token.top_logprobs or []:
                if not alt.token.startswith(lead):
                    continue
                rest = alt.token[len(lead):]
                if not rest:
                    continue
                for action in mass:
                    if action.value.startswith(rest):
                        mass[action] += math.exp(alt.logprob)
            total = sum(mass.values())
            return mass[Action.AUTO_SEND] / total if total else None
        offset = end
    return None


def check_available(model_name: str, host: str = DEFAULT_HOST) -> None:
    try:
        installed = [m.model for m in ollama.Client(host=host).list().models]
    except Exception as exc:
        raise SystemExit(
            f"Cannot reach Ollama at {host} ({type(exc).__name__}). Is it running?"
        ) from exc

    if not any(m == model_name or m.startswith(f"{model_name}:") for m in installed):
        raise SystemExit(
            f"Model {model_name!r} not installed. Run:  ollama pull {model_name}\n"
            f"Installed: {', '.join(installed) or 'none'}"
        )
