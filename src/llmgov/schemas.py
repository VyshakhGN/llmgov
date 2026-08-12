from __future__ import annotations
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Literal
from pydantic import BaseModel, Field


class Action(str, Enum):

    AUTO_SEND = "AUTO_SEND"
    NEEDS_REVIEW = "NEEDS_REVIEW"


class RiskCategory(str, Enum):

    PII = "PII"
    LEGAL_THREAT = "LEGAL_THREAT"
    POLICY_MISMATCH_RISK = "POLICY_MISMATCH_RISK"
    FRAUD_RISK = "FRAUD_RISK"
    MISSING_INFO = "MISSING_INFO"
    BUSINESS_RISK = "BUSINESS_RISK"
    NONE = "NONE"


class PolicyDecision(str, Enum):

    APPROVE_REFUND = "APPROVE_REFUND"
    PARTIAL_REFUND = "PARTIAL_REFUND"
    DENY_REFUND = "DENY_REFUND"
    REQUEST_INFO = "REQUEST_INFO"


class Mode(str, Enum):

    PROMPT_ONLY = "prompt_only"
    FULL_CONTEXT = "full_context"
    RAG = "rag"



class Guideline(BaseModel):

    guideline_id: str
    text: str
    risk_category: RiskCategory
    # Where the rule comes from — a regulation, or our own operational policy.
    # Documentation only; not shown to the model.
    source: str = ""


class Category(str, Enum):

    CLOTHING = "clothing"
    ELECTRONICS = "electronics"


class ReturnCondition(str, Enum):

    ACCEPTABLE = "acceptable"
    DAMAGED = "damaged"


class SystemFacts(BaseModel):
    order_id: str | None = None
    order_status: str | None = None
    category: Category | None = None
    delivered_days_ago: int | None = None
    is_faulty: bool = False
    return_condition: ReturnCondition | None = None
    order_value_eur: float | None = None
    account_flags: list[str] = Field(default_factory=list)


class RiskSignals(BaseModel):

    pii_matches: list[str] = Field(default_factory=list)
    legal_keyword_matches: list[str] = Field(default_factory=list)
    has_iban_like: bool = False
    has_card_like: bool = False
    has_email: bool = False
    has_phone: bool = False
    policy_draft_mismatch: bool = False
    missing_order_id: bool = False


class GoldLabel(BaseModel):

    action: Action
    risk_category: RiskCategory
    reasoning: str = ""
    annotator_id: str = "primary"
    secondary_action: Action | None = None
    secondary_risk_category: RiskCategory | None = None


class Case(BaseModel):

    case_id: str
    user_message: str
    policy_decision: PolicyDecision
    draft_response: str
    system_facts: SystemFacts = Field(default_factory=SystemFacts)

    gold: GoldLabel | None = None


class RetrievedGuideline(BaseModel):

    guideline_id: str
    text: str
    score: float | None = None
    rank: int | None = None


class RoutingDecision(BaseModel):

    action: Action
    risk_category: RiskCategory
    confidence: float | None = Field(default=None, ge=0.0, le=1.0)
    justification: str = ""
    reviewer_hints: str = ""
    retrieved_guidelines: list[RetrievedGuideline] = Field(default_factory=list)
    raw_model_output: str | None = None
    parse_error: str | None = None


class HumanReview(BaseModel):

    reviewer_id: str
    outcome: Literal["APPROVED_UNCHANGED", "APPROVED_EDITED", "REJECTED"]
    edited_response: str | None = None
    reviewer_agreed_with_risk_category: bool | None = None
    notes: str = ""
    reviewed_at: datetime


class TraceRecord(BaseModel):

    trace_id: str
    case_id: str
    run_id: str
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))

    mode: Mode
    prompt_version: str = "n/a"
    model_name: str = "n/a"
    model_params: dict[str, Any] = Field(default_factory=dict)
    seed: int | None = None
    guideline_corpus_version: str = "n/a"
    retriever_config: dict[str, Any] = Field(default_factory=dict)

    risk_signals: RiskSignals = Field(default_factory=RiskSignals)
    decision: RoutingDecision
    enforced_action: Action
    override_reason: str | None = None

    latency_ms: float | None = None
    prompt_tokens: int | None = None
    completion_tokens: int | None = None
    cost_usd: float | None = None
    gold: GoldLabel | None = None
    human_review: HumanReview | None = None
