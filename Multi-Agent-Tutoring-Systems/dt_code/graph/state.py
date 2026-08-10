"""LangGraph state for the tutoring core (Tutor → Verifier → Compare → Recovery)."""

from __future__ import annotations

import operator
from typing import Annotated, Any, Literal, Optional, TypedDict


class StudentStep(TypedDict, total=False):
    next_step: str
    rule: str
    parents: str


class AgentResult(TypedDict, total=False):
    verdict: Optional[str]
    feedback: Optional[str]
    confidence: Optional[float]
    raw: Optional[str]
    latency_ms: float
    model: str
    parse_ok: bool


class RecoveryResult(TypedDict, total=False):
    final_verdict: Optional[str]
    reasoning: Optional[str]
    raw: Optional[str]
    latency_ms: float
    model: str
    parse_ok: bool


class CompareResult(TypedDict, total=False):
    agreed: bool


Status = Literal["ok", "parse_error", "quota", "failed"]


class TutoringState(TypedDict, total=False):
    """Full graph state. Ground truth must never be read by agent prompt builders.

    Optional `eval_only` may hold harness-side fields (e.g. ground_truth_label);
    no node may read it for prompting.
    """

    # --- Input (harness) ---
    case_id: str
    givens: Any
    intermediates: Any
    conclusion: Any
    student_step: StudentStep

    # --- Agent outputs ---
    tutor: AgentResult
    verifier: AgentResult
    compare: CompareResult
    recovery: RecoveryResult

    # --- Final ---
    final_verdict: Optional[str]
    final_feedback: Optional[str]
    recovery_flag: bool
    status: Status
    errors: Annotated[list[str], operator.add]

    # --- Eval only (never used in prompts) ---
    eval_only: dict[str, Any]
