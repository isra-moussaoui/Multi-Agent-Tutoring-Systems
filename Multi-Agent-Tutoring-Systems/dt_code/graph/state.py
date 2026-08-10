"""LangGraph state for the tutoring core.

Architecture:

    Tutor || Verifier
          ↓
        Compare
          ↓
    Recovery? 
          ↓
       Finalize

Ground truth is kept in eval_only and must never be used by
any agent prompt builder.
"""

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
    provider: str
    parse_ok: bool


class RecoveryResult(TypedDict, total=False):
    final_verdict: Optional[str]
    reasoning: Optional[str]
    raw: Optional[str]
    latency_ms: float
    model: str
    provider: str
    parse_ok: bool


class CompareResult(TypedDict, total=False):
    agreed: bool
    tutor_verdict: Optional[str]
    verifier_verdict: Optional[str]
    tutor_parse_ok: bool
    verifier_parse_ok: bool


Status = Literal[
    "ok",
    "parse_error",
    "quota",
    "failed",
]


class TutoringState(TypedDict, total=False):

    # =========================================================
    # INPUT
    # =========================================================

    case_id: str

    givens: Any
    intermediates: Any
    conclusion: Any

    student_step: StudentStep

    # =========================================================
    # AGENT OUTPUTS
    # =========================================================

    tutor: AgentResult

    verifier: AgentResult

    compare: CompareResult

    recovery: RecoveryResult

    # =========================================================
    # FINAL
    # =========================================================

    final_verdict: Optional[str]

    final_feedback: Optional[str]

    recovery_flag: bool

    status: Status

    errors: Annotated[
        list[str],
        operator.add,
    ]

    # =========================================================
    # EVALUATION ONLY
    # =========================================================

    # NEVER read this from prompt-building code.
    eval_only: dict[str, Any]