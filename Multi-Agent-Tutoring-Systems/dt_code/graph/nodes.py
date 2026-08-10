"""LangGraph nodes for the tutoring system.

Execution:

    Tutor || Verifier
          ↓
        Compare
          ↓
    Recovery? 
          ↓
       Finalize

Important research invariant:

The Verifier NEVER reads tutor output when constructing its prompt.
This preserves independent evaluation and avoids conformity bias.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

import prompts

from graph.config import role_from_configurable

from graph.state import (
    AgentResult,
    RecoveryResult,
    TutoringState,
)

from llm_client import (
    DailyQuotaExceeded,
    call_llm,
    extract_json_object,
)


_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY a single valid JSON object "
    "matching the required schema. "
    "No markdown fences and no commentary."
)


# ============================================================
# STUDENT FIELDS
# ============================================================

def _student_fields(
    state: TutoringState,
) -> tuple[Any, Any, Any, str, str, str]:

    step = state.get("student_step") or {}

    return (

        state.get("givens"),

        state.get("intermediates"),

        state.get("conclusion"),

        step.get(
            "next_step",
            "",
        ) or "",

        step.get(
            "rule",
            "",
        ) or "",

        step.get(
            "parents",
            "",
        ) or "",
    )


# ============================================================
# LLM CALL + JSON RETRY
# ============================================================

def _call_with_optional_retry(
    prompt: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    sleep_s: float,
    require_key: str,
) -> tuple[
    dict | None,
    str,
    float,
]:

    started = time.perf_counter()

    # --------------------------------------------------------
    # First attempt
    # --------------------------------------------------------

    raw = call_llm(

        prompt,

        provider=provider,

        model=model,

        temperature=temperature,

        max_tokens=max_tokens,
    )

    parsed = extract_json_object(raw)

    if sleep_s > 0:
        time.sleep(sleep_s)

    if parsed and require_key in parsed:

        latency_ms = (
            time.perf_counter()
            - started
        ) * 1000.0

        return (
            parsed,
            raw,
            latency_ms,
        )

    # --------------------------------------------------------
    # JSON/schema retry
    # --------------------------------------------------------

    retry_temp = max(
        0.0,
        temperature - 0.1,
    )

    raw2 = call_llm(

        prompt + _JSON_RETRY_SUFFIX,

        provider=provider,

        model=model,

        temperature=retry_temp,

        max_tokens=max_tokens,
    )

    parsed2 = extract_json_object(raw2)

    if sleep_s > 0:
        time.sleep(sleep_s)

    latency_ms = (
        time.perf_counter()
        - started
    ) * 1000.0

    if parsed2 and require_key in parsed2:

        return (
            parsed2,
            raw2,
            latency_ms,
        )

    return (
        parsed2 or parsed,
        raw2 or raw,
        latency_ms,
    )


# ============================================================
# TUTOR
# ============================================================

def tutor_node(
    state: TutoringState,
    config: RunnableConfig,
) -> dict[str, Any]:

    cfg = role_from_configurable(
        config.get("configurable"),
        "tutor",
    )

    (
        givens,
        intermediates,
        conclusion,
        next_step,
        rule,
        parents,
    ) = _student_fields(state)

    prompt = prompts.blind_tutor_prompt(

        givens,

        intermediates,

        conclusion,

        next_step,

        rule,

        parents,
    )

    try:

        (
            parsed,
            raw,
            latency_ms,
        ) = _call_with_optional_retry(

            prompt,

            provider=cfg.provider,

            model=cfg.model,

            temperature=cfg.temperature,

            max_tokens=cfg.max_tokens,

            sleep_s=cfg.sleep_s,

            require_key="VERDICT",
        )

    except DailyQuotaExceeded:

        raise

    parse_ok = bool(
        parsed
        and "VERDICT" in parsed
    )

    result: AgentResult = {

        "verdict": (
            parsed.get("VERDICT")
            if parsed
            else None
        ),

        "feedback": (
            parsed.get("FEEDBACK")
            if parsed
            else None
        ),

        "confidence": (
            parsed.get("CONFIDENCE")
            if parsed
            else None
        ),

        "raw": raw,

        "latency_ms":
            latency_ms,

        "model":
            cfg.model,

        "provider":
            cfg.provider,

        "parse_ok":
            parse_ok,
    }

    # IMPORTANT:
    #
    # Do NOT write "status" here.
    #
    # Tutor and Verifier execute concurrently.
    # Both writing status could create a concurrent
    # state update conflict.
    #
    return {
        "tutor": result,
    }


# ============================================================
# VERIFIER
# ============================================================

def verifier_node(
    state: TutoringState,
    config: RunnableConfig,
) -> dict[str, Any]:

    """Independent audit.

    IMPORTANT:
    This function deliberately NEVER reads state["tutor"].

    Tutor and Verifier therefore receive exactly the same
    information and independently produce their judgments.
    """

    cfg = role_from_configurable(

        config.get("configurable"),

        "verifier",
    )

    (
        givens,
        intermediates,
        conclusion,
        next_step,
        rule,
        parents,
    ) = _student_fields(state)

    prompt = prompts.verifier_prompt(

        givens,

        intermediates,

        conclusion,

        next_step,

        rule,

        parents,
    )

    try:

        (
            parsed,
            raw,
            latency_ms,
        ) = _call_with_optional_retry(

            prompt,

            provider=cfg.provider,

            model=cfg.model,

            temperature=cfg.temperature,

            max_tokens=cfg.max_tokens,

            sleep_s=cfg.sleep_s,

            require_key="VERDICT",
        )

    except DailyQuotaExceeded:

        raise

    parse_ok = bool(
        parsed
        and "VERDICT" in parsed
    )

    result: AgentResult = {

        "verdict": (
            parsed.get("VERDICT")
            if parsed
            else None
        ),

        "feedback": (
            parsed.get("FEEDBACK")
            if parsed
            else None
        ),

        "confidence": (
            parsed.get("CONFIDENCE")
            if parsed
            else None
        ),

        "raw": raw,

        "latency_ms":
            latency_ms,

        "model":
            cfg.model,

        "provider":
            cfg.provider,

        "parse_ok":
            parse_ok,
    }

    # Again: do NOT modify status here.

    return {
        "verifier": result,
    }


# ============================================================
# COMPARE
# ============================================================

def compare_node(
    state: TutoringState,
) -> dict[str, Any]:

    """Pure Python synchronization/decision node.

    This node executes only after BOTH Tutor and Verifier
    have completed.
    """

    tutor = state.get(
        "tutor"
    ) or {}

    verifier = state.get(
        "verifier"
    ) or {}

    tutor_parse_ok = bool(
        tutor.get("parse_ok")
    )

    verifier_parse_ok = bool(
        verifier.get("parse_ok")
    )

    tv = (
        str(
            tutor.get(
                "verdict"
            ) or ""
        )
        .strip()
        .lower()
    )

    vv = (
        str(
            verifier.get(
                "verdict"
            ) or ""
        )
        .strip()
        .lower()
    )

    agreed = bool(
        tutor_parse_ok
        and verifier_parse_ok
        and tv
        and vv
        and tv == vv
    )

    # --------------------------------------------------------
    # IMPORTANT:
    #
    # This is now the ONLY place that determines parse_error.
    # This prevents parallel nodes from simultaneously writing
    # the scalar "status" field.
    # --------------------------------------------------------

    if (
        not tutor_parse_ok
        or not verifier_parse_ok
    ):

        status = "parse_error"

    else:

        status = "ok"

    return {

        "compare": {

            "agreed":
                agreed,

            "tutor_verdict":
                tv or None,

            "verifier_verdict":
                vv or None,

            "tutor_parse_ok":
                tutor_parse_ok,

            "verifier_parse_ok":
                verifier_parse_ok,
        },

        "status":
            status,

        "recovery_flag":
            not agreed,
    }


# ============================================================
# RECOVERY
# ============================================================

def recovery_node(
    state: TutoringState,
    config: RunnableConfig,
) -> dict[str, Any]:

    cfg = role_from_configurable(

        config.get("configurable"),

        "recovery",
    )

    (
        givens,
        intermediates,
        conclusion,
        next_step,
        rule,
        parents,
    ) = _student_fields(state)

    tutor = state.get(
        "tutor"
    ) or {}

    verifier = state.get(
        "verifier"
    ) or {}

    prompt = prompts.recovery_prompt(

        givens,

        intermediates,

        conclusion,

        next_step,

        rule,

        parents,

        tutor.get(
            "verdict"
        ),

        tutor.get(
            "feedback"
        ),

        verifier.get(
            "verdict"
        ),

        verifier.get(
            "feedback"
        ),
    )

    try:

        (
            parsed,
            raw,
            latency_ms,
        ) = _call_with_optional_retry(

            prompt,

            provider=cfg.provider,

            model=cfg.model,

            temperature=cfg.temperature,

            max_tokens=cfg.max_tokens,

            sleep_s=cfg.sleep_s,

            require_key="FINAL_VERDICT",
        )

    except DailyQuotaExceeded:

        raise

    parse_ok = bool(
        parsed
        and "FINAL_VERDICT" in parsed
    )

    result: RecoveryResult = {

        "final_verdict": (
            parsed.get(
                "FINAL_VERDICT"
            )
            if parsed
            else None
        ),

        "reasoning": (
            parsed.get(
                "REASONING"
            )
            if parsed
            else None
        ),

        "raw":
            raw,

        "latency_ms":
            latency_ms,

        "model":
            cfg.model,

        "provider":
            cfg.provider,

        "parse_ok":
            parse_ok,
    }

    return {
        "recovery": result,
    }


# ============================================================
# FINALIZE
# ============================================================

def finalize_node(
    state: TutoringState,
) -> dict[str, Any]:

    """Produce the final standardized output."""

    status = (
        state.get(
            "status"
        )
        or "ok"
    )

    tutor = state.get(
        "tutor"
    ) or {}

    verifier = state.get(
        "verifier"
    ) or {}

    recovery = state.get(
        "recovery"
    ) or {}

    compare = state.get(
        "compare"
    ) or {}

    agreed = bool(
        compare.get(
            "agreed"
        )
    )

    # --------------------------------------------------------
    # PARSE ERROR
    # --------------------------------------------------------

    if status == "parse_error":

        # If recovery somehow ran and succeeded,
        # prefer its result.

        if (
            recovery.get(
                "parse_ok"
            )
            and recovery.get(
                "final_verdict"
            )
        ):

            return {

                "final_verdict":
                    recovery.get(
                        "final_verdict"
                    ),

                "final_feedback":
                    recovery.get(
                        "reasoning"
                    ),

                "recovery_flag":
                    True,

                "status":
                    "parse_error",
            }

        # Otherwise return the Tutor result if available.

        return {

            "final_verdict":
                tutor.get(
                    "verdict"
                ),

            "final_feedback":
                tutor.get(
                    "feedback"
                ),

            "recovery_flag":
                False,

            "status":
                "parse_error",
        }

    # --------------------------------------------------------
    # AGREEMENT
    # --------------------------------------------------------

    if agreed:

        return {

            "final_verdict":
                tutor.get(
                    "verdict"
                ),

            "final_feedback":
                tutor.get(
                    "feedback"
                ),

            "recovery_flag":
                False,

            "status":
                "ok",
        }

    # --------------------------------------------------------
    # DISAGREEMENT → RECOVERY
    # --------------------------------------------------------

    if (
        recovery.get(
            "parse_ok"
        )
        and recovery.get(
            "final_verdict"
        )
    ):

        return {

            "final_verdict":
                recovery.get(
                    "final_verdict"
                ),

            "final_feedback":
                recovery.get(
                    "reasoning"
                ),

            "recovery_flag":
                True,

            "status":
                "ok",
        }

    # --------------------------------------------------------
    # RECOVERY FAILED
    # --------------------------------------------------------

    return {

        "final_verdict":
            None,

        "final_feedback":
            None,

        "recovery_flag":
            True,

        "status":
            "parse_error",

        "errors": [
            "finalize_missing_recovery_verdict"
        ],
    }