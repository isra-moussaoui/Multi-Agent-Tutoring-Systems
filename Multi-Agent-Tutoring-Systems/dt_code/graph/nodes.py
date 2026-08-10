"""LangGraph nodes for Tutor → Verifier → Compare → Recovery → Finalize.

Hard invariant: verifier_node must never read tutor verdict/feedback when
building its prompt — only problem fields + student_step.
"""

from __future__ import annotations

import time
from typing import Any

from langchain_core.runnables import RunnableConfig

import prompts
from graph.config import role_from_configurable
from graph.state import AgentResult, RecoveryResult, TutoringState
from llm_client import DailyQuotaExceeded, call_llm, extract_json_object

_JSON_RETRY_SUFFIX = (
    "\n\nIMPORTANT: Respond with ONLY a single valid JSON object matching the "
    "required schema. No markdown fences, no commentary."
)


def _student_fields(state: TutoringState) -> tuple[Any, Any, Any, str, str, str]:
    step = state.get("student_step") or {}
    return (
        state.get("givens"),
        state.get("intermediates"),
        state.get("conclusion"),
        step.get("next_step", "") or "",
        step.get("rule", "") or "",
        step.get("parents", "") or "",
    )


def _call_with_optional_retry(
    prompt: str,
    *,
    provider: str,
    model: str,
    temperature: float,
    max_tokens: int,
    sleep_s: float,
    require_key: str,
) -> tuple[dict | None, str, float]:
    """Call LLM, parse JSON; on missing key, retry once with lower temp + suffix."""
    t0 = time.perf_counter()
    raw = call_llm(
        prompt, provider=provider, model=model,
        temperature=temperature, max_tokens=max_tokens,
    )
    parsed = extract_json_object(raw)
    if sleep_s > 0:
        time.sleep(sleep_s)

    if parsed and require_key in parsed:
        return parsed, raw, (time.perf_counter() - t0) * 1000.0

    # One retry for parse / schema failures
    retry_temp = max(0.0, temperature - 0.1)
    raw2 = call_llm(
        prompt + _JSON_RETRY_SUFFIX,
        provider=provider, model=model,
        temperature=retry_temp, max_tokens=max_tokens,
    )
    parsed2 = extract_json_object(raw2)
    if sleep_s > 0:
        time.sleep(sleep_s)
    latency_ms = (time.perf_counter() - t0) * 1000.0
    if parsed2 and require_key in parsed2:
        return parsed2, raw2, latency_ms
    return parsed2 or parsed, raw2 if raw2 else raw, latency_ms


def tutor_node(state: TutoringState, config: RunnableConfig) -> dict[str, Any]:
    """Blind tutor: grades from problem + student step only (no answer key)."""
    cfg = role_from_configurable(config.get("configurable"), "tutor")
    givens, intermediates, conclusion, next_step, rule, parents = _student_fields(state)
    prompt = prompts.blind_tutor_prompt(
        givens, intermediates, conclusion, next_step, rule, parents,
    )
    try:
        parsed, raw, latency_ms = _call_with_optional_retry(
            prompt,
            provider=cfg.provider, model=cfg.model,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            sleep_s=cfg.sleep_s, require_key="VERDICT",
        )
    except DailyQuotaExceeded:
        raise

    parse_ok = bool(parsed and "VERDICT" in parsed)
    result: AgentResult = {
        "verdict": (parsed or {}).get("VERDICT") if parsed else None,
        "feedback": (parsed or {}).get("FEEDBACK") if parsed else None,
        "confidence": (parsed or {}).get("CONFIDENCE") if parsed else None,
        "raw": raw,
        "latency_ms": latency_ms,
        "model": cfg.model,
        "parse_ok": parse_ok,
    }
    out: dict[str, Any] = {"tutor": result}
    if not parse_ok:
        out["status"] = "parse_error"
        out["errors"] = ["tutor_parse_failed"]
    return out


def verifier_node(state: TutoringState, config: RunnableConfig) -> dict[str, Any]:
    """Independent audit — NEVER reads tutor.* from state for prompting.

    Research invariant against conformity bias: the Verifier only sees the
    same inputs as the Tutor (problem + student step), never the Tutor's
    verdict or feedback.
    """
    # Deliberately do not access state["tutor"] here.
    cfg = role_from_configurable(config.get("configurable"), "verifier")
    givens, intermediates, conclusion, next_step, rule, parents = _student_fields(state)
    prompt = prompts.verifier_prompt(
        givens, intermediates, conclusion, next_step, rule, parents,
    )
    try:
        parsed, raw, latency_ms = _call_with_optional_retry(
            prompt,
            provider=cfg.provider, model=cfg.model,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            sleep_s=cfg.sleep_s, require_key="VERDICT",
        )
    except DailyQuotaExceeded:
        raise

    parse_ok = bool(parsed and "VERDICT" in parsed)
    result: AgentResult = {
        "verdict": (parsed or {}).get("VERDICT") if parsed else None,
        "feedback": (parsed or {}).get("FEEDBACK") if parsed else None,
        "confidence": (parsed or {}).get("CONFIDENCE") if parsed else None,
        "raw": raw,
        "latency_ms": latency_ms,
        "model": cfg.model,
        "parse_ok": parse_ok,
    }
    out: dict[str, Any] = {"verifier": result}
    if not parse_ok:
        out["status"] = "parse_error"
        out["errors"] = ["verifier_parse_failed"]
    return out


def compare_node(state: TutoringState) -> dict[str, Any]:
    """Pure code: case-insensitive verdict equality. No LLM call."""
    tutor = state.get("tutor") or {}
    verifier = state.get("verifier") or {}
    tv = (tutor.get("verdict") or "").strip().lower()
    vv = (verifier.get("verdict") or "").strip().lower()
    if not tv or not vv:
        # Missing verdicts → do not claim agreement; routing skips recovery if status=parse_error
        return {"compare": {"agreed": False}}
    return {"compare": {"agreed": tv == vv}}


def recovery_node(state: TutoringState, config: RunnableConfig) -> dict[str, Any]:
    """Tie-break when Tutor and Verifier disagree. Sees both verdicts."""
    cfg = role_from_configurable(config.get("configurable"), "recovery")
    givens, intermediates, conclusion, next_step, rule, parents = _student_fields(state)
    tutor = state.get("tutor") or {}
    verifier = state.get("verifier") or {}
    prompt = prompts.recovery_prompt(
        givens, intermediates, conclusion, next_step, rule, parents,
        tutor.get("verdict"), tutor.get("feedback"),
        verifier.get("verdict"), verifier.get("feedback"),
    )
    try:
        parsed, raw, latency_ms = _call_with_optional_retry(
            prompt,
            provider=cfg.provider, model=cfg.model,
            temperature=cfg.temperature, max_tokens=cfg.max_tokens,
            sleep_s=cfg.sleep_s, require_key="FINAL_VERDICT",
        )
    except DailyQuotaExceeded:
        raise

    parse_ok = bool(parsed and "FINAL_VERDICT" in parsed)
    result: RecoveryResult = {
        "final_verdict": (parsed or {}).get("FINAL_VERDICT") if parsed else None,
        "reasoning": (parsed or {}).get("REASONING") if parsed else None,
        "raw": raw,
        "latency_ms": latency_ms,
        "model": cfg.model,
        "parse_ok": parse_ok,
    }
    out: dict[str, Any] = {"recovery": result}
    if not parse_ok:
        out["status"] = "parse_error"
        out["errors"] = ["recovery_parse_failed"]
    return out


def finalize_node(state: TutoringState) -> dict[str, Any]:
    """Collapse tutor/recovery into a single exit shape for the harness/JSONL."""
    status = state.get("status") or "ok"
    tutor = state.get("tutor") or {}
    verifier = state.get("verifier") or {}
    recovery = state.get("recovery") or {}
    compare = state.get("compare") or {}
    agreed = bool(compare.get("agreed"))

    if status == "parse_error":
        # Prefer whatever we have; may leave final_verdict None
        if recovery.get("parse_ok") and recovery.get("final_verdict"):
            return {
                "final_verdict": recovery.get("final_verdict"),
                "final_feedback": recovery.get("reasoning"),
                "recovery_flag": True,
                "status": "parse_error",
            }
        return {
            "final_verdict": tutor.get("verdict"),
            "final_feedback": tutor.get("feedback"),
            "recovery_flag": False,
            "status": "parse_error",
        }

    if agreed:
        return {
            "final_verdict": tutor.get("verdict"),
            "final_feedback": tutor.get("feedback"),
            "recovery_flag": False,
            "status": "ok" if tutor.get("parse_ok") and verifier.get("parse_ok") else status,
        }

    if recovery.get("parse_ok") and recovery.get("final_verdict"):
        return {
            "final_verdict": recovery.get("final_verdict"),
            "final_feedback": recovery.get("reasoning"),
            "recovery_flag": True,
            "status": "ok",
        }

    return {
        "final_verdict": None,
        "final_feedback": None,
        "recovery_flag": True,
        "status": "parse_error",
        "errors": ["finalize_missing_recovery_verdict"],
    }
