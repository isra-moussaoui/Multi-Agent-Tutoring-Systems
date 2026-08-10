"""Conditional routing after the Compare node."""

from __future__ import annotations

from typing import Literal

from graph.state import TutoringState

RouteAfterCompare = Literal["finalize", "recovery"]


def after_compare(state: TutoringState) -> RouteAfterCompare:
    """Agree or prior parse failure → finalize; else → recovery."""
    status = state.get("status")
    if status in ("parse_error", "failed", "quota"):
        return "finalize"
    compare = state.get("compare") or {}
    if compare.get("agreed"):
        return "finalize"
    return "recovery"
