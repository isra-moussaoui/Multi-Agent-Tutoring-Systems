"""Conditional routing after the Compare node."""

from __future__ import annotations

from typing import Literal

from graph.state import TutoringState


RouteAfterCompare = Literal[
    "finalize",
    "recovery",
]


def after_compare(
    state: TutoringState,
) -> RouteAfterCompare:

    status = state.get(
        "status"
    )

    # Parse problems should not trigger another
    # disagreement-analysis LLM call.

    if status in (
        "parse_error",
        "failed",
        "quota",
    ):

        return "finalize"

    compare = state.get(
        "compare"
    ) or {}

    if compare.get(
        "agreed"
    ):

        return "finalize"

    return "recovery"