"""Compile the tutoring-core StateGraph."""

from __future__ import annotations

from typing import Optional

from langgraph.graph import END, START, StateGraph

from graph.nodes import (
    compare_node,
    finalize_node,
    recovery_node,
    tutor_node,
    verifier_node,
)
from graph.routing import after_compare
from graph.state import TutoringState


def build_tutoring_graph(checkpointer=None):
    """Build Tutor → Verifier → Compare → (Recovery?) → Finalize.

    Pass a LangGraph checkpointer (e.g. MemorySaver()) for thread_id resume.
    """
    g = StateGraph(TutoringState)
    g.add_node("tutor", tutor_node)
    g.add_node("verifier", verifier_node)
    g.add_node("compare", compare_node)
    g.add_node("recovery", recovery_node)
    g.add_node("finalize", finalize_node)

    g.add_edge(START, "tutor")
    g.add_edge("tutor", "verifier")
    g.add_edge("verifier", "compare")
    g.add_conditional_edges(
        "compare",
        after_compare,
        {"finalize": "finalize", "recovery": "recovery"},
    )
    g.add_edge("recovery", "finalize")
    g.add_edge("finalize", END)

    return g.compile(checkpointer=checkpointer)


def build_tutoring_graph_with_memory():
    """Compile with in-memory checkpointer (useful for debugging single cases)."""
    from langgraph.checkpoint.memory import MemorySaver

    return build_tutoring_graph(checkpointer=MemorySaver())
