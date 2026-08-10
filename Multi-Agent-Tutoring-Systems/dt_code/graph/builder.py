"""Compile the tutoring-core StateGraph.

Graph execution:

                    START
                      |
             +--------+--------+
             |                 |
             v                 v
          TUTOR             VERIFIER
             |                 |
             +--------+--------+
                      |
                      v
                   COMPARE
                      |
              +-------+-------+
              |               |
            AGREE         DISAGREE
              |               |
              |            RECOVERY
              |               |
              +-------+-------+
                      |
                      v
                  FINALIZE
                      |
                      v
                     END

Tutor and Verifier are intentionally independent and therefore
can execute concurrently.
"""

from __future__ import annotations

from langgraph.graph import (
    END,
    START,
    StateGraph,
)

from graph.nodes import (
    compare_node,
    finalize_node,
    recovery_node,
    tutor_node,
    verifier_node,
)

from graph.routing import after_compare

from graph.state import TutoringState


def build_tutoring_graph(
    checkpointer=None,
):
    """Build the parallel tutoring graph.

    Tutor and Verifier both start directly from START.

    LangGraph therefore schedules them as parallel branches
    and Compare acts as the synchronization barrier.

    Args:
        checkpointer:
            Optional LangGraph checkpointer, e.g. MemorySaver().

    Returns:
        Compiled LangGraph.
    """

    g = StateGraph(
        TutoringState
    )

    # ========================================================
    # NODES
    # ========================================================

    g.add_node(
        "tutor",
        tutor_node,
    )

    g.add_node(
        "verifier",
        verifier_node,
    )

    g.add_node(
        "compare",
        compare_node,
    )

    g.add_node(
        "recovery",
        recovery_node,
    )

    g.add_node(
        "finalize",
        finalize_node,
    )

    # ========================================================
    # PARALLEL START
    # ========================================================

    # IMPORTANT:
    #
    # DO NOT do:
    #
    # START → tutor → verifier
    #
    # Instead:
    #
    # START → tutor
    # START → verifier
    #
    # This makes the two independent LLM calls
    # executable in parallel.

    g.add_edge(
        START,
        "tutor",
    )

    g.add_edge(
        START,
        "verifier",
    )

    # ========================================================
    # SYNCHRONIZATION BARRIER
    # ========================================================

    # Compare is reached only after both branches
    # have completed.

    g.add_edge(
        "tutor",
        "compare",
    )

    g.add_edge(
        "verifier",
        "compare",
    )

    # ========================================================
    # CONDITIONAL ROUTING
    # ========================================================

    g.add_conditional_edges(

        "compare",

        after_compare,

        {
            "finalize":
                "finalize",

            "recovery":
                "recovery",
        },
    )

    # ========================================================
    # RECOVERY
    # ========================================================

    g.add_edge(
        "recovery",
        "finalize",
    )

    # ========================================================
    # END
    # ========================================================

    g.add_edge(
        "finalize",
        END,
    )

    return g.compile(
        checkpointer=checkpointer
    )


def build_tutoring_graph_with_memory():
    """Compile with in-memory checkpointer.

    Useful for debugging individual cases.
    """

    from langgraph.checkpoint.memory import (
        MemorySaver,
    )

    return build_tutoring_graph(
        checkpointer=MemorySaver()
    )