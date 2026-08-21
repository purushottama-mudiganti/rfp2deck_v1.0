from __future__ import annotations

from langgraph.graph import END, StateGraph

from rfp2deck.agent.nodes import (
    build_narrative,
    compress_bullets,
    derive_sections,
    derive_visual_briefs,
    derive_technology_recommendations,
    extract_source_evidence,
    generate_notes,
    plan_deck,
    qa_and_report,
    reconcile_sources,
    understand_rfp,
)
from rfp2deck.agent.state import AgentState
from rfp2deck.core.logging import get_logger

log = get_logger(__name__)


def build_graph():
    log.info("Building agent graph")
    g = StateGraph(AgentState)
    g.add_node("reconcile_sources", reconcile_sources)
    g.add_node("extract_source_evidence", extract_source_evidence)
    g.add_node("understand_rfp", understand_rfp)
    g.add_node("derive_sections", derive_sections)
    g.add_node("build_narrative", build_narrative)
    g.add_node("derive_visual_briefs", derive_visual_briefs)
    g.add_node("derive_technology_recommendations", derive_technology_recommendations)
    g.add_node("plan_deck", plan_deck)
    g.add_node("compress_bullets", compress_bullets)
    g.add_node("generate_notes", generate_notes)
    g.add_node("qa_and_report", qa_and_report)

    g.set_entry_point("reconcile_sources")
    g.add_edge("reconcile_sources", "extract_source_evidence")
    g.add_edge("extract_source_evidence", "understand_rfp")

    # These analyses depend only on the RFP understanding and update distinct
    # state fields, so run them in one fan-out instead of paying their latency
    # sequentially.
    g.add_edge("understand_rfp", "derive_sections")
    g.add_edge("understand_rfp", "build_narrative")
    g.add_edge("understand_rfp", "derive_technology_recommendations")
    g.add_edge(["derive_sections", "build_narrative"], "derive_visual_briefs")
    g.add_edge(
        ["derive_visual_briefs", "derive_technology_recommendations"],
        "plan_deck",
    )
    g.add_edge("plan_deck", "compress_bullets")
    g.add_edge("compress_bullets", "generate_notes")
    g.add_edge("generate_notes", "qa_and_report")
    g.add_edge("qa_and_report", END)
    compiled = g.compile()
    log.info("Agent graph compiled (11 nodes)")
    return compiled
