from __future__ import annotations

from langgraph.graph import END, StateGraph

from rfp2deck.agent.nodes import (
    build_narrative,
    compress_bullets,
    derive_sections,
    plan_deck,
    qa_and_report,
    understand_rfp,
)
from rfp2deck.agent.state import AgentState


def build_graph():
    g = StateGraph(AgentState)
    g.add_node("understand_rfp", understand_rfp)
    g.add_node("derive_sections", derive_sections)
    g.add_node("build_narrative", build_narrative)
    g.add_node("plan_deck", plan_deck)
    g.add_node("compress_bullets", compress_bullets)
    g.add_node("qa_and_report", qa_and_report)

    g.set_entry_point("understand_rfp")
    g.add_edge("understand_rfp", "derive_sections")
    g.add_edge("derive_sections", "build_narrative")
    g.add_edge("build_narrative", "plan_deck")
    g.add_edge("plan_deck", "compress_bullets")
    g.add_edge("compress_bullets", "qa_and_report")
    g.add_edge("qa_and_report", END)
    return g.compile()
