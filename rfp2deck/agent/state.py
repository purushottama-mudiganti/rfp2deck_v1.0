from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel

from rfp2deck.core.schemas import (
    DeckPlan,
    ExecutiveNarrative,
    RFPUnderstanding,
    SectionPlan,
    TraceabilityReport,
)


class AgentState(BaseModel):
    narrative: Optional[ExecutiveNarrative] = None
    rfp_text: str
    template_info: Dict[str, Any]
    retrieved_context: Optional[str] = None
    rag_context: Optional[str] = None
    understanding: Optional[RFPUnderstanding] = None
    section_map: Optional[Dict[str, Any]] = None
    section_plan: Optional[SectionPlan] = None
    deck_plan: Optional[DeckPlan] = None
    pptx_path: Optional[str] = None
    report: Optional[TraceabilityReport] = None
    deck_mode: Optional[str] = None
    enable_notes: bool = True
    debug: Dict[str, Any] = {}

    def model_post_init(self, __context: Any) -> None:
        if self.rag_context is None:
            self.rag_context = self.retrieved_context
