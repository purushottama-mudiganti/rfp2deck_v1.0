from __future__ import annotations

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field

from rfp2deck.core.schemas import (
    ClarificationRecord,
    DeckPlan,
    DiagramBrief,
    ExecutiveNarrative,
    RFPUnderstanding,
    SectionPlan,
    SourceDocument,
    SourceEvidenceBatch,
    SourceReconciliation,
    TraceabilityReport,
    TechnologyRecommendationSet,
)


class AgentState(BaseModel):
    narrative: Optional[ExecutiveNarrative] = None
    visual_briefs: List[DiagramBrief] = Field(default_factory=list)
    technology_recommendations: Optional[TechnologyRecommendationSet] = None
    customer_technology_context: Dict[str, Any] = Field(default_factory=dict)
    rfp_text: str
    template_info: Dict[str, Any]
    source_documents: List[SourceDocument] = Field(default_factory=list)
    clarification_records: List[ClarificationRecord] = Field(default_factory=list)
    source_reconciliation: Optional[SourceReconciliation] = None
    source_evidence: List[SourceEvidenceBatch] = Field(default_factory=list)
    evidence_text: Optional[str] = None
    # Architecture research and other supporting references are advisory, not
    # requirements.  Keep a bounded, explicitly-labelled copy available after
    # RFPUnderstanding so visual/technology planners do not silently lose it.
    contextual_reference_context: str = ""
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
    # Per-run override for the specialist-architect fan-out. ``None`` means
    # "fall back to settings.deck_plan_specialists" (env default / headless
    # callers); the UI sets an explicit bool so it can be toggled without a
    # redeploy.
    deck_plan_specialists: Optional[bool] = None
    debug: Dict[str, Any] = Field(default_factory=dict)

    def model_post_init(self, __context: Any) -> None:
        if self.rag_context is None:
            self.rag_context = self.retrieved_context
