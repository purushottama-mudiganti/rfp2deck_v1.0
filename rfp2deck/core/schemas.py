from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["must", "should", "may"]
DocumentType = Literal[
    "base_rfp",
    "annexure",
    "customer_addendum",
    "customer_clarification",
    "supporting_reference",
    "commercial",
    "unknown",
]
SourceAuthority = Literal["binding", "authoritative", "contextual", "non_authoritative"]
EvidenceStatus = Literal["active", "clarified", "superseded", "unresolved"]


class SourceDocument(BaseModel):
    document_id: str
    name: str
    document_type: DocumentType = "unknown"
    authority: SourceAuthority = "contextual"
    issue_date: Optional[str] = None
    text: str
    locator_format: str = "document"
    character_count: int = 0
    metadata: Dict[str, Any] = Field(default_factory=dict)
    warnings: List[str] = Field(default_factory=list)


class ClarificationRecord(BaseModel):
    clarification_id: str
    document_id: str
    question: str = ""
    customer_response: str = ""
    source_ref: str
    authority: SourceAuthority = "authoritative"
    status: EvidenceStatus = "active"
    supersedes: List[str] = Field(default_factory=list)


class SourceConflict(BaseModel):
    topic: str
    statements: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)
    status: Literal["resolved", "unresolved"] = "unresolved"
    resolution: str = ""


class SourceReconciliation(BaseModel):
    precedence_summary: List[str] = Field(default_factory=list)
    clarifications: List[ClarificationRecord] = Field(default_factory=list)
    conflicts: List[SourceConflict] = Field(default_factory=list)
    unresolved_questions: List[str] = Field(default_factory=list)


class ClarificationOutcome(BaseModel):
    topic: str
    question: str = ""
    customer_response: str = ""
    effect_on_requirements: str = ""
    source_refs: List[str] = Field(default_factory=list)
    status: EvidenceStatus = "clarified"


class Requirement(BaseModel):
    id: str
    text: str
    priority: Priority = "should"
    source_ref: Optional[str] = None  # e.g., "RFP p12, section 3.2"
    source_refs: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)
    source_text: Optional[str] = None
    authority: SourceAuthority = "authoritative"
    status: EvidenceStatus = "active"
    supersedes: List[str] = Field(default_factory=list)
    confidence: float = Field(default=1.0, ge=0.0, le=1.0)


class SBOMItem(BaseModel):
    component: str
    category: str = ""
    purpose: str = ""
    source_or_basis: str = ""
    version_or_constraint: str = ""
    deployment_scope: str = ""
    notes: str = ""


class SourceEvidenceBatch(BaseModel):
    source_document_id: str
    chunk_id: str
    context_facts: List[str] = Field(default_factory=list)
    summary_points: List[str] = Field(default_factory=list)
    project_scope_points: List[str] = Field(default_factory=list)
    in_scope_work: List[str] = Field(default_factory=list)
    requirements: List[Requirement] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    submission_instructions: List[str] = Field(default_factory=list)
    procurement_or_submission_tools: List[str] = Field(default_factory=list)
    non_solution_references: List[str] = Field(default_factory=list)
    solution_technologies: List[str] = Field(default_factory=list)
    software_bill_of_materials: List[SBOMItem] = Field(default_factory=list)
    clarification_outcomes: List[ClarificationOutcome] = Field(default_factory=list)
    source_conflicts: List[SourceConflict] = Field(default_factory=list)


class RFPUnderstanding(BaseModel):
    customer_name: Optional[str] = None
    opportunity_title: Optional[str] = None
    due_date: Optional[str] = None
    summary: str
    project_scope: str = ""
    in_scope_work: List[str] = Field(default_factory=list)
    requirements: List[Requirement] = Field(default_factory=list)
    superseded_requirements: List[Requirement] = Field(default_factory=list)
    unresolved_requirements: List[Requirement] = Field(default_factory=list)
    clarification_outcomes: List[ClarificationOutcome] = Field(default_factory=list)
    source_conflicts: List[SourceConflict] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    submission_instructions: List[str] = Field(default_factory=list)
    procurement_or_submission_tools: List[str] = Field(default_factory=list)
    non_solution_references: List[str] = Field(default_factory=list)
    # Named technologies/platforms/tools explicitly mentioned in solution scope
    # or current/target technical requirements. Exclude procurement portals or
    # tools mentioned only for tender administration/submission.
    key_technologies: List[str] = Field(default_factory=list)
    # Preferred technology list for architecture and diagram prompts. Kept
    # separate from key_technologies so future prompts can preserve all mentions
    # while diagrams stay grounded in solution components only.
    solution_technologies: List[str] = Field(default_factory=list)
    software_bill_of_materials: List[SBOMItem] = Field(default_factory=list)


SlideArchetype = Literal[
    "Title",
    "Agenda",
    "Customer Context",
    "Requirements",
    "Solution Overview",
    "Architecture",
    "Deployment Architecture",
    "High Availability & DR",
    "Software Bill of Materials",
    "Assumptions & Dependencies",
    "Requirements Mapping",
    "Value & Differentiators",
    "Delivery Plan",
    "Timeline",
    "Risks",
    "Team",
    "Case Studies",
    "Commercials",
    "Next Steps",
    "Content",
]


# --------------------------
# Dynamic section taxonomy
# --------------------------
SectionType = Literal[
    "context",
    "requirements",
    "solution",
    "architecture",
    "delivery",
    "timeline",
    "risks",
    "team",
    "commercials",
    "case_study",
    "other",
]


class SectionSpec(BaseModel):
    section_title: str
    section_type: SectionType = "other"
    section_goal: str = ""
    slide_titles: List[str] = Field(default_factory=list)
    priority: Priority = "should"


class SectionPlan(BaseModel):
    slide_count_target: int = 16
    sections: List[SectionSpec] = Field(default_factory=list)


SectionTaxonomyCategory = Literal[
    "context",
    "requirements",
    "approach",
    "architecture",
    "delivery",
    "governance",
    "commercials",
    "team",
    "risk",
    "timeline",
    "other",
]


class SectionTaxonomyItem(BaseModel):
    section_id: str
    title: str
    summary: str
    category: SectionTaxonomyCategory = "other"
    key_topics: List[str] = Field(default_factory=list)
    source_refs: List[str] = Field(default_factory=list)


class SectionTaxonomy(BaseModel):
    sections: List[SectionTaxonomyItem] = Field(default_factory=list)


class ExecutiveNarrative(BaseModel):
    value_proposition: str
    strategic_outcomes: List[str] = Field(default_factory=list)
    solution_themes: List[str] = Field(default_factory=list)
    executive_summary_points: List[str] = Field(default_factory=list)
    mandatory_sections: List[str] = Field(default_factory=list)
    milestone_mapping: dict = Field(default_factory=dict)


class DiagramSpec(BaseModel):
    kind: Literal[
        "architecture",
        "deployment",
        "hadr",
        "timeline",
        "process",
        "org",
        "data_model",
        "testing",
        "ams",
        "generic",
    ] = "generic"
    prompt: str
    approved: bool = False  # UI gate; renderer inserts image only if approved
    image_path: Optional[str] = None  # filled by diagram generator


class BulletPoint(BaseModel):
    """A headline bullet with optional supporting sub-points.

    Used for narrative/context-heavy slides where a top-level point needs
    concrete substantiation (e.g. "Legacy estate constrains delivery" with
    sub-points naming the specific systems and pain points). The renderer
    prefers `detailed_points` over flat `bullets` when present.
    """

    text: str
    sub_points: List[str] = Field(default_factory=list)


# --------------------------
# Modern (card-based) layout structures
# --------------------------
# Semantic accent keys map to brand colours in the renderer; a raw 6-hex value
# is also accepted. Keeping this as a free string keeps the LLM output forgiving.
CardAccent = str  # one of: "challenge","solution","why","outcome","info","neutral", or "RRGGBB"


class Card(BaseModel):
    """A titled content card (rounded rectangle with a coloured left stripe).

    Used to render capability grids, executive-summary quadrants, and any
    "headline + supporting detail" block in the modern card style.
    """

    heading: str
    body: str = ""  # one short paragraph
    bullets: List[str] = Field(default_factory=list)  # optional in-card bullets
    accent: Optional[CardAccent] = None  # semantic colour key or hex; renderer cycles if empty


class ComparisonColumn(BaseModel):
    """One side of a two-column comparison (e.g. challenge vs. goal)."""

    heading: str
    items: List[str] = Field(default_factory=list)
    accent: Optional[CardAccent] = None


class Comparison(BaseModel):
    """A two-column "before vs. after / problem vs. goal" comparison slide."""

    left: ComparisonColumn
    right: ComparisonColumn


class SlideSpec(BaseModel):
    slide_id: str
    title: str
    archetype: SlideArchetype = "Content"
    rfp_section: Optional[str] = None
    milestone: Optional[str] = None
    bullets: List[str] = Field(default_factory=list)
    # Optional two-level bullets; when non-empty the renderer uses these instead
    # of `bullets` so headline points carry concrete supporting detail.
    detailed_points: List[BulletPoint] = Field(default_factory=list)
    # --- Modern card-based layout (preferred when present) ---
    # A short emphasised "so what" line shown in a banner under the title.
    key_message: Optional[str] = None
    # Titled cards rendered as a responsive grid (2-up / 2x2). When non-empty
    # the renderer uses a cards layout instead of plain bullets.
    cards: List[Card] = Field(default_factory=list)
    # Two-column comparison (problem vs. goal). Takes precedence over cards.
    comparison: Optional[Comparison] = None
    # Short KPI / stat callouts shown as a chip row along the bottom
    # (e.g. "340+ flights/day", "RTO 12h / RPO 8h").
    kpis: List[str] = Field(default_factory=list)
    table: Optional[Dict[str, Any]] = None  # {headers:[], rows:[[]]}
    notes: Optional[str] = None
    rfps: List[str] = Field(default_factory=list)  # refs for traceability
    layout_hint: Optional[str] = None  # layout name (optional)
    diagram: Optional[DiagramSpec] = None  # optional diagram generation + insertion
    preferred_font_pt: Optional[int] = 18  # renderer may shrink to fit


class DeckPlan(BaseModel):
    deck_title: str
    slides: List[SlideSpec]


class SlideNote(BaseModel):
    slide_id: str
    notes: str


class DeckNotes(BaseModel):
    """Speaker notes for the deck, keyed by slide_id."""

    notes: List[SlideNote] = Field(default_factory=list)


class TraceabilityItem(BaseModel):
    requirement_id: str
    requirement_text: str
    source_refs: List[str] = Field(default_factory=list)
    source_document_ids: List[str] = Field(default_factory=list)
    authority: SourceAuthority = "authoritative"
    status: EvidenceStatus = "active"
    covered_on_slides: List[str] = Field(default_factory=list)


class TraceabilityReport(BaseModel):
    deck_title: str
    generated_at: str
    coverage: List[TraceabilityItem] = Field(default_factory=list)
    uncovered_requirements: List[str] = Field(default_factory=list)
