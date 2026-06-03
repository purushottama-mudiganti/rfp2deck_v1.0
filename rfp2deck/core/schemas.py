from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

Priority = Literal["must", "should", "may"]


class Requirement(BaseModel):
    id: str
    text: str
    priority: Priority = "should"
    source_ref: Optional[str] = None  # e.g., "RFP p12, §3.2"


class RFPUnderstanding(BaseModel):
    customer_name: Optional[str] = None
    opportunity_title: Optional[str] = None
    due_date: Optional[str] = None
    summary: str
    requirements: List[Requirement] = Field(default_factory=list)
    assumptions: List[str] = Field(default_factory=list)
    risks: List[str] = Field(default_factory=list)
    # Named technologies/platforms/tools explicitly mentioned in the RFP
    # (e.g., "AKS", "PostgreSQL", "Kafka"). Used to ground diagram prompts so
    # generated visuals reference real systems rather than generic placeholders.
    key_technologies: List[str] = Field(default_factory=list)


SlideArchetype = Literal[
    "Title",
    "Agenda",
    "Customer Context",
    "Requirements",
    "Solution Overview",
    "Architecture",
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
    kind: Literal["architecture", "timeline", "process", "org", "generic"] = "generic"
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
    covered_on_slides: List[str] = Field(default_factory=list)


class TraceabilityReport(BaseModel):
    deck_title: str
    generated_at: str
    coverage: List[TraceabilityItem] = Field(default_factory=list)
    uncovered_requirements: List[str] = Field(default_factory=list)
