from __future__ import annotations

import re
from typing import Any, Dict, List, Optional

from rfp2deck.agent.prompts import (
    DECK_PLAN_V2_PROMPT,
    EXEC_NARRATIVE_PROMPT,
    RFP_UNDERSTAND_PROMPT,
    SECTION_TAXONOMY_PROMPT,
    SLIDE_COMPRESSION_PROMPT,
)
from rfp2deck.agent.state import AgentState
from rfp2deck.core.schemas import (
    DeckPlan,
    DiagramSpec,
    ExecutiveNarrative,
    RFPUnderstanding,
    SectionTaxonomy,
    SlideSpec,
    TraceabilityReport,
)
from rfp2deck.llm.structured import response_as_schema
from rfp2deck.qa.coverage import build_traceability_report


def understand_rfp(state: AgentState) -> Dict[str, Any]:
    """Extract a structured understanding of the RFP."""
    prompt = RFP_UNDERSTAND_PROMPT.format(
        rfp_text=state.rfp_text or "",
        rag_context=state.rag_context or "",
    )
    understanding = response_as_schema(prompt, RFPUnderstanding, reasoning_effort="high")
    state.understanding = understanding
    return {"understanding": understanding}


def classify_sections(state: AgentState) -> Dict[str, Any]:
    """Classify RFP into section taxonomy for better subtitle generation & narrative."""
    prompt = SECTION_TAXONOMY_PROMPT.format(
        rfp_text=state.rfp_text or "",
        rag_context=state.rag_context or "",
    )
    section_map = response_as_schema(prompt, SectionTaxonomy, reasoning_effort="medium")
    state.section_map = section_map.model_dump()
    return {"section_map": state.section_map}


def build_narrative(state: AgentState) -> Dict[str, Any]:
    """Build an executive narrative spine for the proposal."""
    prompt = EXEC_NARRATIVE_PROMPT.format(
        understanding_json=state.understanding.model_dump() if state.understanding else {},
        rag_context=state.rag_context or "",
    )
    narrative = response_as_schema(prompt, ExecutiveNarrative, reasoning_effort="high")
    state.narrative = narrative
    return {"narrative": narrative}


def derive_sections(state: AgentState) -> Dict[str, Any]:
    """Back-compat wrapper for older graph wiring."""
    return classify_sections(state)


def _tight_id(text: str) -> str:
    """Create a stable, safe identifier from free-form text."""
    t = (text or "").strip().lower()
    t = t.replace("—", "-").replace("→", "-")
    # Keep letters, numbers, underscore, hyphen, and space.
    # IMPORTANT: Put '-' at the end of the character class to avoid any chance
    # of it being interpreted as a range (some environments accidentally ended
    # up with patterns like "\\-" which can break).
    t = re.sub(r"[^a-z0-9_ -]+", "", t)
    t = t.replace(" ", "_")
    return t[:64] if len(t) > 64 else t


# Curated technology vocabulary used to ground diagram prompts when the model
# did not populate `key_technologies` explicitly.
_TECH_KEYWORDS = [
    "kubernetes", "aks", "eks", "gke", "openshift", "docker", "helm", "kustomize",
    "terraform", "ansible", "jenkins", "github actions", "gitlab", "argocd",
    "postgresql", "postgres", "mysql", "mariadb", "oracle", "sql server", "mongodb",
    "dynamodb", "cosmos db", "cassandra", "redis", "memcached",
    "kafka", "rabbitmq", "sqs", "event hub", "pub/sub", "kinesis",
    "elasticsearch", "opensearch", "solr",
    "datadog", "prometheus", "grafana", "splunk", "new relic", "appdynamics",
    "snowflake", "databricks", "spark", "airflow", "dbt", "tableau", "power bi",
    "aws", "azure", "gcp", "google cloud", "lambda", "fargate", "ec2", "s3",
    "react", "angular", "node.js", "spring", "django", ".net", "java", "python",
    "graphql", "rest", "grpc", "okta", "auth0", "keycloak", "active directory",
]


def _extract_tech_terms(understanding: Optional[RFPUnderstanding], limit: int = 6) -> List[str]:
    """Return technologies mentioned in the RFP.

    Prefers the model-extracted `key_technologies`; falls back to scanning the
    summary and requirement text for a curated keyword vocabulary.
    """
    if understanding is None:
        return []
    explicit = [t.strip() for t in (getattr(understanding, "key_technologies", []) or []) if (t or "").strip()]
    if explicit:
        # De-duplicate while preserving order.
        seen: set[str] = set()
        out: List[str] = []
        for t in explicit:
            k = t.lower()
            if k not in seen:
                seen.add(k)
                out.append(t)
        return out[:limit]

    corpus = (getattr(understanding, "summary", "") or "").lower()
    for r in getattr(understanding, "requirements", []) or []:
        corpus += " " + (getattr(r, "text", "") or "").lower()

    found: List[str] = []
    for kw in _TECH_KEYWORDS:
        if kw not in corpus:
            continue
        # Skip near-duplicates where one term contains another (e.g. postgres/postgresql).
        if any(kw in f or f in kw for f in found):
            continue
        found.append(kw)
        if len(found) >= limit:
            break
    return found


def _diagram_context(understanding: Optional[RFPUnderstanding]) -> str:
    """Build a short, grounded context string for diagram prompts."""
    if understanding is None:
        return ""
    customer = getattr(understanding, "customer_name", None) or "the client"
    techs = _extract_tech_terms(understanding)
    parts = [f"Client: {customer}."]
    if techs:
        parts.append("Reference these named technologies where relevant: " + ", ".join(techs) + ".")
    return " ".join(parts)


_SAFE_MARGIN_NOTE = (
    "Style: consulting-grade, white background, readable 14pt+ labels, minimal clutter, "
    "no logos, no gradients, no sketch effects. Use labeled boxes with directional arrows. "
    "Keep all text and shapes inside a 5–8% safe margin; do not place content at the edges."
)


def _build_diagram_prompt(kind: str, understanding: Optional[RFPUnderstanding]) -> str:
    """Create a context-rich diagram prompt grounded in the RFP.

    `kind` is one of: architecture, delivery, timeline, team, solution.
    """
    ctx = _diagram_context(understanding)
    techs = _extract_tech_terms(understanding)
    tech_clause = (" featuring " + ", ".join(techs)) if techs else ""

    if kind == "architecture":
        body = (
            f"Create a target architecture diagram{tech_clause}. Show layered components "
            "(presentation, services/APIs, data stores, messaging, observability), the key "
            "integrations between them, and primary data flows with directional arrows. "
            "Group related services and label each box clearly."
        )
    elif kind == "delivery":
        body = (
            "Create a delivery & governance diagram showing client and vendor roles, a steering "
            "committee, delivery squads, and escalation/reporting cadence with directional arrows."
        )
    elif kind == "timeline":
        body = (
            "Create a horizontal phased roadmap with 4–6 phases (Mobilize, Discovery & Design, "
            "Build & Integrate, Test & Launch, Hypercare). Show milestones and rough durations along a timeline."
        )
    elif kind == "team":
        body = (
            "Create a team org chart with 6–10 roles (Engagement Lead, Solution Architect, "
            "Delivery Lead/PM, Tech Leads, QA Lead, Security/Data SME) showing reporting lines "
            "and the client interface."
        )
    elif kind == "solution":
        body = (
            f"Create a target solution overview{tech_clause}. Show the major building blocks as "
            "labeled layers (platform, application services, data, integration, observability) and "
            "how value flows across them with directional arrows."
        )
    else:
        body = "Create a clear, professional consulting diagram with labeled boxes and directional arrows."

    return f"{ctx}\n{body}\n{_SAFE_MARGIN_NOTE}".strip()


def _appendix_arch_diagram(view_name: str, understanding: RFPUnderstanding) -> DiagramSpec:
    """Create a DiagramSpec for appendix architecture deep dives."""
    customer = getattr(understanding, "customer_name", None) or "Customer"
    # Some schema versions don't include 'goal' – keep this defensive.
    goal = getattr(understanding, "goal", None) or getattr(understanding, "business_objective", None) or ""
    context = f"Customer: {customer}. Goal: {str(goal).strip()[:240]}."
    prompt = (
        f"Create a clear, professional {view_name} architecture diagram.\n"
        f"{context}\n"
        "Style: consulting-grade, readable labels, minimal clutter, white background.\n"
        "Use 6–12 nodes max, grouped, with directional arrows.\n"
        "Add a short title inside the diagram.\n"
        "Keep all text and shapes inside a 5–8% safe margin; do not place content at the edges.\n"
    )
    return DiagramSpec(
        prompt=prompt,
        approved=False,
        image_path=None,
    )


def _exec_summary_diagram_prompt(slide: SlideSpec) -> str:
    """Create a high-quality prompt for the Executive Summary 3-card graphic."""
    bullets = [b for b in (slide.bullets or []) if (b or "").strip()]
    # Map up to 3 bullets to the three cards; fall back to generic hints if missing.
    body_a = bullets[0] if len(bullets) > 0 else "Key opportunity and client context"
    body_b = bullets[1] if len(bullets) > 1 else "Recommended approach and solution highlights"
    body_c = bullets[2] if len(bullets) > 2 else "Expected business outcomes and impact"

    return (
        "Design a clean, consulting-style Executive Summary graphic with three equal cards.\n"
        "Cards (left to right) titled: OPPORTUNITY, RECOMMENDATION, BUSINESS IMPACT.\n"
        "Use the following body text exactly (no lorem ipsum, no placeholders):\n"
        f"- OPPORTUNITY: {body_a}\n"
        f"- RECOMMENDATION: {body_b}\n"
        f"- BUSINESS IMPACT: {body_c}\n"
        "Style: white background, subtle light-gray card borders, blue header bars, "
        "simple sans-serif font, left-aligned text, generous spacing.\n"
        "Do not add extra icons, charts, or decorative elements. No gradients or shadows. "
        "No hand-drawn or sketch effects. Keep text crisp and readable.\n"
        "Keep all text and shapes inside a 5–8% safe margin; do not place content at the edges.\n"
        "Output a single slide-like image sized for 16:9."
    )


def _first_sentence(text: str, max_len: int = 200) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    first_line = t.splitlines()[0].strip()
    parts = first_line.split(". ")
    sent = parts[0].strip()
    if len(sent) > max_len:
        return sent[: max_len - 1].rstrip() + "…"
    return sent


def _clip(text: str, max_len: int = 160) -> str:
    t = (text or "").strip().rstrip(".")
    if len(t) > max_len:
        return t[: max_len - 1].rstrip() + "…"
    return t


def _exec_summary_bullets(
    understanding: RFPUnderstanding | None,
    narrative: "ExecutiveNarrative | None" = None,
) -> List[str]:
    """Build the Executive Summary as a *win thesis*, never RFP logistics.

    Priority order:
      1. The narrative spine (value proposition, strategic outcomes, themes).
      2. A grounded fallback derived from the RFP understanding's summary.

    This function deliberately never emits submission deadlines, question due
    dates, or other proposal logistics as Executive Summary bullets.
    """
    # --- Preferred: narrative-driven win thesis ---
    if narrative is not None:
        pts = [p.strip() for p in (getattr(narrative, "executive_summary_points", []) or []) if (p or "").strip()]
        if len(pts) >= 3:
            return [_clip(p) for p in pts[:3]]

        bullets: List[str] = []
        vp = (getattr(narrative, "value_proposition", "") or "").strip()
        if vp:
            bullets.append(_clip(vp))
        for o in getattr(narrative, "strategic_outcomes", []) or []:
            if (o or "").strip():
                bullets.append(_clip(o))
            if len(bullets) >= 3:
                break
        if len(bullets) < 3:
            for t in getattr(narrative, "solution_themes", []) or []:
                if (t or "").strip():
                    bullets.append(_clip(t))
                if len(bullets) >= 3:
                    break
        if len(bullets) >= 3:
            return bullets[:3]
        if bullets:
            # Top up with the understanding-derived situation line if needed.
            pass
    else:
        bullets = []

    # --- Fallback: grounded, outcome-oriented (no logistics) ---
    if understanding is None and not bullets:
        return [
            "Client opportunity and the outcomes that matter most",
            "Our recommended approach, tailored to the client's priorities",
            "Business impact: faster delivery, reduced risk, measurable value",
        ]

    summary = _first_sentence(getattr(understanding, "summary", "") or "") if understanding else ""
    customer = (getattr(understanding, "customer_name", None) or "").strip() if understanding else ""
    opp = (getattr(understanding, "opportunity_title", None) or "").strip() if understanding else ""

    situation = summary
    if not situation:
        if customer and opp:
            situation = f"{opp} for {customer}"
        elif opp:
            situation = opp
        elif customer:
            situation = f"Strategic opportunity for {customer}"
        else:
            situation = "Client opportunity and the outcomes that matter most"

    approach = (
        f"Recommended approach tailored to {customer}'s priorities"
        if customer
        else "Our recommended approach, tailored to the client's priorities"
    )
    impact = "Business impact: faster delivery, reduced risk, measurable value"

    fallback = [_clip(situation), approach, impact]
    # Merge any partial narrative bullets first, then fill from fallback.
    merged = list(bullets)
    for b in fallback:
        if len(merged) >= 3:
            break
        merged.append(b)
    return merged[:3]


def _is_placeholder_exec_bullets(bullets: Optional[List[str]]) -> bool:
    if not bullets:
        return True
    norm = [b.strip().lower() for b in bullets if (b or "").strip()]
    placeholder = [
        "opportunity and key objectives",
        "our recommended approach and solution highlights",
        "business impact and expected outcomes",
    ]
    return norm[:3] == placeholder


def _is_archetype_present(existing_archetypes: set, target: str) -> bool:
    """Fuzzy archetype match to avoid near-duplicate auto-added slides.

    Catches variants like 'Customer Context' vs 'context' vs 'current state'.
    """
    target_lower = (target or "").lower()
    for a in existing_archetypes:
        a = (a or "").lower()
        if not a:
            continue
        if target_lower == a or target_lower in a or a in target_lower:
            return True
    return False


def ensure_required_slides(
    deck_plan: DeckPlan,
    understanding: RFPUnderstanding | None = None,
    narrative: "ExecutiveNarrative | None" = None,
) -> DeckPlan:
    """Ensure required slides exist. Adds missing ones with sensible defaults."""
    existing = {(s.archetype or "").lower(): s for s in deck_plan.slides}
    existing_keys = set(existing.keys())
    # Helper: add slide
    def add_slide(
        archetype: str,
        title: str,
        bullets: Optional[List[str]] = None,
        diagram: Optional[DiagramSpec] = None,
    ) -> None:
        deck_plan.slides.append(
            SlideSpec(
                slide_id=f"auto_{_tight_id(title)}",
                title=title,
                archetype=archetype,
                bullets=bullets or [],
                diagram=diagram,
            )
        )
        existing_keys.add((archetype or "").lower())

    # Title
    if not _is_archetype_present(existing_keys, "title"):
        add_slide(
            "Title",
            deck_plan.deck_title or "Proposal",
            bullets=[],
        )

    # Agenda
    if not _is_archetype_present(existing_keys, "agenda"):
        add_slide(
            "Agenda",
            "Agenda",
            bullets=[
                "Executive Summary",
                "Our Understanding",
                "Proposed Solution",
                "Architecture",
                "Delivery Plan & Timeline",
                "Commercials",
                "Team",
                "Next Steps",
            ],
        )

    # Exec Summary as Solution Overview (required)
    # (Some models output “Executive Overview” etc; ordering will still handle it.)
    has_exec = any(_is_exec_summary(s) for s in deck_plan.slides)
    if not has_exec:
        add_slide(
            "Solution Overview",
            "Executive Summary",
            bullets=_exec_summary_bullets(understanding, narrative),
            diagram=None,
        )
    else:
        for s in deck_plan.slides:
            if _is_exec_summary(s) and _is_placeholder_exec_bullets(s.bullets):
                s.bullets = _exec_summary_bullets(understanding, narrative)

    # Customer Context
    if not _is_archetype_present(existing_keys, "customer context"):
        add_slide(
            "Customer Context",
            "Current State & Context",
            bullets=[
                "Current environment and constraints",
                "Key stakeholder needs and pain points",
                "Why change / why now",
            ],
        )

    # Requirements
    if not _is_archetype_present(existing_keys, "requirements"):
        add_slide(
            "Requirements",
            "Requirements Summary",
            bullets=[
                "Functional requirements (high level)",
                "Non-functional requirements (security, performance, availability)",
                "Success criteria and acceptance",
            ],
        )

    # Architecture
    if not _is_archetype_present(existing_keys, "architecture"):
        add_slide(
            "Architecture",
            "Target Architecture Overview",
            bullets=[
                "High-level component view and integrations",
                "Data flows and key interfaces",
                "Security and compliance considerations",
            ],
            diagram=DiagramSpec(
                kind="architecture",
                prompt=_build_diagram_prompt("architecture", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Delivery Plan
    if not _is_archetype_present(existing_keys, "delivery plan"):
        add_slide(
            "Delivery Plan",
            "Delivery Model & Governance",
            bullets=[
                "Delivery approach (phased, agile, hybrid)",
                "Governance and stakeholder engagement",
                "Quality assurance and reporting cadence",
            ],
            diagram=DiagramSpec(
                kind="process",
                prompt=_build_diagram_prompt("delivery", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Timeline
    if not _is_archetype_present(existing_keys, "timeline"):
        add_slide(
            "Timeline",
            "Roadmap & Timeline",
            bullets=[
                "Phase 0: Mobilization",
                "Phase 1: Discovery & Design",
                "Phase 2: Build & Integrate",
                "Phase 3: Test & Launch",
                "Phase 4: Hypercare & Transition",
            ],
            diagram=DiagramSpec(
                kind="timeline",
                prompt=_build_diagram_prompt("timeline", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Risks
    if not _is_archetype_present(existing_keys, "risks"):
        add_slide(
            "Risks",
            "Risks & Mitigations",
            bullets=[
                "Key delivery and technical risks",
                "Mitigation actions and owners",
                "Assumptions and dependencies",
            ],
        )

    # Team
    if not _is_archetype_present(existing_keys, "team"):
        add_slide(
            "Team",
            "Proposed Team",
            bullets=[
                "Engagement Lead — governance and stakeholder alignment",
                "Solution Architect — end-to-end design and quality",
                "Delivery Lead / PM — plan, cadence, RAID management",
                "Tech Lead(s) — build and integration",
                "QA Lead — test strategy and execution",
                "Data / Security SME — compliance and data controls",
            ],
            diagram=DiagramSpec(
                kind="org",
                prompt=_build_diagram_prompt("team", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Commercials (always)
    if not _is_archetype_present(existing_keys, "commercials"):
        add_slide(
            "Commercials",
            "Commercials & Pricing",
            bullets=[
                "Commercial model options (T&M / Fixed / Hybrid)",
                "Assumptions and exclusions",
                "Options to accelerate timeline or reduce risk",
            ],
        )

    # Next Steps
    if not _is_archetype_present(existing_keys, "next steps"):
        add_slide(
            "Next Steps",
            "Next Steps",
            bullets=[
                "Confirm scope and success criteria",
                "Align on plan, governance, and resourcing",
                "Kick-off and mobilization",
            ],
        )

    return deck_plan


# --------------------------
# Ordering helpers
# --------------------------
def _is_exec_summary(slide: SlideSpec) -> bool:
    t = (getattr(slide, "title", "") or "").strip().lower()
    # Models sometimes emit variants like "Executive Overview" or "Summary & Recommendation".
    if t == "executive summary" or t.startswith("executive summary"):
        return True
    return (
        "executive overview" in t
        or "summary & recommendation" in t
        or "summary and recommendation" in t
        or ("executive" in t and "summary" in t)
    )


def order_deck(deck_plan: DeckPlan) -> DeckPlan:
    """Order slides into a consulting-style narrative. Keeps relative order within an archetype."""
    order = [
        "Title",
        "Agenda",
        "Solution Overview",  # Exec Summary lives here
        "Customer Context",
        "Requirements",
        "Architecture",
        "Delivery Plan",
        "Timeline",
        "Risks",
        "Case Studies",
        "Team",
        "Commercials",
        "Next Steps",
        "Content",
    ]
    rank = {a.lower(): i for i, a in enumerate(order)}

    def section_priority(slide: SlideSpec) -> int:
        # Within Solution Overview, force Exec Summary first.
        if (slide.archetype or "").lower() == "solution overview":
            return 0 if _is_exec_summary(slide) else 1
        # For Customer Context: prefer "Current State" before generic.
        if (slide.archetype or "").lower() == "customer context":
            t = (slide.title or "").lower()
            return 0 if "current" in t or "context" in t else 1
        return 0

    indexed = list(enumerate(deck_plan.slides))
    indexed.sort(
        key=lambda ix: (
            rank.get((ix[1].archetype or "").lower(), 999),
            section_priority(ix[1]),
            ix[0],
        )
    )
    deck_plan.slides = [s for _, s in indexed]
    return deck_plan


def polish_deck_text(deck_plan: DeckPlan) -> DeckPlan:
    """Light text normalization for a cleaner consulting tone."""
    for s in deck_plan.slides:
        if not s.bullets:
            continue
        new_bullets = []
        for b in s.bullets:
            t = (b or "").strip()
            t = t.replace("  ", " ")
            t = t.rstrip(".")
            if t:
                new_bullets.append(t)
        s.bullets = new_bullets[:8]  # keep crisp
    return deck_plan


def ensure_diagrams_for_key_slides(deck_plan: DeckPlan, understanding: RFPUnderstanding | None = None) -> DeckPlan:
    """Ensure diagrams exist (as guarded approvals) for key slides."""
    for s in deck_plan.slides:
        arch = (s.archetype or "").lower()
        title = (s.title or "").lower()

        if arch in {"architecture", "delivery plan", "timeline", "team", "solution overview"}:
            if s.diagram is None:
                prompt = ""
                kind = "generic"
                if arch == "architecture":
                    prompt = _build_diagram_prompt("architecture", understanding)
                    kind = "architecture"
                elif arch == "delivery plan":
                    prompt = _build_diagram_prompt("delivery", understanding)
                    kind = "process"
                elif arch == "timeline":
                    prompt = _build_diagram_prompt("timeline", understanding)
                    kind = "timeline"
                elif arch == "team":
                    prompt = _build_diagram_prompt("team", understanding)
                    kind = "org"
                elif arch == "solution overview":
                    if _is_exec_summary(s):
                        # Exec Summary is better rendered as native PPTX text, not an image.
                        prompt = ""
                    else:
                        prompt = _build_diagram_prompt("solution", understanding)
                        kind = "generic"

                if prompt:
                    s.diagram = DiagramSpec(kind=kind, prompt=prompt, approved=False, image_path=None)
            elif arch == "solution overview" and _is_exec_summary(s):
                # Remove any legacy Exec Summary diagram to keep it text-native.
                s.diagram = None

        # Appendix architecture deep dives (if present as Content slides)
        if "appendix" in title and understanding is not None:
            if s.diagram is None:
                s.diagram = _appendix_arch_diagram("Appendix Overview", understanding)

    return deck_plan


def build_traceability(state: AgentState) -> Dict[str, Any]:
    """Build a traceability report mapping requirements to slides."""
    if state.understanding is None or state.deck_plan is None:
        return {"traceability_report": None}
    report = build_traceability_report(
        understanding=state.understanding,
        deck=state.deck_plan,
    )
    state.traceability_report = report
    return {"traceability_report": report}


def qa_and_report(state: AgentState) -> Dict[str, Any]:
    """Back-compat wrapper that stores report under the expected key."""
    if state.understanding is None or state.deck_plan is None:
        return {"report": None}
    report = build_traceability_report(
        understanding=state.understanding,
        deck=state.deck_plan,
    )
    state.report = report
    return {"report": report}


def plan_deck(state: AgentState) -> Dict[str, Any]:
    """Plan a deck from RFP + optional RAG context."""
    template_info = state.template_info or {}
    layout_names = template_info.get("slide_layout_names", [])
    placeholder_map = template_info.get("placeholder_map", {})
    understanding_json = (
        state.understanding.model_dump() if state.understanding is not None else {}
    )
    narrative_json = state.narrative.model_dump() if state.narrative is not None else {}

    prompt = DECK_PLAN_V2_PROMPT.format(
        layout_names=layout_names,
        placeholder_map=placeholder_map,
        rag_context=state.rag_context or "",
        understanding_json=understanding_json,
        narrative_json=narrative_json,
    )

    deck_plan = response_as_schema(prompt, DeckPlan, reasoning_effort="high")

    deck_plan = ensure_required_slides(
        deck_plan, understanding=state.understanding, narrative=state.narrative
    )
    deck_plan = order_deck(deck_plan)
    deck_plan = polish_deck_text(deck_plan)
    deck_plan = ensure_diagrams_for_key_slides(deck_plan, understanding=state.understanding)

    state.deck_plan = deck_plan
    return {"deck_plan": deck_plan}


def compress_bullets(state: AgentState) -> Dict[str, Any]:
    """Editorial pass: tighten bullets to executive-grade language.

    Runs the SLIDE_COMPRESSION_PROMPT and merges only the rewritten bullets back
    by slide_id, preserving slide order, diagrams, tables, and traceability. Any
    failure leaves the original deck untouched.
    """
    if state.deck_plan is None:
        return {"deck_plan": None}
    try:
        prompt = SLIDE_COMPRESSION_PROMPT.format(
            deck_plan_json=state.deck_plan.model_dump()
        )
        compressed = response_as_schema(prompt, DeckPlan, reasoning_effort="medium")
        by_id = {s.slide_id: s for s in compressed.slides}
        for s in state.deck_plan.slides:
            cs = by_id.get(s.slide_id)
            if cs and cs.bullets:
                s.bullets = cs.bullets
    except Exception:
        # Compression is a best-effort polish; never fail the pipeline over it.
        pass
    return {"deck_plan": state.deck_plan}


def run(state: AgentState) -> Dict[str, Any]:
    """Entry point used by the LangGraph pipeline."""
    understand_rfp(state)
    classify_sections(state)
    build_narrative(state)
    plan_deck(state)
    compress_bullets(state)
    build_traceability(state)
    return {
        "understanding": state.understanding,
        "deck_plan": state.deck_plan,
        "traceability_report": state.traceability_report,
        "section_map": state.section_map,
    }
