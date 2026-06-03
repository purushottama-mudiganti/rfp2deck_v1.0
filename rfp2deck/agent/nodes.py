from __future__ import annotations

import functools
import re
import time
from typing import Any, Callable, Dict, List, Optional

from rfp2deck.core.logging import get_logger
from rfp2deck.agent.prompts import (
    DECK_PLAN_V2_PROMPT,
    EXEC_NARRATIVE_PROMPT,
    RFP_UNDERSTAND_PROMPT,
    SECTION_TAXONOMY_PROMPT,
    SLIDE_COMPRESSION_PROMPT,
    SPEAKER_NOTES_PROMPT,
)
from rfp2deck.agent.state import AgentState
from rfp2deck.core.config import settings
from rfp2deck.core.schemas import (
    BulletPoint,
    DeckNotes,
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

log = get_logger(__name__)


def _logged_node(func: Callable[[AgentState], Dict[str, Any]]) -> Callable[[AgentState], Dict[str, Any]]:
    """Wrap a graph node so its start, duration, and failures are logged.

    LangGraph nodes otherwise run silently; this surfaces which node was active
    when an error (e.g. an LLM timeout) occurred and how long each step took.
    """

    @functools.wraps(func)
    def wrapper(state: AgentState) -> Dict[str, Any]:
        log.info("Node START: %s", func.__name__)
        start = time.perf_counter()
        try:
            result = func(state)
        except Exception:
            elapsed = time.perf_counter() - start
            log.exception("Node FAILED: %s after %.1fs", func.__name__, elapsed)
            raise
        elapsed = time.perf_counter() - start
        log.info("Node DONE: %s in %.1fs", func.__name__, elapsed)
        return result

    return wrapper


@_logged_node
def understand_rfp(state: AgentState) -> Dict[str, Any]:
    """Extract a structured understanding of the RFP."""
    prompt = RFP_UNDERSTAND_PROMPT.format(
        rfp_text=state.rfp_text or "",
        rag_context=state.rag_context or "",
    )
    understanding = response_as_schema(prompt, RFPUnderstanding, reasoning_effort="high")
    state.understanding = understanding
    return {"understanding": understanding}


@_logged_node
def classify_sections(state: AgentState) -> Dict[str, Any]:
    """Classify RFP into section taxonomy for better subtitle generation & narrative."""
    prompt = SECTION_TAXONOMY_PROMPT.format(
        rfp_text=state.rfp_text or "",
        rag_context=state.rag_context or "",
    )
    section_map = response_as_schema(prompt, SectionTaxonomy, reasoning_effort="medium")
    state.section_map = section_map.model_dump()
    return {"section_map": state.section_map}


@_logged_node
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


def _context_detailed_points(understanding: RFPUnderstanding | None) -> List[BulletPoint]:
    """Build grounded sub-points for the Customer Context slide.

    Turns the three generic context headers into substantive points by hanging
    concrete sub-points off each, drawn from the RFP understanding (assumptions,
    requirements, risks). Returns [] when there is nothing grounded to say, so
    the caller can fall back to flat bullets.
    """
    if understanding is None:
        return []

    points: List[BulletPoint] = []

    # Current environment & constraints — from stated assumptions, else summary.
    env_subs = [_clip(a, 120) for a in (getattr(understanding, "assumptions", []) or []) if (a or "").strip()][:3]
    if not env_subs:
        first = _first_sentence(getattr(understanding, "summary", "") or "", 140)
        if first:
            env_subs = [first]
    if env_subs:
        points.append(BulletPoint(text="Current environment and constraints", sub_points=env_subs))

    # Stakeholder needs & pain points — from must/should requirements.
    reqs = getattr(understanding, "requirements", []) or []
    ranked = sorted(reqs, key=lambda r: {"must": 0, "should": 1, "may": 2}.get(getattr(r, "priority", "should"), 1))
    need_subs = [_clip(getattr(r, "text", ""), 120) for r in ranked if (getattr(r, "text", "") or "").strip()][:4]
    if need_subs:
        points.append(BulletPoint(text="Stakeholder needs and pain points", sub_points=need_subs))

    # Why change now — from stated risks/pressures driving the initiative.
    why_subs = [_clip(r, 120) for r in (getattr(understanding, "risks", []) or []) if (r or "").strip()][:3]
    if why_subs:
        points.append(BulletPoint(text="Why change now", sub_points=why_subs))

    return points


def _requirements_detailed_points(understanding: RFPUnderstanding | None) -> List[BulletPoint]:
    """Group RFP requirements into functional vs non-functional sub-points."""
    if understanding is None:
        return []
    reqs = getattr(understanding, "requirements", []) or []
    if not reqs:
        return []
    nf_kw = (
        "security", "performance", "availability", "compliance", "scalab", "latency",
        "throughput", "uptime", "encrypt", "privacy", "resilien", "audit", "sla",
    )
    functional: List[str] = []
    nonfunctional: List[str] = []
    for r in reqs:
        text = (getattr(r, "text", "") or "").strip()
        if not text:
            continue
        target = nonfunctional if any(k in text.lower() for k in nf_kw) else functional
        target.append(_clip(text, 120))

    points: List[BulletPoint] = []
    if functional:
        points.append(BulletPoint(text="Functional scope", sub_points=functional[:4]))
    if nonfunctional:
        points.append(BulletPoint(text="Non-functional requirements", sub_points=nonfunctional[:4]))
    return points


def _next_steps_bullets(understanding: RFPUnderstanding | None) -> List[str]:
    """Supplier-driven calls to action for the customer.

    Next Steps must express what WE (the supplier) propose the customer does
    next to move the engagement forward — never proposal logistics (submission
    dates, RFP numbers) or a restatement of what the customer wants.
    """
    customer = (getattr(understanding, "customer_name", None) or "").strip() if understanding else ""
    who = customer or "your team"
    return [
        f"Schedule a solution deep-dive workshop with {who}",
        "Confirm priority use cases and success criteria for Phase 1",
        "Align on governance, delivery model, and key stakeholders",
        "Agree commercial model and contracting approach",
        "Approve mobilization to begin within two weeks of award",
    ]


# Phrases that don't belong on a supplier's Next Steps slide: proposal logistics,
# the customer's own evaluation activities, or restatements of what the customer wants.
_NON_ACTION_MARKERS = (
    # Proposal / bid logistics
    "proposal due", "submission deadline", "due date", "questions due", "q&a due",
    "rfp ref", "rfp number", "rfp no", "bid due", "closing date", "deadline for",
    "submit by", "submission date", "clarification question",
    # Customer-side evaluation activities (not OUR next step)
    "evaluat", "shortlist", "award decision", "select a vendor", "vendor selection",
    "vendor response", "scoring", "score the", "review responses",
    # Restating what the customer wants
    "customer want", "client want", "they want", "customer would like",
    "client would like", "customer needs", "client needs", "customer requires",
)


def _sanitize_next_steps(slide: SlideSpec, understanding: RFPUnderstanding | None) -> None:
    """Keep a Next Steps slide to supplier-driven calls to action.

    Drops proposal logistics, customer-side evaluation activities, and
    restatements of what the customer wants. If too little actionable content
    remains, replace it with supplier-driven calls to action.
    """
    cleaned: List[str] = []
    for b in (slide.bullets or []):
        low = (b or "").strip().lower()
        if not low:
            continue
        if any(m in low for m in _NON_ACTION_MARKERS):
            continue
        cleaned.append(b)
    if len(cleaned) < 3:
        cleaned = _next_steps_bullets(understanding)
    slide.bullets = cleaned
    # Next Steps reads best as a clean action list, not nested detail.
    slide.detailed_points = []


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
        detailed_points: Optional[List[BulletPoint]] = None,
    ) -> None:
        deck_plan.slides.append(
            SlideSpec(
                slide_id=f"auto_{_tight_id(title)}",
                title=title,
                archetype=archetype,
                bullets=bullets or [],
                detailed_points=detailed_points or [],
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
        ctx_points = _context_detailed_points(understanding)
        add_slide(
            "Customer Context",
            "Current State & Context",
            bullets=[] if ctx_points else [
                "Current environment and constraints",
                "Key stakeholder needs and pain points",
                "Why change / why now",
            ],
            detailed_points=ctx_points,
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

    # Next Steps — supplier-driven calls to action (not proposal logistics).
    if not _is_archetype_present(existing_keys, "next steps"):
        add_slide(
            "Next Steps",
            "Recommended Next Steps",
            bullets=_next_steps_bullets(understanding),
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


def enrich_slide_detail(deck_plan: DeckPlan, understanding: RFPUnderstanding | None = None) -> DeckPlan:
    """Upgrade thin content slides with grounded sub-points and fix Next Steps.

    Applies to both model-generated and auto-added slides:
      - Customer Context / "Current State" slides lacking sub-points get
        substantive `detailed_points` drawn from the RFP understanding.
      - Requirements slides get functional vs non-functional sub-points.
      - Next Steps slides are sanitized of proposal logistics and, if needed,
        replaced with supplier-driven calls to action.
    """
    for s in deck_plan.slides:
        arch = (s.archetype or "").lower()
        title = (s.title or "").lower()

        if arch == "next steps" or "next step" in title:
            _sanitize_next_steps(s, understanding)
            continue

        # Skip slides that already carry structured detail.
        if getattr(s, "detailed_points", None):
            continue

        is_context = arch == "customer context" or "current state" in title or (
            "context" in title and arch not in {"title", "agenda"}
        )
        is_requirements = arch == "requirements" or "requirement" in title

        if is_context:
            pts = _context_detailed_points(understanding)
            if pts:
                s.detailed_points = pts
                s.bullets = []  # detailed_points supersede flat bullets in the renderer
        elif is_requirements:
            pts = _requirements_detailed_points(understanding)
            if pts:
                s.detailed_points = pts
                s.bullets = []

    return deck_plan


def polish_deck_text(deck_plan: DeckPlan) -> DeckPlan:
    """Light text normalization for a cleaner consulting tone."""
    def _clean(text: str) -> str:
        t = (text or "").strip()
        t = t.replace("  ", " ")
        t = t.rstrip(".")
        return t

    for s in deck_plan.slides:
        if s.bullets:
            s.bullets = [c for c in (_clean(b) for b in s.bullets) if c][:8]
        # Normalize nested points too, preserving their structure.
        if getattr(s, "detailed_points", None):
            cleaned_points = []
            for point in s.detailed_points:
                text = _clean(getattr(point, "text", ""))
                if not text:
                    continue
                subs = [c for c in (_clean(sp) for sp in (point.sub_points or [])) if c][:5]
                point.text = text
                point.sub_points = subs
                cleaned_points.append(point)
            s.detailed_points = cleaned_points[:6]
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


@_logged_node
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


@_logged_node
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
    deck_plan = enrich_slide_detail(deck_plan, understanding=state.understanding)
    deck_plan = polish_deck_text(deck_plan)
    deck_plan = ensure_diagrams_for_key_slides(deck_plan, understanding=state.understanding)

    state.deck_plan = deck_plan
    return {"deck_plan": deck_plan}


@_logged_node
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
        # Compression is a best-effort polish; never fail the pipeline over it,
        # but record why it was skipped so silent quality regressions are visible.
        log.warning("Bullet compression skipped due to error; keeping original deck.", exc_info=True)
    return {"deck_plan": state.deck_plan}


def _fallback_note(slide: SlideSpec) -> str:
    """Build a simple, deterministic speaker note from the slide's own content.

    Used when the LLM notes pass is disabled or fails, so every slide still
    carries at least basic talking points.
    """
    parts: List[str] = []
    title = (slide.title or "").strip()
    if title:
        parts.append(f'This slide covers "{title}."')

    points: List[str] = []
    if getattr(slide, "detailed_points", None):
        for p in slide.detailed_points:
            if (p.text or "").strip():
                points.append(p.text.strip())
            for sp in (p.sub_points or []):
                if (sp or "").strip():
                    points.append(sp.strip())
    else:
        points = [b.strip() for b in (slide.bullets or []) if (b or "").strip()]

    if points:
        parts.append("Walk the audience through each point: " + "; ".join(points[:6]) + ".")
        parts.append(
            "For each, explain why it matters to the client and connect it to their priorities."
        )
    return " ".join(parts).strip()


# Archetypes that are self-explanatory and don't warrant speaker notes.
_NO_NOTES_ARCHETYPES = {"title", "agenda", "next steps"}


def _slide_wants_notes(slide: SlideSpec) -> bool:
    """Whether a slide should carry speaker notes (skip cover/agenda/next-steps)."""
    arch = (slide.archetype or "").lower()
    title = (slide.title or "").lower()
    if arch in _NO_NOTES_ARCHETYPES:
        return False
    if "next step" in title or title == "agenda":
        return False
    return True


@_logged_node
def generate_notes(state: AgentState) -> Dict[str, Any]:
    """Generate presenter speaker notes for content slides.

    Best-effort: a single fast LLM pass writes notes that unpack the reasoning
    behind each slide so a human can present it confidently. Any slide the model
    misses (or the whole pass, if it fails) falls back to deterministic notes
    derived from the slide content. Self-explanatory slides (Title, Agenda, Next
    Steps) are skipped. Never fails the pipeline.
    """
    if state.deck_plan is None:
        return {"deck_plan": None}

    slides = state.deck_plan.slides
    notes_slides = [s for s in slides if _slide_wants_notes(s)]
    notes_by_id: Dict[str, str] = {}

    if notes_slides and getattr(state, "enable_notes", True):
        try:
            compact = [
                {
                    "slide_id": s.slide_id,
                    "title": s.title,
                    "archetype": s.archetype,
                    "bullets": s.bullets,
                    "detailed_points": [
                        {"text": p.text, "sub_points": p.sub_points}
                        for p in (s.detailed_points or [])
                    ],
                }
                for s in notes_slides
            ]
            prompt = SPEAKER_NOTES_PROMPT.format(
                deck_plan_json=compact,
                narrative_json=state.narrative.model_dump() if state.narrative else {},
                understanding_summary=(
                    getattr(state.understanding, "summary", "") if state.understanding else ""
                )
                or "",
            )
            deck_notes = response_as_schema(
                prompt, DeckNotes, model=settings.model_fast, reasoning_effort="low"
            )
            for n in deck_notes.notes:
                if n.slide_id and (n.notes or "").strip():
                    notes_by_id[n.slide_id] = n.notes.strip()
        except Exception:
            # Notes are an enhancement; never fail the deck over them.
            log.warning(
                "Speaker-notes LLM pass failed; using deterministic fallback notes.",
                exc_info=True,
            )

    for s in slides:
        if not _slide_wants_notes(s):
            s.notes = None
            continue
        note = notes_by_id.get(s.slide_id) or _fallback_note(s)
        s.notes = note or None

    return {"deck_plan": state.deck_plan}
