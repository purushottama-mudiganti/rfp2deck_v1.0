from __future__ import annotations

import functools
import hashlib
import json
import re
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional

from rfp2deck.core.logging import get_logger
from rfp2deck.agent.prompts import (
    DECK_SECTION_EXPANSION_PROMPT,
    DECK_PLAN_V2_PROMPT,
    EXEC_NARRATIVE_PROMPT,
    RFP_UNDERSTAND_PROMPT,
    SECTION_TAXONOMY_PROMPT,
    SLIDE_COMPRESSION_PROMPT,
    SOURCE_EVIDENCE_PROMPT,
    SPEAKER_NOTES_PROMPT,
    TECHNOLOGY_RECOMMENDATION_PROMPT,
    VISUAL_BRIEF_PROMPT,
)
from rfp2deck.agent.evidence import (
    merge_evidence_batches,
    render_evidence_for_prompt,
    split_source_document,
)
from rfp2deck.agent.rfp_focus import build_rfp_focus_guide
from rfp2deck.agent.state import AgentState
from rfp2deck.core.config import settings
from rfp2deck.core.schemas import (
    BulletCompressionSet,
    BulletPoint,
    Card,
    Comparison,
    ComparisonColumn,
    DeckNotes,
    DeckPlan,
    DiagramBrief,
    DiagramBriefSet,
    DiagramSpec,
    EngagementProfile,
    EngagementTypeAssessment,
    ExecutiveNarrative,
    LifecycleStageAssessment,
    RFPUnderstanding,
    SectionTaxonomy,
    SlideSpec,
    TechnologyRecommendation,
    TechnologyRecommendationSet,
    SourceDocument,
    SourceEvidenceBatch,
    TraceabilityReport,
)
from rfp2deck.llm.structured import response_as_schema
from rfp2deck.ingestion.source_package import (
    build_source_reconciliation,
    render_reconciliation_summary,
)
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
def reconcile_sources(state: AgentState) -> Dict[str, Any]:
    """Prepare deterministic source precedence and clarification evidence."""
    reconciliation = build_source_reconciliation(
        state.source_documents,
        state.clarification_records,
    )
    state.source_reconciliation = reconciliation
    return {"source_reconciliation": reconciliation}


def _extract_evidence_chunk(chunk) -> SourceEvidenceBatch:
    document = chunk.document
    if _is_contextual_document(document) and not bool(
        getattr(settings, "understanding_contextual_evidence_llm_enabled", False)
    ):
        return _contextual_evidence_from_source(chunk)
    prompt = SOURCE_EVIDENCE_PROMPT.format(
        document_id=document.document_id,
        document_name=document.name,
        document_type=document.document_type,
        authority=document.authority,
        issue_date=document.issue_date or "unknown",
        chunk_id=chunk.chunk_id,
        source_chunk=chunk.text,
    )
    cache_path = None
    if getattr(settings, "understanding_evidence_cache", False):
        cache_key = hashlib.sha256(
            f"source-evidence-v1|{settings.model_fast}|{prompt}".encode("utf-8")
        ).hexdigest()
        cache_dir = settings.data_dir / "evidence_cache"
        cache_dir.mkdir(parents=True, exist_ok=True)
        cache_path = cache_dir / f"{cache_key}.json"
        if cache_path.exists():
            try:
                evidence = SourceEvidenceBatch.model_validate_json(
                    cache_path.read_text(encoding="utf-8")
                )
                log.info("Evidence cache HIT: %s", chunk.chunk_id)
            except Exception:
                log.warning("Ignoring invalid evidence cache entry: %s", cache_path)
                evidence = None
        else:
            evidence = None
    else:
        evidence = None
    if evidence is None:
        evidence_timeout_s = float(settings.understanding_evidence_timeout_s)
        evidence_grace_s = float(
            getattr(settings, "understanding_evidence_grace_s", 60.0)
        )
        is_contextual = _is_contextual_document(document)
        if is_contextual:
            evidence_timeout_s = float(
                getattr(
                    settings,
                    "understanding_contextual_evidence_timeout_s",
                    min(evidence_timeout_s, 60.0),
                )
            )
            evidence_grace_s = float(
                getattr(
                    settings,
                    "understanding_contextual_evidence_grace_s",
                    min(evidence_grace_s, 30.0),
                )
            )
        evidence = response_as_schema(
            prompt,
            SourceEvidenceBatch,
            model=settings.model_fast,
            reasoning_effort=settings.reasoning_effort_low,
            timeout_seconds=evidence_timeout_s,
            background_grace_seconds=evidence_grace_s,
            background=False if is_contextual else None,
            recoverable_failure=is_contextual,
        )
        if cache_path is not None:
            try:
                cache_path.write_text(evidence.model_dump_json(), encoding="utf-8")
            except OSError:
                log.warning("Unable to write evidence cache entry: %s", cache_path)
    evidence.source_document_id = document.document_id
    evidence.chunk_id = chunk.chunk_id
    for requirement in evidence.requirements:
        if document.document_id not in requirement.source_document_ids:
            requirement.source_document_ids.append(document.document_id)
        requirement.authority = document.authority
        if requirement.source_ref and not requirement.source_refs:
            requirement.source_refs = [requirement.source_ref]
    return evidence


_CONTEXTUAL_EVIDENCE_TERMS = (
    "architecture", "azure", "aws", "cloud", "platform", "cots", "saas",
    "application", "service", "component", "integration", "interface", "api",
    "event", "file", "data", "database", "storage", "catalog", "master data",
    "mdm", "pricing", "warehouse", "security", "identity", "network",
    "deployment", "runtime", "monitor", "backup", "recovery", "devops",
)


def _is_contextual_document(document: SourceDocument) -> bool:
    return (
        document.document_type == "supporting_reference"
        or document.authority in {"contextual", "non_authoritative"}
    )


def _contextual_evidence_from_source(
    chunk,
    exc: BaseException | None = None,
) -> SourceEvidenceBatch:
    """Preserve bounded excerpts from a non-authoritative source.

    Contextual sources may inform the architecture but cannot create authoritative
    requirements. This therefore emits source text only as context facts, either
    as the normal fast path or as recovery after an explicitly enabled LLM call.
    """
    raw_units = re.split(r"(?:\r?\n)+|(?<=[.!?])\s+(?=[A-Z0-9])", chunk.text or "")
    units: List[str] = []
    seen: set[str] = set()
    for raw in raw_units:
        clean = re.sub(r"\s+", " ", raw or "").strip()
        while clean:
            if len(clean) <= 600:
                piece, clean = clean, ""
            else:
                boundary = clean.rfind(" ", 0, 600)
                boundary = boundary if boundary >= 300 else 600
                piece, clean = clean[:boundary].strip(), clean[boundary:].strip()
            key = piece.lower()
            if piece and key not in seen:
                seen.add(key)
                units.append(piece)

    def signal_score(item: tuple[int, str]) -> tuple[int, int]:
        index, text = item
        lower = text.lower()
        score = sum(1 for term in _CONTEXTUAL_EVIDENCE_TERMS if term in lower)
        return (-score, index)

    selected: List[str] = []
    selected_chars = 0
    for _, text in sorted(enumerate(units), key=signal_score):
        if len(selected) >= 24 or selected_chars + len(text) > 8000:
            continue
        selected.append(text)
        selected_chars += len(text)

    if exc is None:
        log.info(
            "Prepared contextual evidence locally: document=%r chunk=%s chars=%d "
            "excerpts=%d",
            chunk.document.name,
            chunk.chunk_id,
            len(chunk.text or ""),
            len(selected),
        )
    else:
        warning = (
            f"LLM evidence extraction failed for contextual chunk {chunk.chunk_id}; "
            f"preserved {len(selected)} bounded source excerpts instead and continued "
            "without inferring authoritative requirements."
        )
        if warning not in chunk.document.warnings:
            chunk.document.warnings.append(warning)
        log.warning(
            "Recovered contextual evidence without LLM; proposal generation will continue: "
            "document=%r chunk=%s chars=%d excerpts=%d cause=%s",
            chunk.document.name,
            chunk.chunk_id,
            len(chunk.text or ""),
            len(selected),
            type(exc).__name__,
        )
    return SourceEvidenceBatch(
        source_document_id=chunk.document.document_id,
        chunk_id=chunk.chunk_id,
        context_facts=selected,
        summary_points=selected[:6],
    )


def _build_contextual_reference_context(
    documents: List[SourceDocument],
    max_chars: int | None = None,
) -> str:
    """Keep supporting architecture research available to downstream agents.

    Supporting references are intentionally not promoted to requirements.  The
    ordinary understanding reduction can therefore omit their design detail;
    this separate, labelled channel preserves bounded architecture excerpts for
    visual and technology decisions in both the direct and chunked paths.
    """
    references = [document for document in documents if _is_contextual_document(document)]
    if not references:
        return ""
    budget = max(4000, int(max_chars or getattr(settings, "contextual_reference_max_chars", 18000)))
    lines = [
        "ADVISORY SUPPORTING REFERENCE CONTEXT - use for architecture options and rationale only; "
        "do not convert it into customer scope, a mandate, or a factual current-state claim."
    ]
    seen: set[str] = set()
    used = len(lines[0])
    for document in references:
        heading = f"Supporting reference: {document.name}"
        if used + len(heading) + 2 > budget:
            break
        lines.append(heading)
        used += len(heading) + 1
        for chunk in split_source_document(document, 16000):
            evidence = _contextual_evidence_from_source(chunk)
            for fact in evidence.context_facts:
                clean = re.sub(r"\s+", " ", fact or "").strip()
                key = clean.lower()
                if not clean or key in seen:
                    continue
                item = "- " + clean
                if used + len(item) + 1 > budget:
                    return "\n".join(lines)
                seen.add(key)
                lines.append(item)
                used += len(item) + 1
    return "\n".join(lines)


def _append_unique(items: List[str], item: str, limit: int = 8) -> None:
    clean = re.sub(r"\s+", " ", (item or "").strip())
    if not clean:
        return
    seen = {existing.lower() for existing in items}
    if clean.lower() not in seen and len(items) < limit:
        items.append(clean)


def enrich_understanding_risks(understanding: RFPUnderstanding) -> RFPUnderstanding:
    """Populate proposal delivery risks from RFP-grounded signals.

    The LLM prompt avoids invention, which can make `risks` empty when the RFP
    does not explicitly contain a risk section. This deterministic pass derives
    proposal risks only from visible scope/requirement/assumption signals and
    labels them as inferred.
    """
    risks = [
        re.sub(r"\s+", " ", (item or "").strip())
        for item in (understanding.risks or [])
        if (item or "").strip()
    ]
    signal_parts = [
        understanding.project_scope or "",
        " ".join(understanding.in_scope_work or []),
        " ".join(understanding.assumptions or []),
        " ".join(getattr(req, "text", "") or "" for req in understanding.requirements or []),
        " ".join(getattr(req, "text", "") or "" for req in understanding.unresolved_requirements or []),
        " ".join(understanding.solution_technologies or []),
        " ".join(understanding.key_technologies or []),
    ]
    signal_text = " ".join(signal_parts).lower()

    if any(term in signal_text for term in ("integration", "interface", "api", "sftp", "source system", "upstream", "downstream")):
        _append_unique(
            risks,
            "Inferred from integration scope: source-system access, interface readiness, and contract clarity may delay build validation.",
        )
    if any(term in signal_text for term in ("data", "migration", "quality", "cleansing", "lake", "warehouse", "report", "analytics", "extract")):
        _append_unique(
            risks,
            "Inferred from data scope: data quality, mapping, and reconciliation gaps may affect reporting accuracy and acceptance.",
        )
    if any(term in signal_text for term in ("security", "privacy", "compliance", "access", "identity", "role", "audit")):
        _append_unique(
            risks,
            "Inferred from security/compliance needs: access, control evidence, and approval cycles may extend readiness timelines.",
        )
    if any(term in signal_text for term in ("availability", "disaster", "recovery", "backup", "rto", "rpo", "resilien", "sla")):
        _append_unique(
            risks,
            "Inferred from resilience requirements: backup, recovery, availability, and support ownership must be evidenced before go-live.",
        )
    if any(term in signal_text for term in ("deployment", "environment", "production", "cutover", "release", "devops", "pipeline")):
        _append_unique(
            risks,
            "Inferred from deployment scope: environment readiness, release gates, and cutover windows may constrain production rollout.",
        )
    if understanding.unresolved_requirements:
        _append_unique(
            risks,
            "Inferred from unresolved requirements: pending customer decisions may change scope, estimates, acceptance criteria, or delivery sequencing.",
        )
    if understanding.assumptions:
        _append_unique(
            risks,
            "Inferred from stated assumptions: delayed closure of dependencies may affect mobilisation, sprint throughput, or acceptance sign-off.",
        )
    if any(term in signal_text for term in ("timeline", "milestone", "deadline", "week", "month", "phase", "sprint")):
        _append_unique(
            risks,
            "Inferred from delivery timeline: stakeholder availability and timely approvals are critical to protect planned milestones.",
        )

    if not risks and any(part.strip() for part in signal_parts):
        risks = [
            "Inferred from delivery scope: requirement clarifications and acceptance criteria must be closed early to avoid rework.",
            "Inferred from implementation dependencies: customer access, environments, and decision owners must be ready during mobilisation.",
            "Inferred from proposal scope: delivery governance must actively manage dependencies, changes, and acceptance evidence.",
        ]
    understanding.risks = risks[:8]
    return understanding


@_logged_node
def extract_source_evidence(state: AgentState) -> Dict[str, Any]:
    """Reduce large RFP packages through bounded, source-aware extraction."""
    source_chars = len(state.rfp_text or "")
    contextual_reference_context = _build_contextual_reference_context(
        list(state.source_documents)
    )
    state.contextual_reference_context = contextual_reference_context
    if source_chars <= settings.understanding_direct_max_chars:
        log.info(
            "RFP package fits direct understanding budget (%d <= %d chars)",
            source_chars,
            settings.understanding_direct_max_chars,
        )
        state.source_evidence = []
        state.evidence_text = None
        return {
            "source_evidence": [],
            "evidence_text": None,
            "contextual_reference_context": contextual_reference_context,
        }

    documents = list(state.source_documents)
    if not documents:
        documents = [
            SourceDocument(
                document_id="doc-legacy-rfp",
                name="Legacy RFP input",
                document_type="base_rfp",
                authority="authoritative",
                text=state.rfp_text or "",
                character_count=source_chars,
            )
        ]
    chunks = [
        chunk
        for document in documents
        for chunk in split_source_document(
            document,
            settings.understanding_evidence_chunk_chars,
        )
    ]
    workers = max(1, min(settings.understanding_evidence_workers, len(chunks)))
    log.info(
        "Large RFP package detected: extracting evidence from %d chars across %d chunks "
        "with %d worker(s) using model=%s",
        source_chars,
        len(chunks),
        workers,
        settings.model_fast,
    )
    indexed_results: Dict[int, SourceEvidenceBatch] = {}
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rfp-evidence") as executor:
        futures = {
            executor.submit(_extract_evidence_chunk, chunk): (index, chunk)
            for index, chunk in enumerate(chunks)
        }
        for future in as_completed(futures):
            index, chunk = futures[future]
            try:
                indexed_results[index] = future.result()
            except Exception as exc:
                if _is_contextual_document(chunk.document):
                    indexed_results[index] = _contextual_evidence_from_source(chunk, exc)
                    continue
                raise RuntimeError(
                    "Evidence extraction failed for "
                    f"{chunk.document.name} ({chunk.chunk_id}, {len(chunk.text)} chars)"
                ) from exc

    evidence_batches = [indexed_results[index] for index in range(len(chunks))]
    merged = merge_evidence_batches(evidence_batches)
    evidence_text = render_evidence_for_prompt(
        merged,
        settings.understanding_evidence_max_chars,
    )
    log.info(
        "RFP evidence extraction complete: source_chars=%d evidence_chars=%d chunks=%d "
        "requirements=%d format=%s",
        source_chars,
        len(evidence_text),
        len(chunks),
        len(merged.requirements),
        "indexed" if "indexed-rfp-evidence-v1" in evidence_text[:500] else "compact-json",
    )
    state.source_evidence = evidence_batches
    state.evidence_text = evidence_text
    return {
        "source_evidence": evidence_batches,
        "evidence_text": evidence_text,
        "contextual_reference_context": contextual_reference_context,
    }


@_logged_node
def understand_rfp(state: AgentState) -> Dict[str, Any]:
    """Extract a structured understanding of the RFP."""
    analysis_text = state.evidence_text or state.rfp_text or ""
    rfp_focus_guide = (
        "Source evidence was extracted in bounded chunks; use the structured evidence package."
        if state.evidence_text
        else build_rfp_focus_guide(analysis_text)
    )
    prompt = RFP_UNDERSTAND_PROMPT.format(
        rfp_text=analysis_text,
        rag_context=state.rag_context or "",
        rfp_focus_guide=rfp_focus_guide,
        source_reconciliation=(
            render_reconciliation_summary(state.source_reconciliation)
            if state.source_reconciliation
            else "No separate source reconciliation metadata was supplied."
        ),
    )
    hard_prompt_limit = max(
        settings.understanding_direct_max_chars,
        settings.understanding_evidence_max_chars,
    ) + 50000
    if len(prompt) > hard_prompt_limit:
        raise RuntimeError(
            "RFPUnderstanding prompt exceeded the guarded budget after evidence extraction "
            f"({len(prompt)} > {hard_prompt_limit} characters)."
        )
    understanding = response_as_schema(
        prompt, RFPUnderstanding, reasoning_effort=settings.reasoning_effort_high
    )
    understanding = enrich_understanding_risks(understanding)
    understanding.engagement_profile = _effective_engagement_profile(understanding)
    for requirement_group in (
        understanding.requirements,
        understanding.superseded_requirements,
        understanding.unresolved_requirements,
    ):
        for requirement in requirement_group:
            if requirement.source_ref and not requirement.source_refs:
                requirement.source_refs = [requirement.source_ref]
            elif requirement.source_refs and not requirement.source_ref:
                requirement.source_ref = requirement.source_refs[0]
    state.understanding = understanding
    return {"understanding": understanding}


@_logged_node
def classify_sections(state: AgentState) -> Dict[str, Any]:
    """Classify RFP into section taxonomy for better subtitle generation & narrative."""
    analysis_text = state.evidence_text or state.rfp_text or ""
    rfp_focus_guide = (
        "Source evidence was extracted in bounded chunks; classify the effective evidence."
        if state.evidence_text
        else build_rfp_focus_guide(analysis_text)
    )
    prompt = SECTION_TAXONOMY_PROMPT.format(
        rfp_text=analysis_text,
        rag_context=state.rag_context or "",
        rfp_focus_guide=rfp_focus_guide,
    )
    section_map = response_as_schema(
        prompt,
        SectionTaxonomy,
        model=settings.model_fast,
        reasoning_effort=settings.reasoning_effort_low,
        background=False,
    )
    state.section_map = section_map.model_dump()
    return {"section_map": state.section_map}


@_logged_node
def build_narrative(state: AgentState) -> Dict[str, Any]:
    """Build an executive narrative spine for the proposal."""
    prompt = EXEC_NARRATIVE_PROMPT.format(
        understanding_json=state.understanding.model_dump() if state.understanding else {},
        rag_context=state.rag_context or "",
    )
    narrative = response_as_schema(
        prompt,
        ExecutiveNarrative,
        reasoning_effort=settings.reasoning_effort_high,
        background=False,
    )
    state.narrative = narrative
    return {"narrative": narrative}


def _diagram_brief_input(
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    section_map: Dict[str, Any] | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> Dict[str, Any]:
    return {
        "understanding": _compact_understanding_for_plan(understanding),
        "narrative": narrative.model_dump() if narrative else {},
        "sections": section_map or {},
        "proposal_skeleton": _proposal_section_skeleton(understanding),
        "customer_technology_context": customer_technology_context or {},
        "contextual_reference_context": contextual_reference_context,
    }


def _technology_recommendation_input(
    understanding: RFPUnderstanding | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> Dict[str, Any]:
    """Keep the technology decision call grounded without unrelated narrative payload."""
    relevant_section_ids = {
        "sk_solution", "sk_arch", "sk_technical_arch", "sk_integration",
        "sk_data_model", "sk_reporting", "sk_security", "sk_deployment", "sk_tech",
    }
    return {
        "understanding": _compact_understanding_for_plan(understanding),
        "proposal_architecture_sections": [
            section for section in _proposal_section_skeleton(understanding)
            if str(section.get("slide_id", "")) in relevant_section_ids
        ],
        "customer_technology_context": customer_technology_context or {},
        "contextual_reference_context": contextual_reference_context,
    }


_INTERNAL_SOURCE_NOTE_TERMS = (
    "for this run",
    "bounded merged extract",
    "blank or truncated",
    "no customer addendum",
    "no addendum",
    "provided evidence",
    "evidence extract",
    "base brd-style source",
    "effective authority",
    "detail exists outside",
    "extracted evidence",
)


def _is_internal_source_note(text: str | None) -> bool:
    clean = re.sub(r"\s+", " ", (text or "").strip()).lower()
    return bool(clean and any(term in clean for term in _INTERNAL_SOURCE_NOTE_TERMS))


def _visible_assumptions(understanding: RFPUnderstanding | None) -> List[str]:
    return [
        item.strip()
        for item in (getattr(understanding, "assumptions", []) or [])
        if (item or "").strip() and not _is_internal_source_note(item)
    ]


_OPEN_VISUAL_DECISION_RE = re.compile(
    r"\b(?:tbc|tbd|to be confirmed|to be agreed|subject to confirmation|"
    r"unknown|unspecified)\b",
    flags=re.I,
)


def _is_open_visual_decision(text: str | None) -> bool:
    """Return True for unresolved wording that belongs outside a diagram."""
    return bool(_OPEN_VISUAL_DECISION_RE.search(text or ""))


def _source_grounded_technical_architecture_elements(
    understanding: RFPUnderstanding | None,
) -> tuple[List[str], List[str]]:
    """Build a non-inferential fallback from exact source-derived content.

    The visual-brief agent owns the layer taxonomy.  If that LLM call fails, this
    helper supplies named technologies and scope statements only; it deliberately
    does not substitute a standard Experience/API/Services/Data/Cloud hierarchy.
    """
    entities: List[str] = []
    flows: List[str] = []
    seen_entities: set[str] = set()
    seen_flows: set[str] = set()

    def add_entity(value: str | None) -> None:
        clean = _clip(re.sub(r"\s+", " ", value or "").strip(), 110)
        key = clean.lower()
        if clean and key not in seen_entities:
            seen_entities.add(key)
            entities.append(clean)

    def add_flow(value: str | None) -> None:
        clean = _clip(re.sub(r"\s+", " ", value or "").strip(), 150)
        key = clean.lower()
        if clean and key not in seen_flows:
            seen_flows.add(key)
            flows.append(clean)

    for item in (getattr(understanding, "software_bill_of_materials", []) or []):
        component = (getattr(item, "component", "") or "").strip()
        category = (getattr(item, "category", "") or "").strip()
        if component and not _is_excluded_solution_tool(component, understanding):
            add_entity(f"{category}: {component}" if category else component)
    for technology in (
        list(getattr(understanding, "solution_technologies", []) or [])
        + list(getattr(understanding, "key_technologies", []) or [])
    ):
        if not _is_excluded_solution_tool(technology, understanding):
            add_entity(technology)
    for item in (getattr(understanding, "in_scope_work", []) or []):
        add_entity(item)
    for requirement in (getattr(understanding, "requirements", []) or []):
        text = (getattr(requirement, "text", "") or "").strip()
        if any(
            signal in text.lower()
            for signal in (
                "source", "interface", "integration", "api", "event", "file",
                "workflow", "data", "document", "report", "security", "identity",
                "deploy", "hosting", "cloud", "availability", "recovery",
            )
        ):
            add_flow(text)
    if len(entities) < 4:
        add_entity(getattr(understanding, "project_scope", ""))
        add_entity(getattr(understanding, "summary", ""))
    return entities[:8], flows[:5]


def _fallback_visual_briefs(understanding: RFPUnderstanding | None) -> List[DiagramBrief]:
    """Build exact, type-safe briefs for required proposal visuals."""
    if understanding is None:
        return []
    customer = _customer_label(understanding)
    techs = _extract_tech_terms(understanding, limit=8)
    scope = [item for item in (getattr(understanding, "in_scope_work", []) or []) if item][:5]
    requirements = [r for r in (getattr(understanding, "requirements", []) or []) if getattr(r, "text", "")]
    req_refs = [
        getattr(r, "source_ref", None) or getattr(r, "id", "")
        for r in requirements[:6]
        if (getattr(r, "source_ref", None) or getattr(r, "id", ""))
    ]
    entities = [customer] + techs + [_clip(item, 70) for item in scope[:4]]
    flows = [
        _clip(r.text, 120)
        for r in requirements
        if any(t in r.text.lower() for t in ("integration", "interface", "data", "api", "file", "report"))
    ][:5]
    controls = [
        _clip(r.text, 120)
        for r in requirements
        if any(t in r.text.lower() for t in ("security", "audit", "monitor", "availability", "backup", "access"))
    ][:5]
    base_entities = [e for e in entities if e][:8]
    for label in ("Customer users", "Solution services", "Governed data", "Approved outputs"):
        if len(base_entities) >= 4:
            break
        base_entities.append(label)
    base_flows = flows[:5] or [
        "Source inputs -> validation and business rules -> governed data",
        "Governed data -> application/API services -> approved outputs",
    ]
    if len(base_flows) == 1:
        base_flows.append("Operational events -> monitoring and support -> controlled resolution")
    assumptions = _visible_assumptions(understanding)[:4]
    scope_text = " ".join([
        getattr(understanding, "summary", "") or "",
        getattr(understanding, "project_scope", "") or "",
        " ".join(getattr(understanding, "in_scope_work", []) or []),
        " ".join(getattr(requirement, "text", "") or "" for requirement in requirements),
    ]).lower()
    profile = _effective_engagement_profile(understanding)
    managed = _profile_is_managed_operations(profile)
    if managed:
        managed_roles = [
            label for label in (
                "Service Delivery Lead", "Incident Manager", "Problem Manager",
                "Change Manager", "Service Reporting Analyst", "Customer service owners",
            )
            if label.lower() in scope_text or label in (
                "Service Delivery Lead", "Customer service owners"
            )
        ]
        base_entities = list(dict.fromkeys(
            [customer]
            + managed_roles
            + ["Incident Management", "Problem Management", "Change Management"]
            + ["Governance and service reporting", "Continuous improvement backlog"]
        ))[:9]
        base_flows = [
            "Operational event -> incident triage, escalation and resolution -> post-incident review",
            "Recurring incident trend -> problem analysis and known error -> corrective action",
            "Corrective action or planned change -> risk assessment and approval -> controlled implementation",
            "Service evidence -> operational and executive reviews -> prioritized improvement",
        ]
    common = dict(
        evidence_refs=req_refs,
        open_assumptions=assumptions,
        must_not_show=["generic stock diagram", "procurement or submission portals", "invented technologies"],
    )
    separate_hadr = _has_explicit_hadr_need(understanding)
    deployment_entities = [
        "Build and release pipeline", "Development/test environment", "UAT environment",
        "Production application runtime", "Production integration/API runtime",
        "Production data services", "Identity boundary", "Monitoring and support",
    ]
    deployment_flows = [
        "Versioned artifact -> automated assurance -> controlled promotion -> production",
        "Users and source systems -> secured ingress -> application/API runtime -> data services",
        "Runtime and data services -> logs, metrics and audit events -> monitoring and support",
    ]
    if not separate_hadr:
        deployment_entities.extend(["Backup repository", "Recovery environment"])
        deployment_flows.append("Production data -> verified backup -> restore or recovery environment")
    scope_text = " ".join([
        getattr(understanding, "summary", "") or "",
        getattr(understanding, "project_scope", "") or "",
        " ".join(getattr(understanding, "in_scope_work", []) or []),
        " ".join(getattr(requirement, "text", "") or "" for requirement in requirements),
    ]).lower()
    if any(token in scope_text for token in ("catalogue", "catalog ", "product", "sku")):
        data_model_entities = [
            "Product, service and SKU catalogue",
            "Customer requirements and briefs",
            "Solution packages and shortlists",
            "Validation and compliance outcomes",
            "Pricing and commercial decisions",
            "Approved content and enquiries",
            "Domain owners and stewards",
            "Data quality, lineage and audit",
        ]
        data_model_flows = [
            "Regional and business-unit records -> validation and mapping -> mastered catalogue",
            "Customer brief -> matching and shortlisting -> solution package",
            "Solution package -> compliance, feasibility and pricing -> approved outcome",
            "Domain change -> owner and steward approval -> versioned publication",
        ]
    else:
        data_model_entities = [
            "Authoritative source records",
            "Master and reference domains",
            "Operational transaction and event domains",
            "Decision and validation evidence",
            "Governed analytical and consumer products",
            "Domain owners and stewards",
            "Data quality, lineage and audit",
        ]
    data_model_flows = [
            "Authoritative sources -> validation and mapping -> canonical domains",
            "Canonical domains -> operational decisions and evidence -> governed outputs",
            "Domain change -> owner and steward approval -> versioned publication",
        ]
    technical_architecture_entities, technical_architecture_flows = (
        _source_grounded_technical_architecture_elements(understanding)
    )
    briefs: List[DiagramBrief] = [
        DiagramBrief(
            slide_id="sk_solution", title="Proposed solution at a glance", visual_type="generic",
            purpose="Summarise the proposal-specific solution building blocks and value flow.",
            entities=base_entities, flows=base_flows[:3], controls=controls[:4], must_show=scope[:4], **common,
        ),
        DiagramBrief(
            slide_id="sk_flow", title="End-to-end solution flow", visual_type="process",
            purpose="Show the proposal-specific journey from source inputs through controls to approved outcomes.",
            entities=base_entities, flows=base_flows, controls=controls[:4], must_show=scope[:4], **common,
        ),
        DiagramBrief(
            slide_id="sk_operating_model",
            title="The operating model connects accountability, process and evidence",
            visual_type="process",
            purpose="Show the accountable service model across customer, provider, operational practices, governance and improvement.",
            entities=base_entities,
            flows=base_flows,
            controls=(controls[:4] or ["Named accountability", "RACI and escalation", "Service reviews", "Audit evidence"]),
            must_show=scope[:4],
            **common,
        ),
        DiagramBrief(
            slide_id="sk_service_lifecycle",
            title="Integrated service practices turn operational signals into improvement",
            visual_type="process",
            purpose="Show how incident, problem and change practices interact and feed measurable continuous improvement.",
            entities=[
                "Operational signal", "Incident Management", "Problem Management",
                "Change Management", "Service reporting", "Continuous improvement backlog",
            ],
            flows=base_flows,
            controls=(controls[:4] or ["Severity and escalation", "Root-cause evidence", "Change authorization", "Action closure"]),
            **common,
        ),
        DiagramBrief(
            slide_id="sk_arch", title="Concrete solution architecture", visual_type="architecture",
            purpose="Show how proposal capabilities, systems, data and controls form the target solution.",
            entities=base_entities, flows=base_flows, controls=controls[:6], must_show=scope[:4], **common,
        ),
        DiagramBrief(
            slide_id="sk_technical_arch",
            title="Layered technical architecture connects systems, products and custom services",
            visual_type="technical_architecture",
            purpose=(
                "Show the proposed technical layers, external systems and their data, COTS/build/integrate "
                "boundaries, platform services and cross-cutting controls."
            ),
            entities=technical_architecture_entities,
            flows=technical_architecture_flows,
            controls=(controls[:4] or ["Identity and access", "Audit and lineage", "Monitoring and support"]),
            must_show=scope[:4],
            **common,
        ),
        DiagramBrief(
            slide_id="sk_integration", title="Integration architecture connects source and consumer systems", visual_type="architecture",
            purpose="Show named interfaces, source/consumer boundaries, validation, error handling and directional exchange.",
            entities=base_entities, flows=base_flows, controls=(controls[:4] or ["Authentication and authorization", "Audit and error handling"]), **common,
        ),
        DiagramBrief(
            slide_id="sk_data_model", title="Core data domains and ownership", visual_type="data_model",
            purpose="Show the canonical data domains, their relationships, and accountable ownership and stewardship boundaries.",
            entities=data_model_entities,
            flows=data_model_flows,
            controls=["Authoritative source", "Named domain owner", "Steward approval", "Quality, lineage and audit"],
            **common,
        ),
        DiagramBrief(
            slide_id="sk_reporting",
            title="One governed semantic layer serves decision-ready reporting",
            visual_type="process",
            purpose=(
                "Show a simple lineage from trusted catalogue and operational data through governed measures "
                "to four decision audiences; make the business message obvious without a dense report matrix."
            ),
            entities=[
                "Trusted catalogue, pricing, availability and workflow data",
                "Quality and reconciliation controls",
                "Governed semantic measures",
                "Operational dashboards",
                "Commercial and pricing insights",
                "Compliance and data-quality evidence",
                "Executive outcome reporting",
            ],
            flows=[
                "Trusted domain data -> quality controls -> governed semantic measures",
                "Governed semantic measures -> role-based dashboards and evidence",
                "Reporting insight -> owner action -> corrected governed data",
            ],
            controls=["Metric ownership", "Role-based access", "Lineage and refresh monitoring"],
            **common,
        ),
        DiagramBrief(
            slide_id="sk_deployment", title="Deployment and resilience protect operations", visual_type="deployment",
            purpose="Show environment separation, production runtime topology, controlled promotion, secured access, telemetry and support boundaries.",
            entities=deployment_entities,
            flows=deployment_flows,
            controls=(controls[:5] or ["Identity and access", "Release approval", "Monitoring and audit"]), **common,
        ),
        DiagramBrief(
            slide_id="sk_roadmap",
            title=(
                "Mobilization, transition and stabilization establish the live service"
                if managed else "Incremental delivery releases value through controlled outcomes"
            ),
            visual_type="timeline",
            purpose=(
                "Show mobilization, knowledge transfer, process validation, service readiness, stabilization and improvement."
                if managed else "Show proposed increments, feedback, assurance and release-readiness gates."
            ),
            entities=(
                ["Mobilize and align", "Transition knowledge and access", "Validate processes and reporting", "Stabilize live operations", "Operate and improve"]
                if managed else
                ["Mobilisation and backlog", "Architecture runway", "Incremental releases", "Integrated assurance", "Transition and improvement"]
            ),
            flows=(
                ["Mobilization -> transition acceptance -> live-service readiness", "Stabilization evidence -> governance review -> steady-state operation", "Service trends -> improvement backlog -> measurable action"]
                if managed else
                ["Mobilisation -> thin end-to-end increment -> customer demonstration", "Feedback -> reprioritised backlog -> next release", "Assurance evidence -> release decision"]
            ),
            controls=(
                ["Named transition owners", "Readiness criteria", "Weekly stabilization review", "Service acceptance"]
                if managed else ["Customer feedback", "Security/testing gates", "Operational readiness"]
            ),
            **common,
        ),
        DiagramBrief(
            slide_id="sk_testing",
            title="Acceptance evidence proves the solution is ready",
            visual_type="testing",
            purpose="Show requirement-led evidence streams converging on customer acceptance and release readiness.",
            entities=[
                "Requirement and acceptance traceability",
                "Unit, API and contract evidence",
                "Interface and data reconciliation evidence",
                "End-to-end, performance and security evidence",
                "Customer UAT evidence",
                "Operational-readiness and cutover evidence",
                "Release decision and evidence pack",
            ],
            flows=[
                "Acceptance criteria -> automated and integrated evidence streams",
                "Evidence streams -> defect/retest feedback -> accepted evidence pack",
                "Accepted evidence pack -> customer release decision",
            ],
            controls=["Named acceptance owner", "Defect and retest traceability", "No release without agreed evidence"],
            **common,
        ),
        DiagramBrief(
            slide_id="sk_governance",
            title=(
                "Governance and service leadership make accountability explicit"
                if managed else "Governance resolves decisions without slowing delivery"
            ),
            visual_type="org",
            purpose=(
                "Show service leadership, process ownership, RACI, escalation paths and operational, monthly and executive review forums."
                if managed else "Show proposed decision rights and collaboration between customer ownership, delivery and enabling governance."
            ),
            entities=(
                ["Customer accountable owner", "Provider Service Delivery Lead", "Incident Manager", "Problem Manager", "Change Manager", "Service Reporting Analyst", "Operational review", "Service performance review", "Executive review"]
                if managed else
                ["Customer Product Owner", "Business SMEs", "Cross-functional product squad", "Architecture/security/data chapters", "Steering forum"]
            ),
            flows=(
                ["Operational roles -> Service Delivery Lead -> customer accountable owner", "Service evidence -> review forums -> decisions and actions", "Major incident or change risk -> named escalation and authorization path"]
                if managed else
                ["Product Owner and SMEs -> prioritised outcomes -> product squad", "Product squad -> demonstrations and evidence -> customer feedback", "Escalated decisions -> steering forum -> resolved dependencies"]
            ),
            controls=(
                ["Cross-party RACI", "Named decision owners", "Escalation thresholds", "Recorded actions and due dates"]
                if managed else ["Architecture and security standards", "RAID and dependency decisions", "Outcome reporting"]
            ),
            **common,
        ),
    ]
    if _has_explicit_hadr_need(understanding):
        briefs.append(DiagramBrief(
            slide_id="auto_ha_and_dr_protect_business_continuity",
            title="HA and DR protect business continuity",
            visual_type="hadr",
            purpose="Show availability, replication, backup, failover and recovery responsibilities without inventing RTO/RPO values.",
            entities=["Primary application/runtime", "Redundant runtime", "Primary data store", "Replicated/backup data", "Monitoring and operations"],
            flows=["Primary runtime -> redundant runtime failover", "Primary data -> replication/backup -> restore or DR", "Health event -> alert -> controlled failover and recovery"],
            controls=(controls[:5] or ["Health monitoring", "Backup verification", "Recovery testing"]),
            must_not_show=["data model", "business process flow", "invented RTO/RPO commitments"],
            evidence_refs=req_refs,
            open_assumptions=assumptions,
        ))
    planned_visual_ids = {
        str(section.get("slide_id", ""))
        for section in _proposal_section_skeleton(understanding)
        if section.get("diagram_kind")
    }
    return [
        brief for brief in briefs
        if brief.slide_id in planned_visual_ids
        or brief.visual_type == "hadr" and _has_explicit_hadr_need(understanding)
    ]


@_logged_node
def derive_visual_briefs(state: AgentState) -> Dict[str, Any]:
    """Analyze the proposal and decide which visuals are genuinely grounded."""
    input_json = json.dumps(
        _diagram_brief_input(
            state.understanding,
            state.narrative,
            state.section_map,
            state.customer_technology_context,
            state.contextual_reference_context,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    prompt = VISUAL_BRIEF_PROMPT.format(input_json=input_json)
    try:
        brief_set = response_as_schema(
            prompt,
            DiagramBriefSet,
            reasoning_effort=settings.reasoning_effort_medium,
            timeout_seconds=float(getattr(settings, "deck_plan_timeout_s", 600.0) or 600.0),
            background=False,
        )
        briefs = list(brief_set.briefs)
        supplements = _fallback_visual_briefs(state.understanding)
        existing = {
            (re.sub(r"^(sk|auto|fallback)_", "", (brief.slide_id or "").lower()), brief.visual_type)
            for brief in briefs
        }
        for brief in supplements:
            key = (re.sub(r"^(sk|auto|fallback)_", "", (brief.slide_id or "").lower()), brief.visual_type)
            if key not in existing:
                briefs.append(brief)
                existing.add(key)
    except Exception:
        log.warning("Visual brief LLM call failed; using deterministic visual briefs.", exc_info=True)
        briefs = _fallback_visual_briefs(state.understanding)
    allowed_visual_ids = {
        str(section.get("slide_id", ""))
        for section in _proposal_section_skeleton(state.understanding)
        if section.get("diagram_kind")
    }
    briefs = [brief for brief in briefs if brief.slide_id in allowed_visual_ids]
    state.visual_briefs = briefs
    return {"visual_briefs": briefs}


def _align_recommendations_to_customer_platform(
    recommendations: TechnologyRecommendationSet,
    customer_technology_context: Dict[str, Any] | None,
) -> TechnologyRecommendationSet:
    """Apply an explicit customer platform choice as the final provider guard."""
    context = customer_technology_context or {}
    platform = str(context.get("platform") or "").strip()
    status = str(context.get("status") or "").strip().lower()
    customer_provider = _selected_provider_family(platform)
    if (
        not customer_provider
        or status not in {"customer-preferred", "customer-mandated", "existing estate"}
    ):
        return recommendations

    prior_platform = recommendations.selected_platform
    recommendations.selected_platform = platform
    if recommendations.hosting_model == "customer-decision":
        recommendations.hosting_model = "public-cloud"
    recommendations.recommendations = [
        item for item in recommendations.recommendations
        if not _conflicts_with_selected_provider(
            f"{item.proposed_technology} {item.technology_category}",
            customer_provider,
        )
    ]
    recommendations.component_decisions = [
        item for item in recommendations.component_decisions
        if not _conflicts_with_selected_provider(
            f"{item.recommendation} {item.role}",
            customer_provider,
        )
    ]
    recommendations.platform_assumptions = [
        item for item in recommendations.platform_assumptions
        if not _conflicts_with_selected_provider(item, customer_provider)
    ]
    if _conflicts_with_selected_provider(
        recommendations.deployment_rationale,
        customer_provider,
    ) or _selected_provider_family(prior_platform) not in {"", customer_provider}:
        recommendations.deployment_rationale = (
            f"{platform} is the {status.replace('-', ' ')} platform supplied by the customer."
        )
    if _conflicts_with_selected_provider(
        recommendations.primary_region_strategy,
        customer_provider,
    ):
        recommendations.primary_region_strategy = ""
    return recommendations


def _source_grounded_region_strategy(
    customer_technology_context: Dict[str, Any] | None,
    contextual_reference_context: str = "",
) -> str:
    """Preserve an explicit primary/recovery region statement without inferring one."""
    details = str((customer_technology_context or {}).get("details") or "").strip()
    # Supporting research may inform the LLM recommendation, but the emergency
    # fallback must not elevate an advisory example into a selected topology.
    source = details
    if not source:
        return ""
    sentences = [
        re.sub(r"\s+", " ", item).strip()
        for item in re.split(r"(?<=[.!?])\s+|[\r\n]+", source)
        if (item or "").strip()
    ]
    explicit = [
        sentence for sentence in sentences
        if re.search(r"\bprimary\b", sentence, flags=re.I)
        and re.search(r"\b(?:secondary|recovery|dr)\b", sentence, flags=re.I)
        and re.search(r"\b(?:region|location|geograph)\w*\b", sentence, flags=re.I)
    ]
    return _clip(" ".join(explicit[:2]), 320) if explicit else ""


def _source_grounded_technology_fallback(
    understanding: RFPUnderstanding | None,
    customer_technology_context: Dict[str, Any] | None,
    contextual_reference_context: str = "",
) -> TechnologyRecommendationSet:
    """Degrade safely to named source technologies; never select a default stack."""
    context = customer_technology_context or {}
    platform = str(context.get("platform") or "").strip()
    if platform.lower() == "not specified":
        platform = ""
    provider = _selected_provider_family(platform)
    if not platform:
        source_techs = list(getattr(understanding, "solution_technologies", []) or [])
        provider = _cloud_signal(" ".join(source_techs))
        platform = {
            "azure": "Microsoft Azure",
            "aws": "Amazon Web Services (AWS)",
            "gcp": "Google Cloud Platform",
        }.get(provider, "")

    hosting_model = "customer-decision"
    normalized_platform = platform.lower()
    if provider:
        hosting_model = "public-cloud"
    elif "private cloud" in normalized_platform or "on-prem" in normalized_platform:
        hosting_model = "private-cloud" if "private cloud" in normalized_platform else "on-premises"
    elif "hybrid" in normalized_platform:
        hosting_model = "hybrid"

    recommendations: List[TechnologyRecommendation] = []
    seen: set[str] = set()
    for item in (getattr(understanding, "software_bill_of_materials", []) or []):
        technology = (getattr(item, "component", "") or "").strip()
        if not technology or _is_excluded_solution_tool(technology, understanding):
            continue
        category = (getattr(item, "category", "") or "Source-referenced technology").strip()
        role = (getattr(item, "purpose", "") or "Role described in the supplied proposal material").strip()
        basis = (getattr(item, "source_or_basis", "") or "Named in supplied material; authority and fit require validation").strip()
        basis_lower = basis.lower()
        status = (
            "RFP-mandated" if any(term in basis_lower for term in ("mandat", "required", "must"))
            else "RFP-referenced" if "rfp" in basis_lower
            else "customer-decision"
        )
        sourcing_model = (
            "COTS/SaaS" if any(term in category.lower() for term in ("cots", "saas", "product"))
            else "customer-decision"
        )
        recommendations.append(TechnologyRecommendation(
            architecture_layer=category,
            proposed_technology=technology,
            technology_category=category,
            role=role,
            status=status,
            rationale=basis,
            sourcing_model=sourcing_model,
            build_vs_buy_rationale="Preserve the named source technology until the architecture agent validates its role and sourcing decision.",
        ))
        seen.add(technology.lower())

    for technology in (
        list(getattr(understanding, "solution_technologies", []) or [])
        + list(getattr(understanding, "key_technologies", []) or [])
    ):
        technology = (technology or "").strip()
        if (
            not technology
            or technology.lower() in seen
            or _is_excluded_solution_tool(technology, understanding)
        ):
            continue
        recommendations.append(TechnologyRecommendation(
            architecture_layer="Source-referenced technology",
            proposed_technology=technology,
            technology_category="Named technology requiring role classification",
            role="Role and architecture layer must be derived from the supplied requirements before proposal use",
            status="RFP-referenced",
            rationale="The technology is present in the structured proposal understanding; no additional product choice is inferred.",
            sourcing_model="customer-decision",
            build_vs_buy_rationale="No sourcing decision is inferred by the fallback path.",
        ))
        seen.add(technology.lower())

    status = str(context.get("status") or "").strip().lower()
    details = str(context.get("details") or "").strip()
    deployment_rationale = ""
    if platform:
        deployment_rationale = (
            f"{platform} was supplied before Step 1 as {status.replace('-', ' ') or 'customer technology context'}."
            + (f" Customer detail: {_clip(details, 180)}." if details else "")
        )
    return TechnologyRecommendationSet(
        recommendations=recommendations,
        component_decisions=[],
        hosting_model=hosting_model,
        selected_platform=platform,
        deployment_rationale=deployment_rationale,
        primary_region_strategy=_source_grounded_region_strategy(
            customer_technology_context,
            contextual_reference_context,
        ),
    )


def _complete_technology_recommendations(
    recommendations: TechnologyRecommendationSet,
    fallback: TechnologyRecommendationSet,
) -> TechnologyRecommendationSet:
    """Merge source-named items and explicit context without adding default products."""
    existing = " ".join(
        f"{item.architecture_layer} {item.technology_category} {item.proposed_technology}"
        for item in recommendations.recommendations
    ).lower()
    for item in fallback.recommendations:
        item_text = f"{item.architecture_layer} {item.technology_category} {item.proposed_technology}".lower()
        if (item.proposed_technology or "").strip().lower() in existing:
            continue
        recommendations.recommendations.append(item)
        existing += " " + item_text
    if not recommendations.component_decisions:
        recommendations.component_decisions = fallback.component_decisions
    if not recommendations.selected_platform:
        recommendations.selected_platform = fallback.selected_platform
    if recommendations.hosting_model == "customer-decision":
        recommendations.hosting_model = fallback.hosting_model
    if not recommendations.deployment_rationale:
        recommendations.deployment_rationale = fallback.deployment_rationale
    if not recommendations.primary_region_strategy:
        recommendations.primary_region_strategy = fallback.primary_region_strategy
    return recommendations


def _technology_recommendation_quality_issues(
    recommendations: TechnologyRecommendationSet,
    understanding: RFPUnderstanding | None,
) -> List[str]:
    """Identify missing proposal-relevant decisions without prescribing products."""
    usable = [
        item for item in (recommendations.recommendations or [])
        if (item.proposed_technology or "").strip()
        and "no product selected" not in item.proposed_technology.lower()
        and item.proposed_technology.lower() not in {
            "application service", "database capability", "cloud service",
            "catalogue platform", "integration capability",
        }
    ]
    issues: List[str] = []
    if len(usable) < 5:
        issues.append("fewer than five concrete technology/product decisions")
    scope = _understanding_text(understanding).lower()
    recommendation_text = " ".join(
        f"{item.architecture_layer} {item.technology_category} {item.role}"
        for item in usable
    ).lower()
    expected = [
        (("portal", "web", "mobile", "user experience", "ui "), ("frontend", "web ui", "experience", "user interface"), "required user-experience technology"),
        (("api", "workflow", "application", "business service"), ("backend", "api", "application framework", "service runtime"), "required API/application implementation technology"),
        (("integration", "interface", "sftp", "event", "message"), ("integration", "api management", "message", "event", "connector"), "required integration technology"),
        (("data", "database", "catalog", "repository", "master data"), ("database", "data store", "repository", "lake", "master data"), "required data technology"),
        (("test", "uat", "quality", "acceptance"), ("test", "quality"), "required quality-engineering toolchain"),
        (("deploy", "release", "devops", "pipeline", "infrastructure"), ("deploy", "ci/cd", "devops", "infrastructure as code", "runtime"), "required deployment/DevSecOps technology"),
    ]
    for scope_signals, recommendation_signals, label in expected:
        if any(signal in scope for signal in scope_signals) and not any(
            signal in recommendation_text for signal in recommendation_signals
        ):
            issues.append(label)
    return issues


@_logged_node
def derive_technology_recommendations(state: AgentState) -> Dict[str, Any]:
    """Select a concrete stack independently from proposal constraints."""
    profile = _effective_engagement_profile(state.understanding)
    if not _profile_is_technical_delivery(profile):
        recommendations = TechnologyRecommendationSet()
        state.technology_recommendations = recommendations
        log.info(
            "Skipping technology-stack recommendation for engagement=%s delivery_mode=%s",
            profile.primary_type,
            profile.delivery_mode,
        )
        return {"technology_recommendations": recommendations}
    input_json = json.dumps(
        _technology_recommendation_input(
            state.understanding,
            state.customer_technology_context,
            state.contextual_reference_context,
        ),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    llm_completed = False
    try:
        recommendations = response_as_schema(
            TECHNOLOGY_RECOMMENDATION_PROMPT.format(input_json=input_json),
            TechnologyRecommendationSet,
            reasoning_effort=settings.reasoning_effort_medium,
            timeout_seconds=float(getattr(settings, "deck_plan_timeout_s", 600.0) or 600.0),
            background=True,
        )
        llm_completed = True
    except Exception:
        log.warning(
            "Technology recommendation LLM call failed; preserving only source-named technologies and customer context.",
            exc_info=True,
        )
        recommendations = TechnologyRecommendationSet()
    recommendations = _align_recommendations_to_customer_platform(
        recommendations,
        state.customer_technology_context,
    )
    issues = _technology_recommendation_quality_issues(
        recommendations,
        state.understanding,
    )
    if llm_completed and issues:
        repair_prompt = (
            TECHNOLOGY_RECOMMENDATION_PROMPT.format(input_json=input_json)
            + "\n\nREPAIR REQUIRED:\nThe previous result was incomplete for this proposal: "
            + "; ".join(issues)
            + ". Re-derive the complete stack and applicable architecture layers from INPUT_JSON. "
            "Do not fill the gaps with a familiar default stack. Preserve justified prior decisions and return a complete replacement object.\n"
            + json.dumps(recommendations.model_dump(), ensure_ascii=False, separators=(",", ":"))
        )
        try:
            repaired = response_as_schema(
                repair_prompt,
                TechnologyRecommendationSet,
                reasoning_effort=settings.reasoning_effort_medium,
                timeout_seconds=float(getattr(settings, "deck_plan_timeout_s", 600.0) or 600.0),
                background=True,
            )
            recommendations = _align_recommendations_to_customer_platform(
                repaired,
                state.customer_technology_context,
            )
        except Exception:
            log.warning(
                "Technology recommendation repair call failed; retaining the source-grounded initial result.",
                exc_info=True,
            )
    fallback = _source_grounded_technology_fallback(
        state.understanding,
        state.customer_technology_context,
        state.contextual_reference_context,
    )
    recommendations = _complete_technology_recommendations(recommendations, fallback)
    recommendations = _align_recommendations_to_customer_platform(
        recommendations,
        state.customer_technology_context,
    )
    state.technology_recommendations = recommendations
    return {"technology_recommendations": recommendations}


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

_PROCUREMENT_TOOL_ALIASES = {
    "ariba", "sap ariba", "coupa", "jaggaer", "ivalua", "oracle procurement cloud",
    "procurement portal", "supplier portal", "tender portal", "sourcing portal",
}


def _excluded_solution_tools(understanding: Optional[RFPUnderstanding]) -> set[str]:
    values = set(_PROCUREMENT_TOOL_ALIASES)
    if understanding is not None:
        for bucket in (
            getattr(understanding, "procurement_or_submission_tools", []) or [],
            getattr(understanding, "non_solution_references", []) or [],
        ):
            values.update((item or "").strip().lower() for item in bucket if (item or "").strip())
    return values


def _is_excluded_solution_tool(name: str, understanding: Optional[RFPUnderstanding]) -> bool:
    normalized = re.sub(r"\s+", " ", (name or "").strip().lower())
    if not normalized:
        return True
    return any(
        excluded == normalized or excluded in normalized or normalized in excluded
        for excluded in _excluded_solution_tools(understanding)
        if excluded
    )


def _ai_ml_opportunities(understanding: Optional[RFPUnderstanding]) -> List[Dict[str, str]]:
    """Return defensible, low-infrastructure AI opportunities from scope signals."""
    if understanding is None:
        return []
    text = _understanding_text(understanding).lower()
    opportunities: List[Dict[str, str]] = []

    def add(name: str, value: str, trigger: tuple[str, ...], approach: str) -> None:
        if any(term in text for term in trigger):
            opportunities.append({"name": name, "value": value, "approach": approach})

    add(
        "Intelligent intake and classification",
        "Classify email, spreadsheet, document, and file inputs; extract fields and route uncertain records for review.",
        ("email", "spreadsheet", "xlsx", "pdf", "document", "manual upload", "elp"),
        "Managed document intelligence or a small model, confidence threshold, human review, deterministic import fallback.",
    )
    add(
        "Anomaly and data-quality detection",
        "Prioritise unusual quantities, missing feeds, SLA exceptions, and inconsistent operational records for investigation.",
        ("data", "interface", "integration", "quality", "accuracy", "sla", "milestone"),
        "Rules and statistical baselines first; lightweight managed inference only where it improves precision.",
    )
    add(
        "Forecasting and operational insight",
        "Use historical patterns to support demand, workload, uplift, and exception forecasting without controlling live operations.",
        ("flight", "demand", "uplift", "productivity", "forecast", "capacity", "inventory"),
        "Batch inference on curated data; start with interpretable models and no dedicated accelerator hardware.",
    )
    add(
        "Operations and support copilot",
        "Summarise incidents, retrieve governed knowledge, and answer natural-language questions over approved operational data.",
        ("report", "dashboard", "analytics", "support", "ams", "incident", "knowledge"),
        "Retrieval over approved content with access controls, citations, usage limits, and no autonomous write-back.",
    )
    return opportunities[:4]


def _ai_ml_is_applicable(understanding: Optional[RFPUnderstanding]) -> bool:
    if understanding is None:
        return False
    scope = _scope_text(understanding)
    explicit_ai = _contains_any(
        scope,
        (
            "artificial intelligence", "machine learning", "generative ai", "genai",
            "predictive model", "natural language processing", " ai ", "ml model",
        ),
    )
    analytical_use_case = _contains_any(
        scope,
        (
            "forecast", "predict", "anomaly", "optimisation", "optimization",
            "document extraction", "email extraction", "elp extraction",
            "intelligent classification",
        ),
    )
    profile = _effective_engagement_profile(understanding)
    technical_fit = any(
        _profile_type_score(profile, engagement_type) >= 0.50
        for engagement_type in ("data_analytics", "application_development", "platform_implementation")
    )
    return len(_ai_ml_opportunities(understanding)) >= 2 and (
        explicit_ai or (technical_fit and analytical_use_case)
    )


def _ai_ml_cards(understanding: Optional[RFPUnderstanding]) -> List[Card]:
    cards: List[Card] = []
    accents = ["solution", "info", "outcome", "why"]
    for idx, opportunity in enumerate(_ai_ml_opportunities(understanding)):
        cards.append(
            Card(
                heading=opportunity["name"],
                body=opportunity["value"],
                bullets=[opportunity["approach"]],
                accent=accents[idx % len(accents)],
            )
        )
    return cards


def _ai_ml_architecture_clause(understanding: Optional[RFPUnderstanding]) -> str:
    opportunities = _ai_ml_opportunities(understanding)
    if len(opportunities) < 2:
        return ""
    names = ", ".join(item["name"] for item in opportunities)
    return (
        " Add one clearly labelled optional 'AI-assisted services' sidecar containing: "
        f"{names}. Keep core transactions, validation, and operational decisions deterministic. "
        "Show curated-data access, confidence thresholds, human review, deterministic fallback using rules, "
        "model/usage monitoring, and no autonomous write-back. Depict managed consumption or "
        "small-model inference without dedicated GPU infrastructure."
    )


def _extract_tech_terms(understanding: Optional[RFPUnderstanding], limit: int = 6) -> List[str]:
    """Return technologies mentioned in the RFP.

    Prefers model-extracted `solution_technologies`. Submission/procurement-only
    tools are explicitly excluded so tender portals do not leak into architecture
    diagrams.
    """
    if understanding is None:
        return []
    excluded = _excluded_solution_tools(understanding)
    explicit = [
        t.strip()
        for t in (
            getattr(understanding, "solution_technologies", []) or []
            or getattr(understanding, "key_technologies", []) or []
        )
        if (t or "").strip() and not _is_excluded_solution_tool(t, understanding)
    ]
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

    corpus = " ".join(
        [
            getattr(understanding, "project_scope", "") or "",
            getattr(understanding, "summary", "") or "",
            " ".join(getattr(understanding, "in_scope_work", []) or []),
        ]
    ).lower()
    for r in getattr(understanding, "requirements", []) or []:
        corpus += " " + (getattr(r, "text", "") or "").lower()

    found: List[str] = []
    for kw in _TECH_KEYWORDS:
        if kw in excluded or kw not in corpus:
            continue
        # Skip near-duplicates where one term contains another (e.g. postgres/postgresql).
        if any(kw in f or f in kw for f in found):
            continue
        found.append(kw)
        if len(found) >= limit:
            break
    return found


def _recommended_architecture_tech_terms(understanding: Optional[RFPUnderstanding]) -> List[str]:
    """Do not inject a preselected stack into proposal-grounded diagrams.

    Technology recommendations belong to the LLM deck-planning decision, where
    the complete requirements and trade-offs are available. Diagram grounding
    uses only technologies extracted from authoritative proposal content.
    """
    return []


def _diagram_context(
    understanding: Optional[RFPUnderstanding],
    *,
    include_technologies: bool = True,
) -> str:
    """Build a short, grounded context string for diagram prompts."""
    if understanding is None:
        return ""
    customer = getattr(understanding, "customer_name", None) or "the client"
    techs: List[str] = []
    if include_technologies:
        techs = _extract_tech_terms(understanding, limit=10)
        for proposed in _recommended_architecture_tech_terms(understanding):
            if proposed.lower() not in {item.lower() for item in techs}:
                techs.append(proposed)
    techs = techs[:10]
    parts = [f"Client: {customer}."]
    scope = (getattr(understanding, "project_scope", "") or "").strip()
    if scope:
        parts.append(f"Project scope: {scope}")
    in_scope = [
        item.strip()
        for item in (getattr(understanding, "in_scope_work", []) or [])
        if (item or "").strip()
    ][:5]
    if in_scope:
        parts.append("In-scope capabilities: " + "; ".join(in_scope) + ".")
    if techs:
        parts.append("Use only these named solution technologies where relevant: " + ", ".join(techs) + ".")
    return " ".join(parts)


_SAFE_MARGIN_NOTE = (
    "This diagram will be displayed at roughly 12 by 5.5 inches on a 16:9 slide. "
    "Style: consulting-grade, white background, readable 18pt+ labels, minimal clutter, "
    "no logos, no gradients, no sketch effects. Use at most 8 primary groups and 10 labeled "
    "boxes in total, with no more than two short text lines per box. Do not add an internal "
    "diagram title, descriptive paragraphs, footnotes, citations, evidence references, reviewer "
    "context, source-processing notes, assumption sidebars, or legends unless essential. "
    "Translate requirements into confident proposed-solution labels. Never reproduce 'the platform "
    "shall', 'the system shall', 'should', or 'must' statements, and never show requirement text as "
    "input panels. Use active target-state wording such as 'Catalogue consolidation' or 'The solution "
    "consolidates'. The image is unacceptable if it contains more than 10 labeled boxes. "
    "Show the proposed solution only. Keep open decisions, assumptions, dependencies, placeholder "
    "labels, and customer-confirmation qualifiers out of the image; place them on the Assumptions "
    "and Dependencies slide instead. "
    "Keep labels to five words where possible. Use labeled boxes with directional arrows. "
    "Keep all text and shapes inside a 5-8% safe margin; do not place content at the edges."
)


def _matched_requirement_texts(
    understanding: Optional[RFPUnderstanding],
    terms: tuple[str, ...],
    limit: int = 4,
) -> List[str]:
    if understanding is None:
        return []
    candidates = list(getattr(understanding, "in_scope_work", []) or []) + [
        (getattr(req, "text", "") or "").strip()
        for req in (getattr(understanding, "requirements", []) or [])
    ]
    result: List[str] = []
    for text in candidates:
        clean = re.sub(r"\s+", " ", (text or "").strip())
        if clean and any(term in clean.lower() for term in terms) and clean not in result:
            result.append(clean)
        if len(result) >= limit:
            break
    return result


def _testing_proposal_points(understanding: Optional[RFPUnderstanding]) -> List[BulletPoint]:
    customer = (getattr(understanding, "customer_name", None) or "customer").strip()
    interfaces = _matched_requirement_texts(
        understanding,
        ("interface", "api", "sftp", "file", "email", "fih", "gp4", "sap", "ksms", "elp"),
        3,
    )
    functionality = _matched_requirement_texts(
        understanding,
        ("testing", "test ", "acceptance", "uat", "sit", "functional parity", "regression", "cutover", "cut over", "replace "),
        3,
    )
    controls = _matched_requirement_texts(
        understanding,
        ("security", "access", "audit", "performance", "high availability", "uptime", "resilience", "recovery", "accuracy", "reconciliation", "retention"),
        3,
    )
    return [
        BulletPoint(
            text="Prove every source and target interface",
            sub_points=interfaces or [
                "Validate representative files, APIs, mappings, retries, duplicate handling, exception routing, and reconciliation across the named integration estate."
            ],
        ),
        BulletPoint(
            text="Demonstrate functional parity and target outcomes",
            sub_points=functionality or [
                "Trace priority scenarios to acceptance criteria and prove functional parity or agreed improvement before business sign-off."
            ],
        ),
        BulletPoint(
            text="Evidence data and non-functional controls",
            sub_points=controls or [
                "Evidence data accuracy, role-based access, auditability, security, performance, recovery, and monitoring against confirmed acceptance thresholds."
            ],
        ),
        BulletPoint(
            text="Gate cutover on business-ready evidence",
            sub_points=[
                f"Combine SIT, end-to-end integration, {customer}-led UAT, defect closure, reconciliation, operational runbooks, rollback readiness, and release approval into one evidence pack."
            ],
        ),
    ]


def _ams_proposal_points(understanding: Optional[RFPUnderstanding]) -> List[BulletPoint]:
    support_scope = _matched_requirement_texts(
        understanding,
        ("ams", "warranty", "support", "maintenance", "incident", "service request", "sla"),
        3,
    )
    integration_scope = _matched_requirement_texts(
        understanding,
        ("interface", "api", "sftp", "fih", "gp4", "sap", "ksms", "elp", "power bi"),
        3,
    )
    return [
        BulletPoint(
            text="Support the complete live-service boundary",
            sub_points=(support_scope or [
                "Cover the application, its integrations and interfaces, data stores, scheduled jobs, and supporting services across the live estate."
            ])[:3],
        ),
        BulletPoint(
            text="Monitor business flows, not only infrastructure",
            sub_points=[
                "Correlate application, interface, data-freshness, validation, audit, and reporting signals so incidents are tied to the affected business flows rather than isolated component alerts."
            ],
        ),
        BulletPoint(
            text="Resolve incidents with interface ownership",
            sub_points=(integration_scope or [
                "Use runbooks and ownership paths for source failures, malformed inputs, mapping errors, downstream rejection, data correction, replay, and reconciliation."
            ])[:3],
        ),
        BulletPoint(
            text="Transition warranty into measurable AMS",
            sub_points=[
                "Use hypercare trends, known-error records, knowledge transfer, support acceptance, agreed service measures, and a prioritised minor-enhancement backlog to stabilise and improve the platform."
            ],
        ),
    ]


def _build_diagram_prompt(
    kind: str,
    understanding: Optional[RFPUnderstanding],
    technology_recommendations: TechnologyRecommendationSet | None = None,
) -> str:
    """Create a context-rich diagram prompt grounded in the RFP.

    `kind` is one of: architecture, technical_architecture, data_model,
    delivery, timeline, team, solution, testing, ams.
    """
    has_authoritative_platform = bool(
        technology_recommendations is not None
        and (technology_recommendations.selected_platform or "").strip()
        and kind in {"technical_architecture", "deployment", "hadr"}
    )
    ctx = _diagram_context(
        understanding,
        include_technologies=not has_authoritative_platform,
    )
    techs = [] if has_authoritative_platform else _extract_tech_terms(understanding)
    tech_clause = (" featuring " + ", ".join(techs)) if techs else ""
    ai_clause = _ai_ml_architecture_clause(understanding)

    if kind == "integration":
        interface_details = _matched_requirement_texts(
            understanding,
            ("integration", "interface", "api", "rest", "soap", "sftp", "file", "csv", "xlsx", "webhook", "event", "queue", "source", "external"),
            8,
        )
        grounded = "\nGrounded interface requirements:\n- " + "\n- ".join(interface_details) if interface_details else ""
        body = (
            "Create a concrete integration architecture, not a capability map. Lay it out left to right as "
            "named internal and external source systems -> labelled interface channels and protocols -> secured "
            "ingress and integration services -> validation, transformation, orchestration, retry/dead-letter and "
            "reconciliation controls -> catalogue/core services -> named consumers and external systems. Label every "
            "connection with direction and the protocol or exchange mechanism where known, and distinguish inbound, "
            "outbound and bidirectional flows. Show API management, file/SFTP intake, messaging/events, identity, "
            "monitoring and error handling only when relevant. Never invent source-system names, protocols, frequencies "
            "or products. When a system or protocol is unnamed, use a role-based label such as "
            "'Source system' or 'Approved interface' instead of an unresolved-status placeholder. "
            "Do not use business capabilities as source systems."
            + grounded
        )
    elif kind == "architecture":
        body = (
            f"Create a concrete solution architecture diagram{tech_clause}. Show source systems "
            "and document inputs, ingestion/extraction, validation and business rules, operational "
            "application services, APIs/integration, central operational data store or lakehouse, "
            "reporting/BI, security, monitoring, and support boundaries. For a data hub, make the "
            "data pipeline explicit: capture -> validate -> curate -> serve -> report. Show key "
            "integrations and primary data flows with directional arrows. Group related services "
            "and label each box clearly. Keep AI as one bounded component only when it is explicitly "
            "part of the proposed architecture; do not add a separate AI sidecar or repeat AI controls."
        )
    elif kind == "technical_architecture":
        source_requirements = _matched_requirement_texts(
            understanding,
            (
                "source system", "system of record", "integration", "interface", "api", "file",
                "catalogue", "catalog", "master data", "product", "pricing", "cost", "inventory",
                "warehouse", "availability", "document", "image", "content", "analytics", "report",
            ),
            10,
        )
        grounded = (
            "\nGrounded system and data requirements:\n- " + "\n- ".join(source_requirements)
            if source_requirements else ""
        )
        body = (
            "Create a layered technical architecture that is clearly distinct from a logical capability map "
            "and the solution-architecture overview. Derive the layer sequence from the proposal's channels, "
            "interfaces, domain behavior, selected enterprise/COTS products, information shapes, hosting constraints, "
            "and operating requirements. Do not start from a standard Experience/API/Services/Data/Cloud template: "
            "include, omit, split, merge or rename layers according to the supplied requirements and authoritative "
            "technology decisions. Show external systems and source-system roles only when grounded, and label the "
            "data each supplies. Show identity, security, governance, observability, DevSecOps and support only where "
            "the proposal requires them, either in the applicable layer or as cross-cutting controls. Label directional flows with the business data being "
            "exchanged and identify the system-of-record role wherever proposal evidence supports it. Distinguish "
            "customer-existing, COTS/SaaS, managed-cloud, maintained open-source, custom-build and integration-only "
            "components using a small legend. Do not invent customer system names or imply that one product owns "
            "master data, binary content, pricing, inventory and analytics unless the recommendation explicitly "
            "supports that consolidation. Do not print rejected alternatives, open decisions, assumptions, "
            "dependencies, TBC/TBD wording or customer-confirmation qualifiers inside the visual."
            + grounded
        )
    elif kind == "data_model":
        domain_requirements = _matched_requirement_texts(
            understanding,
            (
                "catalogue", "catalog", "product", "service", "sku", "ingredient",
                "customer", "solution", "shortlist", "validation", "compliance",
                "pricing", "content", "enquiry", "availability", "facility", "market",
                "master data", "reference data", "transaction", "event", "reporting",
            ),
            8,
        )
        grounded = (
            "\nGrounded domain requirements:\n- " + "\n- ".join(domain_requirements)
            if domain_requirements else ""
        )
        body = (
            "Create a conceptual core data-domain and ownership map, not a physical database schema or field-level ERD. "
            "Place the canonical data model at the centre. Group no more than seven proposal-supported domains around it, "
            "such as master/reference data, customer or demand inputs, operational transactions, solution or decision records, "
            "validation/control evidence, commercial data, and governed outputs only when grounded below. Show directional "
            "relationships from authoritative sources through canonical domains to decisions/evidence and approved consumption. "
            "Add a concise ownership band that distinguishes source owner, accountable domain owner, data steward, control owner, "
            "and authorised consumer without inventing named roles. Show quality, lineage, versioning and audit as cross-domain "
            "controls. Do not invent table names, attributes, keys, cardinalities, systems, or ownership assignments."
            + grounded
        )
    elif kind == "reporting":
        body = (
            "Create a simple reporting value-flow diagram with one clear message: trusted governed data is reused "
            "through a semantic layer for consistent decisions. Use at most seven boxes in a left-to-right flow: "
            "trusted catalogue/pricing/availability/workflow data -> quality and reconciliation -> governed semantic "
            "measures -> four concise audience outcomes (operational, commercial, compliance/data quality, executive). "
            "Add a small feedback arrow from insight to the accountable data owner. Keep labels to short noun phrases. "
            "Do not create a 4x4 matrix, dashboard mock-up, report inventory, miniature charts, paragraph text, or more "
            "than four outcome boxes. The audience should understand what is governed, who consumes it, and why it matters "
            "within five seconds."
        )
    elif kind == "deployment":
        separate_hadr = _has_explicit_hadr_need(understanding)
        body = (
            f"Create a deployment topology diagram{tech_clause}. Show environment separation, the "
            "build/artifact and controlled promotion path, production application/API/data runtime "
            "boundaries, secured user and source-system ingress, identity/secrets, telemetry, alerting, "
            "and support ownership. Make runtime placement and directional connections the dominant visual. "
            "Do not show business-process steps, functional modules, AI use cases, requirement text, evidence "
            "references, or reviewer notes. "
            + (
                "A separate HA/DR diagram exists, so include only a small continuity boundary marker; do not "
                "show backup, replication, failover, restore procedures, RTO or RPO on this deployment diagram."
                if separate_hadr else
                "Include backup/recovery placement only where supported, without inventing RTO/RPO targets."
            )
        )
    elif kind == "hadr":
        body = (
            "Create a focused high availability and disaster recovery topology, not a deployment architecture. "
            "Show a primary region/failure domain with multi-zone redundant runtime and data paths, health probes "
            "and failover; then a clearly separate recovery region/failure domain with replication direction, "
            "backup and restore validation, controlled regional failover, recovery validation and failback. Show "
            "operations ownership. Do not display uncommitted RTO/RPO values or qualifier labels; "
            "capture those decisions on the Assumptions and Dependencies slide. "
            "Do not show functional requirements, business capabilities, data models, delivery environments, "
            "the full application-layer inventory, AI use cases, source-document assumptions, paragraph references, evidence IDs, reviewer context, "
            "or document-processing notes."
        )
    elif kind == "delivery":
        body = (
            "Create an Agile delivery and governance operating-model diagram. Show the customer "
            "Product Owner and business SMEs prioritizing an outcome backlog; one or more persistent, "
            "cross-functional product squads containing analysis, architecture, engineering, data or "
            "integration, QA automation, security, DevSecOps/platform, and change skills; and a Scrum "
            "Master or Agile Delivery Lead enabling flow. Show recurring refinement, sprint planning, "
            "daily coordination, integrated demo, retrospective, release decision, and feedback loops. "
            "Place steering governance, architecture/security chapters, dependency resolution, RAID, "
            "and outcome reporting around the squads without depicting functional handoffs or waterfall stages."
        )
    elif kind == "timeline":
        body = (
            "Create a horizontal Agile delivery roadmap. Show mobilisation and backlog inception, "
            "architecture runway, repeated sprint cycles delivering thin end-to-end slices, integrated "
            "testing/security/operations in every sprint, MVP or pilot release, subsequent production "
            "increments, transition, and continuous improvement. Overlay customer demonstrations, "
            "feedback and reprioritisation, dependency decisions, and lightweight release gates. Do not "
            "show separate design, build, test, and deployment phases. Include durations only when grounded "
            "in the RFP; otherwise label the cadence as proposed."
            + (
                " Include a gated AI discovery and pilot thread: validate data readiness and baseline value, "
                "pilot one low-risk use case, measure accuracy/cost, then scale only after human acceptance."
                if ai_clause else ""
            )
        )
    elif kind == "testing":
        customer = (getattr(understanding, "customer_name", None) or "customer").strip()
        testing_focus = " ".join(
            f"{point.text}: {'; '.join(point.sub_points)}"
            for point in _testing_proposal_points(understanding)
        )
        body = (
            "Create a proposal-specific testing and acceptance evidence map, not a generic testing lifecycle. "
            f"Use these acceptance themes: {testing_focus} "
            "Organise the visual as evidence streams for named interfaces and files, functional parity and target-outcome scenarios, "
            f"data reconciliation and control evidence, {customer}-led UAT, and cutover/release readiness. Connect each "
            "stream to an explicit evidence artefact and acceptance owner. Show defect/retest only as a feedback "
            "loop. Do not show a textbook test pyramid, generic sequential test phases, squads, or steering committees."
            + (
                " For optional AI use cases, include representative-data evaluation, accuracy and false-positive thresholds, "
                "human acceptance, security testing, unit-cost evidence, model monitoring, and deterministic fallback verification."
                if ai_clause else ""
            )
        )
    elif kind == "ams":
        ams_focus = " ".join(
            f"{point.text}: {'; '.join(point.sub_points)}"
            for point in _ams_proposal_points(understanding)
        )
        body = (
            "Create a proposal-specific warranty and AMS service map for the proposed platform, not a generic "
            f"support-process tutorial. Use these service themes: {ams_focus} "
            "Show the live-service boundary and named integrations feeding business-flow observability; connect "
            "alerts and user-reported issues to accountable resolution paths, runbooks, correction/replay, and "
            "service evidence. Show warranty-to-AMS transition, knowledge acceptance, known errors, and the minor-"
            "enhancement backlog. Do not display uncommitted service levels or qualifier labels; keep "
            "those decisions on the Assumptions and Dependencies slide. Do not use a "
            "generic L1/L2/L3 pyramid, Agile squad diagram, or steering committee as the main image."
        )
    elif kind == "team":
        body = (
            "Create an Agile squad-topology diagram rather than a functional hierarchy. Show the customer "
            "Product Owner and business SMEs connected to persistent cross-functional squads. Within each "
            "squad show an Agile Delivery Lead or Scrum Master, solution/technical leadership, engineers, "
            "data/integration specialists as relevant, QA automation, security, DevSecOps/platform, and "
            "change/adoption skills. Show shared architecture, security, data, and engineering chapters "
            "providing standards and coaching across squads, plus an executive steering forum for outcomes, "
            "risks, dependencies, and commercial decisions. Do not show separate analysis, build, test, or deployment teams."
        )
    elif kind == "solution":
        body = (
            f"Create a target solution overview{tech_clause}. Show the major building blocks as "
            "labeled layers (source systems and document inputs, ingestion/extraction, integration, "
            "centralized data repository/data lake, reporting/analytics, governance/security, "
            f"observability) and how value flows across them with directional arrows.{ai_clause}"
        )
    else:
        body = "Create a clear, professional consulting diagram with labeled boxes and directional arrows."

    prompt = f"{ctx}\n{body}\n{_SAFE_MARGIN_NOTE}".strip()
    if kind == "technical_architecture":
        technology_context = _technical_architecture_context(technology_recommendations)
        if technology_context:
            prompt = f"{prompt}\n\n{technology_context}".strip()
    elif kind == "deployment":
        technology_context = _deployment_technology_context(technology_recommendations)
        if technology_context:
            prompt = f"{prompt}\n\n{technology_context}".strip()
    elif kind == "hadr":
        technology_context = _hadr_technology_context(technology_recommendations)
        if technology_context:
            prompt = f"{prompt}\n\n{technology_context}".strip()
    return prompt


def _deployment_bullets(
    understanding: RFPUnderstanding | None,
    technology_recommendations: TechnologyRecommendationSet | None = None,
) -> List[str]:
    """Grounded deployment/release defaults for the deployment architecture slide."""
    technology_points = _deployment_recommendation_points(technology_recommendations)
    if technology_points:
        return technology_points
    scope = (getattr(understanding, "project_scope", "") or "").lower() if understanding else ""
    req_text = " ".join(
        (getattr(r, "text", "") or "") for r in (getattr(understanding, "requirements", []) or [])
    ).lower() if understanding else ""
    text = f"{scope} {req_text}"

    bullets = [
        "Separate lower and production environments with controlled promotion",
        "Secure connectivity to source systems, data repository, and reporting consumers",
        "Centralize observability across ingestion, integration, repository, and analytics layers",
    ]
    if any(k in text for k in ("critical", "24x7", "high availability", "minimal downtime", "sla")):
        bullets.append("Use blue-green or rolling deployment to reduce production downtime")
    else:
        bullets.append("Use phased rollout or pilot cutover when business risk is higher")
    bullets.append("Confirm hosting platform, environment topology, and release gates during discovery")
    return bullets


def _agile_delivery_points() -> List[BulletPoint]:
    """Customer-pre-read content for an Agile delivery operating model."""
    return [
        BulletPoint(
            text="Prioritise outcomes through one backlog",
            sub_points=[
                "The customer Product Owner owns priorities and acceptance, while squads refine requirements into thin, demonstrable end-to-end slices.",
                "Backlog decisions remain traceable to RFP outcomes, dependencies, risks, and measurable acceptance evidence.",
            ],
        ),
        BulletPoint(
            text="Deliver through cross-functional sprints",
            sub_points=[
                "Persistent squads combine analysis, architecture, engineering, data or integration, quality automation, security, DevSecOps, and change skills.",
                "A proposed two-week cadence uses planning, daily coordination, integrated demonstrations, and retrospectives to maintain flow and transparency.",
            ],
        ),
        BulletPoint(
            text="Build quality and operations into increments",
            sub_points=[
                "The Definition of Done includes automated testing, security controls, documentation, observability, deployment readiness, and support acceptance.",
                "Architecture, test, security, and operational work progress within each sprint rather than becoming downstream approval or remediation phases.",
            ],
        ),
        BulletPoint(
            text="Release value and learn continuously",
            sub_points=[
                "Usable increments move through controlled release gates using the deployment pattern suited to business risk and operational constraints.",
                "Customer feedback, delivery measures, production evidence, and emerging risks drive backlog reprioritisation and the next release forecast.",
            ],
        ),
    ]


def _agile_roadmap_points() -> List[BulletPoint]:
    """Iterative roadmap horizons that avoid design-build-test handoffs."""
    return [
        BulletPoint(
            text="Mobilise around outcomes and flow",
            sub_points=[
                "Establish product ownership, squad topology, governance, environments, initial backlog, dependency map, Definition of Done, and delivery measures."
            ],
        ),
        BulletPoint(
            text="Create the architecture runway",
            sub_points=[
                "Validate priority use cases, source interfaces, security controls, deployment automation, and thin end-to-end slices while discovery continues."
            ],
        ),
        BulletPoint(
            text="Deliver and demonstrate increments",
            sub_points=[
                "Recurring sprints build, integrate, test, secure, document, and demonstrate usable capabilities; feedback immediately reprioritises the backlog."
            ],
        ),
        BulletPoint(
            text="Release, operate and improve",
            sub_points=[
                "Move accepted increments through controlled production releases, hypercare and service transition, then use operational evidence to guide continuous improvement."
            ],
        ),
    ]


def _agile_squad_points() -> List[BulletPoint]:
    """Roles and decision rights for product-aligned, cross-functional squads."""
    return [
        BulletPoint(
            text="Customer-led product ownership",
            sub_points=[
                "The Product Owner sets outcome priorities, accepts increments, resolves business decisions, and connects squads to users and subject-matter experts."
            ],
        ),
        BulletPoint(
            text="Persistent cross-functional squads",
            sub_points=[
                "Each squad contains the analysis, architecture, engineering, data/integration, quality, security, DevSecOps, and change skills needed to complete usable increments."
            ],
        ),
        BulletPoint(
            text="Flow-oriented delivery leadership",
            sub_points=[
                "The Scrum Master or Agile Delivery Lead removes impediments, manages dependencies, improves predictability, and protects empowered squad decision-making."
            ],
        ),
        BulletPoint(
            text="Enabling chapters and outcome governance",
            sub_points=[
                "Architecture, security, data, quality, and engineering chapters provide standards and coaching; steering forums resolve cross-squad risks and outcome decisions."
            ],
        ),
    ]


_AGILE_DELIVERY_MARKERS = (
    "agile",
    "sprint",
    "backlog",
    "product owner",
    "scrum",
    "cross-functional",
    "increment",
    "retrospective",
    "definition of done",
    "devsecops",
    "continuous delivery",
    "iterative",
)


def _slide_visible_text(slide: SlideSpec) -> str:
    parts = [slide.title or "", slide.key_message or ""]
    parts.extend(slide.bullets or [])
    for point in slide.detailed_points or []:
        parts.append(point.text or "")
        parts.extend(point.sub_points or [])
    for card in slide.cards or []:
        parts.extend([card.heading or "", card.body or ""])
        parts.extend(card.bullets or [])
    return " ".join(parts).lower()


def _has_agile_delivery_language(slide: SlideSpec) -> bool:
    text = _slide_visible_text(slide)
    strong_markers = ("product owner", "cross-functional", "sprint", "backlog", "scrum")
    if any(marker in text for marker in strong_markers):
        return True
    return sum(marker in text for marker in _AGILE_DELIVERY_MARKERS) >= 2


def _hadr_bullets(understanding: RFPUnderstanding | None) -> List[str]:
    reqs = [
        (getattr(r, "text", "") or "").strip()
        for r in (getattr(understanding, "requirements", []) or [])
        if (getattr(r, "text", "") or "").strip()
    ] if understanding else []
    availability_reqs = [
        r for r in reqs
        if any(k in r.lower() for k in ("availability", "disaster", "backup", "restore", "rto", "rpo", "failover", "sla"))
    ][:3]
    bullets = availability_reqs or [
        "Redundant application, integration, and data layers sized for business criticality",
        "Automated backup and tested restore procedures for repository and configuration data",
        "Monitoring, alerting, and incident response across ingestion, processing, and reporting",
    ]
    bullets.append("RTO/RPO values to be confirmed if not explicitly specified in the RFP")
    return bullets[:6]


def _has_explicit_hadr_need(understanding: RFPUnderstanding | None) -> bool:
    if understanding is None:
        return False
    text = " ".join(
        [
            getattr(understanding, "project_scope", "") or "",
            " ".join(getattr(understanding, "risks", []) or []),
            " ".join((getattr(r, "text", "") or "") for r in getattr(understanding, "requirements", []) or []),
        ]
    ).lower()
    return any(
        token in text
        for token in (
            "high availability",
            "disaster recovery",
            "dr environment",
            "backup",
            "restore",
            "failover",
            "rto",
            "rpo",
            "business continuity",
        )
    )


def _sbom_table(understanding: RFPUnderstanding | None) -> Dict[str, Any]:
    headers = ["Component", "Category", "Purpose", "Basis", "Version / Constraint"]
    rows: List[List[str]] = []
    if understanding is not None:
        for item in getattr(understanding, "software_bill_of_materials", []) or []:
            component = (getattr(item, "component", "") or "").strip()
            if not component or _is_excluded_solution_tool(component, understanding):
                continue
            rows.append([
                component,
                (getattr(item, "category", "") or "Solution component").strip(),
                (getattr(item, "purpose", "") or "Supports proposed solution").strip(),
                (getattr(item, "source_or_basis", "") or "RFP-derived / to confirm").strip(),
                (getattr(item, "version_or_constraint", "") or "not specified in RFP").strip(),
            ])

        seen = {r[0].lower() for r in rows}
        for tech in getattr(understanding, "solution_technologies", []) or []:
            tech = (tech or "").strip()
            if tech and not _is_excluded_solution_tool(tech, understanding) and tech.lower() not in seen:
                rows.append([
                    tech,
                    "Named solution technology",
                    "Required or referenced by solution scope",
                    "Explicit RFP mention",
                    "not specified in RFP",
                ])
                seen.add(tech.lower())

    if not rows:
        rows = [
            ["Source systems and documents", "Source/input", "Provide operational and document data", "Derived from integration scope", "to be confirmed"],
            ["Ingestion and extraction services", "Application service", "Extract, validate, and transform required data", "Derived from data extraction scope", "to be confirmed"],
            ["Integration/orchestration layer", "Integration", "Coordinate APIs, workflows, mappings, and error handling", "Derived from integration scope", "to be confirmed"],
            ["Centralized data repository", "Data store", "Store curated operational, document, and reference data", "Derived from repository scope", "to be confirmed"],
            ["Reporting and dashboard layer", "Analytics", "Enable operational reporting, dashboards, and insights", "Derived from reporting scope", "to be confirmed"],
            ["Security, monitoring, and audit services", "Operations/security", "Provide access controls, logging, alerting, and auditability", "Derived from NFR scope", "to be confirmed"],
        ]
    return {"headers": headers, "rows": rows[:14]}


def _source_grounded_technology_table(
    understanding: RFPUnderstanding | None,
) -> Dict[str, Any]:
    """Fallback table containing source facts only.

    Product selection belongs to the technology-recommendation agent. This
    table is used only when that agent cannot provide a complete result, so it
    must never inject a cloud ecosystem, framework, database, toolchain or
    region merely because a platform was selected.
    """
    rows: List[List[str]] = []
    seen: set[str] = set()
    for item in (getattr(understanding, "software_bill_of_materials", []) or []):
        technology = (getattr(item, "component", "") or "").strip()
        if not technology or _is_excluded_solution_tool(technology, understanding):
            continue
        rows.append([
            (getattr(item, "category", "") or "Source-referenced technology").strip(),
            technology,
            (getattr(item, "purpose", "") or "Role described in the supplied material").strip(),
            (getattr(item, "source_or_basis", "") or "Named in supplied material; validate authority and fit").strip(),
        ])
        seen.add(technology.lower())
    for technology in (
        list(getattr(understanding, "solution_technologies", []) or [])
        + list(getattr(understanding, "key_technologies", []) or [])
    ):
        technology = (technology or "").strip()
        if (
            not technology
            or technology.lower() in seen
            or _is_excluded_solution_tool(technology, understanding)
        ):
            continue
        rows.append([
            "Source-referenced technology",
            technology,
            "Architecture role must be derived from the proposal requirements",
            "Named in supplied material; no product or layer inference applied",
        ])
        seen.add(technology.lower())

    if not rows:
        rows.append([
            "Agent recommendation unavailable",
            "No product selected by fallback",
            "Regenerate so the architecture agent can derive the applicable layers and products from proposal-specific inputs",
            "No source-named technology was available for a safe fallback",
        ])
    return {
        "headers": ["Source classification", "Named technology", "Source-described role", "Source / decision status"],
        "rows": rows[:15],
    }


def _is_usable_technology_table(table: Dict[str, Any] | None) -> bool:
    """Accept a substantive planner recommendation without vendor allowlists."""
    if not table or len(table.get("headers") or []) < 3:
        return False
    rows = table.get("rows") or []
    if len(rows) < 4 or not all(isinstance(row, (list, tuple)) and len(row) >= 3 for row in rows):
        return False
    headers_text = " ".join(str(value) for value in table["headers"]).lower()
    table_text = " ".join(str(cell) for row in rows for cell in row).lower()
    if "sdlc phase" in headers_text:
        return False
    capability_signals = (
        "develop", "test", "build", "deploy", "ci/cd", "devsecops",
        "ingest", "integration", "database", "data", "runtime", "compute",
        "security", "identity", "observability", "monitor", "report",
    )
    return sum(signal in table_text for signal in capability_signals) >= 3


def _technology_recommendation_table(
    recommendation_set: TechnologyRecommendationSet | None,
) -> Dict[str, Any] | None:
    recommendations = list(getattr(recommendation_set, "recommendations", None) or [])
    if len(recommendations) < 4:
        return None
    generic_only = (
        "digital catalogue", "catalogue platform", "ai-enabled engine",
        "compliance module", "pricing module", "customer portal",
        "application service", "database capability", "cloud service",
    )
    rows = []
    for item in recommendations:
        technology = (item.proposed_technology or "").strip()
        if not technology or technology.lower() in generic_only:
            continue
        category = (item.technology_category or "").strip()
        proposed = f"{technology} ({category})" if category and category.lower() not in technology.lower() else technology
        rationale = _clip(
            (item.build_vs_buy_rationale or item.rationale or "").strip(),
            105,
        )
        sourcing = (
            (item.sourcing_model or "").replace("-", " ")
            if item.sourcing_model != "customer-decision" else ""
        )
        basis_parts = [item.status.title(), sourcing, rationale]
        rows.append([
            _clip(item.architecture_layer, 55),
            proposed,
            _clip(item.role, 105),
            ": ".join(part for part in basis_parts if part),
        ])
    if len(rows) < 4:
        return None
    return {
        "headers": ["Architecture layer", "Proposed technology / service", "Role in the solution", "Status / rationale"],
        "rows": rows,
    }


def _selected_provider_family(platform: str | None) -> str:
    value = (platform or "").strip().lower()
    if "azure" in value:
        return "azure"
    if "amazon web services" in value or re.search(r"\baws\b", value):
        return "aws"
    if "google cloud" in value or re.search(r"\bgcp\b", value):
        return "gcp"
    return ""


_PROVIDER_PRODUCT_SIGNALS = {
    "azure": (
        "azure", "entra id", "cosmos db", "microsoft sentinel",
    ),
    "aws": (
        "amazon web services", "aws ", "amazon s3", "amazon ecs", "amazon eks",
        "route 53", "fargate", "cloudfront", "dynamodb", "elasticache",
        "amazon aurora", "amazon rds", "cloudwatch", "amazon cognito",
    ),
    "gcp": (
        "google cloud", "gcp ", "bigquery", "cloud run", "cloud sql", "gke",
        "pub/sub", "firestore", "cloud spanner", "google kubernetes engine",
    ),
}


def _conflicts_with_selected_provider(text: str, selected_provider: str) -> bool:
    if not selected_provider:
        return False
    value = f" {(text or '').lower()} "
    return any(
        signal in value
        for provider, signals in _PROVIDER_PRODUCT_SIGNALS.items()
        if provider != selected_provider
        for signal in signals
    )


def _deployment_recommendations(
    recommendation_set: TechnologyRecommendationSet | None,
) -> List[TechnologyRecommendation]:
    deployment_terms = (
        "host", "cloud", "region", "network", "vnet", "vpc", "subnet", "dns",
        "edge", "ingress", "gateway", "firewall", "waf", "load balanc", "runtime",
        "compute", "container", "kubernetes", "function", "application", "api",
        "integration", "messag", "event", "database", "data store", "storage",
        "search", "cache", "identity", "secret", "key", "certificate", "security",
        "siem", "monitor", "observ", "logging", "backup", "recovery", "availability",
        "ci/cd", "pipeline", "artifact", "infrastructure as code", "iac", "deploy",
    )
    recommendations = list(getattr(recommendation_set, "recommendations", None) or [])
    selected_provider = _selected_provider_family(
        getattr(recommendation_set, "selected_platform", "")
    )
    selected = [
        item for item in recommendations
        if item.status != "customer-decision"
        and not _is_open_visual_decision(
            f"{item.architecture_layer} {item.proposed_technology} {item.role}"
        )
        and any(
            term in f"{item.architecture_layer} {item.technology_category} {item.role}".lower()
            for term in deployment_terms
        )
        and not _conflicts_with_selected_provider(
            f"{item.proposed_technology} {item.technology_category}",
            selected_provider,
        )
    ]
    return selected[:16]


def _technical_architecture_context(
    recommendation_set: TechnologyRecommendationSet | None,
) -> str:
    """Render authoritative build/buy and technology decisions into one visual context."""
    if recommendation_set is None:
        return ""
    platform = (recommendation_set.selected_platform or "").strip()
    selected_provider = _selected_provider_family(platform)
    decisions = [
        item for item in (recommendation_set.component_decisions or [])
        if item.decision_status != "customer-decision"
        and item.sourcing_model != "customer-decision"
        and (item.capability or "").strip()
        and (item.recommendation or "").strip()
        and not _is_open_visual_decision(
            f"{item.capability} {item.recommendation} {item.role} {item.system_of_record}"
        )
        and not _conflicts_with_selected_provider(
            f"{item.recommendation} {item.role}", selected_provider
        )
    ][:10]
    technologies = [
        item for item in (recommendation_set.recommendations or [])
        if item.status != "customer-decision"
        and (item.proposed_technology or "").strip()
        and not _is_open_visual_decision(
            f"{item.architecture_layer} {item.proposed_technology} {item.role}"
        )
        and not _conflicts_with_selected_provider(
            f"{item.proposed_technology} {item.technology_category}", selected_provider
        )
    ][:14]
    if not platform and not decisions and not technologies:
        return ""

    parts = [
        "AUTHORITATIVE LAYERED TECHNICAL ARCHITECTURE DECISIONS - these override provisional visual labels:"
    ]
    if platform and not _is_open_visual_decision(platform):
        parts.append(f"Selected hosting platform: {platform}.")
    if decisions:
        parts.append("Map these capability sourcing decisions into the relevant technical layers:")
        for item in decisions:
            detail = [
                f"{item.capability}: {item.recommendation}",
                f"sourcing={item.sourcing_model}",
                f"status={item.decision_status}",
            ]
            if item.system_of_record:
                detail.append(f"system-of-record={_clip(item.system_of_record, 80)}")
            if item.data_inputs:
                detail.append("inputs=" + ", ".join(_clip(value, 55) for value in item.data_inputs[:4]))
            if item.data_outputs:
                detail.append("outputs=" + ", ".join(_clip(value, 55) for value in item.data_outputs[:4]))
            if item.role:
                detail.append("role=" + _clip(item.role, 100))
            if item.rationale:
                detail.append("design rationale=" + _clip(item.rationale, 120))
            parts.append("- " + "; ".join(detail) + ".")
    if technologies:
        parts.append("Use these selected implementation products/services in the corresponding layers:")
        parts.extend(
            f"- {item.architecture_layer}: {item.proposed_technology} "
            f"({item.technology_category}; {item.role}; {item.status}; {item.sourcing_model})."
            for item in technologies
        )
    parts.append(
        "Draw only selected decisions, not evaluated or rejected alternatives. Use the sourcing model as a compact "
        "legend/tag and use the rationale only to place the component correctly, not as visible prose. Connect each "
        "named or role-based system of record to the component it supplies and label the flow with the listed data. "
        "Do not introduce another hyperscaler's services. Keep customer decisions and open assumptions on the "
        "Assumptions and Dependencies slide, outside this visual."
    )
    return "\n".join(parts)


def _deployment_technology_context(
    recommendation_set: TechnologyRecommendationSet | None,
) -> str:
    selected = _deployment_recommendations(recommendation_set)
    platform = (getattr(recommendation_set, "selected_platform", "") or "").strip()
    if (
        recommendation_set is None
        or not platform
        or _is_open_visual_decision(platform)
    ):
        return ""
    hosting_model = (recommendation_set.hosting_model or "customer-decision").replace("-", " ")
    parts = [
        "AUTHORITATIVE DEPLOYMENT TECHNOLOGY DECISION - this overrides any earlier visual draft:",
        f"Hosting model: {hosting_model}.",
        f"Selected platform: {platform}.",
    ]
    if recommendation_set.deployment_rationale and not _is_open_visual_decision(
        recommendation_set.deployment_rationale
    ):
        parts.append("Decision rationale: " + _clip(recommendation_set.deployment_rationale, 240) + ".")
    if recommendation_set.primary_region_strategy and not _is_open_visual_decision(
        recommendation_set.primary_region_strategy
    ):
        parts.append("Region and resilience strategy: " + _clip(recommendation_set.primary_region_strategy, 200) + ".")
    if selected:
        parts.append("Map these chosen services to their deployment layers:")
        for item in selected:
            parts.append(
                f"- {item.architecture_layer}: {item.proposed_technology} ({item.role}; {item.status})."
            )
    else:
        parts.append(
            "Show the application, integration, data, identity, observability, and release capabilities "
            "inside the selected-platform boundary using role-based labels; do not invent service names."
        )
    parts.append(
        "Group services into no more than 8 deployable layers. Show provider-native service names, "
        "network boundaries and directional traffic flows. Do not replace the selected services with generic boxes. "
        "Do not introduce services from another hyperscaler. Do not print open decisions, assumptions, dependencies, "
        "placeholder labels, or customer-confirmation qualifiers inside the visual."
    )
    return "\n".join(parts)


def _hadr_technology_context(
    recommendation_set: TechnologyRecommendationSet | None,
) -> str:
    platform = (getattr(recommendation_set, "selected_platform", "") or "").strip()
    if recommendation_set is None or not platform or _is_open_visual_decision(platform):
        return ""
    resilience_terms = (
        "availability", "zone", "region", "failover", "load balanc", "traffic", "dns",
        "replication", "backup", "restore", "recovery", "monitor", "health",
    )
    selected_provider = _selected_provider_family(platform)
    selected = [
        item for item in (recommendation_set.recommendations or [])
        if item.status != "customer-decision"
        and not _is_open_visual_decision(
            f"{item.architecture_layer} {item.proposed_technology} {item.role}"
        )
        and any(term in f"{item.architecture_layer} {item.technology_category} {item.role}".lower() for term in resilience_terms)
        and not _conflicts_with_selected_provider(
            f"{item.proposed_technology} {item.technology_category}",
            selected_provider,
        )
    ][:10]
    parts = [
        "AUTHORITATIVE HA/DR TECHNOLOGY DECISION - use only these resilience services; do not draw the full deployment stack:",
        f"Selected platform: {platform}.",
    ]
    if recommendation_set.primary_region_strategy and not _is_open_visual_decision(
        recommendation_set.primary_region_strategy
    ):
        parts.append(
            "Named region strategy: "
            + _clip(recommendation_set.primary_region_strategy, 240)
            + "."
        )
    if selected:
        parts.extend(f"- {item.architecture_layer}: {item.proposed_technology} ({item.role})." for item in selected)
    else:
        parts.append(
            "Use role-based runtime, data, replication, backup, health, and recovery labels within the "
            "selected-platform boundary; do not invent service names or recovery commitments."
        )
    parts.append(
        "Do not introduce services from another hyperscaler. Keep open recovery targets, assumptions, "
        "dependencies, placeholder labels, and customer-confirmation qualifiers outside the visual."
    )
    return "\n".join(parts)


def _deployment_recommendation_points(
    recommendation_set: TechnologyRecommendationSet | None,
) -> List[str]:
    selected = _deployment_recommendations(recommendation_set)
    if recommendation_set is None:
        return []
    platform = recommendation_set.selected_platform or "the selected platform"
    hosting = (recommendation_set.hosting_model or "customer-decision").replace("-", " ")
    points = [f"Use a {hosting} deployment baseline on {platform}"]
    if not selected:
        if not (recommendation_set.selected_platform or "").strip():
            return []
        return points + [
            "Separate lower and production environments with controlled promotion",
            "Secure user and source-system ingress through the customer-approved landing zone",
            "Centralize runtime, integration, data, security, and operational telemetry",
        ]
    buckets = (
        ("Network and secure ingress", ("network", "vnet", "vpc", "dns", "edge", "gateway", "firewall", "waf", "load balanc")),
        ("Application and integration runtime", ("runtime", "compute", "container", "kubernetes", "function", "application", "api", "integration", "messag", "event")),
        ("Data services", ("database", "data store", "storage", "search", "cache")),
        ("Security and operations", ("identity", "secret", "key", "certificate", "security", "siem", "monitor", "observ", "logging", "backup", "recovery", "ci/cd", "artifact", "iac")),
    )
    for heading, terms in buckets:
        technologies = []
        for item in selected:
            haystack = f"{item.architecture_layer} {item.technology_category} {item.role}".lower()
            if any(term in haystack for term in terms):
                technologies.append(item.proposed_technology)
        technologies = list(dict.fromkeys(technologies))[:4]
        if technologies:
            points.append(f"{heading}: " + ", ".join(technologies))
    return points[:5]


def _apply_technology_recommendations(
    deck_plan: DeckPlan,
    recommendation_set: TechnologyRecommendationSet | None,
) -> DeckPlan:
    table = _technology_recommendation_table(recommendation_set)
    if table is None:
        return deck_plan
    for slide in deck_plan.slides:
        if (slide.archetype or "").strip().lower() == "software bill of materials":
            slide.table = table
            slide.title = "Proposed solution stack maps technologies to architecture layers"
            slide.key_message = (
                "The proposed stack preserves stated constraints and selects concrete products for unspecified layers using workload, NFR, operability, portability, and cost fit."
            )
    return deck_plan


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


def _truncate_on_word(text: str, max_len: int) -> str:
    """Truncate to <= max_len at a clean boundary, with NO trailing ellipsis.

    A visible "…" reads as a broken sentence on a customer-facing slide, so we
    end at the last complete sentence in range, else the last clause
    (comma/semicolon/colon), else the last whole word — always leaving a clause
    that reads as finished, never "…extraction qual…".
    """
    window = text[:max_len]
    # 1) last complete sentence within a sensible portion of the window
    sentence = re.search(r"^.*[.!?](?=\s|$)", window)
    if sentence and len(sentence.group(0)) >= max_len * 0.45:
        return sentence.group(0).strip()
    # 2) last clause boundary
    for sep in ("; ", ": ", ", "):
        idx = window.rfind(sep)
        if idx >= max_len * 0.5:
            return window[:idx].rstrip(" ,;:")
    # 3) last whole word
    space = window.rstrip().rfind(" ")
    if space > 0:
        window = window[:space]
    return window.rstrip(" ,;:")


def _first_sentence(text: str, max_len: int = 200) -> str:
    t = (text or "").strip()
    if not t:
        return ""
    first_line = t.splitlines()[0].strip()
    parts = first_line.split(". ")
    sent = parts[0].strip()
    if len(sent) > max_len:
        return _truncate_on_word(sent, max_len)
    return sent


def _clip(text: str, max_len: int = 160) -> str:
    original = re.sub(r"\s+", " ", (text or "").strip())
    if len(original) <= max_len:
        # Already fits — keep its own closing punctuation. Stripping the
        # trailing period unconditionally (even when nothing was cut) made a
        # complete sentence read as a dangling fragment for no reason.
        return original
    t = original.rstrip(".")
    if len(t) > max_len:
        return _truncate_on_word(t, max_len)
    return t


def _concise_heading(text: str, max_len: int = 72) -> str:
    """A short, complete card heading — never a mid-sentence truncation.

    Prefers the label before a colon (e.g. "Integration scope readiness: …"),
    then the first sentence, then a clean word-boundary cut. Strips meta
    lead-ins like "Inferred from " so the heading reads as a label.
    """
    t = re.sub(r"\s+", " ", (text or "").strip())
    t = re.sub(r"^inferred from\s+", "", t, flags=re.I).strip()
    if not t:
        return ""
    t = t[0].upper() + t[1:]
    if ":" in t:
        lead = t.split(":", 1)[0].strip()
        if 6 <= len(lead) <= max_len:
            return lead
    first = re.split(r"(?<=[.!?])\s+", t, maxsplit=1)[0].strip()
    if len(first) <= max_len:
        return first.rstrip(".")
    return _truncate_on_word(first, max_len).rstrip(".")


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
            return [re.sub(r"\s+", " ", p).strip() for p in pts[:3]]

        bullets: List[str] = []
        vp = (getattr(narrative, "value_proposition", "") or "").strip()
        if vp:
            bullets.append(re.sub(r"\s+", " ", vp).strip())
        for o in getattr(narrative, "strategic_outcomes", []) or []:
            if (o or "").strip():
                bullets.append(re.sub(r"\s+", " ", o).strip())
            if len(bullets) >= 3:
                break
        if len(bullets) < 3:
            for t in getattr(narrative, "solution_themes", []) or []:
                if (t or "").strip():
                    bullets.append(re.sub(r"\s+", " ", t).strip())
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
    env_subs = [re.sub(r"\s+", " ", a).strip() for a in _visible_assumptions(understanding)][:3]
    if not env_subs:
        first = _first_sentence(getattr(understanding, "summary", "") or "", 140)
        if first:
            env_subs = [first]
    if env_subs:
        points.append(BulletPoint(text="Current environment and constraints", sub_points=env_subs))

    # Stakeholder needs & pain points — from must/should requirements.
    reqs = getattr(understanding, "requirements", []) or []
    ranked = sorted(reqs, key=lambda r: {"must": 0, "should": 1, "may": 2}.get(getattr(r, "priority", "should"), 1))
    need_subs = [re.sub(r"\s+", " ", getattr(r, "text", "")).strip() for r in ranked if (getattr(r, "text", "") or "").strip()][:4]
    if need_subs:
        points.append(BulletPoint(text="Stakeholder needs and pain points", sub_points=need_subs))

    # Why change now — from stated risks/pressures driving the initiative.
    why_subs = [re.sub(r"\s+", " ", r).strip() for r in (getattr(understanding, "risks", []) or []) if (r or "").strip()][:3]
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
        target.append(re.sub(r"\s+", " ", text).strip())

    points: List[BulletPoint] = []
    if functional:
        points.append(BulletPoint(text="Functional scope", sub_points=functional[:4]))
    if nonfunctional:
        points.append(BulletPoint(text="Non-functional requirements", sub_points=nonfunctional[:4]))
    return points


def _complete_sentences(text: str, limit: int = 2) -> str:
    """Select complete leading sentences without cutting a sentence mid-word."""
    clean = re.sub(r"\s+", " ", (text or "").strip())
    if not clean:
        return ""
    sentences = [part.strip() for part in re.split(r"(?<=[.!?])\s+", clean) if part.strip()]
    return " ".join(sentences[:limit])


def _exec_summary_cards(
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
) -> List[Card]:
    """Build a self-contained executive summary when the model returns slogans."""
    if understanding is None and narrative is None:
        return []

    # Each card is a scannable set of distinct bullets — NOT a dense paragraph
    # echoed by bullets. The three cards draw from *different* angles (situation
    # vs response vs outcomes) and are de-duplicated across cards so the summary
    # never repeats the same sentence in every box.
    def _clean(items, limit=3):
        # Exec Summary is exempt from the renderer's card-pagination escape
        # hatch (kept to one slide by design), so a bullet has to fit the box
        # it's committed to — _clip's word/clause-boundary truncation is what
        # guarantees that instead of silently overflowing. The cap is sized to
        # rarely trigger (the renderer's own font-fit floor, not this cap, is
        # what actually keeps dense cards inside the box) — a lower cap here
        # just clips a sentence before the box ever needed it to, and a clause
        # boundary inside an enumerated list ("Incident, Problem, Change...")
        # isn't a safe place to cut anyway, so the box-fit shrink is the
        # correct backstop, not a tighter character budget.
        return [_clip(i.strip(), 170) for i in (items or []) if (i or "").strip()][:limit]

    def _sentences(text, limit=3):
        parts = re.split(r"(?<=[.!?])\s+", re.sub(r"\s+", " ", (text or "").strip()))
        return _clean([p for p in parts if len(p.split()) >= 4], limit)

    # Situation = the client context/problem (summary), NOT the scope of work
    # (that is the response) — this is what removes the cross-card repetition.
    situation_bullets = _sentences(getattr(understanding, "summary", "") if understanding else "")
    if not situation_bullets and understanding is not None:
        situation_bullets = _clean(getattr(understanding, "in_scope_work", []))

    response_bullets = _clean(getattr(narrative, "solution_themes", []) if narrative else [])
    if not response_bullets:
        response_bullets = _clean(getattr(understanding, "in_scope_work", []) if understanding else [])
    if not response_bullets and narrative is not None:
        response_bullets = _sentences(getattr(narrative, "value_proposition", ""))

    outcome_bullets = _clean(getattr(narrative, "strategic_outcomes", []) if narrative else [])
    if not outcome_bullets and understanding is not None:
        outcome_bullets = _clean(
            [getattr(r, "text", "") for r in (getattr(understanding, "requirements", []) or [])
             if getattr(r, "priority", "should") == "must"]
        )

    # De-duplicate near-identical points across the three cards, but never let a
    # card fall below two bullets when its source had more (backfill with the
    # least-overlapping remaining items).
    seen: set[str] = set()

    def _key(item):
        return " ".join(re.sub(r"[^a-z0-9 ]", "", item.lower()).split()[:8])

    def _dedupe(items, min_keep=2):
        out = []
        for item in items:
            k = _key(item)
            if k and not any(k in s or s in k for s in seen):
                seen.add(k)
                out.append(item)
        for item in items:  # backfill to keep the card substantial
            if len(out) >= min(min_keep, len(items)):
                break
            if item not in out:
                out.append(item)
        return out

    situation_bullets = _dedupe(situation_bullets)
    response_bullets = _dedupe(response_bullets)
    outcome_bullets = _dedupe(outcome_bullets)

    cards = [
        Card(heading="Client situation and stakes", body="", bullets=situation_bullets, accent="challenge"),
        Card(heading="Our proposed response", body="", bullets=response_bullets, accent="solution"),
        Card(heading="Outcomes and delivery confidence", body="", bullets=outcome_bullets, accent="outcome"),
    ]
    return [card for card in cards if card.bullets]


def _technology_dependency_items(
    recommendation_set: TechnologyRecommendationSet | None,
) -> List[str]:
    if recommendation_set is None:
        return []
    items = [
        _clip(item, 180)
        for item in (recommendation_set.platform_assumptions or [])
        if (item or "").strip() and not _is_internal_source_note(item)
    ]
    for recommendation in recommendation_set.recommendations or []:
        if recommendation.status != "customer-decision":
            continue
        layer = _clip(recommendation.architecture_layer, 60)
        role = _clip(recommendation.role, 130)
        decision = f"Confirm {layer}: {role}" if layer and role else (layer or role)
        if decision:
            items.append(decision)
    for component in recommendation_set.component_decisions or []:
        for assumption in component.open_assumptions or []:
            if (assumption or "").strip() and not _is_internal_source_note(assumption):
                items.append(_clip(assumption, 180))
        if component.decision_status == "customer-decision" or component.sourcing_model == "customer-decision":
            capability = _clip(component.capability, 70)
            role = _clip(component.role or component.recommendation, 110)
            decision = (
                f"Confirm sourcing and product decision for {capability}: {role}"
                if capability and role else capability or role
            )
            if decision:
                items.append(decision)
    return list(dict.fromkeys(item for item in items if item))[:5]


def _visual_dependency_items(
    visual_briefs: List[DiagramBrief] | None,
    recommendation_set: TechnologyRecommendationSet | None,
) -> List[str]:
    selected_provider = _selected_provider_family(
        getattr(recommendation_set, "selected_platform", "")
    )
    items: List[str] = []
    for brief in visual_briefs or []:
        if (brief.visual_type or "").lower() not in {
            "architecture", "technical_architecture", "deployment", "hadr", "topology", "testing", "ams"
        }:
            continue
        for item in brief.open_assumptions or []:
            clean = _clip(item, 180)
            if (
                clean
                and not _is_internal_source_note(clean)
                and not _conflicts_with_selected_provider(clean, selected_provider)
            ):
                items.append(clean)
    return list(dict.fromkeys(items))[:8]


def _assumptions_dependency_points(
    understanding: RFPUnderstanding | None,
    technology_recommendations: TechnologyRecommendationSet | None = None,
    visual_briefs: List[DiagramBrief] | None = None,
) -> List[BulletPoint]:
    if understanding is None:
        return []
    assumptions = _visible_assumptions(understanding)[:5]
    dependency_terms = (
        "access", "availability", "provide", "readiness", "interface", "source",
        "sample", "approval", "hosting", "environment", "third party", "customer",
    )
    dependencies = []
    for requirement in getattr(understanding, "requirements", []) or []:
        text = (getattr(requirement, "text", "") or "").strip()
        if text and any(term in text.lower() for term in dependency_terms):
            dependencies.append(text)
        if len(dependencies) >= 5:
            break
    constraints = [
        item.strip()
        for item in (getattr(understanding, "risks", []) or [])
        if (item or "").strip()
    ][:5]
    platform = _matched_requirement_texts(
        understanding,
        ("hosting", "cloud", "azure", "environment", "security", "access", "network", "identity"),
        3,
    )
    technology_dependencies = _technology_dependency_items(technology_recommendations)
    acceptance = _matched_requirement_texts(
        understanding,
        ("acceptance", "uat", "approval", "sign-off", "cutover", "warranty", "support", "sla"),
        3,
    )
    if _ai_ml_is_applicable(understanding):
        platform = list(platform[:2]) + [
            "AI-assisted use cases proceed beyond pilot only after data readiness, accuracy, human-control, security, and unit-cost thresholds are accepted."
        ]
    dependency_candidates = technology_dependencies + _visual_dependency_items(
        visual_briefs,
        technology_recommendations,
    )
    for item in dependency_candidates:
        value = item.lower()
        if any(term in value for term in ("interface", "source system", "source data", "data volume", "access")):
            dependencies.insert(0, item)
        elif any(term in value for term in ("recovery", "backup", "rto", "rpo", "support", "service level", "monitoring", "retention")):
            acceptance.insert(0, item)
        elif any(term in value for term in (
            "hosting", "cloud", "landing-zone", "subscription", "connectivity",
            "environment", "runtime", "release", "deployment", "network", "identity", "security",
        )):
            platform.insert(0, item)
        else:
            assumptions.insert(0, item)
    assumptions = list(dict.fromkeys(assumptions))
    dependencies = list(dict.fromkeys(dependencies))
    platform = list(dict.fromkeys(platform))
    acceptance = list(dict.fromkeys(acceptance))
    return [
        BulletPoint(
            text="Scope and design baseline to validate",
            sub_points=assumptions[:3] or [
                "Confirm requirement priorities, process variants, volumes, retention, service levels, and non-functional acceptance thresholds during mobilisation."
            ],
        ),
        BulletPoint(
            text="Customer inputs and access dependencies",
            sub_points=dependencies[:3] or [
                "Customer owners provide representative data, interface specifications, subject-matter experts, environments, credentials, and timely design decisions."
            ],
        ),
        BulletPoint(
            text="Platform, security and environment decisions",
            sub_points=platform[:3] or [
                "Confirm target hosting, connectivity, identity, security controls, monitoring integration, environment topology, and recovery responsibilities before detailed design."
            ],
        ),
        BulletPoint(
            text="Acceptance and operational readiness dependencies",
            sub_points=acceptance[:3] or constraints[:3] or [
                "Agree acceptance evidence, business UAT ownership, cutover windows, rollback criteria, support handover, warranty measures, and production sign-off authorities."
            ],
        ),
    ]


def _data_domain_points(understanding: RFPUnderstanding | None) -> List[BulletPoint]:
    source_points = _matched_requirement_texts(
        understanding,
        (
            "flight", "fih", "gp4", "elp", "sftp", "email", "source",
            "catalogue", "catalog", "product", "service", "sku", "ingredient",
            "dietary", "allergen", "packaging", "facility", "market",
        ),
        3,
    )
    operational_points = _matched_requirement_texts(
        understanding,
        (
            "uplift", "catering", "productivity", "accuracy", "milestone", "sla", "iccms",
            "customer requirement", "brief", "solution", "shortlist", "matching",
            "validation", "compliance", "feasibility", "pricing", "enquiry", "availability",
        ),
        3,
    )
    control_points = _matched_requirement_texts(
        understanding,
        ("validate", "reconciliation", "audit", "retention", "quality", "security", "lineage"),
        3,
    )
    consumer_points = _matched_requirement_texts(
        understanding,
        (
            "report", "dashboard", "analytics", "output", "consumer", "power bi",
            "approved content", "publish", "customer content", "visualization", "visualisation",
        ),
        3,
    )
    return [
        BulletPoint(
            text="Authoritative source and reference domains",
            sub_points=source_points or ["Separate authoritative source records from derived operational data and retain source lineage."],
        ),
        BulletPoint(
            text="Canonical operational domains",
            sub_points=operational_points or ["Organise the hub around stable business entities, events, milestones, and measurable operational outcomes."],
        ),
        BulletPoint(
            text="Quality, reconciliation and governance controls",
            sub_points=control_points or ["Apply ownership, validation, exception handling, auditability, retention, and reconciliation consistently across domains."],
        ),
        BulletPoint(
            text="Governed serving and consumption domains",
            sub_points=consumer_points or ["Publish trusted operational views and reusable analytical products without duplicating source-system logic."],
        ),
    ]


def _risk_mitigation(risk: str) -> str:
    low = (risk or "").lower()
    if any(term in low for term in ("source", "data", "interface", "extract", "api")):
        return "Validate representative samples, interface contracts, and fallback handling during mobilisation."
    if any(term in low for term in ("availability", "recovery", "backup", "rto", "rpo")):
        return "Confirm recovery ownership, evidence, and test acceptance before production readiness approval."
    if any(term in low for term in ("security", "access", "privacy", "compliance")):
        return "Agree the control model, evidence requirements, and accountable owners before build completion."
    if any(term in low for term in ("cutover", "operation", "disrupt", "change")):
        return "Use phased release, parallel validation, rollback readiness, and visible cutover support."
    if any(
        term in low
        for term in ("ownership", "boundary", "boundaries", "coordinat", "escalation", "third part", "vendor", "raci")
    ):
        return "Agree a RACI across every named party during mobilisation and close boundary gaps before go-live."
    if any(
        term in low
        for term in ("report", "reporting", "repository", "baseline", "kpi", "visibility", "metric", "dashboard")
    ):
        return "Stand up the reporting baseline and repository early, and validate KPI accuracy before it drives decisions."
    return "Assign an accountable owner and validate the mitigation through the programme RAID process."


def _risk_detailed_points(understanding: RFPUnderstanding | None) -> List[BulletPoint]:
    fallback_risks = [
        "Source-system access, sample data, or interface readiness may delay discovery and build validation.",
        "Security, privacy, and production-readiness approvals may take longer than planned if evidence expectations are not agreed early.",
        "Business availability for reviews, clarifications, and acceptance may constrain sprint throughput and sign-off.",
    ]
    if understanding is None:
        risks = fallback_risks
    else:
        risks = [
            item.strip()
            for item in (getattr(understanding, "risks", []) or [])
            if (item or "").strip()
        ][:5]
        if not risks:
            risks = fallback_risks
    return [
        BulletPoint(text=risk, sub_points=[_risk_mitigation(risk)])
        for risk in risks
    ]


def _expansion_detailed_points(understanding: RFPUnderstanding | None) -> List[BulletPoint]:
    """Grounded fallback body for an optional/later-phase expansion slide.

    A slide about optional scope still needs its own visible argument —
    "optional" describes the customer's decision, not licence to leave the
    slide empty. Named later-phase items come from the same
    ``optional_response_topics`` the RFP-understanding step already extracts
    (visible elsewhere, e.g. a commercials slide's "optional expansion" card,
    proving the data exists); this only backfills the *dedicated* expansion
    slide's own body when generation left it blank.
    """
    profile = getattr(understanding, "engagement_profile", None) if understanding else None
    topics = [t.strip() for t in (getattr(profile, "optional_response_topics", None) or []) if (t or "").strip()][:5]
    points = [
        BulletPoint(
            text="Phase 1 stays self-contained",
            sub_points=[
                "Every required capability is delivered and evidenced in Phase 1; nothing in scope depends on an "
                "optional item being taken up."
            ],
        ),
    ]
    if topics:
        points.append(BulletPoint(text="Electable Phase 2 scope", sub_points=topics))
    else:
        points.append(
            BulletPoint(
                text="Electable Phase 2 scope",
                sub_points=["Later-phase options are scoped and priced separately once Phase 1 evidence is available."],
            )
        )
    points.append(
        BulletPoint(
            text="Each option activates only when evidence supports it",
            sub_points=[
                "Service performance evidence, dependencies, business benefit, and commercial terms are confirmed "
                "before an option starts.",
                "The customer decides which options to take up and when — none are assumed or pre-committed.",
            ],
        )
    )
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
    broad_existing = {"architecture", "content", "solution overview"}
    for a in existing_archetypes:
        a = (a or "").lower()
        if not a:
            continue
        if target_lower == a or target_lower in a:
            return True
        if a in broad_existing:
            continue
        if a in target_lower:
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
    planned_sections = _proposal_section_skeleton(understanding)
    planned_ids = {str(section.get("slide_id", "")) for section in planned_sections}
    profile = _effective_engagement_profile(understanding)
    managed = _profile_is_managed_operations(profile)
    # Helper: add slide
    def add_slide(
        archetype: str,
        title: str,
        bullets: Optional[List[str]] = None,
        diagram: Optional[DiagramSpec] = None,
        detailed_points: Optional[List[BulletPoint]] = None,
        table: Optional[Dict[str, Any]] = None,
        key_message: Optional[str] = None,
        cards: Optional[List[Card]] = None,
    ) -> None:
        deck_plan.slides.append(
            SlideSpec(
                slide_id=f"auto_{_tight_id(title)}",
                title=title,
                archetype=archetype,
                bullets=bullets or [],
                detailed_points=detailed_points or [],
                diagram=diagram,
                table=table,
                key_message=key_message,
                cards=cards or [],
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
    else:
        # Native HCLTech agenda layouts reserve a narrow/structural title area.
        # A long claim title turns into an oversized, badly wrapped divider.
        for slide in deck_plan.slides:
            if (slide.archetype or "").strip().lower() == "agenda":
                slide.title = "Agenda"

    # Exec Summary as Solution Overview (required)
    # (Some models output “Executive Overview” etc; ordering will still handle it.)
    has_exec = any(_is_exec_summary(s) for s in deck_plan.slides)
    if not has_exec:
        add_slide(
            "Solution Overview",
            "Executive Summary",
            bullets=[],
            cards=_exec_summary_cards(understanding, narrative),
            key_message=(getattr(narrative, "value_proposition", "") or "").strip() if narrative else None,
            diagram=None,
        )
    else:
        for s in deck_plan.slides:
            if not _is_exec_summary(s):
                continue
            visible_text = " ".join(
                list(s.bullets or [])
                + [point.text for point in (s.detailed_points or [])]
                + [sub for point in (s.detailed_points or []) for sub in (point.sub_points or [])]
                + [card.heading for card in (s.cards or [])]
                + [card.body for card in (s.cards or [])]
                + [bullet for card in (s.cards or []) for bullet in (card.bullets or [])]
            )
            if _is_placeholder_exec_bullets(s.bullets) or len(visible_text.split()) < 90:
                s.bullets = []
                s.detailed_points = []
                s.cards = _exec_summary_cards(understanding, narrative)
                if narrative and not (s.key_message or "").strip():
                    s.key_message = (getattr(narrative, "value_proposition", "") or "").strip()

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
    if "sk_arch" in planned_ids and not _is_archetype_present(existing_keys, "architecture"):
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

    # Deployment Architecture
    if "sk_deployment" in planned_ids and not _is_archetype_present(existing_keys, "deployment architecture"):
        add_slide(
            "Deployment Architecture",
            "Deployment and resilience architecture protects operations",
            bullets=_deployment_bullets(understanding),
            diagram=DiagramSpec(
                kind="deployment",
                prompt=_build_diagram_prompt("deployment", understanding),
                approved=False,
                image_path=None,
            ),
        )

    has_platform_nfr = any(
        any(
            token in ((slide.title or "") + " " + _slide_visible_text(slide)).lower()
            for token in (
                "secured communication",
                "data security",
                "observability",
                "audit",
                "logging",
                "monitoring",
                "non-functional",
            )
        )
        for slide in deck_plan.slides
    )
    if "sk_security" in planned_ids and not has_platform_nfr:
        add_slide(
            "Architecture",
            "Security, observability and NFR controls are built in",
            detailed_points=[
                BulletPoint(
                    text="End-to-end secured communication",
                    sub_points=[
                        "Use encrypted transport for data movement between source systems, integration services, APIs, data stores, and reporting consumers.",
                        "Apply identity-aware access, network segmentation, and controlled service-to-service connectivity aligned to the customer's hosting and security standards.",
                    ],
                ),
                BulletPoint(
                    text="Data security and auditability",
                    sub_points=[
                        "Protect operational data through role-based access, least privilege, secrets management, retention controls, and auditable transaction history.",
                        "Maintain traceability from ingestion through validation, transformation, reporting, and support actions.",
                    ],
                ),
                BulletPoint(
                    text="Observability and resilience",
                    sub_points=[
                        "Centralize logs, metrics, alerts, interface health, pipeline status, and SLA signals so support teams can detect and resolve issues quickly.",
                        "Design backup, recovery, availability, and operational readiness controls as part of the platform rather than as post-build activities.",
                    ],
                ),
            ],
            key_message=(
                "Even when not stated as separate RFP line items, security, observability, auditability, and resilience are mandatory for a production platform."
            ),
        )

    # High Availability and DR
    if "sk_hadr" in planned_ids and not _is_archetype_present(existing_keys, "high availability"):
        add_slide(
            "High Availability & DR",
            "HA and DR protect business continuity",
            bullets=_hadr_bullets(understanding),
            diagram=DiagramSpec(
                kind="hadr",
                prompt=_build_diagram_prompt("hadr", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Software Bill of Materials
    if "sk_tech" in planned_ids and not _is_archetype_present(existing_keys, "software bill of materials"):
        add_slide(
            "Software Bill of Materials",
            "Proposed solution stack maps services to architecture layers",
            key_message=(
                "RFP-mandated technologies are preserved; gaps are completed with qualified ecosystem-aligned recommendations that remain subject to architecture confirmation."
            ),
            table=_source_grounded_technology_table(understanding),
        )
    else:
        # The planner sometimes creates the required SBOM slide but omits the
        # structured table payload. Complete the existing slide instead of
        # accepting an empty table layout.
        for slide in deck_plan.slides:
            if (slide.archetype or "").strip().lower() == "software bill of materials":
                if not _is_usable_technology_table(slide.table):
                    slide.table = _source_grounded_technology_table(understanding)
                elif _ai_ml_is_applicable(understanding):
                    rows = list(slide.table.get("rows") or [])
                    if not any("ai-assisted" in " ".join(str(cell) for cell in row).lower() for row in rows):
                        ai_row = [
                            "AI-assisted capabilities",
                            "Rules/statistical baselines, managed AI services, or small models",
                            "Classification, anomaly detection, forecasting, and governed assistance",
                            "Optional; scale only after value, accuracy, security, and run-cost validation",
                        ]
                        width = len(slide.table.get("headers") or ai_row)
                        slide.table["rows"] = rows + [(ai_row + [""] * width)[:width]]
                slide.title = "Proposed solution stack maps services to architecture layers"
                slide.key_message = (
                    "RFP-mandated technologies are preserved; proposed services close implementation gaps and remain subject to customer architecture confirmation."
                )

    has_sdlc_tech = any(
        (slide.archetype or "").strip().lower() == "software bill of materials"
        for slide in deck_plan.slides
    )
    if "sk_tech" in planned_ids and not has_sdlc_tech:
        add_slide(
            "Software Bill of Materials",
            "Proposed solution stack maps services to architecture layers",
            key_message=(
                "This view names the implementation services behind each architecture layer and distinguishes requirements from proposed choices."
            ),
            table=_source_grounded_technology_table(understanding),
        )

    has_ai_ml_slide = any(
        any(token in ((slide.title or "") + " " + _slide_visible_text(slide)).lower()
            for token in ("ai-assisted", "ai/ml opportunity", "machine learning opportunity"))
        for slide in deck_plan.slides
    )
    if "sk_ai_opportunities" in planned_ids and _ai_ml_is_applicable(understanding):
        if not has_ai_ml_slide:
            add_slide(
                "Value & Differentiators",
                "AI-assisted capabilities target high-value exceptions",
                key_message=(
                    "A rules-first core protects operational continuity while optional AI is introduced only where data readiness, measurable value, and run cost justify it."
                ),
                cards=_ai_ml_cards(understanding),
            )
        else:
            for slide in deck_plan.slides:
                slide_text = ((slide.title or "") + " " + _slide_visible_text(slide)).lower()
                if not any(token in slide_text for token in ("ai-assisted", "ai/ml opportunity", "machine learning opportunity")):
                    continue
                if not slide.cards and not slide.detailed_points and not slide.bullets:
                    slide.cards = _ai_ml_cards(understanding)
                slide.key_message = slide.key_message or (
                    "A rules-first core protects operational continuity while optional AI is introduced only where data readiness, measurable value, and run cost justify it."
                )

    if "sk_ai_opportunities" in planned_ids and _ai_ml_is_applicable(understanding):
        ai_risk = BulletPoint(
            text="AI accuracy, drift, and consumption cost",
            sub_points=[
                "Baseline accuracy and unit cost on representative data; apply confidence thresholds, human review, usage budgets, monitoring, and deterministic fallback before production scale."
            ],
        )
        ai_dependency = BulletPoint(
            text="AI adoption depends on evidence",
            sub_points=[
                "Progress each use case from data-readiness assessment to a measured pilot; production adoption depends on accepted accuracy, explainability, security, and run-cost thresholds."
            ],
        )
        for slide in deck_plan.slides:
            arch = (slide.archetype or "").strip().lower()
            title = (slide.title or "").lower()
            if arch == "risks" or "risk" in title:
                if slide.cards and len(slide.cards) < 4 and not any(
                    "ai accuracy" in (card.heading or "").lower() for card in slide.cards
                ):
                    slide.cards.append(Card(
                        heading=ai_risk.text,
                        bullets=ai_risk.sub_points,
                        accent="risk",
                    ))
                elif len(slide.detailed_points or []) < 4 and not any(
                    "ai accuracy" in (point.text or "").lower() for point in slide.detailed_points or []
                ):
                    slide.detailed_points.append(ai_risk)
            if arch == "assumptions & dependencies":
                if slide.cards and len(slide.cards) < 4 and not any(
                    "ai adoption" in (card.heading or "").lower() for card in slide.cards
                ):
                    slide.cards.append(Card(
                        heading=ai_dependency.text,
                        bullets=ai_dependency.sub_points,
                        accent="info",
                    ))
                elif len(slide.detailed_points or []) < 4 and not any(
                    "ai adoption" in (point.text or "").lower() for point in slide.detailed_points or []
                ):
                    slide.detailed_points.append(ai_dependency)

    # Assumptions and dependencies are essential in a customer pre-read when
    # delivery relies on unresolved inputs, access, hosting, or third parties.
    assumption_points = _assumptions_dependency_points(understanding)
    if "sk_assumptions" in planned_ids and assumption_points and not _is_archetype_present(existing_keys, "assumptions & dependencies"):
        add_slide(
            "Assumptions & Dependencies",
            "Delivery conditions must be confirmed early",
            detailed_points=assumption_points,
            key_message=(
                "Early validation of assumptions and dependencies protects scope, schedule, and operational readiness."
            ),
        )

    # Delivery Plan
    if "sk_testing" in planned_ids and not _is_archetype_present(existing_keys, "delivery plan"):
        add_slide(
            "Delivery Plan",
            "Acceptance evidence proves the solution is ready",
            detailed_points=_testing_proposal_points(understanding),
            key_message=(
                "Requirement-led evidence and named customer decisions control release readiness."
            ),
            diagram=DiagramSpec(
                kind="testing",
                prompt=_build_diagram_prompt("testing", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Timeline
    if "sk_roadmap" in planned_ids and not _is_archetype_present(existing_keys, "timeline"):
        roadmap_title = (
            "Mobilization, transition and stabilization establish the live service"
            if managed else "Incremental delivery releases value through controlled outcomes"
        )
        add_slide(
            "Timeline",
            roadmap_title,
            detailed_points=_agile_roadmap_points(),
            key_message=(
                "Each stage has explicit outcomes, customer decisions and readiness evidence."
            ),
            diagram=DiagramSpec(
                kind="timeline",
                prompt=_build_diagram_prompt("timeline", understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Risks
    if "sk_risks" in planned_ids and not _is_archetype_present(existing_keys, "risks"):
        add_slide(
            "Risks",
            "Risks & Mitigations",
            bullets=[
                "Key delivery and technical risks",
                "Mitigation actions and owners",
                "Assumptions and dependencies",
            ],
        )

    # A roadmap or governance slide does not replace a dedicated delivery-team view.
    has_team_structure = any(
        (s.archetype or "").strip().lower() == "team"
        for s in deck_plan.slides
    )
    if "sk_governance" in planned_ids and not has_team_structure:
        if managed:
            team_title = "Governance and service leadership make accountability explicit"
            team_points = [
                BulletPoint(text="Named service leadership", sub_points=["A Service Delivery Lead owns day-to-day execution, performance, stakeholder alignment and improvement."]),
                BulletPoint(text="Accountable process ownership", sub_points=["Named operational roles execute the required service practices with explicit decision and escalation rights."]),
                BulletPoint(text="Cross-party operating boundaries", sub_points=["A RACI separates customer, provider and third-party responsibilities, approvals and evidence obligations."]),
                BulletPoint(text="Evidence-led governance", sub_points=["Operational, service-performance and executive reviews convert measures, risks and trends into recorded actions."]),
            ]
            team_message = "Named accountability and evidence-led governance keep the service controlled and measurable."
            diagram_kind = "org"
            diagram_prompt_kind = "team"
        else:
            team_title = "Product-aligned Agile team structure supports delivery and transition"
            team_points = _agile_squad_points()
            team_message = "Persistent cross-functional squads own usable outcomes end to end, supported by enabling chapters and lightweight steering governance."
            diagram_kind = "org"
            diagram_prompt_kind = "team"
        add_slide(
            "Team",
            team_title,
            detailed_points=team_points,
            key_message=team_message,
            diagram=DiagramSpec(
                kind=diagram_kind,
                prompt=_build_diagram_prompt(diagram_prompt_kind, understanding),
                approved=False,
                image_path=None,
            ),
        )

    # Commercials (always)
    if "sk_commercials" in planned_ids and not _is_archetype_present(existing_keys, "commercials"):
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

    # Slides added later in this function must receive the same AI governance
    # treatment as model-provided slides handled above.
    if "sk_ai_opportunities" in planned_ids and _ai_ml_is_applicable(understanding):
        governance_items = {
            "risks": (
                "AI accuracy, drift, and consumption cost",
                "Baseline accuracy and unit cost on representative data; use confidence thresholds, human review, usage budgets, monitoring, and deterministic fallback before production scale.",
                "risk",
            ),
            "assumptions & dependencies": (
                "AI adoption depends on evidence",
                "Progress from data-readiness assessment to a measured pilot; production adoption depends on accepted accuracy, explainability, security, and run-cost thresholds.",
                "info",
            ),
        }
        for slide in deck_plan.slides:
            arch = (slide.archetype or "").strip().lower()
            if arch not in governance_items:
                continue
            heading, detail, accent = governance_items[arch]
            if heading.lower() in _slide_visible_text(slide).lower():
                continue
            if slide.cards:
                if len(slide.cards) < 4:
                    slide.cards.append(Card(heading=heading, bullets=[detail], accent=accent))
                else:
                    slide.cards[-1].bullets = [detail] + list(slide.cards[-1].bullets or [])[:2]
            elif slide.detailed_points:
                if len(slide.detailed_points) < 4:
                    slide.detailed_points.append(BulletPoint(text=heading, sub_points=[detail]))
                else:
                    slide.detailed_points[-1].sub_points = list(slide.detailed_points[-1].sub_points or []) + [detail]
            elif slide.bullets:
                slide.bullets = list(slide.bullets[:4]) + [f"{heading}: {detail}"]
            else:
                slide.detailed_points = [BulletPoint(text=heading, sub_points=[detail])]

        for slide in deck_plan.slides:
            title = (slide.title or "").lower()
            if not any(token in title for token in ("security", "observability", "non-functional", "nfr")):
                continue
            if "model monitoring" in _slide_visible_text(slide).lower():
                continue
            heading = "AI governance and model operations"
            detail = (
                "Control authorised data access, prompt/model versions, evaluation evidence, confidence thresholds, "
                "human review, usage cost, drift, and rollback to deterministic processing."
            )
            if slide.cards and len(slide.cards) < 4:
                slide.cards.append(Card(heading=heading, bullets=[detail], accent="why"))
            elif slide.detailed_points and len(slide.detailed_points) < 4:
                slide.detailed_points.append(BulletPoint(text=heading, sub_points=[detail]))
            elif slide.bullets and len(slide.bullets) < 5:
                slide.bullets.append(f"{heading}: {detail}")

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


# The best-in-class proposal arc, sequenced by audience (the "song"):
#   Act 1 (business) -> Act 2 (architecture/engineering) -> Act 3 (delivery/ops)
#   -> Act 4 (why-us + the ask). Commercials come last, after value is built and
#   "why us" is proven; core solution content (data/integration/security) sits
#   in Act 2, never after Commercials.
_DECK_BEATS = [
    "title", "agenda", "exec", "context", "outcomes",                       # Act 1
    "solution", "scope", "flow", "architecture", "data", "integration",
    "security", "deployment", "ha_dr", "ai", "sbom",                        # Act 2
    "delivery", "timeline", "testing", "ams", "team",                        # Act 3
    "whyus", "case_studies", "risks", "mapping", "assumptions",
    "commercials", "next",                                                   # Act 4
]
_BEAT_RANK = {beat: i for i, beat in enumerate(_DECK_BEATS)}


def _deck_beat(slide: SlideSpec) -> str:
    """Map a slide to its narrative beat (title keywords win over archetype)."""
    if _is_exec_summary(slide):
        return "exec"
    arch = (slide.archetype or "").strip().lower()
    t = (slide.title or "").strip().lower()

    def has(*keys: str) -> bool:
        return any(k in t for k in keys)

    if arch == "title":
        return "title"
    if arch == "agenda":
        return "agenda"
    # The win-theme statement slide is spliced in right after the Exec
    # Summary and must stay grouped with it — its title is blank, so it
    # would otherwise fall through to the generic "solution" (Act 2) default
    # and land after the Act-2 divider instead of before it.
    if arch == "win theme":
        return "exec"
    # Terminal archetypes first: their descriptive titles often contain solution
    # keywords (e.g. "Commercials align build, warranty and support") that would
    # otherwise mis-route them into an earlier act.
    if arch == "commercials":
        return "commercials"
    if arch == "risks":
        return "risks"
    if arch == "assumptions & dependencies":
        return "assumptions"
    if arch == "requirements mapping":
        return "mapping"
    if arch == "next steps":
        return "next"
    # Act 1
    if arch == "customer context" or has("current challenge", "current state", "challenges", "pain point"):
        return "context"
    if has("target outcome", "outcomes align", "business outcome", "measurable outcome", "success measure", "business value"):
        return "outcomes"
    # Act 4 "why us" — check before generic value/differentiator archetype
    if has("why hcltech", "why hcl", "differentiator", "proposal value", "value and differentiator", "credential"):
        return "whyus"
    if arch == "case studies" or has("case stud"):
        return "case_studies"
    # Act 2 (solution)
    if has("ai-assisted", "ai/ml", "machine learning", "ai capabilit", "ai opportun"):
        return "ai"
    if arch == "software bill of materials" or has("technology stack", "solution stack", "bill of materials", "sbom"):
        return "sbom"
    if has("integration") or arch == "integration":
        return "integration"
    if has("data domain", "core data", "information model", "data model", "reporting", "analytics"):
        return "data"
    if has("security", "observability", "compliance", "non-functional", " nfr"):
        return "security"
    if arch == "deployment architecture" or has("deployment", "resilience", "environment topology", "release topology"):
        return "deployment"
    if arch == "high availability & dr" or has("high availability", "ha and dr", "ha & dr", "disaster recovery", "business continuity"):
        return "ha_dr"
    if arch == "architecture" or has("architecture"):
        return "architecture"
    if has("end-to-end", "operational flow", "process flow", "solution flow", "data flow"):
        return "flow"
    if arch == "requirements" or has("scope", "boundaries", "in scope", "out of scope"):
        return "scope"
    if arch == "solution overview" or has("proposed solution", "solution at a glance", "target state", "design principle", "design tenet"):
        return "solution"
    # Act 3 (delivery/ops)
    if has("warranty", "ams ", " ams", "application maintenance", "hypercare", "live-service", "support model"):
        return "ams"
    if arch == "timeline" or has("roadmap", "timeline", "increment", "release plan"):
        return "timeline"
    if has("testing", "acceptance", "quality", "uat", "readiness evidence"):
        return "testing"
    if arch == "team" or has("governance", "squad", "team", "operating model", "raci", "decision right"):
        return "team"
    if arch == "delivery plan" or has("delivery", "agile", "sprint", "phased delivery"):
        return "delivery"
    # Act 4 (ask)
    if arch == "value & differentiators":
        return "whyus"
    if arch == "risks" or has("risk", "mitigation"):
        return "risks"
    if arch == "requirements mapping" or has("mapping", "traceability"):
        return "mapping"
    if arch == "assumptions & dependencies" or has("assumption", "dependenc"):
        return "assumptions"
    if arch == "commercials" or has("commercial", "pricing", "investment", "fees", "cost model"):
        return "commercials"
    if arch == "next steps" or has("next step", "call to action"):
        return "next"
    # Generic content defaults into the solution act, never after commercials.
    return "solution"


def order_deck(deck_plan: DeckPlan) -> DeckPlan:
    """Order slides into a best-in-class, audience-sequenced narrative arc.

    Stable within a beat (keeps the model's relative order), and forces the
    Executive Summary first within Act 1's opening.
    """
    def sort_key(indexed):
        idx, slide = indexed
        beat = _deck_beat(slide)
        return (_BEAT_RANK.get(beat, _BEAT_RANK["solution"]), idx)

    ordered = sorted(enumerate(deck_plan.slides), key=sort_key)
    deck_plan.slides = [s for _, s in ordered]
    return deck_plan


def synchronize_agenda(deck_plan: DeckPlan) -> DeckPlan:
    """Build the agenda from sections that actually exist in the final plan."""
    slides = [
        slide for slide in deck_plan.slides
        if (slide.archetype or "").strip().lower() not in {"title", "agenda"}
    ]

    def has_archetype(*names: str) -> bool:
        wanted = {name.lower() for name in names}
        return any((slide.archetype or "").strip().lower() in wanted for slide in slides)

    agenda_items: List[str] = []
    if any(_is_exec_summary(slide) for slide in slides):
        agenda_items.append("Executive Summary")
    if has_archetype("Customer Context", "Requirements"):
        agenda_items.append("Understanding, Scope and Requirements")
    if has_archetype("Solution Overview"):
        agenda_items.append("Proposed Solution and Design Principles")
    if has_archetype(
        "Architecture",
        "Deployment Architecture",
        "High Availability & DR",
        "Requirements Mapping",
        "Software Bill of Materials",
    ):
        agenda_items.append("Architecture, Security and Resilience")
    if has_archetype("Delivery Plan", "Timeline", "Team"):
        agenda_items.append("Delivery, Governance and Team")
    if has_archetype("Assumptions & Dependencies", "Risks"):
        agenda_items.append("Assumptions, Dependencies and Risks")
    if has_archetype("Case Studies", "Value & Differentiators", "Commercials"):
        agenda_items.append("Value, Evidence and Commercials")
    if has_archetype("Next Steps"):
        agenda_items.append("Next Steps")

    for slide in deck_plan.slides:
        if (slide.archetype or "").strip().lower() != "agenda":
            continue
        slide.title = "Agenda"
        slide.bullets = agenda_items
        slide.detailed_points = []
        slide.cards = []
        slide.comparison = None
        slide.table = None
        slide.diagram = None
    return deck_plan


def prune_empty_content_slides(deck_plan: DeckPlan) -> DeckPlan:
    """Remove decorative/title-only content slides that add no proposal value.

    Structural slides and any slide carrying real text, a table, comparison,
    KPI, or diagram specification are retained. This primarily catches model
    output that selects a full-image/section-like layout but supplies only a
    title, which otherwise appears as an unnecessary section header.
    """
    retained: List[SlideSpec] = []
    for slide in deck_plan.slides:
        archetype = (slide.archetype or "content").strip().lower()
        has_payload = bool(
            slide.bullets
            or slide.detailed_points
            or slide.cards
            or slide.comparison is not None
            or slide.kpis
            or slide.table
            or slide.diagram is not None
            or (slide.key_message or "").strip()
        )
        if archetype == "content" and not has_payload:
            log.info("Dropping empty content slide %s (%r)", slide.slide_id, slide.title)
            continue
        retained.append(slide)
    deck_plan.slides = retained
    return deck_plan


def _storyline_bucket(slide: SlideSpec) -> str:
    arch = (slide.archetype or "content").strip().lower()
    title = (slide.title or "").strip().lower()
    text = _slide_visible_text(slide)
    if arch in {"title", "agenda", "next steps", "commercials"}:
        return arch
    if _is_exec_summary(slide):
        return "executive_summary"
    if arch == "software bill of materials" or any(k in title for k in ("technology", "technologies", "sbom", "bill of materials")):
        return "sbom"
    if arch == "assumptions & dependencies" or "assumption" in title or (
        "dependenc" in title and "scope" not in title
    ):
        return "assumptions"
    if arch == "risks" or "risk" in title or "mitigation" in title:
        return "risks"
    if arch == "value & differentiators" or any(
        k in title for k in ("value proposition", "differentiator", "why hcltech", "business value")
    ):
        return "value_differentiators"
    if arch == "requirements mapping" or "mapping" in title:
        return "mapping"
    if arch == "customer context" or any(k in title for k in ("current", "challenge", "pain point")):
        return "current_state"
    if arch == "requirements" or any(k in title for k in ("scope", "requirement", "priority")):
        return "scope_requirements"
    if arch == "solution overview" or any(k in title for k in ("proposed", "target state", "design tenet", "workstream", "data flow")):
        return "solution_overview"
    if arch == "architecture":
        return "solution_architecture"
    if arch == "high availability & dr" or any(
        k in title for k in ("high availability", "ha and dr", "ha & dr", "disaster recovery", "business continuity")
    ):
        return "ha_dr"
    if arch == "deployment architecture" or any(
        k in title for k in ("deployment", "environment topology", "release topology")
    ):
        return "deployment_architecture"
    if arch in {"delivery plan", "timeline", "team"} or any(
        k in text for k in ("agile", "squad", "scrum", "roadmap", "governance")
    ):
        return "delivery_governance"
    return "content"


def _slide_payload_score(slide: SlideSpec) -> int:
    score = 0
    if slide.diagram:
        score += 6
    if slide.table:
        score += 5
    if slide.comparison:
        score += 4
    score += min(len(slide.cards or []), 4) * 2
    score += min(len(slide.detailed_points or []), 6) * 2
    score += min(len(slide.bullets or []), 6)
    score += 2 if (slide.key_message or "").strip() else 0
    return score


def prune_redundant_storyline_slides(
    deck_plan: DeckPlan,
    protected_ids: Optional[set] = None,
) -> DeckPlan:
    """Keep the main deck focused by dropping repeated generic storyline slides.

    ``protected_ids`` are planned keystone sections (from the proposal skeleton)
    that must never be pruned — each is a distinct required beat (value &
    differentiators, governance/team, AMS, etc.). Redundancy pruning only exists
    to remove *extra* slides the model invents, not the one-per-section skeleton.
    """
    protected_ids = protected_ids or set()
    max_by_bucket = {
        "solution_overview": 2,
        "current_state": 2,
        "scope_requirements": 2,
        "solution_architecture": 4,
        "deployment_architecture": 1,
        "ha_dr": 1,
        "delivery_governance": 5,
        "value_differentiators": 2,
        "risks": 2,
        "assumptions": 1,
        "sbom": 2,
        "mapping": 2,
        "content": 6,
    }
    kept: List[SlideSpec] = []
    bucket_counts: Dict[str, int] = {}
    for slide in deck_plan.slides:
        if str(getattr(slide, "slide_id", "") or "") in protected_ids:
            kept.append(slide)
            continue
        bucket = _storyline_bucket(slide)
        limit = max_by_bucket.get(bucket)
        if limit is None:
            kept.append(slide)
            continue
        count = bucket_counts.get(bucket, 0)
        if count >= limit:
            log.info(
                "Dropping redundant %s slide %s (%r)",
                bucket,
                slide.slide_id,
                slide.title,
            )
            continue
        # Drop weak, title/text-only generic content if the bucket already has
        # a stronger proof object.
        if count > 0 and _slide_payload_score(slide) <= 2:
            log.info(
                "Dropping weak duplicate %s slide %s (%r)",
                bucket,
                slide.slide_id,
                slide.title,
            )
            continue
        kept.append(slide)
        bucket_counts[bucket] = count + 1
    deck_plan.slides = kept
    return deck_plan


def prune_profile_misaligned_slides(
    deck_plan: DeckPlan,
    understanding: RFPUnderstanding | None,
) -> DeckPlan:
    """Remove lifecycle slides contradicted by the selected engagement policy.

    This protects the direct-planner and cached-plan paths as well as the
    engagement-specific chunked planner. It targets recognizable lifecycle
    artifacts rather than arbitrary content, preserving useful proposal extras.
    """
    sections = _proposal_section_skeleton(understanding)
    allowed_ids = {str(section.get("slide_id", "")).lower() for section in sections}
    allowed_visual_kinds = {
        str(section.get("diagram_kind", "")).lower()
        for section in sections
        if section.get("diagram_kind")
    }
    profile = _effective_engagement_profile(understanding)
    managed_without_build = _profile_is_managed_operations(profile) and not _profile_is_technical_delivery(profile)
    allowed_flags = {
        "architecture": any(item in allowed_ids for item in ("sk_arch", "sk_technical_arch", "sk_integration")),
        "deployment": "sk_deployment" in allowed_ids,
        "testing": "sk_testing" in allowed_ids,
        "technology": "sk_tech" in allowed_ids,
        "ai": "sk_ai_opportunities" in allowed_ids,
        "ams": "sk_ams" in allowed_ids,
    }

    kept: List[SlideSpec] = []
    for slide in deck_plan.slides:
        sid = (slide.slide_id or "").lower()
        title = (slide.title or "").lower()
        archetype = (slide.archetype or "").lower()
        if sid in allowed_ids:
            kept.append(slide)
            continue

        remove = False
        if archetype == "deployment architecture" or "deployment architecture" in title:
            remove = not allowed_flags["deployment"]
        elif archetype == "software bill of materials" or any(
            token in title for token in ("technology stack", "solution stack", "bill of materials")
        ):
            remove = not allowed_flags["technology"]
        elif archetype == "architecture" or any(
            token in title for token in ("solution architecture", "technical architecture", "integration architecture")
        ):
            remove = not allowed_flags["architecture"]
        elif any(token in title for token in (
            "testing strategy", "testing builds", "test strategy", "release confidence",
            "quality engineering", "quality assurance", "system testing", "integration testing",
            "user acceptance testing", " uat", " sit",
        )):
            remove = not allowed_flags["testing"]
        elif any(token in title for token in ("ai-assisted", "ai/ml", "artificial intelligence")):
            remove = not allowed_flags["ai"]
        elif any(token in title for token in ("warranty and ams", "ams support")):
            remove = not allowed_flags["ams"]
        elif managed_without_build and any(
            token in title for token in ("product-aligned squad", "agile team", "agile squad", "architecture runway")
        ):
            remove = True

        if remove:
            log.info(
                "Pruned profile-misaligned slide: engagement=%s slide_id=%s title=%r",
                profile.primary_type,
                slide.slide_id,
                slide.title,
            )
            continue
        if slide.diagram and not getattr(slide.diagram, "approved", False):
            kind = (getattr(slide.diagram, "kind", "") or "").lower()
            if kind not in allowed_visual_kinds:
                slide.diagram = None
        kept.append(slide)
    deck_plan.slides = kept
    return deck_plan


def enrich_slide_detail(
    deck_plan: DeckPlan,
    understanding: RFPUnderstanding | None = None,
    technology_recommendations: TechnologyRecommendationSet | None = None,
    visual_briefs: List[DiagramBrief] | None = None,
) -> DeckPlan:
    """Upgrade thin content slides with grounded sub-points and fix Next Steps.

    Applies to both model-generated and auto-added slides:
      - Customer Context / "Current State" slides lacking sub-points get
        substantive `detailed_points` drawn from the RFP understanding.
      - Requirements slides get functional vs non-functional sub-points.
      - Next Steps slides are sanitized of proposal logistics and, if needed,
        replaced with supplier-driven calls to action.
    """
    profile = _effective_engagement_profile(understanding)
    managed = _profile_is_managed_operations(profile)
    technical_delivery = _profile_is_technical_delivery(profile)
    for s in deck_plan.slides:
        arch = (s.archetype or "").lower()
        title = (s.title or "").lower()
        slide_id = (getattr(s, "slide_id", "") or "").lower()

        if arch == "assumptions & dependencies" or (
            "assumption" in title and "dependenc" in title
        ):
            s.title = "Assumptions and dependencies need early closure"
            s.key_message = (
                "These are proposal controls to validate during mobilisation, not unqualified statements of fact."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = _assumptions_dependency_points(
                understanding,
                technology_recommendations,
                visual_briefs,
            )
            continue

        is_technical_architecture_slide = (
            slide_id == "sk_technical_arch"
            or "technical architecture" in title
            or "layered architecture" in title
            or (
                getattr(s, "diagram", None) is not None
                and (getattr(s.diagram, "kind", "") or "").lower() == "technical_architecture"
            )
        )
        if is_technical_architecture_slide:
            s.title = "Layered technical architecture connects systems, products and custom services"
            s.key_message = (
                "The proposed composition reuses authoritative enterprise systems, selects products where they fit, "
                "and reserves custom development for differentiated workflows and decision logic."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = []
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "technical_architecture"
                s.diagram.prompt = _build_diagram_prompt(
                    "technical_architecture", understanding, technology_recommendations
                )
            continue

        is_deployment_slide = (
            arch == "deployment architecture"
            or (
                "deployment" in title
                and not any(token in title for token in ("high availability", "ha and dr", "disaster recovery"))
            )
            or (
                getattr(s, "diagram", None) is not None
                and (getattr(s.diagram, "kind", "") or "").lower() == "deployment"
            )
        )
        if is_deployment_slide:
            s.title = "Deployment and resilience protect operations"
            s.key_message = (
                "Environment separation, controlled promotion, secured runtime boundaries, and observability reduce release and operational risk."
            )
            s.bullets = _deployment_bullets(understanding, technology_recommendations)
            s.cards = []
            s.comparison = None
            s.detailed_points = []
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "deployment"
                s.diagram.prompt = _build_diagram_prompt("deployment", understanding)
            continue

        if slide_id == "sk_reporting" or any(
            token in title for token in ("semantic layer", "decision-ready reporting")
        ):
            s.title = "One governed semantic layer serves decision-ready reporting"
            s.key_message = (
                "Trusted catalogue, pricing, availability and workflow data is governed once, then reused consistently for operational, commercial, compliance and executive decisions."
            )
            s.bullets = []
            s.detailed_points = []
            s.cards = []
            s.comparison = None
            s.table = None
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "process"
                s.diagram.prompt = _build_diagram_prompt("reporting", understanding)
            continue

        is_data_model_slide = (
            (getattr(s, "slide_id", "") or "").lower() == "sk_data_model"
            or any(token in title for token in ("data domain", "core data", "data model", "information model"))
        )
        if is_data_model_slide:
            s.key_message = (
                "The data model separates authoritative inputs, canonical operational entities, control evidence, and governed consumption products."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = _data_domain_points(understanding)
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "data_model"
                s.diagram.prompt = _build_diagram_prompt("data_model", understanding)
            continue

        if arch == "delivery plan" and any(token in title for token in ("test", "acceptance", "uat", "sit")):
            customer = (getattr(understanding, "customer_name", None) or "customer").strip()
            s.title = "Acceptance evidence proves the solution is ready"
            s.key_message = (
                f"Testing is organised around replacement outcomes, named interfaces, data reconciliation, operational controls, and {customer} acceptance evidence."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = _testing_proposal_points(understanding)
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "testing"
                s.diagram.prompt = _build_diagram_prompt("testing", understanding)
            continue

        if arch == "delivery plan" and any(token in title for token in ("ams", "support", "warranty", "operate", "operations")):
            s.title = "AMS protects the complete live-service boundary"
            s.key_message = (
                "Support connects business-flow observability, interface ownership, data correction, runbooks, and measurable service evidence across the proposed platform."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = _ams_proposal_points(understanding)
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.kind = "ams"
                s.diagram.prompt = _build_diagram_prompt("ams", understanding)
            continue

        if slide_id == "sk_roadmap_detail":
            s.title = "Each roadmap increment has a clear outcome and exit gate"
            s.key_message = (
                "Every increment combines usable scope, customer decisions, assurance evidence and an explicit readiness outcome."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.table = None
            s.diagram = None
            s.detailed_points = _agile_roadmap_points()
            continue

        if (
            technical_delivery
            and not managed
            and arch in {"delivery plan", "timeline", "team"}
            and not _has_agile_delivery_language(s)
        ):
            if arch == "delivery plan":
                s.title = "Agile delivery turns priorities into usable increments"
                s.key_message = (
                    "Product ownership, persistent cross-functional squads, and evidence-based release decisions maintain speed without weakening governance."
                )
                s.detailed_points = _agile_delivery_points()
                diagram_kind = "delivery"
            elif arch == "timeline":
                s.title = "Agile roadmap releases value through recurring increments"
                s.key_message = (
                    "Discovery, engineering, assurance, security, and operational readiness progress together within each increment."
                )
                s.detailed_points = _agile_roadmap_points()
                diagram_kind = "timeline"
            else:
                s.title = "Product-aligned squads combine business and engineering ownership"
                s.key_message = (
                    "Persistent cross-functional squads own usable outcomes end to end, supported by enabling chapters and lightweight steering governance."
                )
                s.detailed_points = _agile_squad_points()
                diagram_kind = "team"
            s.bullets = []
            s.cards = []
            s.comparison = None
            if s.diagram is not None and not getattr(s.diagram, "approved", False):
                s.diagram.prompt = _build_diagram_prompt(diagram_kind, understanding)
            continue

        if arch == "next steps" or "next step" in title:
            _sanitize_next_steps(s, understanding)
            continue

        if arch == "risks" or "risk" in title:
            point_count = max(
                len(s.detailed_points or []),
                len(s.cards or []),
                len(s.bullets or []),
            )
            if point_count < 3:
                grounded_risks = _risk_detailed_points(understanding)
                if grounded_risks:
                    s.detailed_points = grounded_risks
                    s.cards = []
                    s.bullets = []
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
            s.bullets = [c for c in (_clean(b) for b in s.bullets) if c]
        # Normalize nested points too, preserving their structure.
        if getattr(s, "detailed_points", None):
            cleaned_points = []
            for point in s.detailed_points:
                text = _clean(getattr(point, "text", ""))
                if not text:
                    continue
                subs = [c for c in (_clean(sp) for sp in (point.sub_points or [])) if c]
                point.text = text
                point.sub_points = subs
                cleaned_points.append(point)
            s.detailed_points = cleaned_points
    return deck_plan


def _short_card_body(text: str, limit: int = 135) -> str:
    return re.sub(r"\s+", " ", (_complete_sentences(text, 1) or text).strip())


def _cards_from_detailed_points(points: List[BulletPoint], *, accent: str = "info", max_cards: int = 4) -> List[Card]:
    cards: List[Card] = []
    for point in points[:max_cards]:
        full = re.sub(r"\s+", " ", (getattr(point, "text", "") or "").strip())
        heading = _concise_heading(full)
        bullets = [
            re.sub(r"\s+", " ", item).strip()
            for item in (getattr(point, "sub_points", None) or [])
            if (item or "").strip()
        ][:3]
        # If the heading dropped meaningful detail from the point text, keep the
        # full point as the leading bullet so nothing is silently lost.
        if full and len(full) > len(heading) + 12 and not any(
            full[:40].lower() in b.lower() for b in bullets
        ):
            bullets = ([full] + bullets)[:3]
        if heading or bullets:
            cards.append(
                Card(
                    heading=heading or _concise_heading(bullets[0] if bullets else ""),
                    body="" if bullets else _short_card_body(full),
                    bullets=bullets,
                    accent=accent,
                )
            )
    return cards


def _risk_cards_from_detailed_points(points: List[BulletPoint], *, max_cards: int = 4) -> List[Card]:
    """Like ``_cards_from_detailed_points``, but for Risks specifically: the
    risk and its mitigation must read as visibly distinct, not one run-on
    paragraph. ``_risk_detailed_points`` builds each point as
    ``text=risk, sub_points=[mitigation]`` — put the risk in the card body
    (plain prose under the heading) and the mitigation as its own explicitly
    labelled bullet, so a reader can tell which is which without parsing two
    merged sentences.
    """
    cards: List[Card] = []
    for point in points[:max_cards]:
        risk = re.sub(r"\s+", " ", (getattr(point, "text", "") or "").strip())
        mitigations = [
            re.sub(r"\s+", " ", item).strip()
            for item in (getattr(point, "sub_points", None) or [])
            if (item or "").strip()
        ]
        if not risk and not mitigations:
            continue
        heading = _concise_heading(risk) or "Risk"
        cards.append(
            Card(
                heading=heading,
                body=risk,
                bullets=[f"Mitigation: {m}" for m in mitigations[:2]],
                accent="challenge",
            )
        )
    return cards


def _cards_from_bullets(bullets: List[str], *, accent: str = "info", max_cards: int = 4) -> List[Card]:
    cards: List[Card] = []
    for idx, bullet in enumerate([b for b in bullets if (b or "").strip()][:max_cards], start=1):
        heading = _concise_heading(bullet) or f"Point {idx}"
        body = bullet[len(heading):].strip(" .:-") if bullet.startswith(heading) else ""
        cards.append(
            Card(
                heading=heading,
                body=_short_card_body(body or bullet, 150),
                bullets=[],
                accent=accent,
            )
        )
    return cards


def _slide_line_items(slide: SlideSpec) -> List[str]:
    lines = [b for b in (slide.bullets or []) if (b or "").strip()]
    for point in slide.detailed_points or []:
        if (getattr(point, "text", "") or "").strip():
            lines.append(point.text)
        lines.extend([sp for sp in (point.sub_points or []) if (sp or "").strip()])
    return lines


def consulting_grade_proposal_polish(
    deck_plan: DeckPlan,
    understanding: RFPUnderstanding | None = None,
    narrative: ExecutiveNarrative | None = None,
) -> DeckPlan:
    """Prefer HCLTech-native proposal patterns over document-summary text slides."""
    for slide in deck_plan.slides:
        arch = (slide.archetype or "").strip().lower()
        title = (slide.title or "").strip().lower()

        if arch in {"title", "agenda"} or slide.table or slide.diagram:
            continue

        if _is_exec_summary(slide):
            cards = _exec_summary_cards(understanding, narrative)
            if cards:
                slide.cards = cards[:3]
                slide.bullets = []
                slide.detailed_points = []
                slide.comparison = None
                slide.key_message = slide.key_message or (
                    getattr(narrative, "value_proposition", "") if narrative else None
                )
            continue

        if arch == "customer context" or "challenge" in title or "current" in title:
            current_items = [
                re.sub(r"\s+", " ", text).strip()
                for text in _slide_line_items(slide)
                if (text or "").strip()
            ][:5]
            if current_items:
                slide.comparison = Comparison(
                    left=ComparisonColumn(
                        heading="Current operating challenge",
                        items=current_items[:4],
                        accent="challenge",
                    ),
                    right=ComparisonColumn(
                        heading="Proposal response",
                        items=_response_priorities(understanding, narrative),
                        accent="solution",
                    ),
                )
                slide.bullets = []
                slide.detailed_points = []
            continue

        if arch == "solution overview" and not slide.cards:
            source_lines = [
                text for text in _slide_line_items(slide)
                if (text or "").strip()
            ]
            cards = _cards_from_bullets(source_lines, accent="solution", max_cards=4)
            if cards:
                slide.cards = cards
                slide.bullets = []
                slide.detailed_points = []
            continue

        if (
            arch == "timeline"
            and "optional" in title
            and not slide.cards
            and not slide.detailed_points
            and not slide.bullets
        ):
            # The dedicated optional/later-phase expansion slide sometimes
            # comes back with only a key_message and no visible body — the
            # model pushed the real Phase 2 argument into speaker notes
            # instead. Backfill from RFP-grounded optional_response_topics
            # rather than leaving the slide empty.
            slide.detailed_points = _expansion_detailed_points(understanding)
            continue

        if any(token in title for token in ("security", "observability", "nfr", "non-functional")):
            points = slide.detailed_points or [
                BulletPoint(text="End-to-end secured communication", sub_points=[
                    "Encrypt communication paths between clients, services, data stores, and consumers",
                    "Use identity-aware access and controlled service connectivity",
                ]),
                BulletPoint(text="Data security and auditability", sub_points=[
                    "Apply least privilege, secrets control, retention, and transaction audit history",
                    "Trace ingestion, validation, transformation, reporting, and support actions",
                ]),
                BulletPoint(text="Observability and resilience", sub_points=[
                    "Centralize logs, metrics, alerts, interface health, and SLA signals",
                    "Design backup, recovery, availability, and operational readiness into the platform",
                ]),
            ]
            slide.cards = _cards_from_detailed_points(points, accent="why", max_cards=4)
            slide.bullets = []
            slide.detailed_points = []
            continue

        if arch in {"risks", "assumptions & dependencies", "value & differentiators", "requirements"}:
            if slide.detailed_points:
                slide.cards = (
                    _risk_cards_from_detailed_points(slide.detailed_points, max_cards=4)
                    if arch == "risks"
                    else _cards_from_detailed_points(slide.detailed_points, accent="info", max_cards=4)
                )
                slide.detailed_points = []
                slide.bullets = []
            elif slide.bullets and len(slide.bullets) >= 3:
                slide.cards = _cards_from_bullets(
                    slide.bullets,
                    accent="challenge" if arch == "risks" else "info",
                    max_cards=4,
                )
                slide.bullets = []

    for slide in deck_plan.slides:
        # Store one authored sentence in the DeckPlan itself. The renderer
        # repeats this same sentence when native layouts require continuation.
        slide.key_message = _complete_sentences(slide.key_message, 1) or None
    return deck_plan


def enforce_slide_density(deck_plan: DeckPlan) -> DeckPlan:
    """Keep content structures bounded without truncating proposal copy."""
    for slide in deck_plan.slides:
        if slide.table:
            # Table slides must remain table-first.
            slide.bullets = []
            slide.detailed_points = []
            slide.cards = []
            slide.comparison = None
            continue

        if slide.diagram:
            # The renderer separates the visual and explanation. Retain concise
            # authored copy so the companion page remains customer-facing.
            slide.bullets = [b for b in (slide.bullets or [])[:5] if (b or "").strip()]
            if slide.detailed_points:
                slide.detailed_points = slide.detailed_points[:4]
            slide.cards = []
            slide.comparison = None
            continue

        if slide.bullets:
            slide.bullets = [b for b in slide.bullets[:5] if (b or "").strip()]

        if slide.detailed_points:
            limited_points = []
            for point in slide.detailed_points[:4]:
                point.sub_points = [sp for sp in (point.sub_points or [])[:3] if (sp or "").strip()]
                limited_points.append(point)
            slide.detailed_points = limited_points

        if slide.cards:
            limited_cards = []
            for card in slide.cards[:4]:
                card.bullets = [b for b in (card.bullets or [])[:3] if (b or "").strip()]
                limited_cards.append(card)
            slide.cards = limited_cards

        if slide.comparison:
            slide.comparison.left.items = [item for item in slide.comparison.left.items[:5] if (item or "").strip()]
            slide.comparison.right.items = [item for item in slide.comparison.right.items[:5] if (item or "").strip()]

    return deck_plan


def _diagram_kind_from_visual_type(visual_type: str) -> str:
    mapping = {
        "architecture": "architecture",
        "technical_architecture": "technical_architecture",
        "deployment": "deployment",
        "hadr": "hadr",
        "timeline": "timeline",
        "process": "process",
        "org": "org",
        "data_flow": "data_model",
        "sequence": "process",
        "swimlane": "process",
        "topology": "deployment",
        "data_model": "data_model",
        "testing": "testing",
        "ams": "ams",
    }
    return mapping.get((visual_type or "").lower(), "generic")


def _brief_grounding_score(brief: DiagramBrief) -> tuple[float, List[str]]:
    entities = [e for e in (brief.entities or []) if (e or "").strip()]
    flows = [f for f in (brief.flows or []) if (f or "").strip()]
    evidence = [r for r in (brief.evidence_refs or []) if (r or "").strip()]
    controls = [c for c in (brief.controls or []) if (c or "").strip()]
    score = 0.0
    if len(entities) >= 4:
        score += 0.35
    elif len(entities) >= 2:
        score += 0.18
    if len(flows) >= 2:
        score += 0.30
    elif len(flows) == 1:
        score += 0.15
    if evidence:
        score += 0.20
    if controls:
        score += 0.15
    warnings: List[str] = []
    if len(entities) < 4:
        warnings.append("Brief has fewer than four proposal-specific entities.")
    if len(flows) < 2:
        warnings.append("Brief has fewer than two concrete proposal flows.")
    if not evidence:
        warnings.append("Brief has no requirement/source evidence refs.")
    return min(score, 1.0), warnings


def _brief_to_diagram_prompt(brief: DiagramBrief, understanding: RFPUnderstanding | None = None) -> str:
    customer = _customer_label(understanding, "the customer")
    def solution_label(value: str) -> str:
        clean = re.sub(r"\s+", " ", (value or "").strip())
        clean = re.sub(
            r"^(?:the\s+)?(?:platform|system|solution|project)\s+(?:shall|should|must)\s+",
            "",
            clean,
            flags=re.I,
        )
        return clean[0].upper() + clean[1:] if clean else ""

    entities = [item for item in (brief.entities or []) if not _is_open_visual_decision(item)]
    flows = [item for item in (brief.flows or []) if not _is_open_visual_decision(item)]
    controls = [item for item in (brief.controls or []) if not _is_open_visual_decision(item)]
    must_show = [item for item in (brief.must_show or []) if not _is_open_visual_decision(item)]
    parts = [
        f"Create a proposal-specific {brief.visual_type} diagram for {customer}.",
        f"Purpose: {(brief.purpose or brief.title or 'show the grounded proposal visual').strip()}",
    ]
    if entities:
        parts.append("Use only these named entities/components where relevant: " + "; ".join(solution_label(item) for item in entities[:8]) + ".")
    if flows:
        parts.append("Show these directional flows: " + "; ".join(solution_label(item) for item in flows[:5]) + ".")
    if controls:
        parts.append("Show these controls/boundaries: " + "; ".join(solution_label(item) for item in controls[:4]) + ".")
    if must_show:
        parts.append("Must show: " + "; ".join(solution_label(item) for item in must_show[:4]) + ".")
    banned = list(brief.must_not_show or [])
    banned.extend(["generic stock diagram", "invented tools", "proposal-submission or procurement portals unless explicitly operational"])
    parts.append("Must not show: " + "; ".join(dict.fromkeys(banned)) + ".")
    # Assumptions remain in the traceability model and dedicated assumptions
    # slide; image models tend to turn them into dense reviewer sidebars.
    parts.append(
        "Do not print requirement IDs, paragraph references, evidence citations, reviewer context, "
        "source-processing notes, addendum status, or document-quality observations in the image. "
        "Do not invent requirement-number ranges or label any group with requirement IDs."
    )
    parts.append(_SAFE_MARGIN_NOTE)
    return "\n".join(parts).strip()


def _brief_matches_slide(brief: DiagramBrief, slide: SlideSpec, fallback_kind: str) -> bool:
    slide_id = (getattr(slide, "slide_id", "") or "").lower()
    title = (getattr(slide, "title", "") or "").lower()
    archetype = (getattr(slide, "archetype", "") or "").lower()
    brief_id = (brief.slide_id or "").lower()
    brief_title = (brief.title or "").lower()
    visual_type = (brief.visual_type or "").lower()
    compatible_types = {
        "architecture": {"architecture"},
        "technical_architecture": {"technical_architecture"},
        "deployment": {"deployment", "topology"},
        "hadr": {"hadr"},
        "testing": {"testing"},
        "ams": {"ams"},
        "timeline": {"timeline"},
        "org": {"org"},
        "process": {"process", "sequence", "swimlane", "data_flow"},
        "data_model": {"data_model", "data_flow"},
        "generic": {"generic"},
    }
    if visual_type not in compatible_types.get(fallback_kind, {fallback_kind}):
        return False
    normalized_slide_id = re.sub(r"^(sk|auto|fallback)_", "", slide_id)
    normalized_brief_id = re.sub(r"^(sk|auto|fallback)_", "", brief_id)
    if normalized_brief_id and normalized_brief_id == normalized_slide_id:
        return True
    if brief_title and (brief_title in title or title in brief_title):
        return True
    aliases = {
        "architecture": ("solution architecture", "target architecture"),
        "technical_architecture": ("technical architecture", "layered architecture", "technology architecture"),
        "deployment": ("deployment", "resilience", "runtime"),
        "hadr": ("availability", "disaster", "resilience", "dr"),
        "timeline": ("timeline", "roadmap"),
        "process": ("process", "flow", "journey"),
        "data_model": ("data domain", "core data", "data model", "information model"),
        "org": ("team", "squad", "operating model"),
        "testing": ("testing", "acceptance", "uat", "sit"),
        "ams": ("ams", "support", "warranty", "operate", "operations"),
    }
    tokens = aliases.get(visual_type, aliases.get(fallback_kind, (fallback_kind,)))
    haystack = f"{title} {archetype}"
    return any(token in haystack for token in tokens)


def _select_visual_brief(
    slide: SlideSpec,
    fallback_kind: str,
    visual_briefs: List[DiagramBrief] | None,
) -> DiagramBrief | None:
    matches = [
        brief for brief in (visual_briefs or [])
        if _brief_matches_slide(brief, slide, fallback_kind)
    ]
    if not matches:
        return None
    slide_id = re.sub(r"^(sk|auto|fallback)_", "", (slide.slide_id or "").lower())
    return max(
        matches,
        key=lambda brief: (
            re.sub(r"^(sk|auto|fallback)_", "", (brief.slide_id or "").lower()) == slide_id,
            len(brief.evidence_refs or []),
            len(brief.entities or []) + len(brief.flows or []),
        ),
    )


def _diagram_from_brief(brief: DiagramBrief, understanding: RFPUnderstanding | None = None) -> DiagramSpec:
    score, warnings = _brief_grounding_score(brief)
    return DiagramSpec(
        kind=_diagram_kind_from_visual_type(brief.visual_type),
        prompt=(
            f"Diagram identity: {brief.slide_id or brief.title or brief.visual_type} [{brief.visual_type}].\n"
            + _brief_to_diagram_prompt(brief, understanding)
        ),
        approved=False,
        image_path=None,
        entities=brief.entities,
        flows=brief.flows,
        controls=brief.controls,
        evidence_refs=brief.evidence_refs,
        open_assumptions=brief.open_assumptions,
        grounding_score=score,
        grounding_warnings=warnings,
    )


def ensure_diagrams_for_key_slides(
    deck_plan: DeckPlan,
    understanding: RFPUnderstanding | None = None,
    visual_briefs: List[DiagramBrief] | None = None,
    technology_recommendations: TechnologyRecommendationSet | None = None,
) -> DeckPlan:
    """Attach diagrams only to profile-selected, evidence-backed visual slides."""
    # With an RFP understanding, the engagement-specific skeleton is the
    # eligibility authority.  A few low-level callers deliberately invoke this
    # helper without one (for example rendering/unit utilities); preserve their
    # explicit slide semantics instead of treating "no context" as a classified
    # engagement with no visual sections.
    planned_sections = _proposal_section_skeleton(understanding) if understanding is not None else []
    known_section_ids = {
        str(section.get("slide_id", "")).lower() for section in planned_sections
    }
    planned_visual_by_id = {
        str(section.get("slide_id", "")).lower(): str(section.get("diagram_kind", "")).lower()
        for section in planned_sections
        if section.get("diagram_kind")
    }
    planned_visual_kinds = set(planned_visual_by_id.values())
    if understanding is None:
        planned_visual_kinds = {
            "generic", "architecture", "technical_architecture", "deployment",
            "hadr", "timeline", "org", "testing", "data_model", "process", "ams",
        }
    for s in deck_plan.slides:
        arch = (s.archetype or "").lower()
        title = (s.title or "").lower()
        slide_id = (getattr(s, "slide_id", "") or "").lower()
        exact_kind = planned_visual_by_id.get(slide_id)

        if _is_exec_summary(s):
            s.diagram = None
            continue

        semantic_kind = "generic"
        is_integration = any(token in title for token in ("integration", "interface"))
        is_technical_architecture = (
            slide_id == "sk_technical_arch"
            or "technical architecture" in title
            or "layered architecture" in title
            or "technology architecture" in title
        )
        is_data_model = (
            slide_id == "sk_data_model"
            or any(token in title for token in ("data domain", "core data", "data model", "information model"))
        )
        is_reporting = slide_id == "sk_reporting" or any(
            token in title for token in ("semantic layer", "decision-ready reporting")
        )
        if is_technical_architecture:
            semantic_kind = "technical_architecture"
        elif is_data_model:
            semantic_kind = "data_model"
        elif is_reporting:
            semantic_kind = "process"
        elif arch == "architecture":
            semantic_kind = "architecture"
        elif arch == "deployment architecture":
            semantic_kind = "deployment"
        elif arch == "high availability & dr":
            semantic_kind = "hadr"
        elif arch == "timeline":
            semantic_kind = "timeline"
        elif arch == "team":
            semantic_kind = "org"
        elif arch == "delivery plan":
            if any(token in title for token in ("test", "acceptance", "uat", "sit")):
                semantic_kind = "testing"
            elif any(token in title for token in ("ams", "support", "warranty", "operate", "operations")):
                semantic_kind = "ams"
            else:
                semantic_kind = "process"

        if exact_kind:
            semantic_kind = exact_kind
        elif slide_id in known_section_ids:
            # Text-native skeleton sections must not gain a diagram merely
            # because their broad archetype commonly has one.
            s.diagram = None
            continue
        elif semantic_kind not in planned_visual_kinds:
            if s.diagram is None or not getattr(s.diagram, "approved", False):
                s.diagram = None
            continue

        if exact_kind or is_technical_architecture or is_data_model or is_reporting or arch in {
            "architecture",
            "deployment architecture",
            "high availability & dr",
            "delivery plan",
            "timeline",
            "team",
            "solution overview",
        }:
            if s.diagram is None:
                prompt = ""
                kind = "generic"
                if exact_kind == "process":
                    prompt = _build_diagram_prompt(
                        "delivery" if arch == "delivery plan" else "solution",
                        understanding,
                    )
                    kind = "process"
                elif exact_kind == "org":
                    prompt = _build_diagram_prompt("team", understanding)
                    kind = "org"
                elif exact_kind == "timeline":
                    prompt = _build_diagram_prompt("timeline", understanding)
                    kind = "timeline"
                elif exact_kind == "testing":
                    prompt = _build_diagram_prompt("testing", understanding)
                    kind = "testing"
                elif exact_kind == "data_model":
                    prompt = _build_diagram_prompt("data_model", understanding)
                    kind = "data_model"
                elif exact_kind == "deployment":
                    prompt = _build_diagram_prompt("deployment", understanding)
                    kind = "deployment"
                elif exact_kind == "technical_architecture":
                    prompt = _build_diagram_prompt(
                        "technical_architecture", understanding, technology_recommendations
                    )
                    kind = "technical_architecture"
                elif exact_kind == "architecture" and is_integration:
                    prompt = _build_diagram_prompt("integration", understanding)
                    kind = "architecture"
                elif exact_kind == "architecture":
                    prompt = _build_diagram_prompt("architecture", understanding)
                    kind = "architecture"
                elif is_technical_architecture:
                    prompt = _build_diagram_prompt(
                        "technical_architecture", understanding, technology_recommendations
                    )
                    kind = "technical_architecture"
                elif is_data_model:
                    prompt = _build_diagram_prompt("data_model", understanding)
                    kind = "data_model"
                elif is_reporting:
                    prompt = _build_diagram_prompt("reporting", understanding)
                    kind = "process"
                elif arch == "architecture":
                    prompt = _build_diagram_prompt("integration" if is_integration else "architecture", understanding)
                    kind = "architecture"
                elif arch == "deployment architecture":
                    prompt = _build_diagram_prompt("deployment", understanding)
                    kind = "deployment"
                elif arch == "high availability & dr":
                    prompt = _build_diagram_prompt("hadr", understanding)
                    kind = "hadr"
                elif arch == "delivery plan":
                    if any(token in title for token in ("test", "acceptance", "uat", "sit")):
                        prompt = _build_diagram_prompt("testing", understanding)
                        kind = "testing"
                    elif any(token in title for token in ("ams", "support", "warranty", "operate", "operations")):
                        prompt = _build_diagram_prompt("ams", understanding)
                        kind = "ams"
                    else:
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

                brief = _select_visual_brief(s, kind, visual_briefs)
                if brief is not None:
                    s.diagram = _diagram_from_brief(brief, understanding)
                elif prompt and kind != "ams":
                    score = (
                        0.5
                        if understanding is not None and (
                            is_integration or is_data_model or is_technical_architecture
                        )
                        else (0.25 if understanding is not None else 0.0)
                    )
                    warning = (
                        "Integration topology derived directly from proposal requirements."
                        if is_integration else
                        "Layered technical architecture derived from proposal requirements and sourcing decisions."
                        if is_technical_architecture else
                        "Data-domain topology derived directly from proposal requirements."
                        if is_data_model else
                        "No proposal-derived visual brief matched this slide; using archetype fallback prompt."
                    )
                    s.diagram = DiagramSpec(
                        kind=kind,
                        prompt=prompt,
                        approved=False,
                        image_path=None,
                        grounding_score=score,
                        grounding_warnings=[warning],
                    )
            elif arch == "solution overview" and _is_exec_summary(s):
                # Remove any legacy Exec Summary diagram to keep it text-native.
                s.diagram = None
            elif s.diagram is not None and not getattr(s.diagram, "approved", False):
                brief = None if is_integration else _select_visual_brief(s, semantic_kind, visual_briefs)
                if brief is not None:
                    s.diagram = _diagram_from_brief(brief, understanding)
                elif semantic_kind == "ams":
                    s.diagram = None
                elif semantic_kind != "generic":
                    prompt_kind = {"org": "team", "process": "delivery"}.get(semantic_kind, semantic_kind)
                    s.diagram = DiagramSpec(
                        kind=semantic_kind,
                        prompt=_build_diagram_prompt("integration" if is_integration else prompt_kind, understanding),
                        approved=False,
                        grounding_score=(
                            0.5
                            if understanding is not None and (
                                is_integration or is_data_model or is_technical_architecture
                            )
                            else 0.25
                        ),
                        grounding_warnings=(
                            ["Integration topology derived directly from proposal requirements."]
                            if is_integration else
                            ["Layered technical architecture derived from proposal requirements and sourcing decisions."]
                            if is_technical_architecture else
                            ["Data-domain topology derived directly from proposal requirements."]
                            if is_data_model else
                            ["No proposal-derived visual brief matched this slide; using semantic fallback prompt."]
                        ),
                    )

            if (
                s.diagram is not None
                and _ai_ml_is_applicable(understanding)
                and not getattr(s.diagram, "approved", False)
                and arch in {"timeline", "solution overview"}
                and not _is_exec_summary(s)
                and "ai-assisted" not in (s.diagram.prompt or "").lower()
            ):
                s.diagram.prompt = (
                    (s.diagram.prompt or "").rstrip()
                    + "\n"
                    + _ai_ml_architecture_clause(understanding).strip()
                ).strip()

            if (
                s.diagram is not None
                and not getattr(s.diagram, "approved", False)
                and (getattr(s.diagram, "kind", "") or "").lower() in {
                    "technical_architecture", "deployment", "hadr"
                }
            ):
                diagram_kind = (getattr(s.diagram, "kind", "") or "").lower()
                technology_context = (
                    _technical_architecture_context(technology_recommendations)
                    if diagram_kind == "technical_architecture"
                    else _deployment_technology_context(technology_recommendations)
                    if diagram_kind == "deployment"
                    else _hadr_technology_context(technology_recommendations)
                )
                if technology_context:
                    # Rebuild from the authoritative platform decision instead of
                    # appending to a possibly contradictory model-authored prompt.
                    # This prevents stale AWS/Azure/GCP labels surviving after the
                    # technology recommendation has selected a different provider.
                    s.diagram.prompt = _build_diagram_prompt(
                        diagram_kind,
                        understanding,
                        technology_recommendations,
                    )

            if (
                s.diagram is not None
                and not getattr(s.diagram, "approved", False)
                and _is_open_visual_decision(s.diagram.prompt)
            ):
                prompt_kind = {
                    "org": "team",
                    "process": "delivery",
                    "generic": "solution",
                    "data_model": "data_model",
                    "technical_architecture": "technical_architecture",
                }.get((s.diagram.kind or "").lower(), (s.diagram.kind or "generic").lower())
                if is_reporting:
                    prompt_kind = "reporting"
                if prompt_kind == "architecture" and is_integration:
                    prompt_kind = "integration"
                s.diagram.prompt = _build_diagram_prompt(
                    prompt_kind,
                    understanding,
                    technology_recommendations,
                )

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


_PLAN_LAYOUT_TOKENS = (
    "cover",
    "agenda",
    "key point",
    "table",
    "diagram",
    "org chart",
    "case stud",
    "end plate",
    "title and text",
    "title + text",
)
_PLAN_LAYOUT_EXCLUSIONS = ("instruction", "guidance", "how to use", "example only")


def _compact_plan_template_context(template_info: Dict[str, Any]) -> tuple[List[str], Dict[str, Any]]:
    """Keep only layout metadata that can influence the generated plan.

    The expanded HCLTech template contains hundreds of layouts and more than
    100k characters of placeholder metadata. The renderer maps archetypes to
    layouts deterministically, so the model only needs a representative set of
    usable layout names and compact placeholder signatures.
    """
    raw_names = template_info.get("slide_layout_names", []) or []
    selected: List[str] = []
    seen = set()
    for raw_name in raw_names:
        name = str(raw_name or "").strip()
        low = name.lower()
        if not name or low in seen or any(token in low for token in _PLAN_LAYOUT_EXCLUSIONS):
            continue
        if not any(token in low for token in _PLAN_LAYOUT_TOKENS):
            continue
        selected.append(name)
        seen.add(low)
        if len(selected) >= max(1, settings.deck_plan_layout_limit):
            break

    # Keep a small fallback sample if a non-HCLTech template uses generic names.
    if not selected:
        for raw_name in raw_names:
            name = str(raw_name or "").strip()
            low = name.lower()
            if name and low not in seen:
                selected.append(name)
                seen.add(low)
            if len(selected) >= min(20, max(1, settings.deck_plan_layout_limit)):
                break

    raw_map = template_info.get("placeholder_map", {}) or {}
    compact_map: Dict[str, Any] = {}
    for name in selected:
        placeholder_types = []
        placeholder_count = 0
        for placeholder in raw_map.get(name, []) or []:
            if not isinstance(placeholder, dict):
                continue
            placeholder_count += 1
            placeholder_type = str(placeholder.get("type", "")).split()[0]
            if placeholder_type and placeholder_type not in placeholder_types:
                placeholder_types.append(placeholder_type)
        compact_map[name] = {
            "count": placeholder_count,
            "types": placeholder_types,
        }
    return selected, compact_map


def _bounded_plan_context(text: str) -> str:
    clean = (text or "").strip()
    limit = max(0, settings.deck_plan_rag_max_chars)
    if not limit or len(clean) <= limit:
        return clean
    return clean[:limit].rsplit("\n", 1)[0] + "\n[Reusable context truncated for deck planning]"


def _compact_requirement(requirement: Any) -> Dict[str, Any]:
    return {
        "id": getattr(requirement, "id", ""),
        "text": _clip(getattr(requirement, "text", "") or "", 160),
        "priority": getattr(requirement, "priority", "should"),
        "source_refs": (getattr(requirement, "source_refs", []) or [])[:2],
    }


def _compact_understanding_for_plan(understanding: RFPUnderstanding | None) -> Dict[str, Any]:
    if understanding is None:
        return {}
    requirements = sorted(
        getattr(understanding, "requirements", []) or [],
        key=lambda r: {"must": 0, "should": 1, "may": 2}.get(getattr(r, "priority", "should"), 1),
    )
    profile = _effective_engagement_profile(understanding)
    return {
        "customer_name": getattr(understanding, "customer_name", None),
        "opportunity_title": getattr(understanding, "opportunity_title", None),
        "summary": _clip(getattr(understanding, "summary", "") or "", 900),
        "project_scope": _clip(getattr(understanding, "project_scope", "") or "", 900),
        "in_scope_work": [_clip(item, 160) for item in (getattr(understanding, "in_scope_work", []) or [])[:10]],
        "top_requirements": [_compact_requirement(req) for req in requirements[:24]],
        "clarifications": [
            {
                "topic": _clip(getattr(item, "topic", "") or "", 80),
                "effect": _clip(getattr(item, "effect_on_requirements", "") or "", 160),
            }
            for item in (getattr(understanding, "clarification_outcomes", []) or [])[:8]
        ],
        "assumptions": [_clip(item, 150) for item in (getattr(understanding, "assumptions", []) or [])[:8]],
        "risks": [_clip(item, 150) for item in (getattr(understanding, "risks", []) or [])[:8]],
        "solution_technologies": (getattr(understanding, "solution_technologies", []) or [])[:20],
        "key_technologies": (getattr(understanding, "key_technologies", []) or [])[:20],
        "software_bill_of_materials": [
            {
                "component": _clip(getattr(item, "component", "") or "", 80),
                "category": _clip(getattr(item, "category", "") or "", 60),
                "purpose": _clip(getattr(item, "purpose", "") or "", 120),
                "basis": _clip(getattr(item, "source_or_basis", "") or "", 80),
            }
            for item in (getattr(understanding, "software_bill_of_materials", []) or [])[:14]
        ],
        "engagement_profile": profile.model_dump(),
    }


def _compact_narrative_for_plan(narrative: ExecutiveNarrative | None) -> Dict[str, Any]:
    if narrative is None:
        return {}
    return {
        "value_proposition": _clip(getattr(narrative, "value_proposition", "") or "", 700),
        "strategic_outcomes": [_clip(item, 160) for item in (getattr(narrative, "strategic_outcomes", []) or [])[:8]],
        "solution_themes": [_clip(item, 120) for item in (getattr(narrative, "solution_themes", []) or [])[:8]],
        "executive_summary_points": [
            _clip(item, 180) for item in (getattr(narrative, "executive_summary_points", []) or [])[:4]
        ],
        "mandatory_sections": (getattr(narrative, "mandatory_sections", []) or [])[:10],
    }


def _compact_deck_plan_prompt(
    *,
    layout_names: List[str],
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> str:
    compact_layouts = layout_names[: min(16, len(layout_names))]
    payload = {
        "layouts": compact_layouts,
        "understanding": _compact_understanding_for_plan(understanding),
        "narrative": _compact_narrative_for_plan(narrative),
        "proposal_sections": _proposal_section_skeleton(understanding),
        "customer_technology_context": customer_technology_context or {},
        "contextual_reference_context": contextual_reference_context,
    }
    return (
        "You are a Tier-1 consulting deck architect. Return strict JSON matching the DeckPlan schema.\n"
        "Create a focused HCLTech proposal using only INPUT_JSON.proposal_sections. The supplied "
        "engagement profile and lifecycle scope are authoritative planning constraints. Do not add "
        "architecture, deployment, testing, Agile squads, technology-stack, AI, migration, or managed-"
        "service sections merely because those sections are common in proposal decks.\n"
        "Rules: challenges must precede the response; preserve Phase 1 versus optional/later-phase "
        "boundaries; prioritize mandatory response topics and evaluation criteria; avoid duplicate "
        "slides and tutorial content; exclude procurement/submission tools; keep diagram slides "
        "visual-first; keep bullets short enough to render without clipping.\n"
        "Use these compact inputs only; do not invent facts.\n"
        f"INPUT_JSON:\n{json.dumps(payload, ensure_ascii=False, separators=(',', ':'))}"
    )


# ------------------------------------------------------------------
# Lead Architect bookends (Phase 0): a locked Solution Brief set up-front,
# and a consolidation pass that threads the win theme and removes content
# that repeats across slides — the coherence a real Lead Architect enforces.
# ------------------------------------------------------------------
@dataclass
class SolutionBrief:
    """Cross-cutting decisions locked once, shared by every downstream step.

    Deterministically derived from the RFP understanding + narrative, with an
    explicit customer technology selection taking precedence over inferred
    provider terms. Later phases consume the same brief so sections build on
    one foundation instead of diverging.
    """
    customer: str
    engagement_kind: str
    target_cloud: str
    solution_name: str
    win_theme: str
    # A complete, unclipped sentence for the dedicated hero quote slide — see
    # insert_win_theme_slide. `win_theme` itself stays clipped to banner
    # length (its original job as a compact key_message strip); reusing that
    # pre-clipped string for a full-slide statement produced sentence
    # fragments (a long comma-separated list truncated mid-list reads as
    # broken copy, not a finished thought).
    win_theme_full: str


def _engagement_kind(understanding: RFPUnderstanding | None) -> str:
    profile = _effective_engagement_profile(understanding)
    if profile.primary_type != "other":
        primary = profile.primary_type.replace("_", " ")
        if profile.secondary_types:
            return primary + " with " + ", ".join(
                item.replace("_", " ") for item in profile.secondary_types[:2]
            )
        return primary
    if _is_data_platform_engagement(understanding):
        return "data platform"
    text = _understanding_text(understanding).lower()
    if _contains_any(text, ("migrate", "migration", "cutover", "re-platform", "replatform",
                            "modernization", "modernisation", "lift and shift")):
        return "migration / modernization"
    if _contains_any(text, ("kubernetes", "container", "microservice", "web service", "application platform")):
        return "application / platform"
    if _contains_any(text, ("integration", "interface", "api", "sftp", "message queue", "event")):
        return "integration"
    return "solution delivery"


def _solution_name(understanding: RFPUnderstanding | None) -> str:
    opp = (getattr(understanding, "opportunity_title", None) or "").strip() if understanding else ""
    customer = _customer_label(understanding, "")
    name = opp
    name = re.sub(r"\b(proposal|response|rfp|rft|tender|request for (proposal|tender))\b", " ", name, flags=re.I)
    if customer:
        name = re.sub(re.escape(customer), " ", name, flags=re.I)
    name = re.sub(r"\b[A-Za-z]{1,4}\d{3,}[A-Za-z0-9]*\b", " ", name)  # opportunity codes e.g. CT2605Z078
    name = re.sub(r"\s+", " ", name).strip(" -–—:|,")
    return name or "the proposed solution"


def build_solution_brief(
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    customer_technology_context: Dict[str, Any] | None = None,
) -> SolutionBrief:
    corpus = " ".join(
        [_understanding_text(understanding)]
        + ((getattr(understanding, "solution_technologies", None) or []) if understanding else [])
    )
    win = (getattr(narrative, "value_proposition", "") or "").strip() if narrative else ""
    customer_platform = str(
        (customer_technology_context or {}).get("platform") or ""
    ).strip()
    selected_provider = _selected_provider_family(customer_platform)
    normalized_platform = customer_platform.lower()
    if selected_provider:
        target_cloud = selected_provider
    elif "on-premises" in normalized_platform or "private cloud" in normalized_platform:
        target_cloud = "on-premises / private cloud"
    elif "hybrid" in normalized_platform:
        target_cloud = "hybrid"
    else:
        target_cloud = _cloud_signal(corpus)
    return SolutionBrief(
        customer=_customer_label(understanding),
        engagement_kind=_engagement_kind(understanding),
        target_cloud=target_cloud,
        solution_name=_solution_name(understanding),
        win_theme=_clip(_complete_sentences(win, 1) or win, 180) if win else "",
        win_theme_full=(_complete_sentences(win, 1) or win).strip() if win else "",
    )


def _norm_unit(text: str) -> str:
    return " ".join(re.sub(r"[^a-z0-9 ]", " ", (text or "").lower()).split())


def _dedupe_across_slides(deck_plan: DeckPlan) -> int:
    """Drop points that repeat verbatim on a later slide (keep the first use).

    A real deck says each thing once and references it thereafter; the model
    tends to restate the same operational lines on every slide. Exec Summary,
    Agenda, Title, and Next Steps are exempt (a summary is allowed to preview),
    and no slide is ever reduced below two units.
    """
    seen: set[str] = set()
    exempt = {"title", "agenda", "next steps"}
    dropped = 0
    for slide in deck_plan.slides:
        arch = (getattr(slide, "archetype", "") or "").strip().lower()
        if arch in exempt or _is_exec_summary(slide):
            continue

        def _unit_count() -> int:
            n = len([b for b in (slide.bullets or []) if (b or "").strip()])
            for point in (slide.detailed_points or []):
                n += len([s for s in (getattr(point, "sub_points", None) or []) if (s or "").strip()])
            for card in (slide.cards or []):
                n += len([b for b in (getattr(card, "bullets", None) or []) if (b or "").strip()])
            return n

        remaining = _unit_count()
        local: list[str] = []

        def _keep(text: str) -> bool:
            nonlocal remaining, dropped
            key = _norm_unit(text)
            if not key:
                return True
            if key in seen and remaining > 2:
                remaining -= 1
                dropped += 1
                return False
            local.append(key)
            return True

        if slide.bullets:
            slide.bullets = [b for b in slide.bullets if _keep(b)]
        for point in (slide.detailed_points or []):
            if getattr(point, "sub_points", None):
                point.sub_points = [s for s in point.sub_points if _keep(s)]
        for card in (slide.cards or []):
            if getattr(card, "bullets", None):
                card.bullets = [b for b in card.bullets if _keep(b)]

        seen.update(local)
        # Headings/point-texts anchor the topic; a later slide should not repeat
        # them as a bullet.
        for point in (slide.detailed_points or []):
            if (getattr(point, "text", "") or "").strip():
                seen.add(_norm_unit(point.text))
        for card in (slide.cards or []):
            if (getattr(card, "heading", "") or "").strip():
                seen.add(_norm_unit(card.heading))
    if dropped:
        log.info("Lead consolidation removed %d cross-slide duplicate point(s).", dropped)
    return dropped


def lead_consolidation(deck_plan: DeckPlan, brief: SolutionBrief) -> DeckPlan:
    """Lead Architect down-stream pass: thread the win theme, remove repetition."""
    if brief.win_theme:
        for slide in deck_plan.slides:
            if _is_exec_summary(slide) and not (getattr(slide, "key_message", None) or "").strip():
                slide.key_message = brief.win_theme
                break
    _dedupe_across_slides(deck_plan)
    return deck_plan


# ------------------------------------------------------------------
# Template/layout variety: an engagement-aware cover mood, a win-theme
# statement slide, and section dividers between narrative acts. These are
# purely presentational (no proposal content is added or removed) — they
# exist because a 30-55 slide deck with one fixed cover and no visual break
# between context/solution/delivery/commercials reads as one undifferentiated
# wall of "Two key points" boxes.
# ------------------------------------------------------------------
def _cover_layout_hint(brief: SolutionBrief) -> Optional[str]:
    """Match the cover's mood to the engagement instead of always using the
    same cover. Picks by theme (speed/agility/collaboration/progress), never
    by demographic imagery, so the swap reads as intentional, not arbitrary."""
    kind = (brief.engagement_kind or "").lower()
    if _contains_any(kind, ("migration", "modernization", "modernisation")):
        return "Cover – Speed (Light)"
    if _contains_any(kind, ("data platform", "analytics", "reporting")):
        return "Cover – Progress (Light)"
    if _contains_any(kind, ("integration",)):
        return "Cover – Collaboration (Light)"
    if _contains_any(kind, ("application", "platform")):
        return "Cover – Agility (Light)"
    return None


def apply_cover_mood(deck_plan: DeckPlan, brief: SolutionBrief) -> DeckPlan:
    hint = _cover_layout_hint(brief)
    if not hint:
        return deck_plan
    for slide in deck_plan.slides:
        if (slide.archetype or "").strip().lower() != "title":
            continue
        existing = (slide.layout_hint or "").strip()
        # Respect an existing hint only if it looks like a real cover layout
        # name (all of them read "Cover – ..."); the model sometimes puts
        # free-text description ("Clean customer-facing cover with...") into
        # layout_hint instead, which never resolves to a layout at render time
        # and would otherwise silently block the mood swap below.
        if existing and existing.lower().startswith("cover"):
            break
        slide.layout_hint = hint
        break
    return deck_plan


def insert_win_theme_slide(deck_plan: DeckPlan, brief: SolutionBrief) -> DeckPlan:
    """A full-bleed statement slide right after the Executive Summary.

    Echoes the same win theme ``lead_consolidation`` threads into the Exec
    Summary key_message — giving it a dedicated visual beat is a deliberate
    emphasis choice real proposal decks use, not accidental duplication.
    Attributed to HCLTech, never fabricated as a customer quote. Uses
    ``win_theme_full`` (the complete, unclipped sentence), not ``win_theme``
    (clipped to banner length for its other use as a compact key_message
    strip) — this slide has a full page to work with, and the renderer's own
    box-fit sizing (see ``_fill_quote_slide``) is what should decide how
    small the text gets, not an upstream character cap.
    """
    quote = (brief.win_theme_full or brief.win_theme or "").strip()
    if not quote:
        return deck_plan
    if any((slide.archetype or "").strip().lower() == "win theme" for slide in deck_plan.slides):
        return deck_plan
    exec_idx = next((i for i, s in enumerate(deck_plan.slides) if _is_exec_summary(s)), None)
    if exec_idx is None:
        return deck_plan
    deck_plan.slides.insert(
        exec_idx + 1,
        SlideSpec(slide_id="sk_win_theme", title="", archetype="Win Theme", key_message=quote),
    )
    return deck_plan


_ACT1_BEATS = {"title", "agenda", "exec", "context", "outcomes"}
_ACT2_BEATS = {
    "solution", "scope", "flow", "architecture", "data", "integration",
    "security", "deployment", "ha_dr", "ai", "sbom",
}
_ACT3_BEATS = {"delivery", "timeline", "testing", "ams", "team"}
# Act 4 is everything else (whyus, case_studies, risks, mapping, assumptions,
# commercials, next) — the tail of _DECK_BEATS.
_DIVIDER_LABELS = {
    2: ("Our Proposed Solution", "Architecture, delivery approach and technical design"),
    3: ("Delivery and Governance", "How we mobilize, deliver and support the engagement"),
    4: ("Why HCLTech", "Value, credentials and commercial terms"),
}


def _act_of_beat(beat: str) -> int:
    if beat in _ACT1_BEATS:
        return 1
    if beat in _ACT2_BEATS:
        return 2
    if beat in _ACT3_BEATS:
        return 3
    return 4


def insert_section_dividers(deck_plan: DeckPlan) -> DeckPlan:
    """Splice a section-break slide before each Act transition.

    Reads the beat already assigned by ``order_deck`` to find the three Act
    boundaries and inserts a divider ahead of each — it never re-sorts, so it
    must run after all ordering/pruning passes have settled the final order.
    """
    result: List[SlideSpec] = []
    current_act = 1
    for slide in deck_plan.slides:
        act = _act_of_beat(_deck_beat(slide))
        for missing_act in range(current_act + 1, act + 1):
            label = _DIVIDER_LABELS.get(missing_act)
            if label is None:
                continue
            title, subtitle = label
            result.append(
                SlideSpec(
                    slide_id=f"div_act_{missing_act}",
                    title=title,
                    archetype="Divider",
                    bullets=[subtitle],
                )
            )
        current_act = max(current_act, act)
        result.append(slide)
    deck_plan.slides = result
    return deck_plan


_KPI_PATTERN = re.compile(
    r"\b\d{1,3}(?:\.\d+)?\s?%|\b\d+x\b|\$\s?\d[\d,.]*\s?(?:million|billion|[mbk])?\b|\bin \d+\s?(?:days?|weeks?|months?)\b",
    re.IGNORECASE,
)


def _derive_kpis_for_slide(slide: SlideSpec) -> List[str]:
    """Pull up to two numeric highlights already present on the slide.

    Additive only: the source bullet/point text is left in place, so this can
    never remove proposal content — it only gives the renderer material for a
    stat-forward layout on slides that would otherwise be plain text boxes.
    """
    candidates: List[str] = []
    for bullet in (slide.bullets or []):
        if bullet and _KPI_PATTERN.search(bullet):
            candidates.append(bullet.strip())
    for point in (slide.detailed_points or []):
        text = getattr(point, "text", "") or ""
        if _KPI_PATTERN.search(text):
            candidates.append(text.strip())
    seen: set[str] = set()
    result: List[str] = []
    for candidate in candidates:
        key = _norm_unit(candidate)
        if key in seen:
            continue
        seen.add(key)
        result.append(candidate)
        if len(result) >= 2:
            break
    return result


def enrich_outcome_kpis(deck_plan: DeckPlan) -> DeckPlan:
    """Surface numeric outcome highlights as stat callouts where they exist."""
    for slide in deck_plan.slides:
        if slide.kpis:
            continue
        if _deck_beat(slide) not in {"exec", "outcomes"}:
            continue
        derived = _derive_kpis_for_slide(slide)
        if derived:
            slide.kpis = derived
    return deck_plan


def _sanitize_customer_visible_source_notes(deck_plan: DeckPlan) -> DeckPlan:
    """Keep extraction/audit mechanics out of customer-facing slide content."""
    def clean(text: str | None) -> str:
        value = re.sub(r"\s+", " ", (text or "").strip())
        if _is_internal_source_note(value):
            return ""
        value = re.sub(r"\b(?:RFP|evidence) extract\b", "available requirements", value, flags=re.I)
        value = re.sub(r"\bextracted evidence\b", "available requirements", value, flags=re.I)
        return value

    def sentence(text: str | None) -> str:
        value = clean(text)
        if value and value[-1] not in ".?!":
            value += "."
        return value

    for slide in deck_plan.slides:
        # A key message owns one subtitle band. Preserve a complete authored
        # sentence instead of allowing a paragraph to be split or clipped by
        # template-specific rendering.
        slide.key_message = sentence(_complete_sentences(slide.key_message, 1)) or None
        slide.bullets = [value for item in slide.bullets if (value := sentence(item))]
        slide.kpis = [value for item in slide.kpis if (value := clean(item))]
        points: List[BulletPoint] = []
        for point in slide.detailed_points:
            heading = clean(point.text)
            if heading:
                points.append(BulletPoint(
                    text=heading,
                    sub_points=[value for item in point.sub_points if (value := sentence(item))],
                ))
        slide.detailed_points = points
        cards: List[Card] = []
        for card in slide.cards:
            heading = clean(card.heading)
            body = sentence(card.body)
            bullets = [value for item in card.bullets if (value := sentence(item))]
            if heading and (body or bullets):
                cards.append(Card(heading=heading, body=body, bullets=bullets, accent=card.accent))
        slide.cards = cards
        if slide.comparison is not None:
            slide.comparison.left.items = [
                value for item in slide.comparison.left.items if (value := sentence(item))
            ]
            slide.comparison.right.items = [
                value for item in slide.comparison.right.items if (value := sentence(item))
            ]
        if slide.table:
            slide.table["rows"] = [
                [clean(str(cell)) for cell in row]
                for row in (slide.table.get("rows") or [])
            ]
    return deck_plan


def _post_process_deck_plan(
    deck_plan: DeckPlan,
    *,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    visual_briefs: List[DiagramBrief] | None = None,
    technology_recommendations: TechnologyRecommendationSet | None = None,
    customer_technology_context: Dict[str, Any] | None = None,
) -> DeckPlan:
    brief = build_solution_brief(
        understanding,
        narrative,
        customer_technology_context,
    )
    log.info(
        "Solution brief: customer=%r engagement=%r cloud=%r solution=%r",
        brief.customer, brief.engagement_kind, brief.target_cloud, brief.solution_name,
    )
    keystone_ids = {str(section.get("slide_id", "")) for section in _proposal_section_skeleton(understanding)}
    deck_plan = prune_profile_misaligned_slides(deck_plan, understanding)
    deck_plan = ensure_required_slides(deck_plan, understanding=understanding, narrative=narrative)
    deck_plan = _apply_technology_recommendations(deck_plan, technology_recommendations)
    deck_plan = enrich_slide_detail(
        deck_plan,
        understanding=understanding,
        technology_recommendations=technology_recommendations,
        visual_briefs=visual_briefs,
    )
    deck_plan = prune_empty_content_slides(deck_plan)
    deck_plan = prune_redundant_storyline_slides(deck_plan, protected_ids=keystone_ids)
    deck_plan = consulting_grade_proposal_polish(deck_plan, understanding=understanding, narrative=narrative)
    deck_plan = enrich_outcome_kpis(deck_plan)
    deck_plan = order_deck(deck_plan)
    deck_plan = synchronize_agenda(deck_plan)
    deck_plan = polish_deck_text(deck_plan)
    deck_plan = lead_consolidation(deck_plan, brief)
    deck_plan = apply_cover_mood(deck_plan, brief)
    deck_plan = _sanitize_customer_visible_source_notes(deck_plan)
    deck_plan = ensure_diagrams_for_key_slides(
        deck_plan,
        understanding=understanding,
        visual_briefs=visual_briefs,
        technology_recommendations=technology_recommendations,
    )
    deck_plan = enforce_slide_density(deck_plan)
    # Positional inserts — appended last so no earlier pass (pruning, density
    # limits, agenda sync) has to know about these synthetic, content-free
    # slides. They read the plan's already-final order and never re-sort it.
    deck_plan = insert_win_theme_slide(deck_plan, brief)
    deck_plan = insert_section_dividers(deck_plan)
    return deck_plan


def _fallback_deck_plan(
    *,
    title: str,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
) -> DeckPlan:
    """Build a conservative profile-driven plan if the LLM planner is unavailable."""
    sections = _proposal_section_skeleton(understanding)
    deck = _empty_plan_for_sections(title, sections)
    section_by_id = {str(section.get("slide_id", "")): section for section in sections}
    prompt_kind = {
        "architecture": "architecture",
        "technical_architecture": "technical_architecture",
        "deployment": "deployment",
        "hadr": "hadr",
        "timeline": "timeline",
        "org": "team",
        "testing": "testing",
        "data_model": "data_model",
        "process": "delivery",
    }
    for slide in deck.slides:
        if slide.slide_id == "sk_exec":
            slide.key_message = (
                (getattr(narrative, "value_proposition", "") or "") if narrative else None
            )
            slide.cards = _exec_summary_cards(understanding, narrative)
        elif slide.slide_id == "sk_context":
            slide.detailed_points = _context_detailed_points(understanding)
        elif slide.slide_id == "sk_scope":
            slide.detailed_points = _requirements_detailed_points(understanding)
        elif slide.slide_id == "sk_solution":
            slide.bullets = _exec_summary_bullets(understanding, narrative)
        elif slide.slide_id == "sk_risks":
            slide.detailed_points = _risk_detailed_points(understanding)
        elif slide.slide_id == "sk_assumptions":
            slide.detailed_points = _assumptions_dependency_points(understanding)
        elif slide.slide_id == "sk_tech":
            slide.table = _source_grounded_technology_table(understanding)
        elif slide.slide_id == "sk_next":
            slide.bullets = _next_steps_bullets(understanding)

        diagram_kind = str(section_by_id.get(slide.slide_id, {}).get("diagram_kind", ""))
        if diagram_kind:
            slide.diagram = DiagramSpec(
                kind=diagram_kind,
                prompt=_build_diagram_prompt(prompt_kind.get(diagram_kind, "solution"), understanding),
                approved=False,
            )
    return deck


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    low = (text or "").lower()
    return any(term in low for term in terms)


def _understanding_text(understanding: RFPUnderstanding | None) -> str:
    if understanding is None:
        return ""
    parts = [
        getattr(understanding, "summary", "") or "",
        getattr(understanding, "project_scope", "") or "",
        " ".join(getattr(understanding, "in_scope_work", []) or []),
        " ".join(getattr(understanding, "assumptions", []) or []),
        " ".join(getattr(understanding, "risks", []) or []),
        " ".join((getattr(r, "text", "") or "") for r in getattr(understanding, "requirements", []) or []),
    ]
    return " ".join(parts)


# --------------------------------------------------------------------------
# Generic, RFP-derived content helpers
#
# These keep the proposal skeleton and fallback content engagement-agnostic:
# nothing here may assume a specific customer, a specific legacy system, or
# that every engagement is a data/analytics platform.
# --------------------------------------------------------------------------
_GENERIC_CUSTOMER_NAMES = {"", "customer", "client", "the customer", "the client", "unknown", "n/a", "tbd"}


def _customer_label(understanding: RFPUnderstanding | None, default: str = "the customer") -> str:
    """Return a clean customer name for titles/copy, or a neutral default."""
    name = (getattr(understanding, "customer_name", None) or "").strip()
    if name.lower() in _GENERIC_CUSTOMER_NAMES or len(name) > 60:
        return default
    return name


def _scope_text(understanding: RFPUnderstanding | None) -> str:
    """Positive scope text (what is being built), used to classify the engagement.

    Deliberately excludes the risks/assumptions corpus so that a keyword which
    only appears in an *exclusion* ("X is out of scope") does not misclassify
    the engagement.
    """
    if understanding is None:
        return ""
    parts = [
        getattr(understanding, "project_scope", "") or "",
        " ".join(getattr(understanding, "in_scope_work", []) or []),
        getattr(understanding, "summary", "") or "",
    ]
    return " ".join(parts).lower()


_DATA_PLATFORM_PHRASES = (
    "data hub", "data lake", "lakehouse", "data warehouse", "data platform",
    "single source of truth", "master data", "reporting platform", "analytics platform",
    "data repository", "data mart", "business intelligence", "bi platform", "data catalog",
)


def _is_data_platform_engagement(understanding: RFPUnderstanding | None) -> bool:
    """True only when building a data/analytics platform is a core deliverable.

    Uses the positive scope text and explicit data-platform phrases, so an
    incidental "dashboard" (e.g. observability) or an out-of-scope data
    pipeline does not force data-domain/reporting sections onto every deck.
    """
    scope = _scope_text(understanding)
    if any(phrase in scope for phrase in _DATA_PLATFORM_PHRASES):
        return True
    if "consolidat" in scope and "data" in scope:
        return True
    return False


_TECHNICAL_ENGAGEMENT_TYPES = {
    "application_development",
    "platform_implementation",
    "migration_modernization",
    "data_analytics",
    "infrastructure_cloud",
}

_OPERATING_ENGAGEMENT_TYPES = {
    "managed_service_operations",
    "business_process_transformation",
}


def _phrase_score(text: str, weighted_phrases: tuple[tuple[str, float], ...]) -> tuple[float, List[str]]:
    score = 0.0
    evidence: List[str] = []
    low = (text or "").lower()
    for phrase, weight in weighted_phrases:
        if phrase in low:
            score += weight
            evidence.append(phrase)
    return min(1.0, score), evidence[:6]


def _infer_engagement_profile(understanding: RFPUnderstanding | None) -> EngagementProfile:
    """Conservative local fallback when the understanding has no model profile.

    Strong multi-word scope signals carry the most weight. Broad terms such as
    ``technology``, ``platform``, ``system``, ``implement`` and ``support`` are
    intentionally not sufficient on their own because they occur in nearly
    every technology RFP.
    """
    if understanding is None:
        return EngagementProfile(classification_rationale="No RFP understanding was available.")

    scope = _scope_text(understanding)
    requirements = " ".join(
        (getattr(item, "text", "") or "")
        for item in (getattr(understanding, "requirements", []) or [])
    ).lower()
    corpus = f"{scope} {requirements}"
    score_map: Dict[str, tuple[float, List[str]]] = {
        "managed_service_operations": _phrase_score(corpus, (
            ("managed service", 0.55), ("service management", 0.45),
            ("incident management", 0.30), ("problem management", 0.30),
            ("change management", 0.25), ("service delivery manager", 0.25),
            ("service level", 0.15), ("operating model", 0.20),
            ("operate the", 0.15), ("itil", 0.30), ("service desk", 0.30),
            ("application maintenance", 0.35),
        )),
        "application_development": _phrase_score(corpus, (
            ("application development", 0.55), ("software development", 0.55),
            ("develop and implement", 0.35), ("design and build", 0.30),
            ("build an application", 0.45), ("build a web application", 0.45),
            ("develop an application", 0.45), ("develop a web application", 0.45),
            ("build a portal", 0.45), ("develop a portal", 0.45),
            ("web application", 0.30), ("mobile application", 0.30),
            ("source code", 0.25), ("user stories", 0.20),
            ("software engineering", 0.30),
        )),
        "platform_implementation": _phrase_score(corpus, (
            ("platform implementation", 0.55), ("system implementation", 0.45),
            ("implement a platform", 0.45), ("configure and implement", 0.35),
            ("erp implementation", 0.45), ("crm implementation", 0.45),
            ("servicenow implementation", 0.45),
        )),
        "migration_modernization": _phrase_score(corpus, (
            ("migration", 0.35), ("migrate", 0.30), ("modernization", 0.40),
            ("modernisation", 0.40), ("re-platform", 0.40),
            ("replatform", 0.40), ("cutover", 0.25), ("legacy replacement", 0.40),
        )),
        "data_analytics": _phrase_score(corpus, (
            ("data platform", 0.55), ("data hub", 0.55), ("data lake", 0.50),
            ("lakehouse", 0.50), ("data warehouse", 0.50),
            ("analytics platform", 0.45), ("reporting platform", 0.40),
            ("business intelligence", 0.35), ("master data", 0.35),
        )),
        "infrastructure_cloud": _phrase_score(corpus, (
            ("cloud migration", 0.55), ("cloud infrastructure", 0.50),
            ("infrastructure transformation", 0.45), ("data center", 0.35),
            ("datacentre", 0.35), ("network transformation", 0.40),
            ("landing zone", 0.40), ("infrastructure managed service", 0.35),
        )),
        "advisory_assessment": _phrase_score(corpus, (
            ("advisory services", 0.50), ("strategy and roadmap", 0.45),
            ("maturity assessment", 0.35), ("current state assessment", 0.35),
            ("recommendations report", 0.30), ("consulting services", 0.30),
        )),
        "business_process_transformation": _phrase_score(corpus, (
            ("process transformation", 0.50), ("operating model transformation", 0.50),
            ("process redesign", 0.40), ("business process", 0.25),
            ("continuous improvement", 0.20), ("process governance", 0.25),
        )),
        "training_change_enablement": _phrase_score(corpus, (
            ("training delivery", 0.50), ("instructor-led training", 0.55),
            ("change enablement", 0.45), ("change adoption", 0.35),
            ("learning program", 0.40), ("training programme", 0.40),
        )),
    }
    if _is_data_platform_engagement(understanding):
        score, evidence = score_map["data_analytics"]
        score_map["data_analytics"] = (max(score, 0.75), evidence or ["data-platform scope"])
    if (
        (getattr(understanding, "solution_technologies", []) or [])
        and _contains_any(scope, ("configure", "implement", "deploy"))
    ):
        score, evidence = score_map["platform_implementation"]
        score_map["platform_implementation"] = (
            max(score, 0.55),
            list(dict.fromkeys(evidence + ["named platform implementation scope"])),
        )

    ranked = sorted(score_map.items(), key=lambda item: item[1][0], reverse=True)
    top_type, (top_score, _) = ranked[0]
    if top_score < 0.20:
        top_type = "other"
    secondaries = [
        name for name, (score, _) in ranked[1:]
        if score >= 0.30 and score >= top_score * 0.60
    ][:3]
    assessments = [
        EngagementTypeAssessment(engagement_type=name, score=score, evidence=evidence)
        for name, (score, evidence) in ranked
        if score > 0
    ]

    technical_primary = top_type in _TECHNICAL_ENGAGEMENT_TYPES
    technical_secondary = any(name in _TECHNICAL_ENGAGEMENT_TYPES for name in secondaries)
    stages: List[LifecycleStageAssessment] = []

    def add_stage(stage: str, terms: tuple[str, ...], *, allowed: bool = True) -> None:
        hits = [term for term in terms if term in corpus]
        if hits and allowed:
            stages.append(LifecycleStageAssessment(
                stage=stage,
                in_scope=True,
                confidence=min(0.9, 0.45 + 0.12 * len(hits)),
                evidence=hits[:5],
            ))

    add_stage("discover_assess", ("assessment", "discovery", "current state", "maturity"))
    add_stage("design", ("design", "framework", "operating model", "process model"))
    add_stage(
        "configure_build",
        ("application development", "software development", "design and build", "configure and implement", "source code", "build"),
        allowed=technical_primary or technical_secondary,
    )
    add_stage(
        "integrate_migrate",
        ("migration", "migrate", "cutover", "api integration", "system integration", "interface integration"),
        allowed=technical_primary or technical_secondary,
    )
    add_stage(
        "test_validate",
        ("system testing", "integration testing", "user acceptance testing", "uat", "test automation", "testing"),
        allowed=technical_primary or technical_secondary,
    )
    add_stage(
        "deploy_release",
        ("production deployment", "application deployment", "release pipeline", "ci/cd", "go-live", "deployment"),
        allowed=technical_primary or technical_secondary,
    )
    add_stage("mobilize_transition", ("mobilization", "mobilisation", "transition plan", "service transition", "stabilization", "stabilisation"))
    add_stage(
        "operate_support",
        (
            "managed service", "operate", "operations support", "service management",
            "maintenance", "application maintenance", "ams support",
            "warranty and support", "live-service support", "production support",
        ),
    )
    add_stage("optimize_transform", ("continuous improvement", "optimization", "optimisation", "maturity improvement", "transformation"))

    delivery_mode = "unknown"
    if top_type == "managed_service_operations":
        delivery_mode = "managed_service"
    elif top_type == "advisory_assessment":
        delivery_mode = "advisory"
    elif top_type == "hybrid" or (secondaries and top_score and score_map[secondaries[0]][0] >= top_score * 0.80):
        delivery_mode = "hybrid"
    elif technical_primary:
        delivery_mode = "project_delivery"

    mandatory_topics: List[str] = []
    topic_signals = (
        ("staffing model", ("staffing model", "resource model", "proposed roles")),
        ("governance model", ("governance", "raci", "service review")),
        ("service levels and measures", ("service level", "sla", "kpi", "success measures")),
        ("implementation or transition approach", ("implementation approach", "transition plan", "mobilization plan", "mobilisation plan")),
        ("commercial proposal", ("commercial proposal", "pricing", "rate card")),
        ("references and relevant experience", ("client references", "references", "relevant experience")),
    )
    for label, terms in topic_signals:
        if any(term in corpus for term in terms):
            mandatory_topics.append(label)

    optional_topics: List[str] = []
    for marker in ("optional", "phase 2", "future expansion", "expansion opportunity"):
        if marker in corpus:
            optional_topics.append(marker)

    confidence = max(0.35, min(0.85, top_score if top_type != "other" else 0.35))
    return EngagementProfile(
        primary_type=top_type,
        secondary_types=secondaries,
        type_assessments=assessments,
        delivery_mode=delivery_mode,
        lifecycle_stages=stages,
        mandatory_response_topics=mandatory_topics,
        optional_response_topics=optional_topics,
        phase_labels=[marker for marker in ("Phase 1", "Phase 2") if marker.lower() in corpus],
        classification_rationale=(
            f"Classified as {top_type.replace('_', ' ')} from the strongest scope and requirement signals."
        ),
        confidence=confidence,
    )


def _effective_engagement_profile(understanding: RFPUnderstanding | None) -> EngagementProfile:
    if understanding is None:
        return _infer_engagement_profile(None)
    profile = getattr(understanding, "engagement_profile", None)
    if profile is None or profile.primary_type == "other" or profile.confidence < 0.35:
        return _infer_engagement_profile(understanding)
    return profile


def _profile_type_score(profile: EngagementProfile, engagement_type: str) -> float:
    if profile.primary_type == engagement_type:
        primary = max(0.65, profile.confidence)
    else:
        primary = 0.0
    scored = max(
        (
            assessment.score
            for assessment in profile.type_assessments
            if assessment.engagement_type == engagement_type
        ),
        default=0.0,
    )
    # A declared secondary type is strong enough to select its own evidence-
    # backed sections in a hybrid engagement; lifecycle-stage gates still
    # prevent a secondary label from importing an entire template.
    secondary = 0.55 if engagement_type in profile.secondary_types else 0.0
    return max(primary, scored, secondary)


def _profile_has_stage(
    profile: EngagementProfile,
    stage: str,
    *,
    include_optional: bool = False,
) -> bool:
    return any(
        item.stage == stage
        and item.in_scope
        and (include_optional or not item.optional)
        for item in profile.lifecycle_stages
    )


def _profile_is_technical_delivery(profile: EngagementProfile) -> bool:
    technical_type = any(
        _profile_type_score(profile, engagement_type) >= 0.50
        for engagement_type in _TECHNICAL_ENGAGEMENT_TYPES
    )
    technical_stage = any(
        _profile_has_stage(profile, stage)
        for stage in ("configure_build", "integrate_migrate", "test_validate", "deploy_release")
    )
    return technical_type and technical_stage


def _profile_is_managed_operations(profile: EngagementProfile) -> bool:
    return (
        _profile_type_score(profile, "managed_service_operations") >= 0.50
        or profile.delivery_mode == "managed_service"
        or (
            _profile_type_score(profile, "business_process_transformation") >= 0.50
            and _profile_has_stage(profile, "operate_support")
        )
    )


def _profile_topic_contains(items: List[str], *terms: str) -> bool:
    text = " ".join(items or []).lower()
    return any(term in text for term in terms)


def _profile_explicitly_excludes(profile: EngagementProfile, *terms: str) -> bool:
    return _profile_topic_contains(profile.explicitly_unsupported_topics, *terms)


def _cloud_signal(corpus: str) -> str:
    """Return the dominant target cloud ('azure' | 'aws' | 'gcp' | '')."""
    c = (corpus or "").lower()
    # Reporting or endpoint products do not establish the hosting platform.
    # Require an actual cloud/platform signal before selecting an ecosystem.
    if any(t in c for t in ("azure", "microsoft fabric", "onelake", "aks", "azure app service")):
        return "azure"
    if any(t in c for t in ("aws", "amazon web services", "eks", "redshift", "cloudfront", "lambda")):
        return "aws"
    if any(t in c for t in ("gcp", "google cloud", "bigquery", "gke", "cloud run")):
        return "gcp"
    return ""


def _response_priorities(
    understanding: RFPUnderstanding | None,
    narrative: "ExecutiveNarrative | None" = None,
) -> List[str]:
    """Three short 'how we respond' points, derived from the RFP, never hardcoded."""
    # These often read as "label: elaboration" (the label itself eats into the
    # budget), and a comma inside an enumerated list ("Incident, Problem,
    # Change...") is not a safe clause boundary to cut at — _truncate_on_word
    # will still take it if forced to, leaving a dangling fragment. The
    # rendered comparison column comfortably holds full sentences well past
    # 150 characters, so the cap here is sized to rarely trigger at all
    # rather than to fit a specific box height.
    if narrative is not None:
        for attr in ("solution_themes", "strategic_outcomes"):
            vals = [v.strip() for v in (getattr(narrative, attr, []) or []) if (v or "").strip()]
            if len(vals) >= 2:
                return [_clip(v, 170) for v in vals[:3]]
    scope = [s.strip() for s in (getattr(understanding, "in_scope_work", []) or []) if (s or "").strip()]
    if len(scope) >= 2:
        return [_clip(s, 170) for s in scope[:3]]
    vp = (getattr(narrative, "value_proposition", "") or "").strip() if narrative else ""
    if vp:
        return [_clip(vp, 170)]
    return [
        "Meet the stated requirements with a proven, low-risk delivery approach",
        "Preserve continuity through phased delivery, validation, and cutover controls",
        "Build in security, observability, and operational readiness from the start",
    ]


def _proposal_section_skeleton(understanding: RFPUnderstanding | None) -> List[Dict[str, Any]]:
    """Create an evidence-backed, engagement-specific set of slide candidates.

    This is a planning policy, not a fixed deck template. Common executive
    bookends remain stable, while architecture, testing, deployment, operating
    model, staffing, technology and other lifecycle sections are selected from
    the classified engagement profile and explicit RFP signals.
    """
    text = _understanding_text(understanding)
    scope_text = _scope_text(understanding)
    customer = _customer_label(understanding)
    profile = _effective_engagement_profile(understanding)
    if understanding is not None:
        understanding.engagement_profile = profile

    managed = _profile_is_managed_operations(profile)
    technical_delivery = _profile_is_technical_delivery(profile)
    is_data = _profile_type_score(profile, "data_analytics") >= 0.50
    advisory = _profile_type_score(profile, "advisory_assessment") >= 0.50
    process_transformation = _profile_type_score(profile, "business_process_transformation") >= 0.50
    training = _profile_type_score(profile, "training_change_enablement") >= 0.50

    has_integration = technical_delivery and _contains_any(
        scope_text,
        ("api integration", "system integration", "interface", "sftp", "message queue", "webhook", "connector"),
    )
    has_migration = (
        _profile_type_score(profile, "migration_modernization") >= 0.50
        or (
            technical_delivery
            and _contains_any(scope_text, ("migration", "migrate", "cutover", "re-platform", "replatform"))
        )
    )
    has_support = managed or _profile_has_stage(profile, "operate_support")
    has_security = _contains_any(
        text,
        ("security", "mfa", "audit", "siem", "access control", "encryption", "compliance", "logging"),
    )
    needs_architecture = technical_delivery and not _profile_explicitly_excludes(
        profile, "solution architecture", "technical architecture",
        "target architecture", "solution design"
    ) and any(
        _profile_type_score(profile, engagement_type) >= 0.50
        for engagement_type in _TECHNICAL_ENGAGEMENT_TYPES
    )
    needs_technical_architecture = needs_architecture and any(
        _profile_type_score(profile, engagement_type) >= 0.50
        for engagement_type in (
            "application_development", "platform_implementation", "data_analytics", "infrastructure_cloud"
        )
    )
    needs_deployment = technical_delivery and not _profile_explicitly_excludes(
        profile, "deployment", "release architecture", "runtime architecture"
    ) and (
        _profile_has_stage(profile, "deploy_release")
        or _contains_any(scope_text, ("production deployment", "application deployment", "go-live", "ci/cd"))
    )
    needs_testing = technical_delivery and not _profile_explicitly_excludes(
        profile, "testing", "test strategy", "quality engineering"
    ) and (
        _profile_has_stage(profile, "test_validate")
        or _profile_has_stage(profile, "configure_build")
        or _profile_has_stage(profile, "integrate_migrate")
    )
    needs_technology_stack = (
        needs_technical_architecture
        and not managed
        and not _profile_explicitly_excludes(
            profile, "technology stack", "solution stack", "bill of materials"
        )
    )
    needs_roadmap = bool(profile.lifecycle_stages) or technical_delivery or managed or advisory
    has_governance = managed or _contains_any(text, ("governance", "raci", "steering", "service review", "decision rights"))
    has_metrics = managed or _contains_any(text, ("service level", "sla", "kpi", "dashboard", "success measure", "performance reporting"))
    has_staffing = managed or _profile_topic_contains(
        profile.mandatory_response_topics, "staff", "resource", "role", "team"
    )
    has_optional_expansion = bool(profile.optional_response_topics or profile.phase_labels) and _contains_any(
        text, ("phase 2", "optional", "future expansion", "expansion opportunity")
    )
    wants_references = _profile_topic_contains(
        profile.mandatory_response_topics, "reference", "experience", "case stud"
    ) or _contains_any(text, ("client references", "comparable engagements", "case studies"))
    wants_commercials = _profile_topic_contains(
        profile.mandatory_response_topics, "commercial", "pricing", "rate card"
    ) or _contains_any(text, ("commercial proposal", "pricing", "rate card"))

    outcomes_title = (
        f"Target outcomes align to {customer}'s priorities"
        if customer != "the customer"
        else "Target outcomes and measurable commitments"
    )
    sections: List[Dict[str, Any]] = []

    def add(
        slide_id: str,
        title: str,
        archetype: str,
        purpose: str,
        *,
        diagram_kind: str | None = None,
        inclusion_reason: str = "",
        phase: str = "required scope",
    ) -> None:
        section: Dict[str, Any] = {
            "slide_id": slide_id,
            "title": title,
            "archetype": archetype,
            "purpose": purpose,
            "inclusion_reason": inclusion_reason or "Supports the RFP response and proposal narrative.",
            "phase": phase,
            "engagement_type": profile.primary_type,
        }
        if diagram_kind:
            section["diagram_kind"] = diagram_kind
        sections.append(section)

    add("sk_title", "Proposal title", "Title", "Customer-facing cover")
    add("sk_agenda", "Agenda", "Agenda", "Summarize the selected deck flow")
    add("sk_exec", "Executive Summary", "Solution Overview", "Win thesis, situation, response and outcomes")
    add(
        "sk_context", "Current challenges define the response priorities", "Customer Context",
        "Show understanding of current context, pain, constraints and why action is needed",
    )
    add(
        "sk_outcomes", outcomes_title, "Value & Differentiators",
        "Business, service and technical outcomes supported by the RFP",
    )
    add(
        "sk_scope", "Scope and boundaries are explicit", "Requirements",
        "Separate required, optional, out-of-scope and open-decision items",
        inclusion_reason="Prevents optional or later-phase scope from becoming a delivery commitment.",
    )
    add(
        "sk_solution", "Proposed response at a glance", "Solution Overview",
        "Summarize the engagement-appropriate response pillars without forcing a technical architecture",
    )

    if managed or process_transformation:
        add(
            "sk_operating_model", "The operating model connects accountability, process and evidence",
            "Solution Overview",
            "Show service ownership, operational roles, core practices, governance, reporting and improvement",
            diagram_kind="process",
            inclusion_reason="Managed-service and operating-model scope requires an accountable service view.",
        )
        add(
            "sk_service_lifecycle", "Integrated service practices turn operational signals into improvement",
            "Content",
            "Show how the named operational practices interact, including escalation, learning, governance and closure",
            diagram_kind="process",
            inclusion_reason="Explains the RFP's service processes as one operating system rather than isolated documents.",
        )
    elif advisory:
        add(
            "sk_approach", "The advisory approach moves from evidence to executable decisions", "Delivery Plan",
            "Assessment lenses, stakeholder alignment, recommendations, roadmap and decision gates",
            diagram_kind="process",
            inclusion_reason="Advisory scope needs a method and decision path, not build architecture.",
        )
        add(
            "sk_deliverables", "Each advisory deliverable supports a defined decision", "Content",
            "Map analyses and deliverables to customer decisions, owners and outcomes",
        )
    elif training:
        add(
            "sk_enablement_model", "The enablement model connects learning, practice and adoption", "Solution Overview",
            "Audience segmentation, learning journeys, delivery channels, reinforcement and adoption measurement",
            diagram_kind="process",
        )

    if technical_delivery:
        add(
            "sk_flow", "End-to-end solution flow", "Solution Overview",
            "Visual flow from named inputs through solution controls to outcomes",
            diagram_kind="process",
            inclusion_reason="Technical delivery scope benefits from one end-to-end flow before detailed design.",
        )
    if needs_architecture:
        add(
            "sk_arch", "Concrete solution architecture", "Architecture",
            "Specific build, configuration or migration architecture and major components",
            diagram_kind="architecture",
        )
    if needs_technical_architecture:
        add(
            "sk_technical_arch", "Layered technical architecture connects systems, products and custom services",
            "Architecture",
            "External systems, COTS/build/integrate decisions, technical layers and cross-cutting controls",
            diagram_kind="technical_architecture",
        )
    if has_integration:
        add(
            "sk_integration", "Integration architecture connects source and consumer systems", "Architecture",
            "Named interfaces, protocols, dependencies, validation and error handling",
            diagram_kind="architecture",
        )
    if is_data and technical_delivery:
        add(
            "sk_data_model", "Core data domains and ownership", "Content",
            "Core data entities, relationships, ownership and stewardship",
            diagram_kind="data_model",
        )
        add(
            "sk_reporting", "One governed semantic layer serves decision-ready reporting", "Content",
            "Trusted data, governed measures and decision audiences",
            diagram_kind="process",
        )
    if _ai_ml_is_applicable(understanding) and not _profile_explicitly_excludes(
        profile, "artificial intelligence", "ai/ml", "machine learning"
    ):
        add(
            "sk_ai_opportunities", "AI-assisted capabilities target evidenced use cases", "Value & Differentiators",
            "Prioritize explicit or analytically grounded use cases with value, readiness, human control and cost gates",
            inclusion_reason="Included only because the classified technical scope contains credible AI/analytical signals.",
        )
    if has_security:
        security_title = (
            "Security, auditability and operational controls are embedded in the service"
            if managed and not technical_delivery
            else "Security and observability are built into the solution"
        )
        add(
            "sk_security", security_title, "Content",
            "RFP-grounded access, audit, monitoring, compliance and resilience controls",
        )
    if needs_deployment:
        add(
            "sk_deployment", "Deployment and resilience protect operations", "Deployment Architecture",
            "Runtime environments, release path, recovery, monitoring and support boundaries",
            diagram_kind="deployment",
        )
    if technical_delivery and _has_explicit_hadr_need(understanding):
        add(
            "sk_hadr", "Availability and recovery controls protect service continuity", "High Availability & DR",
            "Redundancy, backup, recovery, failover responsibilities and RTO/RPO decision points",
            diagram_kind="hadr",
        )
    if has_migration:
        add(
            "sk_migration", "Migration and cutover protect continuity", "Delivery Plan",
            "Migration preparation, rehearsal, reconciliation, cutover and stabilization controls",
        )
    if needs_testing:
        add(
            "sk_testing", "Acceptance evidence proves the solution is ready", "Delivery Plan",
            "Requirement-led functional, integration, data, security, UAT and release-readiness evidence",
            diagram_kind="testing",
        )

    if needs_roadmap:
        if managed:
            roadmap_title = "Mobilization, transition and stabilization establish the live service"
            roadmap_purpose = "Mobilization, knowledge transfer, process validation, service readiness, stabilization and improvement"
        elif advisory:
            roadmap_title = "The advisory roadmap turns findings into prioritized action"
            roadmap_purpose = "Assessment, alignment, recommendations, roadmap decisions and enablement"
        elif has_migration:
            roadmap_title = "The migration roadmap controls preparation, cutover and stabilization"
            roadmap_purpose = "Discovery, preparation, rehearsals, migration waves, cutover and stabilization"
        else:
            roadmap_title = "Incremental delivery releases value through controlled outcomes"
            roadmap_purpose = "Outcome increments, feedback, assurance and readiness gates"
        add(
            "sk_roadmap", roadmap_title, "Timeline", roadmap_purpose,
            diagram_kind="timeline",
        )
        if technical_delivery:
            add(
                "sk_roadmap_detail", "Each roadmap increment has a clear outcome and exit gate", "Content",
                "Explain increment outcomes, customer decisions, evidence and exit gates",
            )

    if managed and has_metrics:
        add(
            "sk_service_measures", "Service measures connect operational performance to outcomes", "Content",
            "KPIs, SLAs, dashboards, service reviews, trends and accountable improvement actions",
            inclusion_reason="Service measures and reporting are core managed-service decision evidence.",
        )
        add(
            "sk_improvement", "Continuous improvement converts service evidence into measurable action", "Content",
            "Improvement backlog, prioritization, ownership, benefits and governance closure",
        )
    elif has_support:
        add(
            "sk_ams", "Live-service support protects the complete solution boundary", "Delivery Plan",
            "Service coverage, monitoring, ownership, runbooks, transition and improvement",
        )

    if has_governance or managed or technical_delivery:
        governance_title = (
            "Governance and service leadership make accountability explicit"
            if managed
            else "Governance resolves decisions without slowing delivery"
        )
        governance_purpose = (
            "Service leadership, process ownership, RACI, escalation paths and review forums"
            if managed
            else "Decision rights, customer ownership, delivery teams, enabling roles and forums"
        )
        add(
            "sk_governance", governance_title, "Team", governance_purpose,
            diagram_kind="org",
        )
    if has_staffing:
        add(
            "sk_staffing", "The staffing model aligns accountability, capacity and coverage", "Team",
            "Named roles, responsibilities, coverage assumptions, scaling logic and customer/vendor boundaries",
            inclusion_reason="The RFP explicitly asks for staffing, roles or resource accountability.",
            diagram_kind="org",
        )
    if has_optional_expansion:
        add(
            "sk_expansion", "Optional expansion is sequenced by evidence and customer choice", "Timeline",
            "Separate required scope from optional services, dependencies, benefits, staffing and commercial triggers",
            inclusion_reason="The RFP identifies optional or later-phase scope that must remain separately electable.",
            phase="optional/later phase",
            diagram_kind="timeline",
        )

    add(
        "sk_value", "Proposal value and differentiators", "Value & Differentiators",
        "Why the engagement-appropriate approach is credible, useful and lower risk",
    )
    if needs_technology_stack:
        add(
            "sk_tech", "Proposed solution stack maps services to architecture layers", "Software Bill of Materials",
            "Concrete implementation services with mandated, referenced, recommended and decision-required status",
        )
    if understanding is None or getattr(understanding, "risks", None) or technical_delivery or managed:
        add(
            "sk_risks", "Key risks and mitigations are actively managed", "Risks",
            "Proposal-specific risk, implication, mitigation and owner/control",
        )
    if understanding is None or _visible_assumptions(understanding) or technical_delivery or managed:
        add(
            "sk_assumptions", "Assumptions and dependencies need early closure", "Assumptions & Dependencies",
            "Only engagement-relevant scope, access, people, process, technology and approval dependencies",
        )
    if wants_references:
        add(
            "sk_references", "Relevant experience demonstrates delivery credibility", "Case Studies",
            "Comparable engagements, outcomes, relevance and reference availability",
            inclusion_reason="The RFP explicitly requests references or comparable experience.",
        )
    if wants_commercials:
        commercial_title = (
            "Commercials separate transition, run service and optional expansion"
            if managed
            else "Commercials align scope, delivery and support"
        )
        add(
            "sk_commercials", commercial_title, "Commercials",
            "RFP-required pricing structure, assumptions, exclusions, options and change controls",
        )
    add("sk_next", "Next Steps", "Next Steps", "Recommended customer actions")
    return sections


def _chunked_plan_input(
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> str:
    brief = build_solution_brief(
        understanding,
        narrative,
        customer_technology_context,
    )
    payload = {
        # Locked cross-cutting decisions the Lead Architect sets up-front; every
        # section must write against these so the deck stays consistent.
        "solution_brief": {
            "customer": brief.customer,
            "engagement_kind": brief.engagement_kind,
            "target_cloud": brief.target_cloud or "not stated",
            "solution_name": brief.solution_name,
            "win_theme": brief.win_theme,
        },
        "understanding": _compact_understanding_for_plan(understanding),
        "narrative": _compact_narrative_for_plan(narrative),
        "customer_technology_context": customer_technology_context or {},
        "contextual_reference_context": contextual_reference_context,
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))


def _batched(items: List[Dict[str, Any]], size: int) -> List[List[Dict[str, Any]]]:
    size = max(1, size)
    return [items[index:index + size] for index in range(0, len(items), size)]


def _empty_plan_for_sections(title: str, sections: List[Dict[str, Any]]) -> DeckPlan:
    slides = [
        SlideSpec(
            slide_id=str(section["slide_id"]),
            title=str(section["title"]),
            archetype=section.get("archetype", "Content"),
        )
        for section in sections
    ]
    return DeckPlan(deck_title=title or "Proposal", slides=slides)


def _chunked_deck_plan(
    *,
    title: str,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> DeckPlan:
    sections = _proposal_section_skeleton(understanding)
    batch_size = max(2, min(6, getattr(settings, "deck_plan_batch_size", 4)))
    input_json = _chunked_plan_input(
        understanding,
        narrative,
        customer_technology_context,
        contextual_reference_context,
    )
    all_slides: List[SlideSpec] = []
    for batch_index, batch in enumerate(_batched(sections, batch_size), start=1):
        prompt = DECK_SECTION_EXPANSION_PROMPT.format(
            sections_json=json.dumps(batch, ensure_ascii=False, separators=(",", ":")),
            input_json=input_json,
            role_focus="",
        )
        log.info(
            "DeckPlan chunk %d: sections=%d prompt_chars=%d",
            batch_index,
            len(batch),
            len(prompt),
        )
        try:
            partial = response_as_schema(
                prompt,
                DeckPlan,
                reasoning_effort=settings.reasoning_effort_deck_plan,
                timeout_seconds=settings.deck_plan_timeout_s,
                background=True,
            )
            all_slides.extend(partial.slides)
        except Exception:
            log.warning(
                "DeckPlan chunk %d failed; using local section placeholders for this batch.",
                batch_index,
                exc_info=True,
            )
            all_slides.extend(_empty_plan_for_sections(title, batch).slides)

    return DeckPlan(deck_title=title or "Proposal", slides=all_slides)


# ------------------------------------------------------------------
# Phase 1: specialist-architect fan-out
#
# Instead of one generalist expanding every section, each section is owned by a
# specialist role that expands its slides in parallel with a role-focused
# persona — all consuming the same Solution Brief. The Lead consolidation pass
# (Phase 0) then merges and de-duplicates. Gated behind
# ``settings.deck_plan_specialists`` so it can be A/B'd against the chunked path.
# ------------------------------------------------------------------
_ROLE_APPLICATION = "Application/Solution Architect"
_ROLE_INTEGRATION = "Integration Architect"
_ROLE_DATA = "Data Architect"
_ROLE_LEAD = "Lead Architect"

_ROLE_PERSONAS = {
    _ROLE_APPLICATION: (
        "ROLE FOCUS:\nYou are a senior Application/Solution Architect. For these slides focus on the "
        "target-state application architecture, components and services, runtime, deployment, resilience "
        "and DR, and how the design meets the functional and non-functional requirements. Name only "
        "technologies that appear in the solution brief or the RFP input.\n"
    ),
    _ROLE_INTEGRATION: (
        "ROLE FOCUS:\nYou are a senior Integration Architect. For these slides focus on interfaces and APIs, "
        "protocols (API/file/MQ/SFTP/event), source and consumer systems, sequencing, validation, error "
        "handling, idempotency, and dependency ownership. Name only technologies that appear in the "
        "solution brief or the RFP input.\n"
    ),
    _ROLE_DATA: (
        "ROLE FOCUS:\nYou are a senior Data Architect. For these slides focus on data domains and ownership, "
        "canonical models, lineage, data quality and reconciliation, governance, retention, and how "
        "reporting/analytics reuse the trusted data. Name only technologies that appear in the solution "
        "brief or the RFP input.\n"
    ),
    _ROLE_LEAD: (
        "ROLE FOCUS:\nYou are the Lead Architect assembling the remainder of the proposal. Keep these slides "
        "crisp, outcome-focused, commercially aware, and consistent with the solution brief.\n"
    ),
}


def _section_role(section: Dict[str, Any]) -> str:
    """Assign a proposal section to the specialist who should own it."""
    sid = str(section.get("slide_id", "") or "")
    archetype = (section.get("archetype", "") or "").strip().lower()
    title = (section.get("title", "") or "").strip().lower()
    if sid in {"sk_integration"} or "integration architecture" in title:
        return _ROLE_INTEGRATION
    if sid in {"sk_data_model", "sk_reporting"}:
        return _ROLE_DATA
    if sid in {"sk_solution", "sk_flow", "sk_arch", "sk_technical_arch", "sk_deployment", "sk_security"} or archetype in {
        "architecture", "deployment architecture", "solution overview", "high availability & dr",
    }:
        return _ROLE_APPLICATION
    return _ROLE_LEAD


def _expand_role_sections(
    role: str,
    role_sections: List[Dict[str, Any]],
    *,
    title: str,
    input_json: str,
    batch_size: int,
) -> List[SlideSpec]:
    """Expand one specialist's sections (batched within the role)."""
    persona = _ROLE_PERSONAS.get(role, "")
    slides: List[SlideSpec] = []
    for batch in _batched(role_sections, batch_size):
        prompt = DECK_SECTION_EXPANSION_PROMPT.format(
            sections_json=json.dumps(batch, ensure_ascii=False, separators=(",", ":")),
            input_json=input_json,
            role_focus=persona,
        )
        try:
            partial = response_as_schema(
                prompt,
                DeckPlan,
                reasoning_effort=settings.reasoning_effort_deck_plan,
                timeout_seconds=settings.deck_plan_timeout_s,
                background=True,
            )
            slides.extend(partial.slides)
        except Exception:
            log.warning(
                "Specialist %r failed on a batch; using local placeholders.", role, exc_info=True
            )
            slides.extend(_empty_plan_for_sections(title, batch).slides)
    return slides


def _specialist_deck_plan(
    *,
    title: str,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
    customer_technology_context: Dict[str, Any] | None = None,
    contextual_reference_context: str = "",
) -> DeckPlan:
    sections = _proposal_section_skeleton(understanding)
    batch_size = max(2, min(6, getattr(settings, "deck_plan_batch_size", 4)))
    input_json = _chunked_plan_input(
        understanding,
        narrative,
        customer_technology_context,
        contextual_reference_context,
    )

    # Group sections by owning role (order preserved within each group).
    groups: Dict[str, List[Dict[str, Any]]] = {}
    for section in sections:
        groups.setdefault(_section_role(section), []).append(section)
    log.info(
        "DeckPlan specialist fan-out: %s",
        ", ".join(f"{role}={len(secs)}" for role, secs in groups.items()),
    )

    # Run specialists in parallel; each expands only its own sections.
    role_slides: Dict[str, List[SlideSpec]] = {}
    with ThreadPoolExecutor(max_workers=min(4, len(groups)), thread_name_prefix="rfp-specialist") as executor:
        futures = {
            executor.submit(
                _expand_role_sections, role, secs,
                title=title, input_json=input_json, batch_size=batch_size,
            ): role
            for role, secs in groups.items()
        }
        for future in as_completed(futures):
            role = futures[future]
            try:
                role_slides[role] = future.result()
            except Exception:
                log.warning("Specialist %r crashed entirely; using placeholders.", role, exc_info=True)
                role_slides[role] = _empty_plan_for_sections(title, groups[role]).slides

    # Merge back into the skeleton's order; backfill any section a specialist
    # dropped so no planned section is silently lost.
    by_id: Dict[str, SlideSpec] = {}
    extras: List[SlideSpec] = []
    known_ids = {str(section.get("slide_id", "")) for section in sections}
    for slides in role_slides.values():
        for slide in slides:
            sid = str(getattr(slide, "slide_id", "") or "")
            if sid in known_ids and sid not in by_id:
                by_id[sid] = slide
            elif sid not in known_ids:
                extras.append(slide)

    ordered: List[SlideSpec] = []
    for section in sections:
        sid = str(section.get("slide_id", ""))
        ordered.append(by_id.get(sid) or _empty_plan_for_sections(title, [section]).slides[0])
    ordered.extend(extras)
    return DeckPlan(deck_title=title or "Proposal", slides=ordered)


@_logged_node
def plan_deck(state: AgentState) -> Dict[str, Any]:
    """Plan a deck from RFP + optional RAG context."""
    plan_title = (
        getattr(state.understanding, "opportunity_title", None)
        or getattr(state.understanding, "customer_name", None)
        or "Proposal"
    )
    use_specialists = getattr(state, "deck_plan_specialists", None)
    if use_specialists is None:
        use_specialists = getattr(settings, "deck_plan_specialists", False)
    if use_specialists:
        log.info("DeckPlan using specialist-architect fan-out (Phase 1)")
        deck_plan = _specialist_deck_plan(
            title=plan_title,
            understanding=state.understanding,
            narrative=state.narrative,
            customer_technology_context=state.customer_technology_context,
            contextual_reference_context=state.contextual_reference_context,
        )
        deck_plan = _post_process_deck_plan(
            deck_plan,
            understanding=state.understanding,
            narrative=state.narrative,
            visual_briefs=state.visual_briefs,
            technology_recommendations=state.technology_recommendations,
            customer_technology_context=state.customer_technology_context,
        )
        state.deck_plan = deck_plan
        return {"deck_plan": deck_plan}

    if getattr(settings, "deck_plan_chunked", True):
        log.info(
            "DeckPlan using chunked section expansion (batch_size=%d)",
            getattr(settings, "deck_plan_batch_size", 4),
        )
        deck_plan = _chunked_deck_plan(
            title=plan_title,
            understanding=state.understanding,
            narrative=state.narrative,
            customer_technology_context=state.customer_technology_context,
            contextual_reference_context=state.contextual_reference_context,
        )
        deck_plan = _post_process_deck_plan(
            deck_plan,
            understanding=state.understanding,
            narrative=state.narrative,
            visual_briefs=state.visual_briefs,
            technology_recommendations=state.technology_recommendations,
            customer_technology_context=state.customer_technology_context,
        )
        state.deck_plan = deck_plan
        return {"deck_plan": deck_plan}

    template_info = state.template_info or {}
    layout_names, placeholder_map = _compact_plan_template_context(template_info)
    understanding_json = (
        json.dumps(state.understanding.model_dump(), ensure_ascii=False, separators=(",", ":"))
        if state.understanding is not None
        else "{}"
    )
    narrative_json = (
        json.dumps(state.narrative.model_dump(), ensure_ascii=False, separators=(",", ":"))
        if state.narrative is not None
        else "{}"
    )
    customer_technology_context_json = json.dumps(
        state.customer_technology_context or {},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    rag_context = _bounded_plan_context(state.rag_context or "")

    layout_json = json.dumps(layout_names, ensure_ascii=False, separators=(",", ":"))
    placeholder_json = json.dumps(placeholder_map, ensure_ascii=False, separators=(",", ":"))

    prompt = DECK_PLAN_V2_PROMPT.format(
        layout_names=layout_json,
        placeholder_map=placeholder_json,
        rag_context=rag_context,
        understanding_json=understanding_json,
        narrative_json=narrative_json,
        customer_technology_context_json=customer_technology_context_json,
        contextual_reference_context=state.contextual_reference_context,
    )
    prompt_mode = "full"
    max_prompt_chars = max(12000, getattr(settings, "deck_plan_prompt_max_chars", 30000))
    if len(prompt) > max_prompt_chars:
        prompt = _compact_deck_plan_prompt(
            layout_names=layout_names,
            understanding=state.understanding,
            narrative=state.narrative,
            customer_technology_context=state.customer_technology_context,
            contextual_reference_context=state.contextual_reference_context,
        )
        prompt_mode = "compact"

    log.info(
        "DeckPlan prompt components: total=%d layouts=%d layout_chars=%d "
        "placeholder_chars=%d rag_chars=%d understanding_chars=%d narrative_chars=%d mode=%s",
        len(prompt),
        len(layout_names),
        len(layout_json),
        len(placeholder_json),
        len(rag_context),
        len(understanding_json),
        len(narrative_json),
        prompt_mode,
    )

    try:
        deck_plan = response_as_schema(
            prompt,
            DeckPlan,
            reasoning_effort=settings.reasoning_effort_deck_plan,
            timeout_seconds=settings.deck_plan_timeout_s,
            background=True,
        )
    except Exception:
        log.warning(
            "DeckPlan LLM call failed in %s mode; using deterministic fallback plan.",
            prompt_mode,
            exc_info=True,
        )
        deck_plan = _fallback_deck_plan(
            title=plan_title,
            understanding=state.understanding,
            narrative=state.narrative,
        )

    deck_plan = _post_process_deck_plan(
        deck_plan,
        understanding=state.understanding,
        narrative=state.narrative,
        visual_briefs=state.visual_briefs,
        technology_recommendations=state.technology_recommendations,
        customer_technology_context=state.customer_technology_context,
    )

    state.deck_plan = deck_plan
    return {"deck_plan": deck_plan}


@_logged_node
def compress_bullets(state: AgentState) -> Dict[str, Any]:
    """Editorial pass: tighten bullets to executive-grade language.

    Sends only slide IDs, titles and editable bullets, then merges the rewritten
    bullets back by slide_id. Diagrams, tables, notes and traceability never enter
    this editorial request. Any failure leaves the original deck untouched.
    """
    if state.deck_plan is None:
        return {"deck_plan": None}
    editable_slides = [
        {
            "slide_id": slide.slide_id,
            "title": slide.title,
            "bullets": slide.bullets,
        }
        for slide in state.deck_plan.slides
        if slide.bullets
    ]
    if not editable_slides:
        return {"deck_plan": state.deck_plan}
    try:
        prompt = SLIDE_COMPRESSION_PROMPT.format(
            bullet_input_json=json.dumps(
                {"slides": editable_slides},
                ensure_ascii=False,
                separators=(",", ":"),
            )
        )
        compressed = response_as_schema(
            prompt,
            BulletCompressionSet,
            model=settings.model_fast,
            reasoning_effort=settings.reasoning_effort_low,
            background=False,
        )
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
        parts.append(
            f'Open by framing the central message of "{title}" and why that decision matters to the customer.'
        )

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
        parts.append("Then explain the slide in this order: " + "; ".join(points[:6]) + ".")
        parts.append(
            "For each point, connect the visible statement to the design rationale, the risk it controls, and the customer outcome it enables."
        )
    if slide.diagram is not None:
        diagram = slide.diagram
        entities = [item for item in (diagram.entities or []) if item][:5]
        flows = [item for item in (diagram.flows or []) if item][:3]
        if entities:
            parts.append("Read the visual through its main components: " + "; ".join(entities) + ".")
        if flows:
            parts.append("Trace the important movement or decision path: " + "; ".join(flows) + ".")
    if slide.table:
        headers = [str(value) for value in (slide.table.get("headers") or [])]
        if headers:
            parts.append(
                "Explain how the table links " + ", ".join(headers[:4]) + ", highlighting choices and rationale rather than reading every cell."
            )
    parts.append(
        "Close by separating committed design choices from items that still require normal architecture validation, then transition to the next decision in the story."
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


def _compact_slide_for_notes(
    slide: SlideSpec,
    previous_title: str,
    next_title: str,
) -> Dict[str, Any]:
    diagram = slide.diagram
    comparison = slide.comparison
    return {
        "slide_id": slide.slide_id,
        "title": slide.title,
        "archetype": slide.archetype,
        "previous_slide_title": previous_title,
        "next_slide_title": next_title,
        "key_message": slide.key_message,
        "bullets": slide.bullets,
        "detailed_points": [
            {"text": point.text, "sub_points": point.sub_points}
            for point in (slide.detailed_points or [])
        ],
        "cards": [
            {"heading": card.heading, "body": card.body, "bullets": card.bullets}
            for card in (slide.cards or [])
        ],
        "comparison": (
            {
                "left": {"heading": comparison.left.heading, "items": comparison.left.items},
                "right": {"heading": comparison.right.heading, "items": comparison.right.items},
            }
            if comparison is not None else None
        ),
        "table": (
            {
                "headers": (slide.table.get("headers") or [])[:6],
                "rows": (slide.table.get("rows") or [])[:10],
            }
            if slide.table else None
        ),
        "diagram": (
            {
                "kind": diagram.kind,
                "entities": (diagram.entities or [])[:10],
                "flows": (diagram.flows or [])[:6],
                "controls": (diagram.controls or [])[:6],
                "design_prompt": _clip(diagram.prompt or "", 2200),
            }
            if diagram is not None else None
        ),
    }


def _technology_context_for_notes(
    recommendation_set: TechnologyRecommendationSet | None,
) -> Dict[str, Any]:
    if recommendation_set is None:
        return {}
    return {
        "selected_platform": recommendation_set.selected_platform,
        "hosting_model": recommendation_set.hosting_model,
        "deployment_rationale": recommendation_set.deployment_rationale,
        "primary_region_strategy": recommendation_set.primary_region_strategy,
        "recommendations": [
            {
                "layer": item.architecture_layer,
                "technology": item.proposed_technology,
                "role": item.role,
                "status": item.status,
            }
            for item in (recommendation_set.recommendations or [])[:20]
        ],
        "component_decisions": [
            {
                "capability": item.capability,
                "recommendation": item.recommendation,
                "sourcing": item.sourcing_model,
                "role": item.role,
                "system_of_record": item.system_of_record,
            }
            for item in (recommendation_set.component_decisions or [])[:10]
        ],
    }


@_logged_node
def generate_notes(state: AgentState) -> Dict[str, Any]:
    """Generate presenter speaker notes for content slides.

    Notes are generated in small parallel batches. This gives the model enough
    room to explain visuals and design decisions without one very large request,
    while a failed batch falls back independently instead of degrading the whole
    deck. Self-explanatory slides are skipped. Never fails the pipeline.
    """
    if state.deck_plan is None:
        return {"deck_plan": None}

    slides = state.deck_plan.slides
    notes_slides = [s for s in slides if _slide_wants_notes(s)]
    notes_by_id: Dict[str, str] = {}

    if notes_slides and getattr(state, "enable_notes", True):
        index_by_id = {slide.slide_id: index for index, slide in enumerate(slides)}
        batch_size = max(2, min(10, int(getattr(settings, "notes_batch_size", 6))))
        batches = [notes_slides[index:index + batch_size] for index in range(0, len(notes_slides), batch_size)]
        technology_context = _technology_context_for_notes(state.technology_recommendations)
        narrative_context = state.narrative.model_dump() if state.narrative else {}
        understanding_summary = (
            getattr(state.understanding, "summary", "") if state.understanding else ""
        ) or ""
        reference_context = (state.contextual_reference_context or "")[:6000]

        def generate_batch(batch: List[SlideSpec]) -> DeckNotes:
            compact: List[Dict[str, Any]] = []
            for slide in batch:
                index = index_by_id.get(slide.slide_id, 0)
                previous_title = slides[index - 1].title if index > 0 else ""
                next_title = slides[index + 1].title if index + 1 < len(slides) else ""
                compact.append(_compact_slide_for_notes(slide, previous_title, next_title))
            prompt = SPEAKER_NOTES_PROMPT.format(
                deck_plan_json=json.dumps(compact, ensure_ascii=False, separators=(",", ":")),
                narrative_json=json.dumps(narrative_context, ensure_ascii=False, separators=(",", ":")),
                understanding_summary=understanding_summary,
                technology_context=json.dumps(technology_context, ensure_ascii=False, separators=(",", ":")),
                reference_context=reference_context,
            )
            return response_as_schema(
                prompt,
                DeckNotes,
                model=settings.model_fast,
                reasoning_effort=settings.reasoning_effort_low,
                background=False,
            )

        workers = max(1, min(int(getattr(settings, "notes_workers", 3)), len(batches)))
        with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="rfp-notes") as executor:
            futures = {executor.submit(generate_batch, batch): batch for batch in batches}
            for future in as_completed(futures):
                batch = futures[future]
                try:
                    deck_notes = future.result()
                    for note in deck_notes.notes:
                        if note.slide_id and (note.notes or "").strip():
                            notes_by_id[note.slide_id] = note.notes.strip()
                except Exception:
                    log.warning(
                        "Speaker-notes batch failed for slides %s; using deterministic notes for that batch.",
                        ", ".join(slide.slide_id for slide in batch),
                        exc_info=True,
                    )

    for s in slides:
        if not _slide_wants_notes(s):
            s.notes = None
            continue
        note = notes_by_id.get(s.slide_id) or _fallback_note(s)
        s.notes = note or None

    return {"deck_plan": state.deck_plan}
