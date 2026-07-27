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
    BulletPoint,
    Card,
    Comparison,
    ComparisonColumn,
    DeckNotes,
    DeckPlan,
    DiagramSpec,
    ExecutiveNarrative,
    RFPUnderstanding,
    SectionTaxonomy,
    SlideSpec,
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
        evidence = response_as_schema(
            prompt,
            SourceEvidenceBatch,
            model=settings.model_fast,
            reasoning_effort=settings.reasoning_effort_low,
            timeout_seconds=settings.understanding_evidence_timeout_s,
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
    if source_chars <= settings.understanding_direct_max_chars:
        log.info(
            "RFP package fits direct understanding budget (%d <= %d chars)",
            source_chars,
            settings.understanding_direct_max_chars,
        )
        state.source_evidence = []
        state.evidence_text = None
        return {"source_evidence": [], "evidence_text": None}

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
    return {"source_evidence": evidence_batches, "evidence_text": evidence_text}


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
        prompt, SectionTaxonomy, reasoning_effort=settings.reasoning_effort_medium
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
        prompt, ExecutiveNarrative, reasoning_effort=settings.reasoning_effort_high
    )
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
    return len(_ai_ml_opportunities(understanding)) >= 2


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
    """Fill implementation gaps with a qualified ecosystem-aligned reference stack."""
    if understanding is None:
        return []
    explicit = _extract_tech_terms(understanding, limit=20)
    corpus = " ".join(explicit + [_understanding_text(understanding)]).lower()
    cloud = _cloud_signal(corpus)
    is_data = _is_data_platform_engagement(understanding)
    if is_data:
        if cloud == "aws":
            return ["AWS Glue/Step Functions", "AWS Lambda/ECS", "API Gateway/EventBridge/SQS", "S3/Lake Formation/Redshift", "CloudWatch/Security Hub"]
        if cloud == "gcp":
            return ["Cloud Data Fusion/Dataflow", "Cloud Run/Cloud Functions", "Apigee/Pub/Sub", "Cloud Storage/BigQuery", "Cloud Logging/Monitoring"]
        if cloud == "azure":
            return [
                "Fabric Data Factory pipelines",
                "Fabric OneLake Lakehouse/Warehouse",
                "Fabric Dataflow Gen2/notebooks",
                "Azure Functions/Logic Apps",
                "Azure API Management/Service Bus",
                "Power BI",
                "Microsoft Entra ID/Key Vault",
                "Azure Monitor/Application Insights/Sentinel",
            ]
        return []
    # Application / platform engagement: container, data store, messaging, delivery.
    if cloud == "azure":
        return ["Azure Kubernetes Service (AKS)", "Azure Database for PostgreSQL", "Azure Service Bus/Event Hubs", "Azure Cache for Redis", "Azure Blob Storage/Front Door", "Microsoft Entra ID/Key Vault", "Azure Monitor or Datadog", "GitHub Actions/Terraform"]
    if cloud == "aws":
        return ["Amazon EKS", "Amazon RDS/Aurora PostgreSQL", "Amazon MSK/SNS/SQS", "Amazon ElastiCache (Redis)", "Amazon S3/CloudFront", "IAM/Secrets Manager/KMS", "CloudWatch or Datadog", "GitHub Actions/Terraform"]
    if cloud == "gcp":
        return ["Google Kubernetes Engine (GKE)", "Cloud SQL for PostgreSQL", "Pub/Sub", "Memorystore (Redis)", "Cloud Storage/Cloud CDN", "Cloud IAM/Secret Manager", "Cloud Monitoring or Datadog", "Cloud Build/Terraform"]
    return []


def _diagram_context(understanding: Optional[RFPUnderstanding]) -> str:
    """Build a short, grounded context string for diagram prompts."""
    if understanding is None:
        return ""
    customer = getattr(understanding, "customer_name", None) or "the client"
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
    "This diagram will be displayed at roughly 7.5 by 5 inches on a 16:9 slide. "
    "Style: consulting-grade, white background, readable 18pt+ labels, minimal clutter, "
    "no logos, no gradients, no sketch effects. Use at most 12 primary nodes, no more than "
    "two text lines per node, and no descriptive paragraphs, footnotes, or tiny legends. "
    "Keep labels to five words where possible. Use labeled boxes with directional arrows. "
    "Keep all text and shapes inside a 5–8% safe margin; do not place content at the edges."
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
            result.append(_clip(clean, 165))
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
        ("iccms", "functional", "pre-set", "workflow", "operational process", "milestone", "sla"),
        3,
    )
    controls = _matched_requirement_texts(
        understanding,
        ("security", "access", "audit", "performance", "availability", "accuracy", "reconciliation", "retention"),
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


def _build_diagram_prompt(kind: str, understanding: Optional[RFPUnderstanding]) -> str:
    """Create a context-rich diagram prompt grounded in the RFP.

    `kind` is one of: architecture, delivery, timeline, team, solution, testing, ams.
    """
    ctx = _diagram_context(understanding)
    techs = _extract_tech_terms(understanding)
    tech_clause = (" featuring " + ", ".join(techs)) if techs else ""
    ai_clause = _ai_ml_architecture_clause(understanding)

    if kind == "architecture":
        body = (
            f"Create a concrete solution architecture diagram{tech_clause}. Show source systems "
            "and document inputs, ingestion/extraction, validation and business rules, operational "
            "application services, APIs/integration, central operational data store or lakehouse, "
            "reporting/BI, security, monitoring, and support boundaries. For a data hub, make the "
            "data pipeline explicit: capture -> validate -> curate -> serve -> report. Show key "
            "integrations and primary data flows with directional arrows. Group related services "
            f"and label each box clearly.{ai_clause}"
        )
    elif kind == "deployment":
        body = (
            f"Create one combined deployment and resilience architecture diagram{tech_clause}. "
            "Show production, test/UAT, and DR/backup only when relevant; hosting boundary or "
            "hosting-to-confirm assumption; network/security zones; identity and access controls; "
            "source-system connectivity; release path; monitoring/SIEM; backup/restore; failover; "
            "and support touchpoints. Keep this specific to the proposal and avoid generic cloud "
            "tutorial elements."
            + (
                " If AI-assisted capabilities are proposed, show managed consumption, usage budgets, "
                "model monitoring, and scale-to-zero or scheduled batch processing; do not show GPU clusters."
                if ai_clause else ""
            )
        )
    elif kind == "hadr":
        body = (
            "Create a high availability and disaster recovery topology. Show active/standby or "
            "active/active options as appropriate, redundant application/integration tiers, replicated "
            "data repository, backup/restore flow, monitoring and alerting, incident/failover flow, "
            "RTO/RPO callouts marked as 'not specified in RFP' when not provided, and operations "
            "handover/support boundaries."
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
            "enhancement backlog. Label service levels as 'to be agreed' unless grounded in the RFP. Do not use a "
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

    return f"{ctx}\n{body}\n{_SAFE_MARGIN_NOTE}".strip()


def _deployment_bullets(understanding: RFPUnderstanding | None) -> List[str]:
    """Grounded deployment/release defaults for the deployment architecture slide."""
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


def _sdlc_technology_table(understanding: RFPUnderstanding | None) -> Dict[str, Any]:
    """Build an implementable, qualified solution stack by architecture layer."""
    sbom = getattr(understanding, "software_bill_of_materials", []) or [] if understanding else []
    techs = [
        (getattr(item, "component", "") or "").strip()
        for item in sbom
        if (getattr(item, "component", "") or "").strip()
        and not _is_excluded_solution_tool(getattr(item, "component", ""), understanding)
    ]
    for tech in (getattr(understanding, "solution_technologies", []) or []) if understanding else []:
        if tech and not _is_excluded_solution_tool(tech, understanding) and tech not in techs:
            techs.append(tech)

    corpus = " ".join(techs + [_understanding_text(understanding)]).lower()
    cloud = _cloud_signal(corpus)
    is_data_platform = _is_data_platform_engagement(understanding)

    def basis(products: str, proposed: str) -> str:
        product_names = [part.strip().lower() for part in re.split(r"[,/]", products) if part.strip()]
        if any(any(name in tech.lower() or tech.lower() in name for tech in techs) for name in product_names):
            return "RFP-required or referenced solution technology"
        return proposed

    cloud_label = {"azure": "Azure", "aws": "AWS", "gcp": "Google Cloud"}.get(cloud, "")
    pb = (
        f"Proposed {cloud_label}-aligned option; confirm with customer architecture"
        if cloud_label
        else "Recommended cloud-neutral option; confirm platform standards"
    )

    if is_data_platform:
        # Data / analytics platform build: layers reflect ingestion → curation →
        # serving → reporting.
        if cloud == "aws":
            rows = [
                ["Ingestion and orchestration", "AWS Glue; Step Functions; Transfer Family", "Ingest and orchestrate file, API, and database feeds", pb],
                ["Validation and services", "AWS Lambda; ECS/Fargate", "Run validation, canonical transformations, APIs, and operational services", pb],
                ["Integration", "Amazon API Gateway; EventBridge; SQS", "Secure and decouple synchronous and asynchronous interfaces", pb],
                ["Data repository", "Amazon S3; Lake Formation; Redshift or Aurora", "Store governed lake, analytical, and transactional data", pb],
                ["Reporting", "Amazon QuickSight or RFP-referenced BI tooling", "Serve governed operational dashboards and analytics", pb],
                ["Security and observability", "IAM; KMS; Secrets Manager; CloudWatch; Security Hub", "Protect identities/data and monitor the live service", pb],
                ["DevSecOps", "CodePipeline/CodeBuild or GitHub Actions; Terraform", "Automate quality, security, infrastructure, and releases", pb],
            ]
        elif cloud == "gcp":
            rows = [
                ["Ingestion and orchestration", "Cloud Data Fusion/Dataflow; Cloud Composer", "Ingest and orchestrate file, API, stream, and database feeds", pb],
                ["Validation and services", "Cloud Run; Cloud Functions", "Run validation, canonical transformations, APIs, and operational services", pb],
                ["Integration", "Apigee; Pub/Sub", "Secure APIs and decouple asynchronous interfaces", pb],
                ["Data repository", "Cloud Storage; BigQuery", "Store governed raw, curated, and analytical data", pb],
                ["Reporting", "Looker or RFP-referenced BI tooling", "Serve governed operational dashboards and analytics", pb],
                ["Security and observability", "Cloud IAM; Secret Manager; Cloud Logging/Monitoring; Security Command Center", "Protect identities/data and monitor the live service", pb],
                ["DevSecOps", "Cloud Build or GitHub Actions; Terraform", "Automate quality, security, infrastructure, and releases", pb],
            ]
        else:  # Azure, or cloud not stated -> Azure-shaped data reference stack
            data_pb = pb if cloud == "azure" else "Recommended managed reference stack; platform decision required"
            rows = [
                ["Source connectivity", "Fabric Data Factory pipelines; SFTP/REST connectors", "Ingest files, APIs, databases, and scheduled source extracts", basis("Fabric Data Factory", data_pb)],
                ["Orchestration", "Fabric Data Factory pipelines; Azure Logic Apps", "Coordinate schedules, dependencies, routing, retries, and exception workflows", basis("Azure Logic Apps", data_pb)],
                ["Validation and canonical transformation", "Fabric Dataflow Gen2 / notebooks; Azure Functions", "Validate, standardise, enrich, and transform source records into canonical structures", basis("Azure Functions", data_pb)],
                ["Integration services", "Azure API Management; Azure Service Bus", "Secure APIs, decouple interfaces, manage asynchronous exchange, and control consumers", basis("Azure API Management", data_pb)],
                ["Central data repository", "Microsoft Fabric OneLake Lakehouse/Warehouse; Azure SQL where transactional semantics are required", "Retain raw, curated, reference, and serving data with governed access", basis("Microsoft Fabric OneLake", data_pb)],
                ["Analytics and reporting", "Fabric semantic models; Power BI", "Provide governed operational reporting, KPI analysis, and reusable analytical views", basis("Power BI", data_pb)],
                ["Identity and secrets", "Microsoft Entra ID; Azure Key Vault; managed identities", "Apply role-based access, service identity, secret protection, and least privilege", basis("Microsoft Entra ID", data_pb)],
                ["Observability and security operations", "Azure Monitor; Application Insights; Log Analytics; Microsoft Sentinel", "Monitor applications, pipelines, interfaces, audit events, alerts, and support signals", basis("Microsoft Sentinel", data_pb)],
                ["DevSecOps and infrastructure", "Azure DevOps or GitHub Actions; Bicep/Terraform", "Automate build, test, security checks, environment provisioning, and controlled releases", data_pb],
            ]
    else:
        # Application / platform build (e.g. containerised app, migration,
        # modernization): layers reflect compute, data, messaging, and delivery.
        if cloud == "azure":
            rows = [
                ["Compute and orchestration", "Azure Kubernetes Service (AKS); Azure Container Apps", "Run containerised web, worker, and background services", basis("Azure Kubernetes Service, AKS", pb)],
                ["Application services", "Containerised .NET/Java/Node/Python services", "Host application workloads and APIs with portable packaging", pb],
                ["Database", "Azure Database for PostgreSQL; Azure SQL where required", "Provide the primary transactional data store", basis("Azure Database for PostgreSQL, PostgreSQL", pb)],
                ["Messaging and streaming", "Azure Service Bus / Event Hubs; Kafka on AKS where portability is required", "Decouple services and process asynchronous events", basis("Kafka, Pulsar", pb)],
                ["Caching", "Azure Cache for Redis", "Provide low-latency caching and session state", basis("Redis", pb)],
                ["Object storage and CDN", "Azure Blob Storage; Azure Front Door / CDN", "Store objects and accelerate content delivery", basis("CDN, CloudFront, S3", pb)],
                ["Search", "Azure AI Search or self-managed Elasticsearch/OpenSearch", "Provide search and indexing", basis("Elasticsearch", pb)],
                ["Identity and secrets", "Microsoft Entra ID; Azure Key Vault; managed identities", "Apply access control, service identity, and secret protection", basis("Microsoft Entra ID, OAuth, SAML", pb)],
                ["Observability", "Datadog or Azure Monitor / Application Insights / Log Analytics", "Metrics, logs, tracing, and alerting across the estate", basis("Datadog", pb)],
                ["DevSecOps and infrastructure", "Azure DevOps or GitHub Actions; Terraform/Bicep", "Automate build, test, security, provisioning, and controlled releases", basis("Terraform", pb)],
            ]
        elif cloud == "aws":
            rows = [
                ["Compute and orchestration", "Amazon EKS; ECS/Fargate", "Run containerised web, worker, and background services", basis("Amazon EKS, EKS, Kubernetes", pb)],
                ["Application services", "Containerised .NET/Java/Node/Python services", "Host application workloads and APIs with portable packaging", pb],
                ["Database", "Amazon RDS / Aurora PostgreSQL", "Provide the primary transactional data store", basis("PostgreSQL, RDS", pb)],
                ["Messaging and streaming", "Amazon MSK (Kafka); SNS/SQS", "Decouple services and process asynchronous events", basis("Kafka, Pulsar", pb)],
                ["Caching", "Amazon ElastiCache for Redis", "Provide low-latency caching and session state", basis("Redis", pb)],
                ["Object storage and CDN", "Amazon S3; CloudFront", "Store objects and accelerate content delivery", basis("S3, CloudFront", pb)],
                ["Search", "Amazon OpenSearch or self-managed Elasticsearch", "Provide search and indexing", basis("Elasticsearch, OpenSearch", pb)],
                ["Identity and secrets", "IAM/Cognito; Secrets Manager; KMS", "Apply access control, service identity, and secret protection", basis("OAuth, SAML", pb)],
                ["Observability", "Datadog or Amazon CloudWatch", "Metrics, logs, tracing, and alerting across the estate", basis("Datadog", pb)],
                ["DevSecOps and infrastructure", "CodePipeline/CodeBuild or GitHub Actions; Terraform", "Automate build, test, security, provisioning, and controlled releases", basis("Terraform", pb)],
            ]
        elif cloud == "gcp":
            rows = [
                ["Compute and orchestration", "Google Kubernetes Engine (GKE); Cloud Run", "Run containerised web, worker, and background services", basis("GKE, Kubernetes", pb)],
                ["Application services", "Containerised .NET/Java/Node/Python services", "Host application workloads and APIs with portable packaging", pb],
                ["Database", "Cloud SQL for PostgreSQL", "Provide the primary transactional data store", basis("PostgreSQL", pb)],
                ["Messaging and streaming", "Pub/Sub; Kafka on GKE where portability is required", "Decouple services and process asynchronous events", basis("Kafka, Pulsar", pb)],
                ["Caching", "Memorystore for Redis", "Provide low-latency caching and session state", basis("Redis", pb)],
                ["Object storage and CDN", "Cloud Storage; Cloud CDN", "Store objects and accelerate content delivery", pb],
                ["Search", "Self-managed Elasticsearch/OpenSearch", "Provide search and indexing", basis("Elasticsearch", pb)],
                ["Identity and secrets", "Cloud IAM; Secret Manager", "Apply access control, service identity, and secret protection", pb],
                ["Observability", "Datadog or Cloud Logging/Monitoring", "Metrics, logs, tracing, and alerting across the estate", basis("Datadog", pb)],
                ["DevSecOps and infrastructure", "Cloud Build or GitHub Actions; Terraform", "Automate build, test, security, provisioning, and controlled releases", basis("Terraform", pb)],
            ]
        else:
            rows = [
                ["Compute and orchestration", "Managed Kubernetes; container runtime", "Run containerised web, worker, and background services", basis("Kubernetes", pb)],
                ["Application services", "Containerised .NET/Java/Node/Python services", "Host application workloads and APIs with portable packaging", pb],
                ["Database", "PostgreSQL (managed)", "Provide the primary transactional data store", basis("PostgreSQL", pb)],
                ["Messaging and streaming", "Apache Kafka or cloud-native messaging", "Decouple services and process asynchronous events", basis("Kafka, Pulsar", pb)],
                ["Caching", "Redis", "Provide low-latency caching and session state", basis("Redis", pb)],
                ["Object storage and CDN", "Object storage; CDN", "Store objects and accelerate content delivery", pb],
                ["Search", "Elasticsearch/OpenSearch", "Provide search and indexing", basis("Elasticsearch", pb)],
                ["Identity and secrets", "OIDC/OAuth2; managed secrets/KMS", "Apply access control, service identity, and secret protection", basis("OAuth, SAML", pb)],
                ["Observability", "OpenTelemetry; Prometheus/Grafana or customer standard", "Metrics, logs, tracing, and alerting across the estate", basis("Datadog", pb)],
                ["DevSecOps and infrastructure", "GitHub Actions/GitLab CI; Terraform", "Automate build, test, security, provisioning, and controlled releases", basis("Terraform", pb)],
            ]
    if _ai_ml_is_applicable(understanding):
        ai_tech = {
            "azure": "Azure AI / Document Intelligence, or managed models",
            "aws": "Amazon Bedrock / SageMaker, or managed models",
            "gcp": "Vertex AI, or managed models",
        }.get(cloud, "Managed AI service or small models")
        rows.append(
            [
                "AI-assisted capabilities",
                ai_tech,
                "Support classification, anomaly detection, forecasting, and governed assistance",
                "Optional; requires customer approval, plus validation of value, data readiness, accuracy, and run cost before scale",
            ]
        )
    return {
        "headers": ["Architecture layer", "Proposed technology / service", "Role in the solution", "Status / basis"],
        "rows": rows,
    }


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
    t = re.sub(r"\s+", " ", (text or "").strip()).rstrip(".")
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
        return [_clip(i.strip(), 130) for i in (items or []) if (i or "").strip()][:limit]

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


def _assumptions_dependency_points(
    understanding: RFPUnderstanding | None,
) -> List[BulletPoint]:
    if understanding is None:
        return []
    assumptions = [
        item.strip()
        for item in (getattr(understanding, "assumptions", []) or [])
        if (item or "").strip()
    ][:5]
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
    acceptance = _matched_requirement_texts(
        understanding,
        ("acceptance", "uat", "approval", "sign-off", "cutover", "warranty", "support", "sla"),
        3,
    )
    if _ai_ml_is_applicable(understanding):
        platform = list(platform[:2]) + [
            "AI-assisted use cases proceed beyond pilot only after data readiness, accuracy, human-control, security, and unit-cost thresholds are accepted."
        ]
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
        ("flight", "fih", "gp4", "elp", "sftp", "email", "source"),
        3,
    )
    operational_points = _matched_requirement_texts(
        understanding,
        ("uplift", "catering", "productivity", "accuracy", "milestone", "sla", "iccms"),
        3,
    )
    control_points = _matched_requirement_texts(
        understanding,
        ("validate", "reconciliation", "audit", "retention", "quality", "security", "lineage"),
        3,
    )
    consumer_points = _matched_requirement_texts(
        understanding,
        ("report", "dashboard", "analytics", "output", "consumer", "power bi"),
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

    # Deployment Architecture
    if not _is_archetype_present(existing_keys, "deployment architecture"):
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
    if not has_platform_nfr:
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
    if _has_explicit_hadr_need(understanding) and not _is_archetype_present(existing_keys, "high availability"):
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
    if not _is_archetype_present(existing_keys, "software bill of materials"):
        add_slide(
            "Software Bill of Materials",
            "Proposed solution stack maps services to architecture layers",
            key_message=(
                "RFP-mandated technologies are preserved; gaps are completed with qualified ecosystem-aligned recommendations that remain subject to architecture confirmation."
            ),
            table=_sdlc_technology_table(understanding),
        )
    else:
        # The planner sometimes creates the required SBOM slide but omits the
        # structured table payload. Complete the existing slide instead of
        # accepting an empty table layout.
        for slide in deck_plan.slides:
            if (slide.archetype or "").strip().lower() == "software bill of materials":
                headers_text = " ".join(str(value) for value in ((slide.table or {}).get("headers") or [])).lower()
                table_text = " ".join(
                    str(cell)
                    for row in ((slide.table or {}).get("rows") or [])
                    for cell in row
                ).lower()
                implementation_tokens = (
                    "fabric", "onelake", "azure functions", "logic apps", "api management",
                    "service bus", "app service", "container apps", "postgresql", "kafka",
                    "aws glue", "lambda", "bigquery", "cloud run", "terraform",
                )
                is_generic_lifecycle_table = "sdlc phase" in headers_text or not any(
                    token in table_text for token in implementation_tokens
                )
                if not slide.table or not slide.table.get("headers") or is_generic_lifecycle_table:
                    slide.table = _sdlc_technology_table(understanding)
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
    if not has_sdlc_tech:
        add_slide(
            "Software Bill of Materials",
            "Proposed solution stack maps services to architecture layers",
            key_message=(
                "This view names the implementation services behind each architecture layer and distinguishes requirements from proposed choices."
            ),
            table=_sdlc_technology_table(understanding),
        )

    has_ai_ml_slide = any(
        any(token in ((slide.title or "") + " " + _slide_visible_text(slide)).lower()
            for token in ("ai-assisted", "ai/ml opportunity", "machine learning opportunity"))
        for slide in deck_plan.slides
    )
    if _ai_ml_is_applicable(understanding):
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

    if _ai_ml_is_applicable(understanding):
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
    if assumption_points and not _is_archetype_present(existing_keys, "assumptions & dependencies"):
        add_slide(
            "Assumptions & Dependencies",
            "Delivery conditions must be confirmed early",
            detailed_points=assumption_points,
            key_message=(
                "Early validation of assumptions and dependencies protects scope, schedule, and operational readiness."
            ),
        )

    # Delivery Plan
    if not _is_archetype_present(existing_keys, "delivery plan"):
        add_slide(
            "Delivery Plan",
            "Agile delivery turns priorities into usable increments",
            detailed_points=_agile_delivery_points(),
            key_message=(
                "Product ownership, persistent cross-functional squads, and evidence-based release decisions maintain speed without weakening governance."
            ),
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
            "Agile roadmap releases value through recurring increments",
            detailed_points=_agile_roadmap_points(),
            key_message=(
                "Discovery, engineering, assurance, security, and operational readiness progress together within each increment."
            ),
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
    has_delivery_squad_view = any(
        (s.archetype or "").strip().lower() in {"delivery plan", "timeline"}
        and _has_agile_delivery_language(s)
        for s in deck_plan.slides
    )
    if not has_delivery_squad_view and not _is_archetype_present(existing_keys, "team"):
        add_slide(
            "Team",
            "Product-aligned squads combine business and engineering ownership",
            detailed_points=_agile_squad_points(),
            key_message=(
                "Persistent cross-functional squads own usable outcomes end to end, supported by enabling chapters and lightweight steering governance."
            ),
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

    # Slides added later in this function must receive the same AI governance
    # treatment as model-provided slides handled above.
    if _ai_ml_is_applicable(understanding):
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
            s.detailed_points = _assumptions_dependency_points(understanding)
            continue

        if getattr(s, "diagram", None) is not None and any(
            token in title for token in ("data domain", "data model", "information model")
        ):
            s.key_message = (
                "The data model separates authoritative inputs, canonical operational entities, control evidence, and governed consumption products."
            )
            s.bullets = []
            s.cards = []
            s.comparison = None
            s.detailed_points = _data_domain_points(understanding)

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
            if s.diagram is None:
                s.diagram = DiagramSpec(kind="testing", prompt=_build_diagram_prompt("testing", understanding))
            elif not getattr(s.diagram, "approved", False):
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
            if s.diagram is None:
                s.diagram = DiagramSpec(kind="ams", prompt=_build_diagram_prompt("ams", understanding))
            elif not getattr(s.diagram, "approved", False):
                s.diagram.kind = "ams"
                s.diagram.prompt = _build_diagram_prompt("ams", understanding)
            continue

        if arch in {"delivery plan", "timeline", "team"} and not _has_agile_delivery_language(s):
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
    return _clip(_complete_sentences(text, 1) or text, limit)


def _cards_from_detailed_points(points: List[BulletPoint], *, accent: str = "info", max_cards: int = 4) -> List[Card]:
    cards: List[Card] = []
    for point in points[:max_cards]:
        full = re.sub(r"\s+", " ", (getattr(point, "text", "") or "").strip())
        heading = _concise_heading(full)
        bullets = [
            _clip(item, 140)
            for item in (getattr(point, "sub_points", None) or [])
            if (item or "").strip()
        ][:3]
        # If the heading dropped meaningful detail from the point text, keep the
        # full point as the leading bullet so nothing is silently lost.
        if full and len(full) > len(heading) + 12 and not any(
            full[:40].lower() in b.lower() for b in bullets
        ):
            bullets = ([_clip(full, 150)] + bullets)[:3]
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
                _clip(text, 100)
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
                slide.cards = _cards_from_detailed_points(
                    slide.detailed_points,
                    accent="challenge" if arch == "risks" else "info",
                    max_cards=4,
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


def ensure_diagrams_for_key_slides(deck_plan: DeckPlan, understanding: RFPUnderstanding | None = None) -> DeckPlan:
    """Ensure diagrams exist (as guarded approvals) for key slides."""
    for s in deck_plan.slides:
        arch = (s.archetype or "").lower()
        title = (s.title or "").lower()

        if arch in {
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
                if arch == "architecture":
                    prompt = _build_diagram_prompt("architecture", understanding)
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

                if prompt:
                    s.diagram = DiagramSpec(kind=kind, prompt=prompt, approved=False, image_path=None)
            elif arch == "solution overview" and _is_exec_summary(s):
                # Remove any legacy Exec Summary diagram to keep it text-native.
                s.diagram = None

            if (
                s.diagram is not None
                and _ai_ml_is_applicable(understanding)
                and not getattr(s.diagram, "approved", False)
                and arch in {"architecture", "deployment architecture", "timeline", "solution overview"}
                and not _is_exec_summary(s)
                and "ai-assisted" not in (s.diagram.prompt or "").lower()
            ):
                s.diagram.prompt = (
                    (s.diagram.prompt or "").rstrip()
                    + "\n"
                    + _ai_ml_architecture_clause(understanding).strip()
                ).strip()

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
) -> str:
    compact_layouts = layout_names[: min(16, len(layout_names))]
    payload = {
        "layouts": compact_layouts,
        "understanding": _compact_understanding_for_plan(understanding),
        "narrative": _compact_narrative_for_plan(narrative),
    }
    return (
        "You are a Tier-1 consulting deck architect. Return strict JSON matching the DeckPlan schema.\n"
        "Create a focused HCLTech proposal deck plan with 16-20 main slides plus appendix only when needed.\n"
        "Story order: Title, Agenda, Executive Summary, Current Challenges, Business Outcomes, "
        "Proposed Solution, End-to-End Operational Flow, Concrete Solution Architecture, "
        "Integration/Data/Reporting, selective AI/ML opportunities, Deployment & Resilience, Scope/Assumptions, Delivery Roadmap, "
        "Risks, Commercials, Next Steps.\n"
        "Rules: challenges must precede solution; avoid duplicate slides; avoid tutorial-like deployment "
        "or Agile ceremony slides; include a concrete data-hub architecture and an architecture-layer technology stack naming implementable services; exclude procurement/submission tools; distinguish mandated technologies from proposed choices; assess AI/ML selectively using value, data readiness, human control, infrastructure, and run-cost gates; keep diagram slides visual-first; "
        "keep bullets short enough to render without clipping.\n"
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

    Deterministically derived from the RFP understanding + narrative so it is
    stable and testable. Later phases (specialist agents) consume the same
    brief so sections build on one foundation instead of diverging.
    """
    customer: str
    engagement_kind: str
    target_cloud: str
    solution_name: str
    win_theme: str


def _engagement_kind(understanding: RFPUnderstanding | None) -> str:
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
) -> SolutionBrief:
    corpus = " ".join(
        [_understanding_text(understanding)]
        + ((getattr(understanding, "solution_technologies", None) or []) if understanding else [])
    )
    win = (getattr(narrative, "value_proposition", "") or "").strip() if narrative else ""
    return SolutionBrief(
        customer=_customer_label(understanding),
        engagement_kind=_engagement_kind(understanding),
        target_cloud=_cloud_signal(corpus),
        solution_name=_solution_name(understanding),
        win_theme=_clip(_complete_sentences(win, 1) or win, 180) if win else "",
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


def _post_process_deck_plan(
    deck_plan: DeckPlan,
    *,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
) -> DeckPlan:
    brief = build_solution_brief(understanding, narrative)
    log.info(
        "Solution brief: customer=%r engagement=%r cloud=%r solution=%r",
        brief.customer, brief.engagement_kind, brief.target_cloud, brief.solution_name,
    )
    keystone_ids = {str(section.get("slide_id", "")) for section in _proposal_section_skeleton(understanding)}
    deck_plan = ensure_required_slides(deck_plan, understanding=understanding, narrative=narrative)
    deck_plan = enrich_slide_detail(deck_plan, understanding=understanding)
    deck_plan = prune_empty_content_slides(deck_plan)
    deck_plan = prune_redundant_storyline_slides(deck_plan, protected_ids=keystone_ids)
    deck_plan = consulting_grade_proposal_polish(deck_plan, understanding=understanding, narrative=narrative)
    deck_plan = order_deck(deck_plan)
    deck_plan = synchronize_agenda(deck_plan)
    deck_plan = polish_deck_text(deck_plan)
    deck_plan = lead_consolidation(deck_plan, brief)
    deck_plan = ensure_diagrams_for_key_slides(deck_plan, understanding=understanding)
    deck_plan = enforce_slide_density(deck_plan)
    return deck_plan


def _fallback_deck_plan(
    *,
    title: str,
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
) -> DeckPlan:
    """Build a deterministic plan if the LLM planner cannot be reached."""
    slides = [
        SlideSpec(slide_id="fallback_title", title=title or "Proposal", archetype="Title"),
        SlideSpec(slide_id="fallback_agenda", title="Agenda", archetype="Agenda"),
        SlideSpec(
            slide_id="fallback_exec",
            title="Executive Summary",
            archetype="Solution Overview",
            key_message=(getattr(narrative, "value_proposition", "") or "") if narrative else None,
            cards=_exec_summary_cards(understanding, narrative),
        ),
        SlideSpec(
            slide_id="fallback_context",
            title="Current challenges define the solution priorities",
            archetype="Customer Context",
            detailed_points=_context_detailed_points(understanding),
        ),
        SlideSpec(
            slide_id="fallback_requirements",
            title="Scope and requirements are broad but bounded",
            archetype="Requirements",
            detailed_points=_requirements_detailed_points(understanding),
        ),
        SlideSpec(
            slide_id="fallback_solution",
            title="The proposed data hub creates one trusted flow",
            archetype="Solution Overview",
            bullets=_exec_summary_bullets(understanding, narrative),
        ),
        SlideSpec(
            slide_id="fallback_architecture",
            title="Solution architecture builds the centralized data hub",
            archetype="Architecture",
            diagram=DiagramSpec(kind="architecture", prompt=_build_diagram_prompt("architecture", understanding)),
        ),
        SlideSpec(
            slide_id="fallback_deployment",
            title="Deployment and resilience architecture protects operations",
            archetype="Deployment Architecture",
            diagram=DiagramSpec(kind="deployment", prompt=_build_diagram_prompt("deployment", understanding)),
        ),
        SlideSpec(
            slide_id="fallback_assumptions",
            title="Delivery conditions must be confirmed early",
            archetype="Assumptions & Dependencies",
            detailed_points=_assumptions_dependency_points(understanding),
        ),
        SlideSpec(
            slide_id="fallback_timeline",
            title="Agile roadmap releases value through increments",
            archetype="Timeline",
            detailed_points=_agile_roadmap_points(),
        ),
        SlideSpec(
            slide_id="fallback_risks",
            title="Key risks can be actively managed",
            archetype="Risks",
            detailed_points=_risk_detailed_points(understanding),
        ),
        SlideSpec(
            slide_id="fallback_commercials",
            title="Commercials should align build and support",
            archetype="Commercials",
            bullets=[
                "Confirm implementation, warranty, and maintenance scope as one accountable commercial model",
                "Use delivery gates and acceptance evidence to manage non-conformance and rework exposure",
                "Treat open hosting, migration, and integration assumptions as commercial dependencies",
            ],
        ),
        SlideSpec(
            slide_id="fallback_next",
            title="Recommended Next Steps",
            archetype="Next Steps",
            bullets=_next_steps_bullets(understanding),
        ),
    ]
    return DeckPlan(deck_title=title or "Proposal", slides=slides)


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


def _cloud_signal(corpus: str) -> str:
    """Return the dominant target cloud ('azure' | 'aws' | 'gcp' | '')."""
    c = (corpus or "").lower()
    if any(t in c for t in ("azure", "microsoft fabric", "power bi", "entra", "onelake", "aks", "app service")):
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
    if narrative is not None:
        for attr in ("solution_themes", "strategic_outcomes"):
            vals = [v.strip() for v in (getattr(narrative, attr, []) or []) if (v or "").strip()]
            if len(vals) >= 2:
                return [_clip(v, 90) for v in vals[:3]]
    scope = [s.strip() for s in (getattr(understanding, "in_scope_work", []) or []) if (s or "").strip()]
    if len(scope) >= 2:
        return [_clip(s, 90) for s in scope[:3]]
    vp = (getattr(narrative, "value_proposition", "") or "").strip() if narrative else ""
    if vp:
        return [_clip(vp, 110)]
    return [
        "Meet the stated requirements with a proven, low-risk delivery approach",
        "Preserve continuity through phased delivery, validation, and cutover controls",
        "Build in security, observability, and operational readiness from the start",
    ]


def _proposal_section_skeleton(understanding: RFPUnderstanding | None) -> List[Dict[str, Any]]:
    """Adaptive proposal-completeness skeleton used for chunked planning.

    Titles and section gating are engagement-agnostic: no customer name, legacy
    system, or data-platform framing is assumed. Optional sections are added
    only on genuine signals so a non-data engagement (e.g. a cloud migration)
    does not inherit data-hub/reporting slides.
    """
    text = _understanding_text(understanding)
    customer = _customer_label(understanding)
    is_data = _is_data_platform_engagement(understanding)
    has_integration = _contains_any(text, ("api", "interface", "integration", "sftp", "mq", "message queue", "event", "webhook", "connector"))
    has_migration = _contains_any(text, ("migration", "migrate", "legacy", "cutover", "re-platform", "replatform", "modernization", "modernisation", "lift and shift"))
    has_support = _contains_any(text, ("ams", "maintenance", "support", "warranty", "hypercare", "operate", "run"))
    has_security = _contains_any(text, ("security", "mfa", "audit", "siem", "access control", "encryption", "compliance", "log"))

    # Neutral outcomes title; only name the customer when it is a real name.
    outcomes_title = (
        f"Target outcomes align to {customer}'s priorities"
        if customer != "the customer"
        else "Target outcomes and measurable commitments"
    )

    sections: List[Dict[str, Any]] = [
        {"slide_id": "sk_title", "title": "Proposal title", "archetype": "Title", "purpose": "Customer-facing cover"},
        {"slide_id": "sk_agenda", "title": "Agenda", "archetype": "Agenda", "purpose": "Summarize the deck flow"},
        {"slide_id": "sk_exec", "title": "Executive Summary", "archetype": "Solution Overview", "purpose": "Win thesis, situation, response, outcomes"},
        {"slide_id": "sk_context", "title": "Current challenges define the solution priorities", "archetype": "Customer Context", "purpose": "Show understanding of current context, pain, and constraints"},
        {"slide_id": "sk_outcomes", "title": outcomes_title, "archetype": "Value & Differentiators", "purpose": "Business and technical outcomes the proposal commits to"},
        {"slide_id": "sk_scope", "title": "Scope and boundaries are explicit", "archetype": "Requirements", "purpose": "In scope, out of scope, open decisions"},
        {"slide_id": "sk_solution", "title": "Proposed solution at a glance", "archetype": "Solution Overview", "purpose": "Solution pillars and how they fit together"},
        {"slide_id": "sk_flow", "title": "End-to-end solution flow", "archetype": "Solution Overview", "purpose": "Visual flow from inputs through the solution to outputs", "diagram_kind": "process"},
        {"slide_id": "sk_arch", "title": "Concrete solution architecture", "archetype": "Architecture", "purpose": "Specific build architecture and major components", "diagram_kind": "architecture"},
    ]
    if has_integration:
        sections.append({"slide_id": "sk_integration", "title": "Integration architecture connects source and consumer systems", "archetype": "Architecture", "purpose": "APIs, files, messaging, system dependencies, error handling", "diagram_kind": "architecture"})
    if is_data:
        sections.extend([
            {"slide_id": "sk_data_model", "title": "Core data domains and ownership", "archetype": "Content", "purpose": "Core data entities/domains and ownership"},
            {"slide_id": "sk_reporting", "title": "Reporting and analytics reuse trusted data", "archetype": "Content", "purpose": "Reports, dashboards, analytics, and governed outputs"},
        ])
    if _ai_ml_is_applicable(understanding):
        sections.append({
            "slide_id": "sk_ai_opportunities",
            "title": "AI-assisted capabilities target high-value use cases",
            "archetype": "Value & Differentiators",
            "purpose": "Prioritised AI/ML use cases with value, data-readiness, human-control, infrastructure, cost, and approval gates",
        })
    if has_security:
        sections.append({"slide_id": "sk_security", "title": "Security and observability are built into the platform", "archetype": "Content", "purpose": "Access control, audit, logs, monitoring, resilience"})
    sections.append({"slide_id": "sk_deployment", "title": "Deployment and resilience protect operations", "archetype": "Deployment Architecture", "purpose": "Runtime environments, release path, DR/backup, monitoring", "diagram_kind": "deployment"})
    if has_migration:
        sections.append({"slide_id": "sk_migration", "title": "Migration and cutover protect continuity", "archetype": "Delivery Plan", "purpose": "Data/workload migration, rehearsals, cutover controls"})
    sections.extend([
        {"slide_id": "sk_testing", "title": "Acceptance evidence proves the solution is ready", "archetype": "Delivery Plan", "purpose": "Requirement-led evidence for named interfaces, functional parity, reconciliation, controls, customer UAT, and cutover readiness"},
        {"slide_id": "sk_roadmap", "title": "Agile roadmap releases value through increments", "archetype": "Timeline", "purpose": "Incremental delivery, gates, feedback, readiness", "diagram_kind": "timeline"},
    ])
    if has_support:
        sections.append({"slide_id": "sk_ams", "title": "Warranty and AMS support model", "archetype": "Delivery Plan", "purpose": "Service coverage, monitoring, interface ownership, runbooks, warranty transition, and improvement"})
    sections.extend([
        {"slide_id": "sk_governance", "title": "Governance resolves decisions without slowing delivery", "archetype": "Team", "purpose": "Decision rights, product ownership, squads, forums", "diagram_kind": "org"},
        {"slide_id": "sk_value", "title": "Proposal value and differentiators", "archetype": "Value & Differentiators", "purpose": "Why this approach is credible, useful, and lower risk for the customer"},
        {"slide_id": "sk_tech", "title": "Proposed solution stack maps services to architecture layers", "archetype": "Software Bill of Materials", "purpose": "Concrete implementation services by architecture layer, with mandated/referenced/proposed status"},
        {"slide_id": "sk_risks", "title": "Key risks and mitigations are actively managed", "archetype": "Risks", "purpose": "Risk, implication, mitigation, owner/control"},
        {"slide_id": "sk_assumptions", "title": "Assumptions and dependencies need early closure", "archetype": "Assumptions & Dependencies", "purpose": "Hosting, access, interfaces, migration, approvals"},
        {"slide_id": "sk_commercials", "title": "Commercials align build, warranty and support", "archetype": "Commercials", "purpose": "Commercial scope, sensitivities, boundaries"},
        {"slide_id": "sk_next", "title": "Next Steps", "archetype": "Next Steps", "purpose": "Recommended customer actions"},
    ])
    return sections


def _chunked_plan_input(
    understanding: RFPUnderstanding | None,
    narrative: ExecutiveNarrative | None,
) -> str:
    brief = build_solution_brief(understanding, narrative)
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
) -> DeckPlan:
    sections = _proposal_section_skeleton(understanding)
    batch_size = max(2, min(6, getattr(settings, "deck_plan_batch_size", 4)))
    input_json = _chunked_plan_input(understanding, narrative)
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
    if sid in {"sk_solution", "sk_flow", "sk_arch", "sk_deployment", "sk_security"} or archetype in {
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
) -> DeckPlan:
    sections = _proposal_section_skeleton(understanding)
    batch_size = max(2, min(6, getattr(settings, "deck_plan_batch_size", 4)))
    input_json = _chunked_plan_input(understanding, narrative)

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
        )
        deck_plan = _post_process_deck_plan(
            deck_plan,
            understanding=state.understanding,
            narrative=state.narrative,
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
        )
        deck_plan = _post_process_deck_plan(
            deck_plan,
            understanding=state.understanding,
            narrative=state.narrative,
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
    rag_context = _bounded_plan_context(state.rag_context or "")

    layout_json = json.dumps(layout_names, ensure_ascii=False, separators=(",", ":"))
    placeholder_json = json.dumps(placeholder_map, ensure_ascii=False, separators=(",", ":"))

    prompt = DECK_PLAN_V2_PROMPT.format(
        layout_names=layout_json,
        placeholder_map=placeholder_json,
        rag_context=rag_context,
        understanding_json=understanding_json,
        narrative_json=narrative_json,
    )
    prompt_mode = "full"
    max_prompt_chars = max(12000, getattr(settings, "deck_plan_prompt_max_chars", 30000))
    if len(prompt) > max_prompt_chars:
        prompt = _compact_deck_plan_prompt(
            layout_names=layout_names,
            understanding=state.understanding,
            narrative=state.narrative,
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
    )

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
        compressed = response_as_schema(
            prompt, DeckPlan, reasoning_effort=settings.reasoning_effort_medium
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
                prompt,
                DeckNotes,
                model=settings.model_fast,
                reasoning_effort=settings.reasoning_effort_low,
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
