from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass
from typing import Iterable, List, Sequence

from rfp2deck.core.schemas import SourceDocument, SourceEvidenceBatch

log = logging.getLogger(__name__)


@dataclass(frozen=True)
class EvidenceChunk:
    document: SourceDocument
    chunk_id: str
    text: str


_LOCATOR_MARKER = re.compile(
    r"^(?:---\s+(?:PAGE|SHEET|TABLE)|\[(?:PARAGRAPH|TABLE|SHEET)\b)",
    flags=re.IGNORECASE,
)


def split_source_document(document: SourceDocument, max_chars: int) -> List[EvidenceChunk]:
    """Split source text into bounded chunks while retaining locator context."""
    max_chars = max(4000, int(max_chars))
    lines = document.text.splitlines(keepends=True)
    if not lines:
        return [
            EvidenceChunk(
                document=document,
                chunk_id=f"{document.document_id}-chunk-001",
                text="",
            )
        ]

    chunks: List[str] = []
    current: List[str] = []
    current_chars = 0
    locator_context = ""

    def flush() -> None:
        nonlocal current, current_chars
        text = "".join(current).strip()
        if text:
            chunks.append(text)
        current = []
        current_chars = 0

    for raw_line in lines:
        stripped = raw_line.strip()
        if _LOCATOR_MARKER.match(stripped):
            locator_context = stripped
        remaining = raw_line
        while remaining:
            if current and current_chars + len(remaining) > max_chars:
                capacity = max_chars - current_chars
                if capacity > 0:
                    current.append(remaining[:capacity])
                    remaining = remaining[capacity:]
                    current_chars += capacity
                flush()
                continue
            if not current and locator_context and not _LOCATOR_MARKER.match(stripped):
                continuation = f"[LOCATOR CONTEXT: {locator_context}]\n"
                current.append(continuation)
                current_chars += len(continuation)
            capacity = max_chars - current_chars
            if capacity <= 0:
                flush()
                continue
            current.append(remaining[:capacity])
            current_chars += min(len(remaining), capacity)
            remaining = remaining[capacity:]
            if current_chars >= max_chars:
                flush()

    flush()
    return [
        EvidenceChunk(
            document=document,
            chunk_id=f"{document.document_id}-chunk-{index:03d}",
            text=text,
        )
        for index, text in enumerate(chunks, start=1)
    ]


def _normalise(value: str) -> str:
    return re.sub(r"\s+", " ", (value or "").strip().lower())


def _unique_strings(values: Iterable[str]) -> List[str]:
    seen = set()
    result = []
    for value in values:
        cleaned = re.sub(r"\s+", " ", (value or "").strip())
        key = _normalise(cleaned)
        if not cleaned or key in seen:
            continue
        seen.add(key)
        result.append(cleaned)
    return result


def merge_evidence_batches(batches: Sequence[SourceEvidenceBatch]) -> SourceEvidenceBatch:
    """Merge chunk outputs and remove overlap duplicates without losing provenance."""
    requirements = []
    seen_requirements = set()
    sbom = []
    seen_sbom = set()
    outcomes = []
    seen_outcomes = set()
    conflicts = []
    seen_conflicts = set()

    for batch in batches:
        for requirement in batch.requirements:
            if not requirement.source_document_ids:
                requirement.source_document_ids = [batch.source_document_id]
            if requirement.source_ref and not requirement.source_refs:
                requirement.source_refs = [requirement.source_ref]
            source_key = ",".join(sorted(requirement.source_document_ids))
            key = (_normalise(requirement.text), source_key, requirement.status)
            if not key[0] or key in seen_requirements:
                continue
            seen_requirements.add(key)
            requirements.append(requirement)
        for item in batch.software_bill_of_materials:
            key = (_normalise(item.component), _normalise(item.purpose))
            if not key[0] or key in seen_sbom:
                continue
            seen_sbom.add(key)
            sbom.append(item)
        for outcome in batch.clarification_outcomes:
            key = (_normalise(outcome.question), _normalise(outcome.customer_response))
            if key in seen_outcomes:
                continue
            seen_outcomes.add(key)
            outcomes.append(outcome)
        for conflict in batch.source_conflicts:
            key = (_normalise(conflict.topic), tuple(sorted(conflict.source_refs)))
            if key in seen_conflicts:
                continue
            seen_conflicts.add(key)
            conflicts.append(conflict)

    def strings(field: str) -> List[str]:
        return _unique_strings(value for batch in batches for value in getattr(batch, field))

    return SourceEvidenceBatch(
        source_document_id="rfp-package",
        chunk_id="merged-evidence",
        context_facts=strings("context_facts"),
        summary_points=strings("summary_points"),
        project_scope_points=strings("project_scope_points"),
        in_scope_work=strings("in_scope_work"),
        requirements=requirements,
        assumptions=strings("assumptions"),
        risks=strings("risks"),
        submission_instructions=strings("submission_instructions"),
        procurement_or_submission_tools=strings("procurement_or_submission_tools"),
        non_solution_references=strings("non_solution_references"),
        solution_technologies=strings("solution_technologies"),
        software_bill_of_materials=sbom,
        clarification_outcomes=outcomes,
        source_conflicts=conflicts,
    )


def render_evidence_for_prompt(evidence: SourceEvidenceBatch, max_chars: int) -> str:
    """Render merged evidence under a hard budget without dropping requirements."""
    max_chars = max(40000, int(max_chars))
    compact = evidence.model_copy(deep=True)
    for excerpt_limit in (320, 160, 0):
        for requirement in compact.requirements:
            if requirement.source_text:
                requirement.source_text = (
                    requirement.source_text[:excerpt_limit] if excerpt_limit else None
                )
        rendered = json.dumps(
            compact.model_dump(exclude_none=True, exclude_defaults=True),
            ensure_ascii=False,
            separators=(",", ":"),
        )
        if len(rendered) <= max_chars:
            return rendered

    for field in (
        "context_facts",
        "summary_points",
        "project_scope_points",
        "assumptions",
        "risks",
        "submission_instructions",
    ):
        values = getattr(compact, field)
        setattr(compact, field, values[:100])
    rendered = json.dumps(
        compact.model_dump(exclude_none=True, exclude_defaults=True),
        ensure_ascii=False,
        separators=(",", ":"),
    )
    if len(rendered) <= max_chars:
        return rendered

    rendered = _render_indexed_evidence(compact)
    if len(rendered) > max_chars:
        rendered = _render_budgeted_indexed_evidence(compact, max_chars)
        log.warning(
            "Requirement evidence exceeded final prompt budget after lossless compaction; "
            "using budgeted indexed evidence (requirements=%d, chars=%d/%d)",
            len(compact.requirements),
            len(rendered),
            max_chars,
        )
    return rendered


def _requirement_rank(requirement) -> tuple[int, int, str]:
    priority_rank = {"must": 0, "should": 1, "may": 2}.get(requirement.priority, 1)
    status_rank = {"active": 0, "clarified": 1, "unresolved": 2, "superseded": 3}.get(
        requirement.status, 2
    )
    return (priority_rank, status_rank, requirement.id)


def _render_budgeted_indexed_evidence(evidence: SourceEvidenceBatch, max_chars: int) -> str:
    """Fit evidence under budget by keeping the highest-signal requirements."""
    sorted_requirements = sorted(evidence.requirements, key=_requirement_rank)
    low = 0
    high = len(sorted_requirements)
    best = ""

    while low <= high:
        mid = (low + high) // 2
        candidate = evidence.model_copy(deep=True)
        candidate.requirements = sorted_requirements[:mid]
        rendered = _render_indexed_evidence(
            candidate,
            omitted_requirements=max(0, len(sorted_requirements) - mid),
            budgeted=True,
        )
        if len(rendered) <= max_chars:
            best = rendered
            low = mid + 1
        else:
            high = mid - 1

    if best:
        return best

    empty = evidence.model_copy(deep=True)
    empty.requirements = []
    return _render_indexed_evidence(
        empty,
        omitted_requirements=len(sorted_requirements),
        budgeted=True,
    )[:max_chars]


def _render_indexed_evidence(
    evidence: SourceEvidenceBatch,
    *,
    omitted_requirements: int = 0,
    budgeted: bool = False,
) -> str:
    """Preserve requirements and provenance with short keys and shared indexes."""
    def cleaned(value: str) -> str:
        return re.sub(r"\s+", " ", (value or "").strip())

    document_values = _unique_strings(
        document_id
        for requirement in evidence.requirements
        for document_id in requirement.source_document_ids
    )
    document_index = {value: index for index, value in enumerate(document_values)}
    reference_values = _unique_strings(
        reference
        for requirement in evidence.requirements
        for reference in (
            requirement.source_refs
            or ([requirement.source_ref] if requirement.source_ref else [])
        )
    )
    reference_index = {value: index for index, value in enumerate(reference_values)}
    requirement_rows = []
    for requirement in evidence.requirements:
        row = {"i": requirement.id, "t": requirement.text}
        if requirement.priority != "should":
            row["p"] = requirement.priority
        refs = requirement.source_refs or (
            [requirement.source_ref] if requirement.source_ref else []
        )
        if refs:
            row["r"] = [reference_index[cleaned(reference)] for reference in refs]
        if requirement.source_document_ids:
            row["d"] = [document_index[cleaned(value)] for value in requirement.source_document_ids]
        if requirement.authority != "authoritative":
            row["a"] = requirement.authority
        if requirement.status != "active":
            row["s"] = requirement.status
        if requirement.supersedes:
            row["u"] = requirement.supersedes
        if requirement.confidence != 1.0:
            row["c"] = requirement.confidence
        requirement_rows.append(row)

    payload = {
        "format": "indexed-rfp-evidence-v1",
        "budgeted": budgeted,
        "omitted_requirements": omitted_requirements,
        "legend": {
            "requirements": "i=id,t=text,p=priority,r=source reference indexes,d=document indexes,a=authority,s=status,u=supersedes,c=confidence",
            "defaults": "p=should,a=authoritative,s=active,c=1.0",
        },
        "documents": document_values,
        "source_refs": reference_values,
        "requirements": requirement_rows,
        "context_facts": evidence.context_facts,
        "summary_points": evidence.summary_points,
        "project_scope_points": evidence.project_scope_points,
        "in_scope_work": evidence.in_scope_work,
        "assumptions": evidence.assumptions,
        "risks": evidence.risks,
        "submission_instructions": evidence.submission_instructions,
        "procurement_or_submission_tools": evidence.procurement_or_submission_tools,
        "non_solution_references": evidence.non_solution_references,
        "solution_technologies": evidence.solution_technologies,
        "software_bill_of_materials": [
            item.model_dump(exclude_defaults=True, exclude_none=True)
            for item in evidence.software_bill_of_materials
        ],
        "clarification_outcomes": [
            item.model_dump(exclude_defaults=True, exclude_none=True)
            for item in evidence.clarification_outcomes
        ],
        "source_conflicts": [
            item.model_dump(exclude_defaults=True, exclude_none=True)
            for item in evidence.source_conflicts
        ],
    }
    return json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
