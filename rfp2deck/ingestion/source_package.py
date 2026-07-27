from __future__ import annotations

import hashlib
import re
from datetime import datetime
from pathlib import Path
from typing import Iterable, List, Literal, Sequence, Tuple

from rfp2deck.core.schemas import (
    ClarificationRecord,
    DocumentType,
    SourceAuthority,
    SourceDocument,
    SourceReconciliation,
)
from rfp2deck.ingestion.docx_parser import parse_docx
from rfp2deck.ingestion.pdf_parser import parse_pdf
from rfp2deck.ingestion.xlsx_parser import parse_xlsx

UploadRole = Literal["primary", "clarification", "supporting"]

_ADDENDUM_TERMS = ("addendum", "corrigendum", "amendment", "revision", "revised")
_CLARIFICATION_TERMS = (
    "clarification",
    "q&a",
    "q and a",
    "questions and answers",
    "vendor questions",
    "customer responses",
)
_ANNEXURE_TERMS = ("annex", "annexure", "appendix", "schedule", "scope of work", "sow")
_COMMERCIAL_TERMS = ("commercial", "pricing", "price schedule", "bill of quantities", "boq")

_PRECEDENCE = {
    "customer_addendum": 500,
    "customer_clarification": 400,
    "annexure": 300,
    "base_rfp": 300,
    "commercial": 200,
    "supporting_reference": 100,
    "unknown": 50,
}


def _contains_any(text: str, terms: Sequence[str]) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in terms)


def _document_id(name: str, data: bytes) -> str:
    safe_stem = re.sub(r"[^a-z0-9]+", "-", Path(name).stem.lower()).strip("-")[:36]
    digest = hashlib.sha256(name.encode("utf-8", errors="ignore") + data).hexdigest()[:10]
    return f"doc-{safe_stem or 'source'}-{digest}"


def _parse_date_value(sample: str) -> str | None:
    patterns = (
        (r"\b(20\d{2})[-/.](0?[1-9]|1[0-2])[-/.]([0-2]?\d|3[01])\b", (1, 2, 3)),
        (r"\b([0-2]?\d|3[01])[-/.](0?[1-9]|1[0-2])[-/.](20\d{2})\b", (3, 2, 1)),
    )
    for pattern, indexes in patterns:
        match = re.search(pattern, sample)
        if not match:
            continue
        try:
            year, month, day = (int(match.group(index)) for index in indexes)
            return datetime(year, month, day).date().isoformat()
        except ValueError:
            continue
    month_match = re.search(
        r"\b([0-2]?\d|3[01])\s+"
        r"(Jan(?:uary)?|Feb(?:ruary)?|Mar(?:ch)?|Apr(?:il)?|May|Jun(?:e)?|"
        r"Jul(?:y)?|Aug(?:ust)?|Sep(?:tember)?|Oct(?:ober)?|Nov(?:ember)?|Dec(?:ember)?)"
        r"\s+(20\d{2})\b",
        sample,
        flags=re.IGNORECASE,
    )
    if month_match:
        try:
            parsed = datetime.strptime(
                " ".join(month_match.groups()),
                "%d %B %Y" if len(month_match.group(2)) > 3 else "%d %b %Y",
            )
            return parsed.date().isoformat()
        except ValueError:
            return None
    return None


def _infer_issue_date(name: str, text: str) -> str | None:
    filename_date = _parse_date_value(name)
    if filename_date:
        return filename_date
    explicit_date = re.search(
        r"(?:date of issue|issue date|issued on|published on)\s*[:\-]?\s*"
        r"([^\n]{6,32})",
        text[:4000],
        flags=re.IGNORECASE,
    )
    if explicit_date:
        return _parse_date_value(explicit_date.group(1))
    return None


def classify_source(name: str, text: str, role: UploadRole) -> Tuple[DocumentType, SourceAuthority]:
    name_sample = name.lower()
    if role == "clarification":
        if _contains_any(name_sample, _ADDENDUM_TERMS):
            return "customer_addendum", "binding"
        return "customer_clarification", "authoritative"
    if role == "supporting":
        return "supporting_reference", "contextual"
    if _contains_any(name_sample, _ADDENDUM_TERMS):
        return "customer_addendum", "binding"
    if _contains_any(name_sample, _CLARIFICATION_TERMS):
        return "customer_clarification", "authoritative"
    if Path(name).suffix.lower() == ".xlsx":
        header_sample = text[:2500].lower()
        if "vendor question" in header_sample and (
            "customer response" in header_sample or "customer answer" in header_sample
        ):
            return "customer_clarification", "authoritative"
    if _contains_any(name_sample, _ANNEXURE_TERMS):
        return "annexure", "authoritative"
    if _contains_any(name_sample, _COMMERCIAL_TERMS):
        return "commercial", "contextual"
    return "base_rfp", "authoritative"


def parse_source_document(
    name: str,
    data: bytes,
    *,
    role: UploadRole = "primary",
) -> tuple[SourceDocument, List[ClarificationRecord]]:
    suffix = Path(name).suffix.lower()
    document_id = _document_id(name, data)
    metadata: dict = {"extension": suffix, "upload_role": role}
    warnings: List[str] = []
    clarifications: List[ClarificationRecord] = []

    if suffix == ".pdf":
        parsed = parse_pdf(data)
        text = parsed.text
        metadata["pages"] = parsed.page_count
        warnings.extend(parsed.warnings)
        locator_format = "page"
    elif suffix == ".docx":
        parsed = parse_docx(data)
        text = parsed.text
        metadata["paragraphs"] = parsed.paragraph_count
        metadata["tables"] = parsed.table_count
        locator_format = "paragraph/table row"
    elif suffix == ".xlsx":
        parsed = parse_xlsx(data, document_id=document_id)
        text = parsed.text
        metadata["sheets"] = parsed.sheet_count
        metadata["rows"] = parsed.row_count
        warnings.extend(parsed.warnings)
        clarifications = parsed.clarifications
        locator_format = "sheet/row"
    else:
        raise ValueError(f"Unsupported RFP source type: {suffix or 'no extension'}")

    document_type, authority = classify_source(name, text, role)
    if role == "primary" and clarifications and document_type == "base_rfp":
        document_type = "customer_clarification"
        authority = "authoritative"
    for clarification in clarifications:
        if not clarification.customer_response.strip():
            clarification.authority = "non_authoritative"
            clarification.status = "unresolved"
        elif document_type in {"customer_addendum", "customer_clarification"}:
            clarification.authority = authority
        else:
            clarification.authority = "non_authoritative"
    document = SourceDocument(
        document_id=document_id,
        name=name,
        document_type=document_type,
        authority=authority,
        issue_date=_infer_issue_date(name, text),
        text=text,
        locator_format=locator_format,
        character_count=len(text),
        metadata=metadata,
        warnings=warnings,
    )
    return document, clarifications


def sort_sources_by_precedence(documents: Iterable[SourceDocument]) -> List[SourceDocument]:
    return sorted(
        documents,
        key=lambda document: (
            -_PRECEDENCE.get(document.document_type, 0),
            -(int(document.issue_date.replace("-", "")) if document.issue_date else 0),
            document.name.lower(),
        ),
    )


def build_source_reconciliation(
    documents: Sequence[SourceDocument],
    clarifications: Sequence[ClarificationRecord],
) -> SourceReconciliation:
    unresolved = [
        clarification.source_ref
        for clarification in clarifications
        if not clarification.customer_response.strip()
    ]
    present_types = {document.document_type for document in documents}
    precedence = [
        "A later customer-issued addendum or amendment overrides conflicting earlier material.",
        "An explicit customer clarification response governs the related ambiguity in the base RFP.",
        "The base RFP and its requirement-bearing annexures remain authoritative where not amended.",
        "A vendor question provides context only; it is not a requirement without a customer response.",
        "Supporting and commercial documents do not create solution scope unless an authoritative source incorporates them.",
    ]
    if not present_types.intersection({"customer_addendum", "customer_clarification"}):
        precedence.append("No customer clarification or addendum source was supplied for this run.")
    return SourceReconciliation(
        precedence_summary=precedence,
        clarifications=list(clarifications),
        unresolved_questions=unresolved,
    )


def render_reconciliation_summary(
    reconciliation: SourceReconciliation,
    max_records: int = 100,
) -> str:
    """Render compact control metadata without duplicating full Q&A responses."""
    answered = [item for item in reconciliation.clarifications if item.customer_response.strip()]
    unresolved = [item for item in reconciliation.clarifications if not item.customer_response.strip()]
    lines = ["SOURCE PRECEDENCE:"]
    lines.extend(f"- {rule}" for rule in reconciliation.precedence_summary)
    lines.append(f"ANSWERED Q&A RECORDS: {len(answered)}")
    lines.extend(
        f"- {item.clarification_id} | {item.source_ref} | authority={item.authority}"
        for item in answered[:max_records]
    )
    if len(answered) > max_records:
        lines.append(
            f"- ... {len(answered) - max_records} additional answered records are in the evidence package"
        )
    lines.append(f"UNANSWERED VENDOR QUESTIONS: {len(unresolved)}")
    lines.extend(
        f"- {item.clarification_id} | {item.source_ref} | question={item.question[:240]}"
        for item in unresolved[:max_records]
    )
    if len(unresolved) > max_records:
        lines.append(
            f"- ... {len(unresolved) - max_records} additional unresolved records are in the evidence package"
        )
    return "\n".join(lines)


def render_source_package(documents: Sequence[SourceDocument]) -> str:
    parts = [
        "RFP PACKAGE SOURCE RULES:",
        "- Treat document boundaries and source locators as evidence metadata.",
        "- Customer answers are authoritative; vendor questions alone are not requirements.",
        "- Preserve source references for every effective requirement.",
    ]
    for document in sort_sources_by_precedence(documents):
        parts.extend(
            [
                "",
                (
                    f"=== SOURCE DOCUMENT {document.document_id} | name={document.name} | "
                    f"type={document.document_type} | authority={document.authority} | "
                    f"issue_date={document.issue_date or 'unknown'} | "
                    f"locator={document.locator_format} ==="
                ),
                document.text,
                f"=== END SOURCE DOCUMENT {document.document_id} ===",
            ]
        )
    return "\n".join(parts).strip()
