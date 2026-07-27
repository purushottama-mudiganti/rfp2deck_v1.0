from __future__ import annotations

import re
from dataclasses import dataclass


HIGH_SIGNAL_TERMS = (
    "scope",
    "scope of work",
    "requirements",
    "functional",
    "non-functional",
    "technical",
    "solution",
    "architecture",
    "integration",
    "interface",
    "data",
    "analytics",
    "reporting",
    "dashboard",
    "deliverables",
    "service level",
    "sla",
    "migration",
    "implementation",
    "annex",
    "appendix",
)

LOW_SIGNAL_TERMS = (
    "instructions for vendors",
    "tender procedures",
    "submission",
    "proposal submission",
    "ariba",
    "pricing table",
    "standard contract",
    "confidentiality",
    "non-disclosure",
    "nda",
    "schedule of events",
    "contact person",
    "envelope",
)


@dataclass(frozen=True)
class FocusSection:
    heading: str
    start: int
    end: int
    score: int


def _clean_line(line: str) -> str:
    return re.sub(r"\s+", " ", (line or "").strip())


def _is_heading(line: str) -> bool:
    line = _clean_line(line)
    if not line or len(line) > 140:
        return False
    if re.match(r"^(section|annex|appendix)\s+[a-z0-9]+[\s:.-]", line, re.I):
        return True
    if re.match(r"^\d+(\.\d+)*\s+[A-Z]", line):
        return True
    letters = [c for c in line if c.isalpha()]
    if len(letters) < 4:
        return False
    upper_ratio = sum(1 for c in letters if c.isupper()) / max(1, len(letters))
    return upper_ratio > 0.70


def _score_heading(heading: str) -> int:
    low = heading.lower()
    score = 0
    for term in HIGH_SIGNAL_TERMS:
        if term in low:
            score += 2
    for term in LOW_SIGNAL_TERMS:
        if term in low:
            score -= 3
    return score


def _find_sections(text: str) -> list[FocusSection]:
    line_matches = []
    pos = 0
    for raw_line in text.splitlines(keepends=True):
        line = _clean_line(raw_line)
        if _is_heading(line):
            line_matches.append((line, pos, _score_heading(line)))
        pos += len(raw_line)

    sections: list[FocusSection] = []
    for idx, (heading, start, score) in enumerate(line_matches):
        end = line_matches[idx + 1][1] if idx + 1 < len(line_matches) else len(text)
        if end <= start:
            continue
        sections.append(FocusSection(heading=heading, start=start, end=end, score=score))
    return sections


def build_rfp_focus_guide(rfp_text: str, *, max_chars: int = 9000) -> str:
    """Build a compact, generic guide that tells the LLM where requirements live.

    This is intentionally heuristic. It does not replace the full RFP text; it
    biases the first understanding pass toward requirement-bearing sections and
    away from procurement/admin instructions that often mention tools such as
    Ariba only as submission channels.
    """
    text = rfp_text or ""
    if not text.strip():
        return ""

    sections = _find_sections(text)
    high = [s for s in sections if s.score > 0]
    low = [s for s in sections if s.score < 0]
    high = sorted(high, key=lambda s: (-s.score, s.start))[:8]
    low = sorted(low, key=lambda s: (s.start, s.heading.lower()))[:12]

    parts: list[str] = []
    if high:
        parts.append("LIKELY REQUIREMENT-BEARING SECTIONS (prioritize these):")
        budget_each = max(650, max_chars // max(1, len(high)))
        used = 0
        for sec in high:
            excerpt = _clean_line(text[sec.start : min(sec.end, sec.start + budget_each)])
            if not excerpt:
                continue
            block = f"- {sec.heading}\n  Excerpt: {excerpt[:budget_each]}"
            if used + len(block) > max_chars:
                break
            parts.append(block)
            used += len(block)

    if low:
        parts.append("LIKELY PROCUREMENT / ADMIN / SUBMISSION SECTIONS (do not treat as solution scope unless explicitly repeated in requirement sections):")
        for sec in low:
            parts.append(f"- {sec.heading}")

    if not parts:
        return "No reliable section guide was detected; use the full RFP text as the source of truth."
    return "\n".join(parts)
