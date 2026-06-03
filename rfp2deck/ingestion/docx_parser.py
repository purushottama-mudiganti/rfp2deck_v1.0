from __future__ import annotations

from dataclasses import dataclass
from io import BytesIO
from pathlib import Path
from typing import Union

from docx import Document

from rfp2deck.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ParsedDoc:
    text: str
    paragraph_count: int


def parse_docx(path_or_bytes: Union[Path, bytes]) -> ParsedDoc:
    source = "bytes" if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    try:
        if isinstance(path_or_bytes, bytes):
            d = Document(BytesIO(path_or_bytes))
        else:
            d = Document(path_or_bytes)
    except Exception:
        log.exception("Failed to open DOCX (source=%s)", source)
        raise
    paras = [p.text.strip() for p in d.paragraphs if p.text and p.text.strip()]
    parsed = ParsedDoc(text="\n".join(paras), paragraph_count=len(paras))
    log.info("Parsed DOCX (source=%s): %d paragraphs, %d chars", source, parsed.paragraph_count, len(parsed.text))
    return parsed
