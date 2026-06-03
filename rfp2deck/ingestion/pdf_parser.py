from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Union

import fitz  # PyMuPDF

from rfp2deck.core.logging import get_logger

log = get_logger(__name__)


@dataclass
class ParsedDoc:
    text: str
    page_count: int


def parse_pdf(path_or_bytes: Union[Path, bytes]) -> ParsedDoc:
    source = "bytes" if isinstance(path_or_bytes, bytes) else str(path_or_bytes)
    try:
        if isinstance(path_or_bytes, bytes):
            doc = fitz.open(stream=path_or_bytes, filetype="pdf")
        else:
            doc = fitz.open(path_or_bytes)
    except Exception:
        log.exception("Failed to open PDF (source=%s)", source)
        raise
    texts = []
    for i in range(doc.page_count):
        page = doc.load_page(i)
        texts.append(f"\n\n--- PAGE {i+1} ---\n\n" + page.get_text("text"))
    parsed = ParsedDoc(text="\n".join(texts).strip(), page_count=doc.page_count)
    log.info("Parsed PDF (source=%s): %d pages, %d chars", source, parsed.page_count, len(parsed.text))
    return parsed
