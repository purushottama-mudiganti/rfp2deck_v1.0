from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import List, Tuple

import faiss
import numpy as np

from rfp2deck.core.logging import get_logger
from rfp2deck.rag.embeddings import embed_texts

log = get_logger(__name__)


@dataclass
class RAGIndex:
    index: faiss.IndexFlatIP
    chunks: List[str]


def chunk_text(text: str, max_chars: int = 1800, overlap: int = 200) -> List[str]:
    chunks = []
    i = 0
    while i < len(text):
        j = min(len(text), i + max_chars)
        chunks.append(text[i:j])
        i = j - overlap
        if i < 0:
            i = 0
        if j == len(text):
            break
    return [c.strip() for c in chunks if c.strip()]


def build_faiss_index(texts: List[str]) -> RAGIndex:
    log.info("Building FAISS index from %d chunk(s)", len(texts))
    vecs = embed_texts(texts)
    faiss.normalize_L2(vecs)
    dim = vecs.shape[1]
    idx = faiss.IndexFlatIP(dim)
    idx.add(vecs)
    log.info("FAISS index built (dim=%d, vectors=%d)", dim, idx.ntotal)
    return RAGIndex(index=idx, chunks=texts)


def save_index(rag: RAGIndex, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    faiss.write_index(rag.index, str(out_dir / "index.faiss"))
    (out_dir / "chunks.json").write_text(
        json.dumps(rag.chunks, ensure_ascii=False, indent=2), encoding="utf-8"
    )


def load_index(in_dir: Path) -> RAGIndex:
    import faiss

    idx = faiss.read_index(str(in_dir / "index.faiss"))
    chunks = json.loads((in_dir / "chunks.json").read_text(encoding="utf-8"))
    return RAGIndex(index=idx, chunks=chunks)
