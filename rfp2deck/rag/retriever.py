from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import faiss
import numpy as np

from rfp2deck.core.logging import get_logger
from rfp2deck.rag.embeddings import embed_texts
from rfp2deck.rag.indexer import RAGIndex

log = get_logger(__name__)


@dataclass
class RetrievedChunk:
    score: float
    text: str


def retrieve(rag: RAGIndex, query: str, k: int = 6) -> List[RetrievedChunk]:
    log.info("Retrieving top-%d chunks for query (%d chars)", k, len(query or ""))
    qv = embed_texts([query])
    faiss.normalize_L2(qv)
    scores, ids = rag.index.search(qv, k)
    out: List[RetrievedChunk] = []
    for s, i in zip(scores[0].tolist(), ids[0].tolist()):
        if i == -1:
            continue
        out.append(RetrievedChunk(score=float(s), text=rag.chunks[i]))
    log.info("Retrieved %d chunk(s)", len(out))
    return out
