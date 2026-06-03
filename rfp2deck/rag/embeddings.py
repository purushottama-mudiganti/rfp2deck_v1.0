from __future__ import annotations

import time
from typing import List

import numpy as np
from openai import OpenAIError

from rfp2deck.core.config import settings
from rfp2deck.core.logging import get_logger
from rfp2deck.llm.openai_client import get_client

log = get_logger(__name__)


def embed_texts(texts: List[str], timeout_seconds: float = 120.0,) -> np.ndarray:
    client = get_client(timeout=timeout_seconds, max_retries=2)
    log.info("Embedding %d text(s) with model=%s", len(texts), settings.embeddings_model)
    start = time.perf_counter()
    try:
        resp = client.embeddings.create(model=settings.embeddings_model, input=texts)
    except OpenAIError:
        log.exception(
            "Embedding call FAILED after %.1fs (model=%s, count=%d)",
            time.perf_counter() - start,
            settings.embeddings_model,
            len(texts),
        )
        raise
    vectors = [d.embedding for d in resp.data]
    arr = np.array(vectors, dtype="float32")
    log.info("Embedded %d text(s) in %.1fs (dim=%s)", len(texts), time.perf_counter() - start,
             arr.shape[1] if arr.ndim == 2 else "?")
    return arr
