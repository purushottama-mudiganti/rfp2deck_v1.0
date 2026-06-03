from __future__ import annotations

from typing import List

import numpy as np

from rfp2deck.core.config import settings
from rfp2deck.llm.openai_client import get_client


def embed_texts(texts: List[str], timeout_seconds: float = 120.0,) -> np.ndarray:
    client = get_client(timeout=timeout_seconds, max_retries=2)
    resp = client.embeddings.create(model=settings.embeddings_model, input=texts)
    vectors = [d.embedding for d in resp.data]
    return np.array(vectors, dtype="float32")
