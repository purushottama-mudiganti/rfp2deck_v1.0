from __future__ import annotations

from functools import lru_cache
from openai import OpenAI

from rfp2deck.core.config import settings


@lru_cache(maxsize=8)
def _cached_client(timeout: float, max_retries: int) -> OpenAI:
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_client(timeout: float = 120.0, max_retries: int = 2) -> OpenAI:
    return _cached_client(timeout, max_retries)