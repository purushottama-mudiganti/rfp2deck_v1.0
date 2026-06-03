from __future__ import annotations

from functools import lru_cache
from openai import OpenAI

from rfp2deck.core.config import settings
from rfp2deck.core.logging import get_logger

log = get_logger(__name__)


@lru_cache(maxsize=8)
def _cached_client(timeout: float, max_retries: int) -> OpenAI:
    log.debug(
        "Creating OpenAI client (timeout=%.0fs, max_retries=%d, api_key_set=%s)",
        timeout,
        max_retries,
        bool(settings.openai_api_key),
    )
    if not settings.openai_api_key:
        log.warning("OPENAI_API_KEY is not set; OpenAI requests will fail to authenticate.")
    return OpenAI(
        api_key=settings.openai_api_key,
        timeout=timeout,
        max_retries=max_retries,
    )


def get_client(timeout: float = 120.0, max_retries: int = 2) -> OpenAI:
    return _cached_client(timeout, max_retries)
