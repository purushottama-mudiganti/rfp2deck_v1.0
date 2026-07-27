from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

from openai import APIConnectionError, APITimeoutError, OpenAIError

from rfp2deck.core.config import settings
from rfp2deck.core.logging import get_logger
from rfp2deck.llm.openai_client import get_client

log = get_logger(__name__)


_RETRYABLE_IMAGE_STATUS_CODES = {408, 409, 429, 500, 502, 503, 504, 520, 521, 522, 523, 524}


def _image_retry_delay(exc: OpenAIError, attempt: int) -> float | None:
    """Return retry delay for transient image API failures, else None."""
    status_code = getattr(exc, "status_code", None)
    body = getattr(exc, "body", None)
    retryable = isinstance(exc, (APITimeoutError, APIConnectionError))
    if isinstance(body, dict):
        retryable = retryable or body.get("retryable") is True
    retryable = retryable or status_code in _RETRYABLE_IMAGE_STATUS_CODES
    if not retryable:
        return None

    retry_after = None
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", None)
    if headers:
        retry_after = headers.get("retry-after") or headers.get("Retry-After")
    if retry_after is None and isinstance(body, dict):
        retry_after = body.get("retry_after")
    try:
        if retry_after is not None:
            return max(1.0, min(float(retry_after), 300.0))
    except (TypeError, ValueError):
        pass

    base_wait = float(getattr(settings, "openai_retry_base_wait_s", 5.0))
    max_wait = float(getattr(settings, "openai_retry_max_wait_s", 90.0))
    return min(base_wait * (2 ** max(0, attempt - 1)), max_wait)


def generate_diagram_png(
    prompt: str,
    out_path: Optional[Path],
    model: str | None = None,
    size: str = "auto",
    quality: str = "auto",
    timeout_seconds: float | None = None,
) -> bytes:
    model = model or settings.image_model
    timeout_seconds = float(timeout_seconds or settings.image_timeout_s)
    attempts = max(1, int(settings.image_retry_attempts))
    client = get_client(timeout=timeout_seconds, max_retries=0)
    log.info(
        "Generating diagram: model=%s size=%s quality=%s prompt_chars=%d timeout=%.0fs attempts=%d",
        model,
        size,
        quality,
        len(prompt or ""),
        timeout_seconds,
        attempts,
    )
    start = time.perf_counter()
    last_exc: OpenAIError | None = None
    for attempt in range(1, attempts + 1):
        try:
            if attempts > 1:
                log.info("Diagram generation attempt %d/%d", attempt, attempts)
            resp = client.images.generate(
                model=model,
                prompt=prompt,
                size=size,
                quality=quality,
            )
            break
        except OpenAIError as exc:
            last_exc = exc
            elapsed = time.perf_counter() - start
            delay = _image_retry_delay(exc, attempt)
            if delay is None or attempt >= attempts:
                log.exception(
                    "Diagram generation FAILED after %.1fs on attempt %d/%d (model=%s, status=%s)",
                    elapsed,
                    attempt,
                    attempts,
                    model,
                    getattr(exc, "status_code", None),
                )
                raise
            log.warning(
                "Retryable diagram error on attempt %d/%d after %.1fs "
                "(model=%s, status=%s); retrying in %.0fs",
                attempt,
                attempts,
                elapsed,
                model,
                getattr(exc, "status_code", None),
                delay,
            )
            time.sleep(delay)
    else:
        raise RuntimeError("Diagram generation failed without returning a response") from last_exc

    b64 = resp.data[0].b64_json
    png = base64.b64decode(b64)
    log.info(
        "Diagram generated in %.1fs (%d bytes)", time.perf_counter() - start, len(png)
    )
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
        log.debug("Diagram written to %s", out_path)
    return png
