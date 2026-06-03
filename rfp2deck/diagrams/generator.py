from __future__ import annotations

import base64
import time
from pathlib import Path
from typing import Optional

from openai import OpenAIError

from rfp2deck.core.logging import get_logger
from rfp2deck.llm.openai_client import get_client

log = get_logger(__name__)


def generate_diagram_png(
    prompt: str,
    out_path: Optional[Path],
    model: str = "gpt-image-1",
    size: str = "auto",
    quality: str = "auto",
    timeout_seconds: float = 120.0,
) -> bytes:
    client = get_client(timeout=timeout_seconds, max_retries=2)
    log.info(
        "Generating diagram: model=%s size=%s quality=%s prompt_chars=%d",
        model,
        size,
        quality,
        len(prompt or ""),
    )
    start = time.perf_counter()
    try:
        resp = client.images.generate(
            model=model,
            prompt=prompt,
            size=size,
            quality=quality,
        )
    except OpenAIError:
        elapsed = time.perf_counter() - start
        log.exception("Diagram generation FAILED after %.1fs (model=%s)", elapsed, model)
        raise

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
