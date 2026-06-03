from __future__ import annotations

import base64
from pathlib import Path
from typing import Optional

from rfp2deck.llm.openai_client import get_client


def generate_diagram_png(
    prompt: str,
    out_path: Optional[Path],
    model: str = "gpt-image-1",
    size: str = "auto",
    quality: str = "auto",
    timeout_seconds: float = 120.0,
) -> bytes:
    client = get_client(timeout=timeout_seconds, max_retries=2)
    resp = client.images.generate(
        model=model,
        prompt=prompt,
        size=size,
        quality=quality,
    )
    b64 = resp.data[0].b64_json
    png = base64.b64decode(b64)
    if out_path is not None:
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_bytes(png)
    return png
