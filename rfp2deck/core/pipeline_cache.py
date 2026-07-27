from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

from pydantic import BaseModel

from rfp2deck.core.config import settings


def _json_bytes(value: Any) -> bytes:
    return json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")


def sha256_text(value: str) -> str:
    return hashlib.sha256((value or "").encode("utf-8")).hexdigest()


def sha256_bytes(value: bytes) -> str:
    return hashlib.sha256(value or b"").hexdigest()


def hash_files(paths: Iterable[Path]) -> str:
    digest = hashlib.sha256()
    for path in sorted(paths, key=lambda p: str(p)):
        digest.update(str(path.name).encode("utf-8"))
        digest.update(b"\0")
        try:
            digest.update(path.read_bytes())
        except OSError:
            digest.update(b"<missing>")
    return digest.hexdigest()


def stable_hash(payload: Any) -> str:
    return hashlib.sha256(_json_bytes(payload)).hexdigest()


def cache_root() -> Path:
    root = settings.data_dir / "pipeline_cache"
    root.mkdir(parents=True, exist_ok=True)
    return root


def build_plan_cache_key(payload: dict[str, Any]) -> str:
    return stable_hash(
        {
            "kind": "deck-plan",
            "version": settings.pipeline_cache_version,
            **payload,
        }
    )


def build_diagram_cache_key(
    *,
    prompt: str,
    model: str,
    size: str,
    quality: str,
) -> str:
    return stable_hash(
        {
            "kind": "diagram-image",
            "version": settings.pipeline_cache_version,
            "prompt_hash": sha256_text(prompt),
            "model": model,
            "size": size,
            "quality": quality,
        }
    )


def build_diagram_prompt_cache_key(*, prompt: str) -> str:
    """Stable prompt-level alias for the latest generated diagram image."""
    return stable_hash(
        {
            "kind": "diagram-image-prompt-alias",
            "version": settings.pipeline_cache_version,
            "prompt_hash": sha256_text(prompt),
        }
    )


def read_model_cache(key: str, schema: type[BaseModel]) -> BaseModel | None:
    path = cache_root() / f"{key}.json"
    if not path.exists():
        return None
    return schema.model_validate_json(path.read_text(encoding="utf-8"))


def write_model_cache(key: str, value: BaseModel) -> Path:
    path = cache_root() / f"{key}.json"
    path.write_text(value.model_dump_json(), encoding="utf-8")
    return path


def read_json_cache(key: str) -> dict[str, Any] | None:
    path = cache_root() / f"{key}.json"
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def write_json_cache(key: str, value: dict[str, Any]) -> Path:
    path = cache_root() / f"{key}.json"
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    return path


def read_bytes_cache(key: str, suffix: str = ".bin") -> bytes | None:
    path = cache_root() / f"{key}{suffix}"
    if not path.exists():
        return None
    return path.read_bytes()


def write_bytes_cache(key: str, value: bytes, suffix: str = ".bin") -> Path:
    path = cache_root() / f"{key}{suffix}"
    path.write_bytes(value)
    return path
