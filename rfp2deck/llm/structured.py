from __future__ import annotations

from copy import deepcopy
from typing import Any, Dict, Type, TypeVar

from pydantic import BaseModel

from rfp2deck.core.config import settings
from rfp2deck.llm.openai_client import get_client

T = TypeVar("T", bound=BaseModel)


def _resolve_json_pointer(ref: str, defs: Dict[str, Any]) -> Any:
    if not ref.startswith("#/"):
        raise ValueError(f"Unsupported $ref format: {ref}")
    parts = ref.lstrip("#/").split("/")
    if len(parts) != 2:
        raise ValueError(f"Unsupported $ref pointer depth: {ref}")
    root, name = parts
    if root in ("$defs", "definitions"):
        if name not in defs:
            raise KeyError(f"$ref target not found: {ref}")
        return defs[name]
    raise ValueError(f"Unsupported $ref root: {root} in {ref}")


def _dereference(schema: Any, defs: Dict[str, Any], seen: set[str] | None = None) -> Any:
    if seen is None:
        seen = set()
    if isinstance(schema, list):
        return [_dereference(x, defs, seen) for x in schema]
    if not isinstance(schema, dict):
        return schema
    if "$ref" in schema:
        ref = schema["$ref"]
        if ref in seen:
            return schema
        seen.add(ref)
        target = deepcopy(_resolve_json_pointer(ref, defs))
        for k, v in schema.items():
            if k == "$ref":
                continue
            target[k] = v
        return _dereference(target, defs, seen)

    out: Dict[str, Any] = {}
    for k, v in schema.items():
        if k in ("$defs", "definitions"):
            out[k] = v
        else:
            out[k] = _dereference(v, defs, seen)
    return out


def _make_strict(schema: Any) -> Any:
    if isinstance(schema, list):
        return [_make_strict(x) for x in schema]
    if not isinstance(schema, dict):
        return schema

    for k in ("properties", "items", "anyOf", "oneOf", "allOf", "not"):
        if k in schema:
            schema[k] = _make_strict(schema[k])

    if schema.get("type") == "object":
        props = schema.get("properties") or {}
        schema["required"] = sorted(list(props.keys()))
        schema["additionalProperties"] = False
        for pk, pv in list(props.items()):
            props[pk] = _make_strict(pv)
        schema["properties"] = props
    return schema


def response_as_schema(
    prompt: str,
    schema: Type[T],
    model: str | None = None,
    reasoning_effort: str = "medium",
    timeout_seconds: float = 120.0,
) -> T:
    """Call OpenAI Responses API using STRICT JSON Schema structured output."""
    client = get_client(timeout=timeout_seconds, max_retries=2)
    model = model or settings.model_reasoning

    raw_schema: Dict[str, Any] = schema.model_json_schema()
    defs: Dict[str, Any] = {}
    if isinstance(raw_schema.get("$defs"), dict):
        defs.update(raw_schema["$defs"])
    if isinstance(raw_schema.get("definitions"), dict):
        defs.update(raw_schema["definitions"])

    inlined = _dereference(raw_schema, defs)
    if isinstance(inlined, dict):
        inlined.pop("$defs", None)
        inlined.pop("definitions", None)

    strict_schema = _make_strict(inlined)

    resp = client.responses.create(
        model=model,
        input=[{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        text={
            "format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": strict_schema,
                "strict": True,
            }
        },
        reasoning={"effort": reasoning_effort} if reasoning_effort else None,
    )

    return schema.model_validate_json(resp.output_text)
