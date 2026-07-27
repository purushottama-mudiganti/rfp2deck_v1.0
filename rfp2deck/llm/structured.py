from __future__ import annotations

import time
from copy import deepcopy
from typing import Any, Dict, Type, TypeVar

from openai import APIConnectionError, APITimeoutError, OpenAIError
from pydantic import BaseModel, ValidationError

from rfp2deck.core.config import settings
from rfp2deck.core.logging import get_logger
from rfp2deck.llm.openai_client import get_client

log = get_logger(__name__)

T = TypeVar("T", bound=BaseModel)


class StructuredLLMError(RuntimeError):
    """Raised when a streamed structured-output call ends without completion."""


def _field(obj: Any, name: str, default: Any = None) -> Any:
    if obj is None:
        return default
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _error_body(exc: BaseException) -> dict[str, Any]:
    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        return body

    response = getattr(exc, "response", None)
    if response is None:
        return {}

    try:
        parsed = response.json()
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _error_text(exc: BaseException, max_chars: int = 20000) -> str:
    response = getattr(exc, "response", None)
    if response is not None:
        try:
            text = response.text
            if text:
                return text[:max_chars]
        except Exception:
            pass
    return str(exc)[:max_chars]


def _is_proxy_block_error(exc: BaseException) -> bool:
    text = _error_text(exc).lower()
    openai_host_markers = (
        "api.openai.com",
        "api.openai.com&#x2f;",
        "api.openai.com/v1/responses",
    )
    proxy_markers = (
        "<html",
        "network error",
        "threatpulse",
        "bluecoat",
        "symantec",
        "access restricted website",
        "generative ai",
    )
    return any(marker in text for marker in proxy_markers) and any(
        marker in text for marker in openai_host_markers
    )


def _format_proxy_block_error(
    exc: BaseException,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
) -> str:
    status_code = getattr(exc, "status_code", None)
    return (
        "OpenAI API request was blocked or intercepted by the corporate network proxy "
        f"(schema={schema_name}, model={model}, effort={reasoning_effort}, "
        f"status={status_code or 'unknown'}). The proxy returned an HTML block page for "
        "https://api.openai.com/v1/responses, so this is not a model/schema failure. "
        "Request allow-list/access for api.openai.com under the corporate Generative AI "
        "policy, connect through an approved network/VPN, or configure the approved proxy "
        "for the Python runtime."
    )


def _format_connection_error(
    exc: BaseException,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
) -> str:
    cause = exc.__cause__
    cause_text = f" Underlying error: {cause!r}." if cause else ""
    return (
        "OpenAI API connection failed before a response was received "
        f"(schema={schema_name}, model={model}, effort={reasoning_effort}). "
        "This is a network/proxy/TLS/DNS connectivity issue in the Python runtime, "
        "not a model or structured-output schema failure. If the traceback path starts "
        "with /mnt/c or uses .venv/lib, the app is running under WSL; run it with "
        ".venv-win\\Scripts\\python.exe/.venv-win\\Scripts\\streamlit.exe for native "
        "Windows, or configure WSL proxy/CA/DNS access to api.openai.com."
        f"{cause_text}"
    )


def _retry_after_seconds(exc: BaseException, fallback: float) -> float:
    response = getattr(exc, "response", None)
    if response is not None:
        retry_after = response.headers.get("retry-after")
        if retry_after:
            try:
                return float(retry_after)
            except ValueError:
                pass

    body = _error_body(exc)
    retry_after = body.get("retry_after")
    if retry_after is not None:
        try:
            return float(retry_after)
        except (TypeError, ValueError):
            pass

    return fallback


def _is_retryable_openai_error(exc: BaseException) -> bool:
    body = _error_body(exc)
    if body.get("retryable") is True:
        return True
    if body.get("cloudflare_error") is True:
        return True

    status_code = getattr(exc, "status_code", None)
    return status_code in {408, 409, 429} or (
        isinstance(status_code, int) and status_code >= 500
    )


def _responses_create_with_backoff(
    client: Any,
    request_kwargs: Dict[str, Any],
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
) -> Any:
    max_attempts = max(1, settings.openai_retry_attempts)
    for attempt in range(1, max_attempts + 1):
        try:
            return client.responses.create(**request_kwargs)
        except APITimeoutError:
            raise
        except APIConnectionError as exc:
            raise StructuredLLMError(
                _format_connection_error(
                    exc,
                    schema_name=schema_name,
                    model=model,
                    reasoning_effort=reasoning_effort,
                )
            ) from exc
        except OpenAIError as exc:
            if _is_proxy_block_error(exc):
                raise StructuredLLMError(
                    _format_proxy_block_error(
                        exc,
                        schema_name=schema_name,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                ) from exc
            if attempt >= max_attempts or not _is_retryable_openai_error(exc):
                raise

            fallback = min(
                settings.openai_retry_base_wait_s * (2 ** (attempt - 1)),
                settings.openai_retry_max_wait_s,
            )
            delay = min(
                _retry_after_seconds(exc, fallback),
                settings.openai_retry_max_wait_s,
            )
            status_code = getattr(exc, "status_code", None)
            body = _error_body(exc)
            log.warning(
                "OpenAI response create failed with retryable error; waiting %.1fs "
                "before retry %d/%d (schema=%s, model=%s, effort=%s, status=%s, code=%s)",
                delay,
                attempt + 1,
                max_attempts,
                schema_name,
                model,
                reasoning_effort,
                status_code,
                body.get("error_code") or body.get("code"),
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")


def _raise_if_response_not_completed(
    resp: Any,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
    event_types: list[str],
) -> None:
    status = _field(resp, "status")
    if status and status != "completed":
        raise StructuredLLMError(
            _format_response_failure(
                {"type": f"response.{status}", "response": resp},
                schema_name=schema_name,
                model=model,
                reasoning_effort=reasoning_effort,
                event_types=event_types,
            )
        )


def _format_response_failure(
    event: Any,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
    event_types: list[str],
) -> str:
    event_type = _field(event, "type", "missing")
    response = _field(event, "response")
    status = _field(response, "status")
    response_id = _field(response, "id")
    error = _field(response, "error")
    incomplete = _field(response, "incomplete_details")

    details: list[str] = []
    if response_id:
        details.append(f"response_id={response_id}")
    if status:
        details.append(f"status={status}")
    if error:
        code = _field(error, "code")
        message = _field(error, "message")
        if code:
            details.append(f"error_code={code}")
        if message:
            details.append(f"error_message={message}")
    if incomplete:
        reason = _field(incomplete, "reason")
        if reason:
            details.append(f"incomplete_reason={reason}")
    if event_type == "error":
        code = _field(event, "code")
        message = _field(event, "message")
        param = _field(event, "param")
        if code:
            details.append(f"error_code={code}")
        if message:
            details.append(f"error_message={message}")
        if param:
            details.append(f"param={param}")

    if not details:
        tail = ", ".join(event_types[-8:]) or "none"
        details.append(f"last_events={tail}")

    return (
        "OpenAI structured response did not complete "
        f"(schema={schema_name}, model={model}, effort={reasoning_effort}, "
        f"terminal_event={event_type}; {', '.join(details)})"
    )


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
        next_seen = set(seen)
        next_seen.add(ref)
        target = deepcopy(_resolve_json_pointer(ref, defs))
        for k, v in schema.items():
            if k == "$ref":
                continue
            target[k] = v
        return _dereference(target, defs, next_seen)

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
    timeout_seconds: float = 600.0,
) -> T:
    """Call OpenAI Responses API using STRICT JSON Schema structured output.

    The project owns retry behavior so attempt counts and backoff are explicit.
    Streaming remains optional because some large structured responses are more
    reliable through the non-streaming endpoint.
    """
    # Do not combine SDK retries with the explicit retry loop below. Previously,
    # three configured attempts could expand to as many as nine HTTP attempts.
    client = get_client(timeout=timeout_seconds, max_retries=0)
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

    log.info(
        "LLM structured call: schema=%s model=%s effort=%s prompt_chars=%d timeout=%.0fs",
        schema.__name__,
        model,
        reasoning_effort,
        len(prompt),
        timeout_seconds,
    )
    request_kwargs: Dict[str, Any] = {
        "model": model,
        "input": [{"role": "user", "content": [{"type": "input_text", "text": prompt}]}],
        "text": {
            "format": {
                "type": "json_schema",
                "name": schema.__name__,
                "schema": strict_schema,
                "strict": True,
            }
        },
    }
    if reasoning_effort:
        request_kwargs["reasoning"] = {"effort": reasoning_effort}

    start = time.perf_counter()
    resp: Any = None
    try:
        if not settings.openai_structured_streaming:
            log.info(
                "LLM structured call using non-streaming Responses API "
                "(schema=%s, model=%s, effort=%s)",
                schema.__name__,
                model,
                reasoning_effort,
            )
            resp = _responses_create_with_backoff(
                client,
                request_kwargs,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            _raise_if_response_not_completed(
                resp,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
                event_types=[],
            )
        else:
            with client.responses.stream(**request_kwargs) as stream:
                terminal_event: Any = None
                completed = False
                event_types: list[str] = []

                for event in stream:
                    event_type = _field(event, "type", "unknown")
                    event_types.append(event_type)
                    if len(event_types) > 20:
                        event_types = event_types[-20:]

                    if event_type == "response.completed":
                        terminal_event = event
                        completed = True
                    elif event_type in ("response.failed", "response.incomplete", "error"):
                        terminal_event = event
                        raise StructuredLLMError(
                            _format_response_failure(
                                terminal_event,
                                schema_name=schema.__name__,
                                model=model,
                                reasoning_effort=reasoning_effort,
                                event_types=event_types,
                            )
                        )

                if not completed:
                    if terminal_event is None:
                        log.warning(
                            "LLM structured stream ended without a terminal event for schema=%s "
                            "(model=%s, effort=%s, last_events=%s). Retrying once without streaming.",
                            schema.__name__,
                            model,
                            reasoning_effort,
                            ", ".join(event_types[-8:]) or "none",
                        )
                        resp = _responses_create_with_backoff(
                            client,
                            request_kwargs,
                            schema_name=schema.__name__,
                            model=model,
                            reasoning_effort=reasoning_effort,
                        )
                        _raise_if_response_not_completed(
                            resp,
                            schema_name=schema.__name__,
                            model=model,
                            reasoning_effort=reasoning_effort,
                            event_types=event_types,
                        )
                        completed = True
                    else:
                        raise StructuredLLMError(
                            _format_response_failure(
                                terminal_event,
                                schema_name=schema.__name__,
                                model=model,
                                reasoning_effort=reasoning_effort,
                                event_types=event_types,
                            )
                        )

                if completed and resp is None:
                    resp = stream.get_final_response()
    except APITimeoutError:
        elapsed = time.perf_counter() - start
        log.error(
            "LLM structured call TIMED OUT for schema=%s after %.1fs "
            "(model=%s, effort=%s, timeout=%.0fs). Consider raising timeout_seconds "
            "or lowering reasoning_effort.",
            schema.__name__,
            elapsed,
            model,
            reasoning_effort,
            timeout_seconds,
        )
        raise
    except StructuredLLMError:
        elapsed = time.perf_counter() - start
        log.exception(
            "LLM structured stream ended without completion for schema=%s after %.1fs "
            "(model=%s, effort=%s)",
            schema.__name__,
            elapsed,
            model,
            reasoning_effort,
        )
        raise
    except RuntimeError as exc:
        if "response.completed" not in str(exc):
            raise
        elapsed = time.perf_counter() - start
        log.exception(
            "LLM structured stream ended before response.completed for schema=%s after %.1fs "
            "(model=%s, effort=%s)",
            schema.__name__,
            elapsed,
            model,
            reasoning_effort,
        )
        raise StructuredLLMError(
            "OpenAI structured response stream ended before response.completed "
            f"(schema={schema.__name__}, model={model}, effort={reasoning_effort})"
        ) from exc
    except APIConnectionError as exc:
        elapsed = time.perf_counter() - start
        error_message = _format_connection_error(
            exc,
            schema_name=schema.__name__,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        log.error(
            "LLM structured call failed with API connection error after %.1fs "
            "(schema=%s, model=%s, effort=%s)",
            elapsed,
            schema.__name__,
            model,
            reasoning_effort,
        )
        raise StructuredLLMError(error_message) from exc
    except OpenAIError as exc:
        elapsed = time.perf_counter() - start
        if _is_proxy_block_error(exc):
            error_message = _format_proxy_block_error(
                exc,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            log.error(
                "LLM structured call blocked by corporate proxy after %.1fs "
                "(schema=%s, model=%s, effort=%s)",
                elapsed,
                schema.__name__,
                model,
                reasoning_effort,
            )
            raise StructuredLLMError(error_message) from exc
        if settings.openai_structured_streaming and _is_retryable_openai_error(exc):
            log.warning(
                "LLM structured stream failed with retryable OpenAI error after %.1fs; "
                "retrying without streaming (schema=%s, model=%s, effort=%s)",
                elapsed,
                schema.__name__,
                model,
                reasoning_effort,
            )
            resp = _responses_create_with_backoff(
                client,
                request_kwargs,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            _raise_if_response_not_completed(
                resp,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
                event_types=[],
            )
        else:
            log.exception(
                "LLM structured call FAILED for schema=%s after %.1fs (model=%s, effort=%s)",
                schema.__name__,
                elapsed,
                model,
                reasoning_effort,
            )
            raise

    elapsed = time.perf_counter() - start
    output_text = resp.output_text or ""
    log.info(
        "LLM structured call OK: schema=%s in %.1fs (output_chars=%d)",
        schema.__name__,
        elapsed,
        len(output_text),
    )

    try:
        return schema.model_validate_json(output_text)
    except ValidationError:
        log.exception(
            "Response did not match schema=%s. First 500 chars of output: %s",
            schema.__name__,
            output_text[:500],
        )
        raise
