from __future__ import annotations

import hashlib
import random
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
    """Raised when a structured-output call ends without completion."""

    def __init__(self, message: str, *, retryable: bool = False) -> None:
        super().__init__(message)
        self.retryable = retryable


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


def _is_retryable_proxy_gateway_error(exc: BaseException) -> bool:
    """Treat proxy-generated 5xx pages as transient, not policy denials."""
    status_code = getattr(exc, "status_code", None)
    return (
        _is_proxy_block_error(exc)
        and isinstance(status_code, int)
        and status_code >= 500
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


def _format_transient_proxy_error(
    exc: BaseException,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
) -> str:
    status_code = getattr(exc, "status_code", None)
    return (
        "OpenAI API requests repeatedly failed at the corporate proxy gateway "
        f"(schema={schema_name}, model={model}, effort={reasoning_effort}, "
        f"status={status_code or 'unknown'}). The proxy returned a transient HTML "
        "5xx error page for https://api.openai.com/v1/responses after all configured "
        "attempts. This is a proxy availability/handling failure, not a model or "
        "structured-output schema failure."
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


def _exception_diagnostics(exc: BaseException) -> Dict[str, Any]:
    """Return safe transport diagnostics without logging prompts or credentials."""
    response = getattr(exc, "response", None)
    headers = getattr(response, "headers", {}) or {}
    body = _error_body(exc)
    cause = getattr(exc, "__cause__", None)
    return {
        "exception": type(exc).__name__,
        "cause": type(cause).__name__ if cause is not None else "",
        "status": getattr(exc, "status_code", None),
        "code": body.get("error_code") or body.get("code"),
        "request_id": (
            headers.get("x-request-id")
            or headers.get("request-id")
            or getattr(exc, "request_id", None)
        ),
    }


def _retry_delay_seconds(exc: BaseException, attempt: int) -> float:
    fallback = min(
        settings.openai_retry_base_wait_s * (2 ** max(0, attempt - 1)),
        settings.openai_retry_max_wait_s,
    )
    explicit = _retry_after_seconds(exc, -1.0)
    if explicit >= 0:
        return min(explicit, settings.openai_retry_max_wait_s)
    jitter_ratio = max(0.0, float(getattr(settings, "openai_retry_jitter_ratio", 0.2)))
    jitter = random.uniform(0.0, fallback * jitter_ratio) if fallback > 0 else 0.0
    return min(fallback + jitter, settings.openai_retry_max_wait_s)


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
        attempt_start = time.perf_counter()
        try:
            return client.responses.create(**request_kwargs)
        except (APITimeoutError, APIConnectionError, OpenAIError) as exc:
            elapsed = time.perf_counter() - attempt_start
            proxy_error = _is_proxy_block_error(exc)
            transient_proxy = _is_retryable_proxy_gateway_error(exc)
            if proxy_error and not transient_proxy:
                raise StructuredLLMError(
                    _format_proxy_block_error(
                        exc,
                        schema_name=schema_name,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                ) from exc
            retryable = (
                transient_proxy
                or isinstance(exc, (APITimeoutError, APIConnectionError))
                or _is_retryable_openai_error(exc)
            )
            if attempt >= max_attempts or not retryable:
                if transient_proxy:
                    raise StructuredLLMError(
                        _format_transient_proxy_error(
                            exc,
                            schema_name=schema_name,
                            model=model,
                            reasoning_effort=reasoning_effort,
                        ),
                        retryable=True,
                    ) from exc
                if isinstance(exc, APIConnectionError) and not isinstance(exc, APITimeoutError):
                    raise StructuredLLMError(
                        _format_connection_error(
                            exc,
                            schema_name=schema_name,
                            model=model,
                            reasoning_effort=reasoning_effort,
                        ),
                        retryable=True,
                    ) from exc
                raise

            delay = _retry_delay_seconds(exc, attempt)
            diagnostics = _exception_diagnostics(exc)
            log.warning(
                "OpenAI response create attempt %d/%d failed after %.1fs; waiting %.1fs "
                "before retry (schema=%s, model=%s, effort=%s, exception=%s, cause=%s, "
                "status=%s, code=%s, request_id=%s)",
                attempt,
                max_attempts,
                elapsed,
                delay,
                schema_name,
                model,
                reasoning_effort,
                diagnostics["exception"],
                diagnostics["cause"],
                diagnostics["status"],
                diagnostics["code"],
                diagnostics["request_id"],
            )
            time.sleep(delay)

    raise RuntimeError("unreachable")


def _responses_retrieve_with_backoff(
    client: Any,
    response_id: str,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
) -> Any:
    """Retrieve a background response with the same bounded transport policy."""
    max_attempts = max(1, settings.openai_retry_attempts)
    for attempt in range(1, max_attempts + 1):
        attempt_start = time.perf_counter()
        try:
            return client.responses.retrieve(response_id)
        except (APITimeoutError, APIConnectionError, OpenAIError) as exc:
            elapsed = time.perf_counter() - attempt_start
            proxy_error = _is_proxy_block_error(exc)
            transient_proxy = _is_retryable_proxy_gateway_error(exc)
            if proxy_error and not transient_proxy:
                raise StructuredLLMError(
                    _format_proxy_block_error(
                        exc,
                        schema_name=schema_name,
                        model=model,
                        reasoning_effort=reasoning_effort,
                    )
                ) from exc
            retryable = (
                transient_proxy
                or isinstance(exc, (APITimeoutError, APIConnectionError))
                or _is_retryable_openai_error(exc)
            )
            if attempt >= max_attempts or not retryable:
                if transient_proxy:
                    raise StructuredLLMError(
                        _format_transient_proxy_error(
                            exc,
                            schema_name=schema_name,
                            model=model,
                            reasoning_effort=reasoning_effort,
                        ),
                        retryable=True,
                    ) from exc
                if isinstance(exc, APIConnectionError) and not isinstance(exc, APITimeoutError):
                    raise StructuredLLMError(
                        _format_connection_error(
                            exc,
                            schema_name=schema_name,
                            model=model,
                            reasoning_effort=reasoning_effort,
                        ),
                        retryable=True,
                    ) from exc
                raise
            delay = _retry_delay_seconds(exc, attempt)
            diagnostics = _exception_diagnostics(exc)
            log.warning(
                "OpenAI background retrieve attempt %d/%d failed after %.1fs; waiting %.1fs "
                "before retry (schema=%s, response_id=%s, exception=%s, cause=%s, "
                "status=%s, code=%s, request_id=%s)",
                attempt,
                max_attempts,
                elapsed,
                delay,
                schema_name,
                response_id,
                diagnostics["exception"],
                diagnostics["cause"],
                diagnostics["status"],
                diagnostics["code"],
                diagnostics["request_id"],
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


def _poll_background_response(
    client: Any,
    resp: Any,
    *,
    schema_name: str,
    model: str,
    reasoning_effort: str,
    deadline: float,
    grace_seconds: float | None = None,
) -> Any:
    """Poll a background Response using short, independently retried requests."""
    response_id = _field(resp, "id")
    if not response_id:
        raise StructuredLLMError(
            "OpenAI background response did not return a response ID "
            f"(schema={schema_name}, model={model}, effort={reasoning_effort})"
        )

    status = _field(resp, "status", "unknown")
    log.info(
        "LLM structured background response created: schema=%s response_id=%s status=%s",
        schema_name,
        response_id,
        status,
    )
    last_status = status
    last_progress_log = time.perf_counter()
    poll_s = max(0.0, float(getattr(settings, "openai_structured_background_poll_s", 2.0)))
    grace_s = max(
        0.0,
        float(
            grace_seconds
            if grace_seconds is not None
            else getattr(settings, "openai_structured_background_grace_s", 300.0)
        ),
    )
    hard_deadline = deadline + grace_s
    grace_logged = False
    while status in {"queued", "in_progress"}:
        now = time.perf_counter()
        remaining = hard_deadline - now
        if remaining <= 0:
            try:
                with_options = getattr(client, "with_options", None)
                cancel_client = (
                    with_options(timeout=10.0, max_retries=0)
                    if callable(with_options)
                    else client
                )
                cancelled = cancel_client.responses.cancel(response_id)
                log.info(
                    "Cancelled background response after application deadline: "
                    "schema=%s response_id=%s status=%s",
                    schema_name,
                    response_id,
                    _field(cancelled, "status", "unknown"),
                )
            except Exception as cancel_exc:
                log.warning(
                    "Could not cancel background response after application deadline: "
                    "schema=%s response_id=%s cause=%s",
                    schema_name,
                    response_id,
                    type(cancel_exc).__name__,
                )
            raise StructuredLLMError(
                "OpenAI background response exceeded the configured timeout and grace period "
                f"(schema={schema_name}, model={model}, effort={reasoning_effort}, "
                f"response_id={response_id}, last_status={status}, grace_s={grace_s:.0f})",
                retryable=True,
            )
        if now >= deadline and not grace_logged and grace_s > 0:
            log.warning(
                "LLM structured background response is still %s at the normal deadline; "
                "continuing to poll the existing job for up to %.0fs of grace "
                "(schema=%s, response_id=%s)",
                status,
                grace_s,
                schema_name,
                response_id,
            )
            grace_logged = True
        if poll_s:
            time.sleep(min(poll_s, remaining))
        resp = _responses_retrieve_with_backoff(
            client,
            response_id,
            schema_name=schema_name,
            model=model,
            reasoning_effort=reasoning_effort,
        )
        status = _field(resp, "status", "unknown")
        now = time.perf_counter()
        if status != last_status or now - last_progress_log >= 30.0:
            log.info(
                "LLM structured background progress: schema=%s response_id=%s status=%s "
                "deadline_phase=%s remaining=%.1fs",
                schema_name,
                response_id,
                status,
                "grace" if now >= deadline else "normal",
                max(0.0, (hard_deadline if now >= deadline else deadline) - now),
            )
            last_status = status
            last_progress_log = now

    _raise_if_response_not_completed(
        resp,
        schema_name=schema_name,
        model=model,
        reasoning_effort=reasoning_effort,
        event_types=[],
    )
    return resp


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
    background: bool | None = None,
    background_grace_seconds: float | None = None,
    recoverable_failure: bool = False,
) -> T:
    """Call OpenAI Responses API using STRICT JSON Schema structured output.

    The project owns retry behavior so attempt counts and backoff are explicit.
    Unless a caller makes an explicit transport choice, the shared policy uses
    background creation only for prompts above its configured size threshold.
    ``recoverable_failure`` changes logging only; exceptions still propagate so
    the caller can apply its proposal-safe fallback.
    """
    model = model or settings.model_reasoning
    background_min_chars = max(
        0, int(getattr(settings, "openai_structured_background_min_chars", 0) or 0)
    )
    background_enabled = bool(
        getattr(settings, "openai_structured_background_enabled", True)
    )
    background_all = bool(
        getattr(settings, "openai_structured_background_all", True)
    )
    background_grace_s = max(
        0.0,
        float(
            background_grace_seconds
            if background_grace_seconds is not None
            else getattr(settings, "openai_structured_background_grace_s", 300.0)
        ),
    )
    use_background = background_enabled and (
        bool(background)
        if background is not None
        else (
            background_all
            or bool(background_min_chars and len(prompt) >= background_min_chars)
        )
    )
    # Background create/retrieve requests should remain short even though the
    # server-side job may run through the normal deadline plus its grace period.
    request_timeout = (
        min(timeout_seconds, max(1.0, float(getattr(settings, "openai_timeout_s", 120.0))))
        if use_background
        else timeout_seconds
    )
    # Do not combine SDK retries with the explicit retry loop below. Previously,
    # three configured attempts could expand to as many as nine HTTP attempts.
    client = get_client(timeout=request_timeout, max_retries=0)
    prompt_fingerprint = hashlib.sha256(prompt.encode("utf-8")).hexdigest()[:12]

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
        "LLM structured call: schema=%s model=%s effort=%s prompt_chars=%d "
        "prompt_id=%s timeout=%.0fs background_grace=%.0fs request_timeout=%.0fs "
        "transport=%s attempts=%d",
        schema.__name__,
        model,
        reasoning_effort,
        len(prompt),
        prompt_fingerprint,
        timeout_seconds,
        background_grace_s if use_background else 0.0,
        request_timeout,
        (
            "background-poll"
            if use_background else "streaming"
            if settings.openai_structured_streaming else "non-streaming"
        ),
        max(1, settings.openai_retry_attempts),
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
    deadline = start + max(1.0, timeout_seconds)
    resp: Any = None
    try:
        if use_background:
            log.info(
                "LLM structured call using background Responses API with short polling "
                "(schema=%s, model=%s, effort=%s, default_all=%s, threshold_chars=%d)",
                schema.__name__,
                model,
                reasoning_effort,
                background_all,
                background_min_chars,
            )
            background_kwargs = dict(request_kwargs)
            background_kwargs["background"] = True
            resp = _responses_create_with_backoff(
                client,
                background_kwargs,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
            )
            resp = _poll_background_response(
                client,
                resp,
                schema_name=schema.__name__,
                model=model,
                reasoning_effort=reasoning_effort,
                deadline=deadline,
                grace_seconds=background_grace_s,
            )
        elif not settings.openai_structured_streaming:
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
        message = (
            "Recoverable LLM structured call timed out"
            if recoverable_failure
            else "LLM structured call TIMED OUT"
        )
        log_method = log.warning if recoverable_failure else log.error
        log_method(
            "%s for schema=%s after %.1fs (model=%s, effort=%s, timeout=%.0fs)%s",
            message,
            schema.__name__,
            elapsed,
            model,
            reasoning_effort,
            timeout_seconds,
            "; caller will apply its fallback" if recoverable_failure else "",
        )
        raise
    except StructuredLLMError:
        elapsed = time.perf_counter() - start
        log_method = log.warning if recoverable_failure else log.exception
        log_method(
            "%s for schema=%s after %.1fs "
            "(model=%s, effort=%s, transport=%s, prompt_id=%s)%s",
            (
                "Recoverable LLM structured call ended without completion"
                if recoverable_failure
                else "LLM structured call ended without completion"
            ),
            schema.__name__,
            elapsed,
            model,
            reasoning_effort,
            (
                "background-poll"
                if use_background else "streaming"
                if settings.openai_structured_streaming else "non-streaming"
            ),
            prompt_fingerprint,
            "; caller will apply its fallback" if recoverable_failure else "",
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
        "LLM structured call OK: schema=%s in %.1fs "
        "(response_id=%s, status=%s, prompt_id=%s, output_chars=%d)",
        schema.__name__,
        elapsed,
        _field(resp, "id"),
        _field(resp, "status"),
        prompt_fingerprint,
        len(output_text),
    )

    try:
        return schema.model_validate_json(output_text)
    except ValidationError:
        log_method = log.warning if recoverable_failure else log.exception
        log_method(
            "%s schema=%s. First 500 chars of output: %s",
            (
                "Recoverable response did not match"
                if recoverable_failure
                else "Response did not match"
            ),
            schema.__name__,
            output_text[:500],
        )
        raise
