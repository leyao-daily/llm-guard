"""Extract token usage from provider responses.

The gateway never estimates tokens with a local tokenizer. It reads the usage
object the provider itself reports, which is what you are actually billed for.
When a response carries no usage block, the request is recorded with a NULL
cost so the report can flag it instead of inventing a number.

Supports OpenAI Chat Completions / Responses and Anthropic Messages, both
buffered and streamed (SSE).
"""

from __future__ import annotations

import json
from typing import Any, Dict, Iterable, List, Optional, Tuple

from .pricing import TokenUsage


def _as_int(value: Any) -> int:
    try:
        if value is None:
            return 0
        return int(value)
    except (TypeError, ValueError):
        return 0


def _dig(data: Any, *path: str, default: Any = None) -> Any:
    cur = data
    for key in path:
        if not isinstance(cur, dict):
            return default
        cur = cur.get(key)
        if cur is None:
            return default
    return cur


def parse_openai_response(payload: dict) -> Tuple[str, TokenUsage]:
    """Parse a buffered OpenAI response -> (model, usage)."""
    model = str(payload.get("model") or "")
    usage = payload.get("usage") or {}

    prompt = _as_int(usage.get("prompt_tokens", usage.get("input_tokens")))
    completion = _as_int(usage.get("completion_tokens", usage.get("output_tokens")))

    # Cached tokens are reported *inside* prompt_tokens; strip them out so the
    # cost function can bill them at the cache-read rate instead of the full
    # input rate. Getting this wrong overstates cost on cache-heavy traffic.
    cached = _as_int(_dig(usage, "prompt_tokens_details", "cached_tokens"))
    cached += _as_int(_dig(usage, "input_tokens_details", "cached_tokens"))
    if cached > prompt:
        cached = prompt
    return model, TokenUsage(input=max(prompt - cached, 0), output=completion, cached_input=cached)


def parse_anthropic_response(payload: dict) -> Tuple[str, TokenUsage]:
    """Parse a buffered Anthropic Messages response -> (model, usage)."""
    model = str(payload.get("model") or "")
    usage = payload.get("usage") or {}

    # Anthropic reports input_tokens *excluding* cache reads/writes.
    return model, TokenUsage(
        input=_as_int(usage.get("input_tokens")),
        output=_as_int(usage.get("output_tokens")),
        cached_input=_as_int(usage.get("cache_read_input_tokens")),
        cache_write=_as_int(usage.get("cache_creation_input_tokens")),
    )


def parse_buffered(provider: str, body: bytes) -> Tuple[str, TokenUsage]:
    """Parse a non-streamed response body. Returns ("", zeros) when unknown."""
    if not body:
        return "", TokenUsage()
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return "", TokenUsage()
    if not isinstance(payload, dict):
        return "", TokenUsage()

    if provider == "anthropic":
        return parse_anthropic_response(payload)

    # OpenAI-compatible (also covers Azure, Groq, Together, local servers).
    model, usage = parse_openai_response(payload)
    if usage.total == 0 and "usage" not in payload:
        # Responses API nests usage differently; try it before giving up.
        usage = TokenUsage(
            input=_as_int(_dig(payload, "usage", "input_tokens")),
            output=_as_int(_dig(payload, "usage", "output_tokens")),
        )
    return model, usage


def iter_sse_events(text: str) -> Iterable[dict]:
    """Yield decoded JSON objects from an SSE body.

    Handles both ``data: {json}`` lines and multi-line data blocks. Malformed
    frames are skipped rather than raising: a broken frame must not take down
    usage accounting for the rest of the stream.
    """
    for block in text.split("\n\n"):
        if not block.strip():
            continue
        data_lines: List[str] = []
        for line in block.splitlines():
            if line.startswith("data:"):
                data_lines.append(line[5:].strip())
        if not data_lines:
            continue
        raw = "\n".join(data_lines)
        if raw in ("[DONE]", ""):
            continue
        try:
            obj = json.loads(raw)
        except ValueError:
            continue
        if isinstance(obj, dict):
            yield obj


def parse_streamed(provider: str, text: str) -> Tuple[str, TokenUsage]:
    """Parse an accumulated SSE stream -> (model, usage).

    OpenAI sends final usage in the last chunk only when
    ``stream_options={"include_usage": true}`` is set; we forward whatever the
    client asked for and read it if present. Anthropic splits usage across
    ``message_start`` (input side) and ``message_delta`` (output side), so the
    two are merged.

    A ``message_start`` event looks like::

        {"type":"message_start","message":{"model":...,"usage":{...}}}

    so the usage block lives on the *nested message object*, not on the event.
    """
    model = ""
    input_tokens = 0
    output_tokens = 0
    cached_input = 0
    cache_write = 0

    for obj in iter_sse_events(text):
        if not model:
            model = str(obj.get("model") or _dig(obj, "message", "model") or "")

        # --- Anthropic: usage on the nested message object ---
        nested = _dig(obj, "message", "usage")
        if isinstance(nested, dict):
            input_tokens = _as_int(nested.get("input_tokens"))
            cached_input = _as_int(nested.get("cache_read_input_tokens"))
            cache_write = _as_int(nested.get("cache_creation_input_tokens"))

        # --- OpenAI style: flat usage; also Anthropic message_delta output ---
        flat = obj.get("usage")
        if isinstance(flat, dict):
            prompt = _as_int(flat.get("prompt_tokens", flat.get("input_tokens")))
            completion = _as_int(flat.get("completion_tokens", flat.get("output_tokens")))
            cached = _as_int(_dig(flat, "prompt_tokens_details", "cached_tokens"))
            cached += _as_int(_dig(flat, "input_tokens_details", "cached_tokens"))
            if cached > prompt:
                cached = prompt

            if prompt:
                # A full usage block (OpenAI final chunk) replaces our figures.
                input_tokens = max(prompt - cached, 0)
                cached_input = cached
            if completion:
                # Anthropic message_delta reports a cumulative output count.
                output_tokens = completion

    return model, TokenUsage(
        input=input_tokens,
        output=output_tokens,
        cached_input=cached_input,
        cache_write=cache_write,
    )


def _lower(headers: Dict[str, str]) -> Dict[str, str]:
    """Case-insensitive view of a header dict."""
    return {str(k).lower(): v for k, v in headers.items()}


def is_streaming_response(headers: Dict[str, str]) -> bool:
    ctype = str(_lower(headers).get("content-type", "")).lower()
    return "text/event-stream" in ctype


def extract_end_user(headers: Dict[str, str], body: bytes) -> str:
    """Best-effort end-user attribution.

    Priority: explicit header, then the conventional OpenAI ``user`` field,
    then a common metadata key. Never inspects prompt content -- that would
    make the gateway a data processor for customer PII.
    """
    lower = _lower(headers)
    for header in ("x-end-user", "x-end-user-id", "x-customer-id"):
        if lower.get(header):
            return str(lower[header])[:128]
    if body:
        try:
            payload = json.loads(body)
        except (ValueError, UnicodeDecodeError):
            return ""
        if isinstance(payload, dict):
            for key in ("user", "end_user", "metadata"):
                value = payload.get(key)
                if isinstance(value, str) and value:
                    return value[:128]
                if isinstance(value, dict):
                    for sub in ("end_user", "user_id", "customer_id"):
                        if isinstance(value.get(sub), str) and value[sub]:
                            return str(value[sub])[:128]
    return ""


def extract_project(headers: Dict[str, str]) -> str:
    lower = _lower(headers)
    for header in ("x-project", "x-app", "x-service"):
        if lower.get(header):
            return str(lower[header])[:128]
    return ""
