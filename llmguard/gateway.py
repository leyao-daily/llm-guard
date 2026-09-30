"""The gateway: an HTTP/1.1 reverse proxy for LLM APIs, on the stdlib alone.

Why stdlib and not FastAPI/uvicorn: this process is meant to be dropped onto a
customer's VPC box, next to their app, and started in one command with no venv
and no supply chain. It runs on any Python 3.9+.

Behaviour
---------
* Forwards OpenAI-compatible / Anthropic / Gemini traffic verbatim, streaming
  responses byte-for-byte as they arrive.
* Reads the ``usage`` block the provider reports and records cost per request.
* Enforces per-API-key budgets on the request path (alert or block).
* Serves its own read-only JSON API under ``/-/`` for dashboards and CI.

Deliberately not implemented: TLS termination (put it behind nginx/Caddy or a
cloud LB), HTTP/2, response caching. Each would add dependencies or risk.
"""

from __future__ import annotations

import asyncio
import json
import logging
import re
import ssl
import threading
import time
import uuid
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
from urllib.parse import urlsplit

from .analytics import build_report, evaluate_guard
from .detectors import LiveGuard, detect_all, DetectionConfig
from .parsers import (
    extract_end_user,
    extract_project,
    is_streaming_response,
    parse_buffered,
    parse_streamed,
)
from .pricing import TokenUsage, compute_cost
from .storage import RequestRecord, Store, iso

log = logging.getLogger("llmguard")

MAX_REQUEST_BODY = 32 * 1024 * 1024  # 32 MiB; LLM requests are far smaller

# Headers that must not be forwarded in either direction (RFC 7230 6.1) plus
# ones we recompute ourselves.
HOP_BY_HOP = {
    "connection",
    "keep-alive",
    "proxy-authenticate",
    "proxy-authorization",
    "te",
    "trailer",
    "transfer-encoding",
    "upgrade",
    "host",
    "content-length",
}


@dataclass
class Route:
    """Maps an inbound path prefix to an upstream base URL."""

    prefix: str          # e.g. "/v1" for openai, "/anthropic/v1" for anthropic
    upstream: str        # base, no trailing slash
    provider: str
    strip_prefix: str    # replaced with upstream base

    def resolve(self, path: str) -> Optional[str]:
        if not path.startswith(self.prefix):
            return None
        remainder = path[len(self.strip_prefix):]
        if not remainder.startswith("/"):
            remainder = "/" + remainder
        return self.upstream + remainder


DEFAULT_ROUTES = [
    Route(prefix="/v1", upstream="https://api.openai.com", provider="openai",
          strip_prefix="/v1"),
    Route(prefix="/anthropic", upstream="https://api.anthropic.com", provider="anthropic",
          strip_prefix="/anthropic"),
    Route(prefix="/google", upstream="https://generativelanguage.googleapis.com",
          provider="google", strip_prefix="/google"),
]


@dataclass
class GatewayConfig:
    host: str = "127.0.0.1"
    port: int = 8787
    routes: List[Route] = field(default_factory=lambda: list(DEFAULT_ROUTES))
    request_timeout: float = 300.0
    connect_timeout: float = 15.0
    # Verify upstream TLS certificates. Leave on. The only reason to disable it is
    # a corporate MITM proxy that presents its own CA, in which case prefer
    # pointing --upstream-ca at that CA over turning verification off.
    verify_upstream_tls: bool = True
    upstream_ca_file: str = ""
    # Optional static upstream credentials. When empty, the gateway forwards the
    # caller's own credential, which is what you want for a local dev proxy.
    upstream_keys: Dict[str, str] = field(default_factory=dict)
    # Optional mapping from the caller's bearer token to a stable key id used for
    # attribution and budgets, e.g. {"sk-team-a-xxx": "team-a"}.
    key_map: Dict[str, str] = field(default_factory=dict)
    admin_token: str = ""
    # Batch accounting writes instead of committing one row per request. This is
    # the difference between ~4k and ~14k req/s. Records are flushed on a timer
    # and on shutdown; the exposure window is write_behind_interval seconds.
    write_behind: bool = True
    write_behind_interval: float = 0.25
    write_behind_max_batch: int = 512
    # Live agent-loop tripwire: flags (never blocks) a sustained pathological
    # input/output token ratio while it is happening.
    loop_detection: bool = True
    loop_io_ratio: float = 30.0
    # Mid-stream cancellation. A budget that only fires between requests cannot
    # stop the request that crosses it, and long agent chains are exactly the
    # requests that need stopping -- existing gateways are documented as leaving
    # that gap. These caps abort the response mid-flight instead.
    # Off by default: aborting a stream is a new failure mode, and it needs a
    # token estimate rather than provider-reported usage.
    stream_max_cost_usd: Optional[float] = None
    stream_max_tokens: Optional[int] = None
    # Characters per token for the mid-stream estimate. Deliberately crude and
    # deliberately conservative (over-estimates tokens slightly) so the cap
    # trips early rather than late. This estimate is never used for billing.
    stream_chars_per_token: float = 4.0


class _WriteBehind:
    """Buffers request records and flushes them in batches.

    A cost tracker that loses data is worse than one that is 5 ms slower, so the
    design is deliberately conservative: bounded queue, timer + size flush, an
    explicit ``drain()`` that the server awaits on shutdown, and a synchronous
    fallback if the queue is ever saturated.
    """

    def __init__(self, store: Store, interval: float, max_batch: int, queue_size: int = 20_000):
        self.store = store
        self.interval = interval
        self.max_batch = max_batch
        self.queue: "asyncio.Queue[RequestRecord]" = asyncio.Queue(maxsize=queue_size)
        self.buffer: List[RequestRecord] = []
        self.dropped = 0
        self.flushed = 0
        self._task: Optional[asyncio.Task] = None
        self._stopping = False
        # threading.Lock, not asyncio.Lock: a lock created before a loop exists
        # binds to whichever loop first uses it, which breaks `drain()` when it
        # is called from a different loop (tests, shutdown hooks, embedding).
        self._lock = threading.Lock()

    def submit(self, rec: RequestRecord) -> None:
        """Enqueue for the flusher. Falls back to a synchronous write if full."""
        try:
            self.queue.put_nowait(rec)
        except asyncio.QueueFull:
            # Losing spend data silently is the worst outcome; take the latency.
            self.dropped += 1
            self._store_record(rec)

    async def _flush(self) -> int:
        with self._lock:
            batch = self.buffer
            self.buffer = []
        if not batch:
            return 0
        rows = [Store.record_to_row(r) for r in batch]
        try:
            self.store._insert_rows(rows)
            self.flushed += len(rows)
        except Exception:  # noqa: BLE001
            log.exception("failed to flush %d records", len(rows))
        return len(rows)

    async def run(self) -> None:
        while not self._stopping:
            try:
                rec = await asyncio.wait_for(self.queue.get(), timeout=self.interval)
                self.buffer.append(rec)
                while len(self.buffer) < self.max_batch:
                    try:
                        self.buffer.append(self.queue.get_nowait())
                    except asyncio.QueueEmpty:
                        break
                if len(self.buffer) >= self.max_batch:
                    await self._flush()
            except asyncio.TimeoutError:
                await self._flush()
        # final drain
        while True:
            try:
                self.buffer.append(self.queue.get_nowait())
            except asyncio.QueueEmpty:
                break
        await self._flush()

    def start(self) -> None:
        if self._task is None:
            self._task = asyncio.ensure_future(self.run())

    def stop_nowait(self) -> None:
        """Cancel without awaiting, and drain synchronously.

        Safe to call from a different event loop than the flusher's, which is why
        it exists: awaiting a foreign task deadlocks. Used on interpreter
        shutdown and by tests that tear a loop down without an explicit drain.
        """
        self._stopping = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()
        with self._lock:
            while True:
                try:
                    self.buffer.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        if self.buffer:
            try:
                self.store._insert_rows(
                    [Store.record_to_row(r) for r in self.buffer]
                )
                self.flushed += len(self.buffer)
            except Exception:  # noqa: BLE001
                log.exception("failed to flush %d records on shutdown", len(self.buffer))
            finally:
                self.buffer = []

    def __del__(self):  # pragma: no cover - best effort at interpreter shutdown
        # An abandoned instance must not leave a pending task behind: on newer
        # interpreters that surfaces as "Task was destroyed but it is pending!"
        # long after the test that caused it has passed.
        try:
            if self._task is not None and not self._task.done():
                self._task.cancel()
        except Exception:  # noqa: BLE001
            pass

    async def drain(self) -> None:
        """Flush and stop the flusher.

        Deliberately loop-independent: the flusher task is cancelled without
        awaiting it, then the queue and buffer are emptied synchronously. Awaiting
        a task that belongs to a different event loop (tests, shutdown hooks,
        embedding into another app) hangs forever, and draining data is too
        important to leave to that.
        """
        self._stopping = True
        task, self._task = self._task, None
        if task is not None and not task.done():
            task.cancel()

        with self._lock:
            while True:
                try:
                    self.buffer.append(self.queue.get_nowait())
                except asyncio.QueueEmpty:
                    break
        await self._flush()


class StreamBudget:
    """Tracks spend while a response is still streaming, so it can be cut off.

    Why this exists: a per-request budget check happens *before* the request, and
    a per-key budget check happens *between* requests. Neither can stop the one
    request that crosses the line -- which, for a long agent chain, is precisely
    the request that matters. Existing gateways are documented as leaving this
    gap: the stream that crosses the budget still completes.

    Trade-off, stated plainly: mid-stream we only have a *local estimate* of
    tokens, because both OpenAI and Anthropic report authoritative usage at the
    end. The estimate is therefore deliberately conservative (over-counts), and
    this cap is a safety net rather than an accounting mechanism. Billing always
    uses provider-reported numbers.
    """

    def __init__(
        self,
        *,
        max_cost_usd: Optional[float],
        max_tokens: Optional[int],
        initial_tokens: int,
        model: str,
        chars_per_token: float = 4.0,
    ):
        self.max_cost_usd = max_cost_usd
        self.max_tokens = max_tokens
        self.chars_per_token = max(chars_per_token, 1.0)
        self.model = model
        self.input_tokens = max(initial_tokens, 0)
        self.output_tokens = 0
        self.frames = 0
        self.tripped: Optional[str] = None

    @property
    def total_tokens(self) -> int:
        return self.input_tokens + self.output_tokens

    def estimated_cost(self) -> float:
        """Local estimate in USD. Never used for billing."""
        cost = compute_cost(
            self.model,
            TokenUsage(input=self.input_tokens, output=self.output_tokens),
            quantize=False,
        )
        return float(cost) if cost is not None else 0.0

    def observe_content(self, text: str) -> None:
        """Account for streamed output text using the local estimator."""
        if not text:
            return
        self.output_tokens += max(int(len(text) / self.chars_per_token), 1)

    def observe_usage(self, usage: TokenUsage) -> None:
        """Replace estimates with provider-reported figures when they arrive."""
        if usage.input or usage.cached_input:
            self.input_tokens = usage.input + usage.cached_input + usage.cache_write
        if usage.output:
            self.output_tokens = usage.output

    def check(self) -> Optional[str]:
        """Return a reason string when a cap is crossed, else None."""
        if self.tripped:
            return self.tripped
        self.frames += 1
        if self.max_tokens is not None and self.total_tokens >= self.max_tokens:
            self.tripped = (
                f"stream token cap reached: ~{self.total_tokens:,} tokens "
                f"(cap {self.max_tokens:,}, local estimate)"
            )
            return self.tripped
        if self.max_cost_usd is not None:
            spent = self.estimated_cost()
            if spent >= self.max_cost_usd:
                self.tripped = (
                    f"stream cost cap reached: ~${spent:.4f} "
                    f"(cap ${self.max_cost_usd:.4f}, local estimate)"
                )
                return self.tripped
        return None

    def abort_frame(self) -> bytes:
        """An SSE frame telling the client why the stream stopped.

        Shaped like a provider error so existing clients surface it rather than
        treating truncation as a successful completion.
        """
        payload = {
            "error": {
                "message": f"llm-guard aborted the stream: {self.tripped}",
                "type": "budget_exceeded",
                "code": "stream_budget_exceeded",
                "llm_guard": {
                    "estimated_tokens": self.total_tokens,
                    "estimated_cost_usd": round(self.estimated_cost(), 6),
                    "note": "estimate only; see provider usage for billing",
                },
            }
        }
        return f"data: {json.dumps(payload)}\n\ndata: [DONE]\n\n".encode()


def _request_model(body: bytes) -> str:
    """Best-effort model id from the request body, for pricing the estimate."""
    if not body:
        return ""
    try:
        payload = json.loads(body)
    except (ValueError, UnicodeDecodeError):
        return ""
    if isinstance(payload, dict):
        model = payload.get("model")
        if isinstance(model, str):
            return model
    return ""


class Gateway:
    """HTTP reverse proxy that meters and guards LLM API traffic."""

    def __init__(self, store: Store, config: GatewayConfig):
        self.store = store
        self.config = config
        self.started_at = time.time()
        self.counters = {"requests": 0, "blocked": 0, "errors": 0, "aborted": 0}
        self._server: Optional[asyncio.AbstractServer] = None
        self.writer: Optional[_WriteBehind] = None
        if config.write_behind:
            self.writer = _WriteBehind(
                store, config.write_behind_interval, config.write_behind_max_batch
            )
        self.loop_guard = (
            LiveGuard(io_ratio=config.loop_io_ratio) if config.loop_detection else None
        )
        # Rolling record of suspected loops, surfaced through /-/anomalies.
        self.loop_warnings: List[dict] = []

    def _upstream_ssl_context(self) -> "ssl.SSLContext":
        """TLS context for upstream connections.

        Default is full verification (hostname + CA + cert). An explicit CA file
        is the supported answer to a corporate MITM proxy; disabling verification
        entirely is possible but discouraged, so it is not exposed as a config
        flag -- pass a CA instead.
        """
        if self.config.upstream_ca_file:
            context = ssl.create_default_context(cafile=self.config.upstream_ca_file)
        else:
            context = ssl.create_default_context()
        if not self.config.verify_upstream_tls:
            context.check_hostname = False
            context.verify_mode = ssl.CERT_NONE
            log.warning(
                "upstream TLS verification is DISABLED - do not run this way in production"
            )
        return context

    def _store_record(self, rec: RequestRecord) -> None:
        """Persist one accounted request, buffered or synchronously."""
        if self.writer is not None:
            self.writer.submit(rec)
        else:
            self.store.insert(rec)

    async def drain(self) -> None:
        """Flush buffered accounting rows. Call before the process exits."""
        if self.writer is not None:
            await self.writer.drain()

    # ------------------------------------------------------------------ HTTP
    async def handle(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        try:
            await self._handle_inner(reader, writer)
        except (ConnectionResetError, BrokenPipeError, asyncio.IncompleteReadError):
            pass
        except Exception:  # noqa: BLE001 - a proxy must never die on one request
            log.exception("unhandled error in request handler")
            self.counters["errors"] += 1
            try:
                await _write_json(
                    writer, 502, {"error": {"message": "gateway error", "type": "gateway_error"}}
                )
            except Exception:  # noqa: BLE001
                pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _handle_inner(
        self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter
    ) -> None:
        request_line = await reader.readline()
        if not request_line:
            return
        try:
            method, target, _version = request_line.decode("latin-1").strip().split(" ", 2)
        except ValueError:
            await _write_json(writer, 400, {"error": {"message": "malformed request line"}})
            return

        headers = await _read_headers(reader)
        body = await _read_body(reader, headers)

        path = urlsplit(target).path or "/"

        # --- gateway's own API ---
        if path.startswith("/-/"):
            await self._handle_own_api(method, path, writer)
            return

        # --- proxy path ---
        route = self._match_route(path)
        if route is None:
            await _write_json(
                writer,
                404,
                {
                    "error": {
                        "message": f"no upstream route for {path}",
                        "type": "invalid_request_error",
                        "hint": "use /v1/* (OpenAI), /anthropic/* or /google/*",
                    }
                },
            )
            return

        api_key_id = self._identify(headers)
        guard = evaluate_guard(self.store, api_key_id)
        if not guard.allowed and guard.status == 429:
            self.counters["blocked"] += 1
            record = RequestRecord(
                provider=route.provider,
                model="",
                status=429,
                api_key_id=api_key_id,
                error=guard.reason,
                project=extract_project({k.lower(): v for k, v in headers.items()}),
            )
            self._store_record(record)
            await _write_json(
                writer,
                429,
                {
                    "error": {
                        "message": f"budget exceeded: {guard.reason}",
                        "type": "budget_exceeded",
                        "api_key_id": api_key_id,
                    }
                },
            )
            return

        upstream_url = route.resolve(path)
        if upstream_url is None:
            await _write_json(writer, 404, {"error": {"message": "unroutable path"}})
            return

        # Preserve the caller's query string. Route.resolve() already carries it
        # through, so it must not be appended a second time here.
        parts = urlsplit(target)
        if parts.query and "?" not in upstream_url:
            upstream_url = f"{upstream_url}?{parts.query}"

        started = time.perf_counter()
        await self._proxy(
            method=method,
            target_path=path,
            upstream_url=upstream_url,
            route=route,
            headers=headers,
            body=body,
            api_key_id=api_key_id,
            writer=writer,
            started=started,
        )

    def _match_route(self, path: str) -> Optional[Route]:
        # Longest prefix wins so /anthropic/v1 does not shadow /anthropic.
        best: Optional[Route] = None
        for route in self.config.routes:
            if path.startswith(route.prefix):
                if best is None or len(route.prefix) > len(best.prefix):
                    best = route
        return best

    def _identify(self, headers: Dict[str, str]) -> str:
        """Derive a stable attribution key for this caller."""
        lower = {k.lower(): v for k, v in headers.items()}
        token = ""
        auth = lower.get("authorization", "")
        if auth.lower().startswith("bearer "):
            token = auth[7:].strip()
        elif lower.get("x-api-key"):
            token = lower["x-api-key"].strip()

        if token:
            # Never store raw credentials. A key_map hit gives a human label;
            # otherwise use a short non-reversible fingerprint.
            if token in self.config.key_map:
                return self.config.key_map[token]
            return "key-" + _fingerprint(token)
        return "anonymous"

    # --------------------------------------------------------------- proxying
    async def _proxy(
        self,
        *,
        method: str,
        target_path: str,
        upstream_url: str,
        route: Route,
        headers: Dict[str, str],
        body: bytes,
        api_key_id: str,
        writer: asyncio.StreamWriter,
        started: float,
    ) -> None:
        parts = urlsplit(upstream_url)
        host = parts.hostname or ""
        port = parts.port or (443 if parts.scheme == "https" else 80)
        request_target = parts.path + (f"?{parts.query}" if parts.query else "")

        lower_headers = {k.lower(): v for k, v in headers.items()}

        ssl_context = None
        server_hostname = None
        if parts.scheme == "https":
            # Outbound TLS is handled in-process with the stdlib `ssl` module.
            # This is the difference between "works against the real APIs" and
            # "needs a TLS-terminating proxy in front of every deployment".
            ssl_context = self._upstream_ssl_context()
            server_hostname = host

        try:
            up_reader, up_writer = await asyncio.wait_for(
                asyncio.open_connection(
                    host,
                    port,
                    ssl=ssl_context,
                    server_hostname=server_hostname,
                ),
                timeout=self.config.connect_timeout,
            )
        except (OSError, asyncio.TimeoutError) as exc:
            self.counters["errors"] += 1
            self._store_record(
                RequestRecord(
                    provider=route.provider,
                    model="",
                    status=502,
                    latency_ms=int((time.perf_counter() - started) * 1000),
                    api_key_id=api_key_id,
                    error=f"upstream connect failed: {exc}",
                    project=extract_project(lower_headers),
                )
            )
            await _write_json(
                writer, 502, {"error": {"message": f"upstream unreachable: {exc}"}}
            )
            return

        try:
            await self._relay(
                up_reader=up_reader,
                up_writer=up_writer,
                method=method,
                request_target=request_target,
                host=host,
                port=port,
                headers=headers,
                lower_headers=lower_headers,
                body=body,
                route=route,
                api_key_id=api_key_id,
                writer=writer,
                started=started,
            )
        finally:
            try:
                up_writer.close()
                await up_writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _relay(
        self,
        *,
        up_reader: asyncio.StreamReader,
        up_writer: asyncio.StreamWriter,
        method: str,
        request_target: str,
        host: str,
        port: int,
        headers: Dict[str, str],
        lower_headers: Dict[str, str],
        body: bytes,
        route: Route,
        api_key_id: str,
        writer: asyncio.StreamWriter,
        started: float,
    ) -> None:
        # ---- build and send the upstream request ----
        out_headers: Dict[str, str] = {}
        for key, value in headers.items():
            if key.lower() in HOP_BY_HOP:
                continue
            out_headers[key] = value
        out_headers["Host"] = host if port in (80, 443) else f"{host}:{port}"
        out_headers["Content-Length"] = str(len(body))
        # One request per connection: makes response framing trivial and avoids
        # a half-closed connection stalling streamed responses.
        out_headers["Connection"] = "close"

        static_key = self.config.upstream_keys.get(route.provider)
        if static_key:
            if route.provider == "anthropic":
                out_headers["x-api-key"] = static_key
                out_headers.pop("Authorization", None)
            else:
                out_headers["Authorization"] = f"Bearer {static_key}"

        request_bytes = _serialize_request(method, request_target, out_headers, body)
        up_writer.write(request_bytes)
        await up_writer.drain()

        # ---- read the upstream status line + headers ----
        status_line = await asyncio.wait_for(
            up_reader.readline(), timeout=self.config.request_timeout
        )
        if not status_line:
            raise ConnectionError("upstream closed before responding")
        try:
            _ver, status_text, _reason = status_line.decode("latin-1").strip().split(" ", 2)
            status = int(status_text)
        except ValueError:
            raise ConnectionError(f"malformed upstream status line: {status_line!r}")

        up_headers = await _read_headers(up_reader)
        up_lower = {k.lower(): v for k, v in up_headers.items()}

        # ---- decide framing ----
        chunked = "chunked" in up_lower.get("transfer-encoding", "").lower()
        content_length = up_lower.get("content-length")
        streaming = is_streaming_response(up_lower)

        pass_headers = {
            k: v for k, v in up_headers.items() if k.lower() not in HOP_BY_HOP
        }
        pass_headers["Connection"] = "close"
        if chunked or content_length is None:
            # The body is delimited by EOF (see _relay_chunked): advertising a
            # length here would be wrong, and Transfer-Encoding is hop-by-hop
            # and already stripped.
            pass_headers.pop("Content-Length", None)

        head = _serialize_head(status, pass_headers)
        writer.write(head)
        await writer.drain()

        # ---- stream the body through, capturing bytes for usage parsing ----
        captured = bytearray()
        # Only streaming responses can be cancelled mid-flight, and only when a
        # cap is configured. Estimate tokens from the request body so the cap has
        # a starting point before any usage frame arrives.
        abort_reason = ""
        enforce = (
            streaming
            and (
                self.config.stream_max_cost_usd is not None
                or self.config.stream_max_tokens is not None
            )
        )
        # Crude starting token estimate from the request payload. Only the
        # provider's own usage figures are ever billed; this exists so the cap
        # has something to measure against before the first usage frame.
        estimate = StreamBudget(
            max_cost_usd=self.config.stream_max_cost_usd,
            max_tokens=self.config.stream_max_tokens,
            initial_tokens=int(
                len(body) / max(self.config.stream_chars_per_token, 1.0)
            ),
            model=_request_model(body),
            chars_per_token=self.config.stream_chars_per_token,
        )
        if enforce and chunked:
            # Chunked upstream: decode frames, reframe as EOF-delimited, and
            # check the cap on every frame.
            captured, _, abort_reason = await _relay_chunked(up_reader, writer, estimate)
        elif enforce and content_length is None:
            # Streamed response with no Content-Length (the common case: the
            # upstream just closes the connection). This must be checked before
            # the plain-EOF path, otherwise the cap is never consulted.
            captured, _, abort_reason = await _relay_eof_with_budget(
                up_reader, writer, estimate
            )
        elif chunked:
            captured, _, _ = await _relay_chunked(up_reader, writer)
        else:
            captured, _ = await _relay_until_eof(up_reader, writer, content_length)

        latency_ms = int((time.perf_counter() - started) * 1000)

        # ---- account for it ----
        self._record(
            route=route,
            status=status,
            streaming=streaming,
            captured=bytes(captured),
            latency_ms=latency_ms,
            api_key_id=api_key_id,
            lower_headers=lower_headers,
            body=body,
            upstream_error=up_lower,
            abort_reason=abort_reason,
        )

    def _record(
        self,
        *,
        route: Route,
        status: int,
        streaming: bool,
        captured: bytes,
        latency_ms: int,
        api_key_id: str,
        lower_headers: Dict[str, str],
        body: bytes,
        upstream_error: Dict[str, str],
        abort_reason: str = "",
    ) -> None:
        model = ""
        usage = TokenUsage()
        error = ""

        try:
            if streaming:
                model, usage = parse_streamed(route.provider, captured.decode("utf-8", "ignore"))
            else:
                model, usage = parse_buffered(route.provider, captured)
        except Exception:  # noqa: BLE001 - accounting must never break the request
            log.exception("usage parsing failed")
            error = "usage parse failed"

        if status >= 400:
            error = error or f"upstream status {status}"
        if abort_reason:
            # Recorded as an error so it surfaces in the report instead of
            # looking like a clean completion.
            error = abort_reason

        cost = None
        if status < 400 and (usage.total > 0):
            try:
                cost = compute_cost(model, usage)
            except Exception:  # noqa: BLE001
                log.exception("cost computation failed")
                error = error or "cost computation failed"

        self._store_record(
            RequestRecord(
                provider=route.provider,
                model=model,
                input_tokens=usage.input,
                cached_input_tokens=usage.cached_input,
                cache_write_tokens=usage.cache_write,
                output_tokens=usage.output,
                cost_usd=None if cost is None else float(cost),
                latency_ms=latency_ms,
                status=status,
                streamed=streaming,
                api_key_id=api_key_id,
                end_user=extract_end_user(lower_headers, body),
                project=extract_project(lower_headers),
                request_id=uuid.uuid4().hex[:16],
                error=error,
            )
        )
        self.counters["requests"] += 1
        if abort_reason:
            self.counters["aborted"] += 1

        # Live loop tripwire: advisory only, never blocks. A detector that
        # silently killed production traffic would be worse than the overspend
        # it is trying to prevent.
        if self.loop_guard is not None and usage.output > 0:
            warning = self.loop_guard.observe(
                api_key_id, usage.input + usage.cached_input, usage.output
            )
            if warning:
                self.loop_warnings.append(
                    {
                        "at": iso(),
                        "api_key_id": api_key_id,
                        "model": model,
                        "warning": warning,
                        "input_tokens": usage.input + usage.cached_input,
                        "output_tokens": usage.output,
                    }
                )
                # Bounded: this is a tripwire log, not a store.
                if len(self.loop_warnings) > 200:
                    del self.loop_warnings[:-200]
                log.warning("suspected agent loop on %s: %s", api_key_id, warning)

    # -------------------------------------------------------------- own API
    async def _handle_own_api(
        self, method: str, path: str, writer: asyncio.StreamWriter
    ) -> None:
        if method != "GET":
            await _write_json(writer, 405, {"error": {"message": "method not allowed"}})
            return

        if path == "/-/health":
            await _write_json(
                writer,
                200,
                {
                    "status": "ok",
                    "uptime_seconds": int(time.time() - self.started_at),
                    "requests_seen": self.counters["requests"],
                    "requests_blocked": self.counters["blocked"],
                    "streams_aborted": self.counters["aborted"],
                    "rows_total": self.store.count(),
                },
            )
            return

        if path == "/-/stats":
            await _write_json(writer, 200, json.loads(_json(build_report(self.store, days=30))))
            return

        match = re.match(r"^/-/stats/(\d{1,3})$", path)
        if match:
            days = max(1, min(int(match.group(1)), 365))
            await _write_json(writer, 200, json.loads(_json(build_report(self.store, days=days))))
            return

        if path == "/-/anomalies":
            config = DetectionConfig()
            anomalies = detect_all(self.store, config)
            await _write_json(
                writer,
                200,
                {
                    "detected": [
                        {
                            "kind": a.kind,
                            "subject": a.subject,
                            "subject_type": a.subject_type,
                            "severity": a.severity,
                            "detail": a.detail,
                            "remedy": a.remedy(),
                            "evidence": a.evidence,
                        }
                        for a in anomalies
                    ],
                    "live_loop_warnings": self.loop_warnings[-20:],
                    "window_minutes": config.window_minutes,
                    "thresholds": {
                        "io_ratio": config.io_ratio,
                        "velocity_multiplier": config.velocity_multiplier,
                        "storm_min_failures": config.storm_min_failures,
                    },
                },
            )
            return

        if path == "/-/budgets":
            rows = [dict(r) for r in self.store.all_budgets()]
            await _write_json(writer, 200, {"budgets": rows})
            return

        if path == "/-/dashboard":
            from .dashboard import render_dashboard

            html = render_dashboard(build_report(self.store, days=30))
            payload = html.encode("utf-8")
            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/html; charset=utf-8\r\n"
                + f"Content-Length: {len(payload)}\r\nConnection: close\r\n\r\n".encode()
                + payload
            )
            await writer.drain()
            return

        await _write_json(writer, 404, {"error": {"message": f"unknown endpoint {path}"}})

    # ---------------------------------------------------------------- serving
    async def serve_forever(self) -> None:
        if self.writer is not None:
            self.writer.start()
        self._server = await asyncio.start_server(
            self.handle, self.config.host, self.config.port
        )
        addrs = ", ".join(str(s.getsockname()) for s in (self._server.sockets or []))
        log.info("llm-guard listening on %s", addrs)
        try:
            async with self._server:
                await self._server.serve_forever()
        finally:
            # Anything still buffered is spend the user expects to see.
            await self.drain()
            if self.writer is not None:
                log.info(
                    "accounting flushed: %d rows written, %d fell back to sync writes",
                    self.writer.flushed,
                    self.writer.dropped,
                )

    def shutdown(self) -> None:
        if self._server is not None:
            self._server.close()


def _json(obj: object) -> str:
    from .report import report_to_dict

    return json.dumps(report_to_dict(obj), ensure_ascii=False, default=str)


# --------------------------------------------------------------------------
# HTTP wire helpers
# --------------------------------------------------------------------------
def _fingerprint(token: str) -> str:
    import hashlib

    return hashlib.sha256(token.encode("utf-8")).hexdigest()[:10]


def _serialize_head(status: int, headers: Dict[str, str]) -> bytes:
    from http import HTTPStatus

    try:
        reason = HTTPStatus(status).phrase
    except ValueError:
        reason = "Unknown"
    lines = [f"HTTP/1.1 {status} {reason}"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    return ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")


def _serialize_request(
    method: str, target: str, headers: Dict[str, str], body: bytes
) -> bytes:
    lines = [f"{method} {target} HTTP/1.1"]
    for key, value in headers.items():
        lines.append(f"{key}: {value}")
    head = ("\r\n".join(lines) + "\r\n\r\n").encode("latin-1")
    return head + body


async def _read_headers(reader: asyncio.StreamReader) -> Dict[str, str]:
    headers: Dict[str, str] = {}
    while True:
        line = await reader.readline()
        if not line or line in (b"\r\n", b"\n"):
            break
        try:
            text = line.decode("latin-1").rstrip("\r\n")
        except UnicodeDecodeError:
            continue
        if ":" not in text:
            continue
        key, value = text.split(":", 1)
        headers[key.strip()] = value.strip()
    return headers


async def _read_body(reader: asyncio.StreamReader, headers: Dict[str, str]) -> bytes:
    lower = {k.lower(): v for k, v in headers.items()}
    if "chunked" in lower.get("transfer-encoding", "").lower():
        chunks: List[bytes] = []
        total = 0
        while True:
            size_line = await reader.readline()
            if not size_line:
                break
            try:
                size = int(size_line.split(b";")[0].strip() or b"0", 16)
            except ValueError:
                break
            if size == 0:
                await reader.readline()  # trailing CRLF
                break
            total += size
            if total > MAX_REQUEST_BODY:
                raise ValueError("request body too large")
            chunks.append(await reader.readexactly(size))
            await reader.readline()  # CRLF after chunk
        return b"".join(chunks)

    length = lower.get("content-length")
    if not length:
        return b""
    try:
        n = int(length)
    except ValueError:
        return b""
    if n <= 0:
        return b""
    if n > MAX_REQUEST_BODY:
        raise ValueError("request body too large")
    return await reader.readexactly(n)


async def _relay_until_eof(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter, content_length: Optional[str]
) -> Tuple[bytes, int]:
    """Forward the body; returns (captured bytes, bytes forwarded)."""
    captured = bytearray()
    forwarded = 0
    if content_length is not None:
        try:
            remaining = int(content_length)
        except ValueError:
            remaining = -1
        while remaining > 0:
            chunk = await reader.read(min(65536, remaining))
            if not chunk:
                break
            remaining -= len(chunk)
            forwarded += len(chunk)
            captured.extend(chunk)
            writer.write(chunk)
            await writer.drain()
    else:
        while True:
            chunk = await reader.read(65536)
            if not chunk:
                break
            forwarded += len(chunk)
            captured.extend(chunk)
            writer.write(chunk)
            await writer.drain()
    await writer.drain()
    return bytes(captured), forwarded


async def _relay_eof_with_budget(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    budget: "StreamBudget",
) -> Tuple[bytes, int, str]:
    """Forward an unbounded body while enforcing a mid-flight cap.

    Used for streamed responses that carry no Content-Length and no chunked
    framing -- the normal shape for SSE from both OpenAI and Anthropic. Bytes are
    relayed as they arrive, and a frame counter decides when to stop.
    """
    captured = bytearray()
    forwarded = 0
    abort_reason = ""
    decoder = _SseUsageDecoder()
    while True:
        chunk = await reader.read(16384)
        if not chunk:
            break
        captured.extend(chunk)
        forwarded += len(chunk)
        writer.write(chunk)
        await writer.drain()

        decoder.feed(chunk)
        budget.observe_content(decoder.take_content())
        reported = decoder.take_usage()
        if reported is not None:
            budget.observe_usage(reported)
        reason = budget.check()
        if reason:
            abort_reason = reason
            writer.write(budget.abort_frame())
            await writer.drain()
            log.warning("aborted streamed response: %s", reason)
            break
    return bytes(captured), forwarded, abort_reason

async def _relay_chunked(
    reader: asyncio.StreamReader,
    writer: asyncio.StreamWriter,
    budget: "Optional[StreamBudget]" = None,
) -> Tuple[bytes, int, str]:
    """Decode an upstream chunked body and forward it delimited by EOF.

    Rationale for EOF framing: re-emitting chunked while also sending
    ``Connection: close`` is ambiguous -- some clients trust the chunked framing,
    others read to EOF, and at least one (Python's test helper and several HTTP
    libraries) mis-frames the result. Since this connection closes anyway, the
    unambiguous encoding is "body, then EOF": strip both ``Content-Length`` and
    ``Transfer-Encoding`` and let the client read to end of stream.

    When ``budget`` is supplied the body is also inspected frame by frame and the
    stream is cut short -- with an explanatory SSE frame -- once a cap is crossed.
    Returns ``(captured, forwarded_bytes, abort_reason)``.
    """
    captured = bytearray()
    forwarded = 0
    abort_reason = ""
    decoder = _SseUsageDecoder() if budget is not None else None
    while True:
        size_line = await reader.readline()
        if not size_line:
            break
        try:
            size = int(size_line.split(b";")[0].strip() or b"0", 16)
        except ValueError:
            break
        if size == 0:
            # consume any trailers
            while True:
                trailer = await reader.readline()
                if not trailer or trailer in (b"\r\n", b"\n"):
                    break
            break
        body = await reader.readexactly(size)
        await reader.readline()  # trailing CRLF
        captured.extend(body)
        forwarded += len(body)
        writer.write(body)
        await writer.drain()

        if budget is not None and decoder is not None:
            decoder.feed(body)
            budget.observe_content(decoder.take_content())
            reported = decoder.take_usage()
            if reported is not None:
                budget.observe_usage(reported)
            reason = budget.check()
            if reason:
                abort_reason = reason
                # Explain the truncation in a shape the client already parses,
                # then stop reading upstream: continuing to drain a response we
                # have already decided not to pay for defeats the purpose.
                writer.write(budget.abort_frame())
                await writer.drain()
                log.warning("aborted streamed response: %s", reason)
                break
    return bytes(captured), forwarded, abort_reason


class _SseUsageDecoder:
    """Extracts streamed content text and usage objects from SSE chunks.

    Frames can split across TCP reads, so this buffers until a blank-line
    terminator before decoding. Without that, a frame straddling two reads is
    silently dropped and mid-stream accounting under-counts.
    """

    def __init__(self) -> None:
        self._buffer = ""
        self._content: List[str] = []
        self._usage: Optional[TokenUsage] = None

    def feed(self, chunk: bytes) -> None:
        self._buffer += chunk.decode("utf-8", "ignore")
        while "\n\n" in self._buffer:
            block, self._buffer = self._buffer.split("\n\n", 1)
            self._handle(block)

    def _handle(self, block: str) -> None:
        data_lines = [
            line[5:].strip() for line in block.splitlines() if line.startswith("data:")
        ]
        if not data_lines:
            return
        raw = "\n".join(data_lines)
        if raw in ("[DONE]", ""):
            return
        try:
            obj = json.loads(raw)
        except ValueError:
            return
        if not isinstance(obj, dict):
            return

        # OpenAI streaming deltas.
        for choice in obj.get("choices") or []:
            if isinstance(choice, dict):
                delta_block = choice.get("delta") or {}
                text = delta_block.get("content")
                if isinstance(text, str):
                    self._content.append(text)
        # Anthropic content_block_delta.
        delta = obj.get("delta")
        if isinstance(delta, dict) and isinstance(delta.get("text"), str):
            self._content.append(delta["text"])
        # Usage: OpenAI's final frame, or Anthropic's message_delta.
        usage = obj.get("usage")
        if isinstance(usage, dict):
            prompt = int(usage.get("prompt_tokens") or usage.get("input_tokens") or 0)
            completion = int(
                usage.get("completion_tokens") or usage.get("output_tokens") or 0
            )
            cached = 0
            details = usage.get("prompt_tokens_details")
            if isinstance(details, dict):
                cached = int(details.get("cached_tokens") or 0)
            if prompt or completion:
                self._usage = TokenUsage(
                    input=max(prompt - cached, 0),
                    output=completion,
                    cached_input=cached,
                )

    def take_content(self) -> str:
        text = "".join(self._content)
        self._content = []
        return text

    def take_usage(self) -> Optional[TokenUsage]:
        usage, self._usage = self._usage, None
        return usage


async def _write_json(writer: asyncio.StreamWriter, status: int, payload: dict) -> None:
    body = json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")
    head = _serialize_head(
        status,
        {
            "Content-Type": "application/json; charset=utf-8",
            "Content-Length": str(len(body)),
            "Connection": "close",
        },
    )
    writer.write(head + body)
    await writer.drain()
