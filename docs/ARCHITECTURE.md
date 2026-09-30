# Architecture and design decisions

This document records *why* the code looks the way it does. It exists so a future
maintainer (or a customer's platform engineer evaluating the tool) can judge the trade-offs
without reverse-engineering them.

---

## 1. The core design constraint: zero runtime dependencies

**Decision.** The gateway imports nothing outside the Python standard library.

**Why this is a product feature, not a stylistic preference.** The buyer profile for this
tool is a platform or infrastructure engineer who has to run it inside a company VPC, often
in a regulated environment. Every third-party dependency is:

- something their security review has to approve,
- something that can be compromised upstream (the supply-chain surface of a proxy that sees
  every prompt is unusually sensitive),
- something that can break their build when `pip install` reaches PyPI at an inconvenient
  moment,
- one more reason to say no.

`FastAPI + uvicorn + SQLAlchemy + Alembic` would have been faster to write and would have
made the tool harder to adopt. The concrete comparison is: `python3 -m llmguard serve` works
on a bare `python:3.12-slim` image with no install step, versus a requirements file, a lock
file, and a vulnerability scan.

**What it costs.** HTTP is hand-rolled on `asyncio.start_server`: request line parsing, header
folding, chunked decoding, and response framing. That is roughly 200 lines of code in
`gateway.py` that a framework would have provided, and it is the part of the codebase with
the most test coverage as a result.

## 2. Token accounting: read the bill, never estimate it

**Decision.** Usage comes exclusively from the `usage` object the provider returns.

**Why.** A local tokenizer (`tiktoken`-style) would introduce a dependency, would drift from
the provider's actual tokenization, and would be *plausibly* wrong. A cost report that is
plausibly wrong is worse than one that admits ignorance: an engineer will act on it.

**Consequences that shaped the code:**

- Unpriced models produce `cost_usd = NULL`, and the report surfaces them explicitly
  ("N models have no price entry"). The alternative — defaulting to zero — would silently
  under-report spend.
- Streaming responses must be captured to read the final usage frame. OpenAI only sends usage
  when the client sets `stream_options: {"include_usage": true}`; Anthropic splits it across
  `message_start` and `message_delta`. `parsers.py` merges both shapes.
- Cached tokens are reported *inside* `prompt_tokens` by OpenAI. Billing them at the full
  input rate overstates cost on cache-heavy traffic by up to 10x on the cached portion, so
  they are subtracted before pricing.

## 3. Cost model: four token classes, not one blended rate

Input, cached input, cache write and output are priced separately. On both OpenAI and
Anthropic a cache read costs ~10% of input and a cache write costs ~125%, so a single
"cost per token" would be wrong for exactly the workloads this tool is most useful for.

Costs use `decimal.Decimal`, quantized to 6 decimal places. Reason: individual calls cost
fractions of a cent, and summing thousands of binary floats accumulates visible error in a
number whose entire job is to be trusted.

## 4. Write-behind accounting, and why it does not lose data

**Problem.** Committing one SQLite transaction per request dominated the latency budget:
5.05 ms mean overhead, 3,984 req/s.

**Decision.** Buffer records in memory and flush in batches (timer + size trigger).
Result: 4,346 ms→**~5 ms** overhead with batching at a 0.25 s interval, and meaningfully
better throughput under load.

**The risk this introduces, and how it is contained.** Buffering spend data means a crash can
lose data — unacceptable for a cost tool. Mitigations, all tested in
`tests/test_write_behind.py`:

- a bounded queue (20k records); when saturated, records fall back to a **synchronous** write
  rather than being dropped. Latency degrades before data does,
- flush on a 0.25 s timer, on batch size, and on an explicit `drain()`,
- `serve_forever()` awaits `drain()` in a `finally` block, so SIGINT/SIGTERM flush,
- `drain()` is deliberately **loop-independent**: it cancels the flusher task without awaiting
  it, then empties the queue synchronously. Awaiting a task owned by another event loop
  (tests, embedding, shutdown hooks) hangs forever, and this was worth designing around
  rather than debugging later.

**Exposure window:** up to 0.25 s of accounting on a hard `kill -9`. Set
`write_behind=False` in `GatewayConfig` if you need per-request durability.

## 5. Streaming passthrough

Streamed responses are forwarded to the client as they arrive, while a copy is accumulated
for usage parsing. The client sees no added time-to-first-token.

Chunked upstream bodies are **decoded and re-delimited by connection close** rather than
re-emitted as chunked. Re-emitting chunked *and* sending `Connection: close` is ambiguous:
some clients honour the chunked framing, others read to EOF, and the result mis-frames. Since
the connection closes anyway, "body then EOF" with no `Content-Length` and no
`Transfer-Encoding` is the only unambiguous encoding.

## 6. Attribution without becoming a data processor

The gateway records *who* and *how much*, never *what*. Prompt and completion content is
never persisted.

Callers are identified by a SHA-256 fingerprint of their credential (never the credential
itself), or by a human label via `--key-map`. End users come from an explicit header or the
standard OpenAI `user` field. This keeps the operator out of the PII path, which is the
difference between a tool a legal team approves and one it does not.

## 7. Runaway-spend detection, and why it is advisory

**The problem budgets do not solve.** A per-key budget catches totals. The failure
mode that produces five-figure bills is a *single chain* looping, retrying, or
fanning out — and by the time a daily cap trips, the money is spent. OWASP's
AISVS C9.1 chapter catalogues 63 confirmed production overrun incidents across 21
frameworks, including a $47K multi-agent loop and a 2.3M-call retry storm.

**Three detectors, each keyed to a documented signature:**

| Detector | Signal | Why that signal |
|---|---|---|
| `io_ratio` | Sustained input/output token ratio above 30:1 | An unbounded-context loop re-sends its whole history every step. Normal traffic runs 5:1–15:1; the documented Claude Code incident hit 74:1 and 175:1. OWASP calls this "a cheap way to surface this exact failure". |
| `velocity` | Spend rate above a *multiple of the account's own baseline* | A fixed dollar threshold would be wrong for everyone. But the baseline must **exclude the detection window** — otherwise a large enough spike raises its own baseline and the test never fires. That bug existed here and was caught by a test. |
| `retry_storm` | Many failing calls in a short window | Retry storms are the second-most-costly documented pattern and still consume latency and tokens. |

**Why advisory, never blocking.** A detector that silently killed production
traffic would be a worse failure than the overspend it prevents. The codebase
enforces this separation explicitly: `detectors.py` only ever *returns* findings;
only a budget with `action=block` or an explicit stream cap stops anything.

**Median, not mean.** `LiveGuard` uses the median ratio over a rolling window, so
one legitimate document-analysis call cannot trip the detector on its own.

## 8. Mid-stream cancellation

**The gap this closes.** A pre-request budget check cannot stop the request it is
checking. A per-key budget check happens *between* requests. Neither can stop the
streaming request that is crossing the line right now — and for a long agent chain
that is precisely the request that matters. (TrueFoundry's published three-layer
gateway pattern identifies the same gap.)

**How it works.** When `stream_max_cost_usd` or `stream_max_tokens` is set, the
relay inspects each SSE frame, accumulates an estimate, and on breach writes an
error frame shaped like a provider error, then `[DONE]`, then stops reading
upstream. Stopping the read matters: continuing to drain a response we have already
decided not to pay for defeats the point.

**The honest limitation.** Both OpenAI and Anthropic report authoritative usage
only at the end of a stream, so mid-flight enforcement necessarily acts on a
**local estimate**. It is deliberately conservative (over-counts at ~4 chars/token)
so the cap trips early rather than late, and it is never used for billing —
`_SseUsageDecoder` replaces the estimate the moment a real usage frame arrives.
This is why the feature is off by default: truncating a stream is a new failure
mode, and it should be an explicit choice.

**One bug worth recording.** The dispatch originally consulted `chunked` before
`content_length is None`. The common real-world SSE shape is *no Content-Length and
no chunked encoding* — just an upstream that closes the connection — so the cap was
never consulted and streams ran to completion. The end-to-end test caught it because
the mock upstream streams the way real providers do.

## 9. What is deliberately not implemented

| Not implemented | Why, and what to do instead |
|---|---|
| **Inbound** TLS termination | Outbound TLS to providers is done in-process (stdlib `ssl`, verified against the live API). *Inbound* termination is left out on purpose: a load balancer or Caddy does it better, and doing it here would mean shipping certificate renewal. |
| HTTP/2 upstream | No benefit for JSON request/response at these volumes; doubles the framing code. |
| Response caching | Changes semantics (a cache hit is not a model call). Belongs in the app or a dedicated cache, not silently in the metering path. |
| Retries | A proxy that retries can double-bill. Retry policy belongs to the caller. |
| Multi-tenant auth | This is infrastructure for one organisation. For SaaS, put it behind your own auth layer. |

## 10. Testing strategy

The suite tests the two things that actually break:

1. **Billing invariants** (`test_pricing.py`) — cache discounts, cache-write premiums,
   datestamped model aliases, clamping, and the requirement that an unknown model returns
   `None`, never a number.
2. **The wire** (`test_gateway_e2e.py`) — a real mock upstream on loopback, real HTTP through
   the socket, asserting forwarded bytes, recorded cost, chunked reframing, streamed usage,
   budget blocking, and that credentials are not stored in plaintext.

Three bugs found by these tests are worth recording, because they are the class of bug that
would have shipped silently:

- **Timestamp format.** Rows were written as `2026-09-30T11:04:06+00:00` while SQLite's
  `datetime('now')` emits `2026-09-30 11:04:06`. Every window filter compared those
  textually and matched nothing, so *all traffic was invisible to the report*. Fix: store
  SQLite-compatible UTC strings.
- **Window upper bound.** `ts < datetime('now')` excluded any row written in the current
  second — precisely when a user first looks at a fresh deployment. Fix: day-aligned bounds.
- **Alias resolution.** Longest-prefix matching returned keys that were not in the price
  table, so correctly-priced models were reported as unpriced. Fix: build match candidates
  only from resolvable keys.
- **Self-defeating baseline.** The velocity detector computed its baseline over a window that
  *included* the burst being detected, so a large spike raised the average enough that the
  ratio never fired. Fix: exclude the detection window from the baseline (plus a regression
  test).
- **Relay dispatch ordering.** Mid-stream caps were bypassed for unbounded, non-chunked
  responses — the standard SSE shape. Fix: check `content_length is None` before the plain-EOF
  path, and add `_relay_eof_with_budget`.
- **No outbound TLS at all.** The first implementation refused every `https://` route with
  "requires TLS termination in front of the gateway", which meant the gateway could not talk
  to the real provider APIs it exists to sit in front of — a fundamental gap that the unit
  tests could not see because they used an HTTP mock. It surfaced the first time a script
  tried a live call. Fix: in-process TLS via the stdlib `ssl` module, verified against
  `api.openai.com` (a 401 from the real API is the success signal).

## 11. Extension points

- **Add a model:** one entry in `PRICES` (`pricing.py`) plus `VERIFIED_AT` if you re-check
  the whole table.
- **Add a provider:** a `Route` in `DEFAULT_ROUTES` and, if its usage shape differs, a branch
  in `parse_buffered` / `parse_streamed`.
- **Change the output:** `report.py` holds the action engine; `analytics.py` holds the
  numbers. They are separate so the "what to do" logic can be tested against fixed metrics.
- **Different storage:** `storage.py` exposes `insert` / `query`; the SQL is vanilla SQLite
  and the schema maps cleanly onto ClickHouse or Postgres with a column rename.
