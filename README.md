# llm-guard


[![tests](https://github.com/leyao-daily/llm-guard/actions/workflows/tests.yml/badge.svg)](https://github.com/leyao-daily/llm-guard/actions/workflows/tests.yml)
[![Python 3.9+](https://img.shields.io/badge/python-3.9%2B-blue.svg)](https://www.python.org/downloads/)
[![zero dependencies](https://img.shields.io/badge/runtime%20dependencies-0-brightgreen.svg)](#why-this-exists)
[![License: MIT](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

**Your agent loop is spending money right now. This stops it.**

A self-hosted gateway that sits in front of your OpenAI / Anthropic / Gemini calls
and does two things a dashboard cannot:

1. **Detects runaway spend while it is still happening.** A sustained pathological
   input/output token ratio (an unbounded-context agent loop re-sending its whole
   history every step), a spend burst far above your own baseline, or a retry
   storm. Normal traffic runs 5:1–15:1; the documented production incident hit
   **74:1 and 175:1**.
2. **Stops it.** Per-key budgets return `429`; optional per-stream caps **abort a
   single streaming response mid-flight** — the request that crosses the line is
   precisely the one a between-requests budget check cannot stop.

And does the accounting underneath properly: cost from the provider's own usage
block, attributed to key / project / end user.

```text
CRITICAL io_ratio  [key: prod-agent]
   prod-agent is running a 74:1 input-to-output ratio across 46 calls in the last
   15 min (825,700 in / 11,136 out, $2.64). Normal traffic sits at 5:1-15:1.
   -> The prompt is being re-sent in full on every step. Cap the context, and
      enable prompt caching so the repeated prefix is billed at the cache rate.
```

## Why this exists

OWASP's AISVS chapter on execution budgets (C9.1, July 2026) catalogues **63
confirmed production budget-overrun incidents** across 21 orchestration
frameworks: a $47K multi-agent loop, a $47K retry storm from 2.3M erroneous calls,
a $1.2M GPU hijack, Uber exhausting its full 2026 AI budget in four months. It
reports Fortune 500 leakage of roughly **$400M** in unbudgeted spend, attributes
**62% of agent bills to re-sent context**, and states plainly that **no major agent
framework ships comprehensive cost ceilings** — enforcement has to live in a
gateway.

Existing gateways are either heavy to self-host (Langfuse needs ClickHouse),
send your data to someone else's cloud, do attribution without enforcement, or —
in LiteLLM's case — *are themselves the attack surface*: the March 2026 PyPI
supply-chain compromise, plus two CVEs added to CISA's known-exploited list.

So this one is deliberately boring underneath:

- **Zero runtime dependencies.** Standard library only. No `pip install`, no venv,
  no supply chain to audit, no base image to trust. Auditable in one sitting.
- **Cost from billing truth, never a guess.** Reads the `usage` block the provider
  returned. Models it cannot price are reported as *unknown*, not estimated —
  a plausibly-wrong cost number is worse than no number.
- **Cache multipliers are per-model**, not a constant (0.025x on Claude Fable 5.1,
  0.5x on gpt-4o). Getting this wrong is a silent 4–5x error.
- **No data leaves your network.** No telemetry, no CDN, no phone-home. The
  dashboard is server-rendered SVG.
- **Inbound TLS termination is left out on purpose** — a load balancer does it
  better. Outbound TLS to providers is handled in-process and verified against the
  live API.

---

## See it work in 30 seconds (no API key required)

```bash
cd mvp/llm-guard
python3 -m llmguard seed --reset --compare-days 30   # a month of realistic traffic
python3 -m llmguard report
```

That prints a spend report with prioritised actions. Then:

```bash
python3 -m llmguard anomalies                    # runaway agent spend, right now
python3 -m llmguard diagnose --client "Acme"     # the written report, print to PDF
python3 -m llmguard dashboard --out dash.html   # self-contained HTML, open in any browser
python3 -m llmguard cost gpt-6.1-sol --input 50000 --output 2000 --count 1000
python3 -m llmguard models                       # the built-in price table
```

The demo dataset deliberately contains a live unbounded-context loop (46 steps at **74:1**)
and a retry storm, so `anomalies` has something real to find on the first run:

```text
CRITICAL io_ratio  [key: prod-agent]
   prod-agent is running a 74:1 input-to-output ratio across 46 calls in the last 15
   min (825,700 in / 11,136 out, $2.64). Normal traffic sits at 5:1-15:1.
   → The prompt is being re-sent in full on every step. Cap the context (summarise
     or truncate history), and enable prompt caching so the repeated prefix is
     billed at the cache rate. OWASP attributes ~62% of agent bills to re-sent context.
```

Requires **Python 3.9+**. Nothing else.

---

## Put it in front of real traffic

```bash
# forward your own key upstream (normal local-dev setup)
python3 -m llmguard serve --openai-key "$OPENAI_API_KEY"

# then point your SDK at it
export OPENAI_BASE_URL=http://127.0.0.1:8787/v1
export ANTHROPIC_BASE_URL=http://127.0.0.1:8787/anthropic
```

Your application code does not change. With Docker:

```bash
docker compose up -d
# live dashboard:  http://127.0.0.1:8787/-/dashboard
# JSON stats:      http://127.0.0.1:8787/-/stats
# health:          http://127.0.0.1:8787/-/health
```

### Attribute spend to something meaningful

Cost per API key is table stakes. The useful dimensions are the ones you send:

```bash
# stable labels instead of key fingerprints
python3 -m llmguard serve --key-map "sk-team-a-xxx=payments-team"

# or per request, from your app
curl http://127.0.0.1:8787/v1/chat/completions \
  -H "Authorization: Bearer $OPENAI_API_KEY" \
  -H "x-project: checkout-assistant" \
  -H "x-end-user: u_1042" \
  -d '{"model":"gpt-4o","messages":[{"role":"user","content":"hi"}]}'
```

An end user is also picked up from the standard `user` field in the request body, so
per-customer unit economics work without changing a line of application code.

### Enforce a budget

```bash
python3 -m llmguard budget set payments-team --daily 40 --monthly 800 --on-exceed block
python3 -m llmguard budget list
```

`--on-exceed block` makes the gateway return `429` once the limit is hit. `alert` records the
breach and lets traffic through.

### Cut off a runaway stream

```bash
# abort any single streaming response that crosses ~$0.50 or ~250k tokens
python3 -m llmguard serve --openai-key "$OPENAI_API_KEY" \
    --stream-max-cost 0.50 --stream-max-tokens 250000
```

The client receives an SSE frame with `"type": "budget_exceeded"` and an `llm_guard` block
showing the estimate, followed by `[DONE]` — so existing clients surface the truncation
instead of treating it as a successful completion. Mid-stream figures are a **local estimate**
(both providers report authoritative usage at the end); billing always uses provider numbers.

### Detection is advisory; only budgets block

Detectors flag, they never silently kill traffic. A per-key budget with `action=block`, or an
explicit `--stream-max-*` cap, are the only things that stop a request. Detection lives at
`GET /-/anomalies` and at the top of `llm-guard report`.

---

## What the report tells you

The output is written as advice, not as a dashboard. Each finding is derived and ordered:

| Finding | What it means |
|---|---|
| *"One model is 43% of your bill"* | Routing the easy traffic to a cheaper tier is your biggest lever |
| *"claude-3-opus costs 20x your average per request"* | That model carries heavy fixed context — trim it and every call gets cheaper |
| *"Prompt caching is worth up to $175/month"* | Your cache hit rate is low; repeated prefixes are the usual cause |
| *"payments-team is projected over budget"* | Fix the budget, or fix the feature, before month-end |
| *"3 models have no price entry"* | Your headline number is under-reporting; one line fixes it |

## Getting a prospect's data in, before they commit to anything

The intake step decides whether any of this is sellable. If the answer is "deploy
our gateway first", the conversation ends, because nobody deploys infrastructure
to find out whether a diagnosis is worth paying for.

Three ways in, least effort first:

```bash
# 1. straight from the provider's own admin API. One read-only key, no deployment.
python3 -m llmguard import --source anthropic --key "$ANTHROPIC_ADMIN_KEY" --days 30
python3 -m llmguard import --source openai    --key "$OPENAI_ADMIN_KEY"    --days 30

# 2. a file, for anyone who would rather not hand over a key
python3 -m llmguard import --source file --path usage.csv
python3 -m llmguard import --sample      # prints the expected columns

# 3. your own gateway database, if they already run it
```

Column names are matched loosely, so a straight export from a provider dashboard
usually works without editing: `date`/`timestamp`/`start_time` all map to
`bucket_start`, `prompt_tokens` to `input_tokens`, and so on. If a row carries
what the provider actually billed in `cost_usd`, that figure replaces our own
arithmetic. It should: the provider's number outranks our price table, always.

**Provider data is aggregated, and the diagnosis adapts rather than pretending
otherwise.** You get daily or hourly buckets, so there is no latency, no status
code and no end-user dimension. The checks that need per-request rows are skipped,
the volume thresholds scale down (six daily buckets is a week of history; six
requests is nothing), and the report says outright that it is looking at buckets.
A context loop is still findable, because the shape is the point.

## The written diagnosis

`report` is for you. `diagnose` is the thing you send to somebody else.

```bash
python3 -m llmguard diagnose --days 30 --client "Acme Corp" --out diagnosis.html
# or: --format text   (same content, terminal)
# or: --format json   (for your own tooling)
```

It runs seven checks and writes a document with a verdict, a ranked list of what
to fix, the evidence behind every number, and an explicit section on what the
analysis cannot tell you. Print it to PDF from a browser; the layout is set for
A4 with page breaks.

Each finding carries an estimate of what it is worth per month and a confidence
level, because some of these are arithmetic on your own data and some are a
judgement about your architecture:

| Label | Means |
|---|---|
| **Certain** | Arithmetic on your data. The number is what it is. |
| **Likely** | Depends on one stated assumption, written next to it. |
| **Worth testing** | A hypothesis. Worth a bounded experiment, not a committed budget. |

The estimates are deliberately conservative and the report says outright that
they overlap: trimming context and improving cache hit rates act on the same
tokens, so they cannot both be collected in full. A diagnosis that overstates
savings gets found out on the next invoice, and then nothing else in it is
believed either.

## Measured overhead

Benchmarked on an Apple-silicon laptop against a **zero-latency local upstream** — the worst
possible case for a ratio, since real LLM calls take 400–4000 ms:

| Scenario | req/s | mean | p95 |
|---|---|---|---|
| Direct to upstream (no gateway) | 10,136 | 2.87 ms | 3.20 ms |
| Through llm-guard | 3,688 | 8.12 ms | 11.81 ms |
| Through llm-guard (SSE streaming) | 3,851 | 7.70 ms | 8.33 ms |

**Added latency: ~5 ms per request**, of which most is the extra loopback hop and the
per-request `sqlite3` write. Streaming responses are metered correctly, including
`usage` delivered in the final SSE frame. Full numbers and method: `benchmarks/results.json`,
`benchmarks/bench.py`.

## How it works

```
your app ──HTTP──▶ llm-guard ──HTTP──▶ api.openai.com / api.anthropic.com
                      │
                      ├─ reads the usage block from the response
                      ├─ prices it from the tables in pricing.py
                      ├─ checks the per-key budget (may return 429)
                      └─ appends one row to SQLite (batched)
```

| Module | Responsibility |
|---|---|
| `llmguard/gateway.py` | HTTP/1.1 reverse proxy, streaming passthrough, budget guard, `/-/` API |
| `llmguard/parsers.py` | Extracts usage from OpenAI/Anthropic payloads: buffered **and** SSE |
| `llmguard/pricing.py` | Price table + cost maths (cached reads, cache writes, aliases) |
| `llmguard/storage.py` | SQLite schema, batched writes |
| `llmguard/analytics.py` | Aggregations and the derived value metrics |
| `llmguard/detectors.py` | Runaway-loop, velocity and retry-storm detection |
| `llmguard/report.py` | Terminal / JSON / CSV rendering and the action engine |
| `llmguard/dashboard.py` | Self-contained HTML + inline SVG |
| `llmguard/demo.py` | Deterministic demo dataset, including a reproducible agent loop |
| `tools/capture_fixtures.py` | Verifies the parsers against real provider payloads |

Design decisions and their trade-offs: **`docs/ARCHITECTURE.md`**.

## Tests

```bash
python3 -m unittest discover -s tests -v      # 115 tests, no network, no API keys
python3 benchmarks/bench.py --requests 3000 --concurrency 32

# Verify the parsers against what the providers ACTUALLY return today.
# Four calls, well under a cent. This is the check that catches a provider
# quietly changing its payload shape.
export OPENAI_API_KEY=sk-...   ANTHROPIC_API_KEY=sk-ant-...
python3 tools/capture_fixtures.py
```

The suite includes full end-to-end proxy tests: a real mock upstream on loopback, real HTTP
through the socket, asserting forwarded bytes, recorded cost, budget blocking, that streamed
usage is priced, and that a runaway stream is **actually truncated on the wire**. It also pins
the accounting invariants that are expensive to get wrong — cached tokens are never
double-billed, credentials are never stored in plaintext, and buffered writes cannot be lost
on shutdown.

## Limitations (read before trusting it)

- **No inbound TLS termination.** Outbound TLS to providers is handled in-process with the
  stdlib `ssl` module (verified against the live `api.openai.com`). Inbound, run it behind
  nginx/Caddy or on a private network — terminating inbound TLS is left out on purpose, since
  that is what a load balancer is for. Behind a corporate MITM proxy, pass `--upstream-ca`.
- **HTTP/1.1 only.** No HTTP/2 upstream; this is fine for the JSON APIs targeted here.
- **SQLite.** Comfortable to a few million rows per host. Beyond that, ship rows to
  ClickHouse — the schema is portable.
- **Prices go stale.** `pricing.py` carries a `VERIFIED_AT` date. Providers change rates;
  re-check quarterly.
- **Detection thresholds are heuristics.** The 30:1 ratio is drawn from the documented
  incident set, not from a global survey. Tune with `--io-ratio` if your workload legitimately
  runs long-input/short-output (batch classification, embeddings-style workloads).
- **Mid-stream limits are estimates.** Both major providers report authoritative usage only at
  the end of a stream, so a mid-flight cap necessarily acts on a local token estimate. It
  over-counts slightly on purpose.
- **Not a data processor.** Prompts and completions are never persisted; only token counts,
  cost, latency and status.

## License

MIT. Copyright (c) 2026 **LYE LABS LIMITED (灵野科技有限公司)**.

You may use, modify and self-host this commercially without asking. If you need a
different licence for a corporate policy reason, open an issue.
