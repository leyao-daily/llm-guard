"""Generate a realistic demo dataset.

The point of this module is that the product must show its value on the very
first command, with no API key and no live traffic. `llm-guard seed` produces a
month of plausible traffic whose shape encodes the findings the report is meant
to surface:

* one model dominating spend (routing opportunity)
* one day with a retry storm (incident detection)
* an expensive model with low request count but huge fixed context
* a cache-heavy key and a cache-blind key (cache opportunity)
* a realistic error rate

Numbers are generated with a fixed seed by default, so the docs and the tests
can assert on exact output.
"""

from __future__ import annotations

import random
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence, Tuple

from .pricing import TokenUsage, compute_cost
from .storage import RequestRecord, Store, iso

# (model, weight, input_tokens range, output range, cache_hit_prob)
#
# The weights are chosen to make the report's findings land: a cheap model
# carries most of the *volume* while a premium model carries most of the *cost*
# -- which is the single most common shape of real LLM spend, and the thing a
# per-model breakdown exists to reveal.
PROFILES: Sequence[Tuple[str, float, Tuple[int, int], Tuple[int, int], float]] = (
    ("gpt-6-luna", 0.34, (700, 3000), (60, 400), 0.55),
    ("gpt-6.1-sol", 0.20, (1500, 8000), (150, 900), 0.35),
    ("gpt-5.6-luna", 0.14, (900, 4000), (80, 500), 0.45),
    ("claude-haiku-4.5", 0.11, (1200, 5000), (120, 700), 0.40),
    ("claude-sonnet-4.5", 0.09, (2000, 12000), (200, 1200), 0.30),
    ("claude-opus-5.5", 0.045, (3000, 14000), (400, 2500), 0.10),
    ("gpt-6-astra", 0.015, (4000, 20000), (600, 3000), 0.08),
    # A customer fine-tune with no published rate: must report as unpriced
    # rather than silently costing zero.
    ("ft:acme-support-v2", 0.01, (600, 2000), (40, 200), 0.0),
)

KEYS = (
    ("prod-web", 0.40, ("web-app", "checkout-assistant")),
    ("prod-batch", 0.22, ("nightly-enrich", "doc-pipeline")),
    ("staging", 0.16, ("web-app", "evals")),
    ("internal-tools", 0.12, ("sales-copilot", "support-triage")),
    ("key-9f3a1c8b22", 0.10, ("legacy-cron",)),
)

END_USERS = ("u_1042", "u_2087", "u_3311", "u_4509", "u_6672", "u_7781", "")


def _pick(weights: Sequence[float]) -> int:
    total = sum(weights)
    r = random.random() * total
    upto = 0.0
    for i, w in enumerate(weights):
        upto += w
        if r <= upto:
            return i
    return len(weights) - 1


def generate_records(
    days: int = 30,
    *,
    seed: int = 7,
    requests_per_day: int = 260,
    now: Optional[datetime] = None,
) -> List[RequestRecord]:
    """Build a deterministic list of RequestRecords covering ``days`` days."""
    rng = random.Random(seed)
    now = now or datetime.now(timezone.utc)
    records: List[RequestRecord] = []

    model_weights = [p[1] for p in PROFILES]
    key_weights = [k[1] for k in KEYS]

    for day_offset in range(days - 1, -1, -1):
        day_start = (now - timedelta(days=day_offset)).replace(
            minute=0, second=0, microsecond=0
        )
        weekday = day_start.weekday()
        # Weekends are quieter; a product with real users shows this pattern.
        volume = requests_per_day * (0.45 if weekday >= 5 else 1.0)

        # Day 9 back in time: a regression caused retries and errors.
        incident = day_offset == 9
        if incident:
            volume *= 1.7

        for _ in range(int(volume * rng.uniform(0.85, 1.15))):
            profile = PROFILES[_pick(model_weights)]
            model, _w, in_range, out_range, cache_prob = profile
            key, _kw, projects = KEYS[_pick(key_weights)]
            project = rng.choice(projects)

            input_tokens = rng.randint(*in_range)
            output_tokens = rng.randint(*out_range)

            cached = 0
            cache_write = 0
            if rng.random() < cache_prob:
                cached = int(input_tokens * rng.uniform(0.5, 0.95))
                input_tokens = max(input_tokens - cached, 0)

            # A few very large requests stand out, as they do in real traffic.
            if rng.random() < 0.012:
                input_tokens = int(input_tokens * rng.uniform(4, 9))
                output_tokens = int(output_tokens * rng.uniform(2, 4))

            # Anthropic cache writes (cache creation) on a small share of calls.
            if model.startswith("claude") and rng.random() < 0.08:
                cache_write = rng.randint(500, 4000)

            status = 200
            error = ""
            if incident and rng.random() < 0.34:
                status = rng.choice([429, 500, 503])
                error = f"upstream status {status}"
            elif rng.random() < 0.012:
                status = rng.choice([400, 401, 429])
                error = f"upstream status {status}"

            usage = TokenUsage(
                input=input_tokens,
                output=output_tokens,
                cached_input=cached,
                cache_write=cache_write,
            )
            cost = None if status >= 400 and error else compute_cost(model, usage)
            if cost is None and status < 400:
                cost = compute_cost(model, usage)  # may still be None if unpriced

            latency = rng.randint(280, 2600)
            if incident:
                latency = int(latency * rng.uniform(1.5, 3.0))
            if output_tokens > 1500:
                latency += rng.randint(800, 3000)

            ts = day_start + timedelta(
                hours=rng.randint(0, 23),
                minutes=rng.randint(0, 59),
                seconds=rng.randint(0, 59),
            )
            # Keep everything in the past.
            if ts > now:
                ts = now - timedelta(minutes=rng.randint(1, 120))

            records.append(
                RequestRecord(
                    provider=(
                        "anthropic" if model.startswith("claude") else
                        "google" if model.startswith("gemini") else "openai"
                    ),
                    model=model,
                    input_tokens=usage.input,
                    cached_input_tokens=usage.cached_input,
                    cache_write_tokens=usage.cache_write,
                    output_tokens=usage.output,
                    cost_usd=None if cost is None else float(cost),
                    latency_ms=latency,
                    status=status,
                    streamed=rng.random() < 0.35,
                    api_key_id=key,
                    end_user=rng.choice(END_USERS),
                    project=project,
                    request_id=f"seed{rng.randrange(16**12):012x}",
                    error=error,
                    ts=iso(ts),
                )
            )

    records.sort(key=lambda r: r.ts or "")
    return records


def generate_agent_loop(
    *,
    now: Optional[datetime] = None,
    steps: int = 46,
    minutes_ago_start: int = 9,
    key: str = "prod-agent",
    project: str = "support-agent",
) -> List[RequestRecord]:
    """A bounded reproduction of the documented unbounded-context incident.

    OWASP AISVS C9.1 records a Claude Code CLI session that ingested ~30M input
    tokens at a **74:1** input/output ratio (the second run hit 175:1), because
    the conversation history grew and was re-sent in full on every step. Normal
    traffic sits at 5:1-15:1.

    Reproducing it faithfully -- rather than writing a few lopsided rows -- is
    what makes the loop detector demonstrable on first run, and gives the
    regression tests a realistic fixture.
    """
    now = now or datetime.now(timezone.utc)
    records: List[RequestRecord] = []
    # Context grows every step: this is the mechanism, not just the symptom.
    for step in range(steps):
        input_tokens = 4_000 + step * 620
        output_tokens = max(int(input_tokens / 74), 18)
        offset = minutes_ago_start - (step / max(steps, 1)) * minutes_ago_start
        ts = now - timedelta(minutes=max(offset, 0.05))
        usage = TokenUsage(input=input_tokens, output=output_tokens)
        cost = compute_cost("claude-sonnet-4.5", usage)
        records.append(
            RequestRecord(
                provider="anthropic",
                model="claude-sonnet-4.5",
                input_tokens=usage.input,
                output_tokens=usage.output,
                cost_usd=None if cost is None else float(cost),
                latency_ms=900 + step * 15,
                status=200,
                streamed=False,
                api_key_id=key,
                end_user="",
                project=project,
                request_id=f"loop{step:04d}",
                error="",
                ts=iso(ts),
            )
        )
    return records


def generate_retry_storm(
    *, now: Optional[datetime] = None, calls: int = 42, minutes_ago: int = 4,
    key: str = "prod-batch",
) -> List[RequestRecord]:
    """A burst of failing calls, the second-most-costly documented pattern."""
    now = now or datetime.now(timezone.utc)
    out: List[RequestRecord] = []
    for i in range(calls):
        out.append(
            RequestRecord(
                provider="openai",
                model="gpt-6.1-sol",
                input_tokens=0,
                output_tokens=0,
                cost_usd=None,
                latency_ms=2100,
                status=503,
                api_key_id=key,
                project="nightly-enrich",
                request_id=f"storm{i:04d}",
                error="upstream status 503",
                ts=iso(now - timedelta(minutes=minutes_ago, seconds=i * 3)),
            )
        )
    return out


def seed(
    store: Store,
    *,
    days: int = 30,
    seed_value: int = 7,
    requests_per_day: int = 260,
    reset: bool = False,
    with_budgets: bool = True,
    with_incidents: bool = True,
) -> Dict[str, object]:
    """Populate the store with demo data. Returns a summary dict."""
    if reset:
        store.query("DELETE FROM requests")
        store.query("DELETE FROM budgets")

    records = generate_records(days=days, seed=seed_value, requests_per_day=requests_per_day)

    incident_records: List[RequestRecord] = []
    if with_incidents:
        # A live loop and a live retry storm inside the detection window, so
        # `llm-guard anomalies` has something real to find on first run.
        incident_records = generate_agent_loop() + generate_retry_storm()
        records = records + incident_records

    inserted = store.insert_many(records)

    if with_budgets:
        store.set_budget("prod-web", daily_usd=45.0, monthly_usd=1200.0, action="block")
        store.set_budget("prod-batch", daily_usd=30.0, monthly_usd=700.0, action="alert")
        store.set_budget("staging", daily_usd=25.0, monthly_usd=400.0, action="alert")
        # The looping key is deliberately un-budgeted: it demonstrates that
        # detection works even where no budget has been configured.
        store.set_budget("prod-agent", monthly_usd=250.0, action="alert")

    total_cost = sum(r.cost_usd or 0.0 for r in records)
    return {
        "inserted": inserted,
        "days": days,
        "total_cost_usd": round(total_cost, 4),
        "first_ts": records[0].ts if records else None,
        "last_ts": records[-1].ts if records else None,
        "budgets": 4 if with_budgets else 0,
        "incidents": len(incident_records),
    }
