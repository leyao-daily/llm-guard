"""The published failure taxonomy, as detectors.

Why this module exists
----------------------
A diagnosis that says "our experience suggests your context is too long" is a
sales pitch. A diagnosis that says "this matches a catalogued production failure
pattern, here is the observable signal, and here is what it cost in the recorded
incidents" is evidence.

The taxonomy is not ours. It comes from two public sources, and both are cited in
the output so a client can check the claim rather than trust us:

  * **OWASP AISVS C9.1** — Execution Budgets, Loop Control and Circuit Breakers
    (last researched 2026-07-13). Source of the qualitative patterns and the
    Fortune 500 leakage figure.
  * **arXiv:2606.04056**, Khan, June 2026 — "Token Budgets: An Empirical Catalog
    of 63 LLM-Agent Budget-Overrun Incidents". Source of the eight-cluster
    taxonomy, the incident counts, and the dollar figures attached to them.

What we add, and what is therefore the only defensible part of this file: the
mapping from a *narrative* pattern to an *observable signal* in somebody's own
usage data. The paper describes incidents; it does not ship detectors. That
mapping is the product, and it gets calibrated against real diagnoses over time.

Two honest caveats, both stated in the report:

  * A pattern match is a hypothesis, not a diagnosis. Several patterns look alike
    from token counts alone.
  * The incident counts are a floor. They are the cases somebody wrote up, not the
    cases that happened. Nobody publishes a best-effort distribution of LLM
    spend, which is exactly the information asymmetry worth having.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Sequence

from .storage import Store

SOURCES = {
    "owasp": "OWASP AISVS C9.1 (Execution Budgets, Loop Control, Circuit Breakers), rev. 2026-07-13",
    "arxiv": "arXiv:2606.04056 (Khan, 2026-06) - catalogue of 63 confirmed production incidents",
}


@dataclass
class Signal:
    """One measurable condition on a customer's own data."""

    name: str
    description: str
    #: Given the aggregated measures, does this signal fire?
    test: Callable[[Dict[str, float]], bool]
    #: The measure's value, for showing the reader what was compared.
    value_of: Callable[[Dict[str, float]], float]


@dataclass
class IncidentPattern:
    """A catalogued production failure pattern and how to spot it."""

    key: str
    name: str
    cluster: str                     # the paper's taxonomy cluster
    summary: str
    signals: List[Signal]
    #: How many recorded incidents exhibited this pattern, where the source says.
    recorded_incidents: int = 0
    #: Typical reported loss, USD, where the source gives a figure.
    typical_loss_usd: Optional[float] = None
    source_keys: Sequence[str] = ("owasp", "arxiv")
    remedy: str = ""
    effort: str = ""
    #: What the pattern looks like when it is NOT present, so the reader can
    #: check we are not simply always firing.
    absent_when: str = ""

    def cites(self) -> List[str]:
        return [SOURCES[k] for k in self.source_keys if k in SOURCES]


# ---------------------------------------------------------------------------
# The patterns
# ---------------------------------------------------------------------------
def _ratio(m: Dict[str, float]) -> float:
    out = m.get("output_tokens", 0.0)
    return (m.get("input_tokens", 0.0) / out) if out else 0.0


PATTERNS: List[IncidentPattern] = [
    IncidentPattern(
        key="context_loop",
        name="Unbounded context loop",
        cluster="runaway tool-calling loop",
        summary=(
            "An agent re-sends its entire conversation history on every step, so input "
            "tokens grow with the square of the step count while output stays flat. The "
            "single most common documented overrun."
        ),
        signals=[
            Signal(
                name="sustained input/output ratio",
                description="Input tokens per output token, which sits at 5:1 to 15:1 in normal chat traffic. The documented incident reached 74:1 and 175:1.",
                test=lambda m: _ratio(m) >= 30.0,
                value_of=_ratio,
            ),
        ],
        recorded_incidents=11,
        source_keys=("owasp", "arxiv"),
        remedy=(
            "Cap the context that is resent: summarise completed turns, drop tool results "
            "that are no longer needed, and stop appending to a history that only grows. "
            "Prompt caching turns the repeated prefix into a cache read, which is billed at "
            "between 2.5% and 50% of input depending on the model."
        ),
        effort="a sprint",
        absent_when="the ratio stays under about 15:1, which normal chat traffic does",
    ),
    IncidentPattern(
        key="delegation_fanout",
        name="Delegation fan-out race",
        cluster="unbounded fan-out across sub-agents",
        summary=(
            "A supervising agent delegates to sub-agents that each retain or re-derive their "
            "own budget, so the aggregate cost is the product of the fan-out rather than the "
            "sum. The case where a simple counter is genuinely insufficient: the paper shows "
            "the same pattern under asyncio overshooting in 30 of 30 runs while a four-line "
            "counter matches the typed implementation at 0 of 30 on single-agent work."
        ),
        signals=[
            Signal(
                name="concurrent burst across many keys or projects",
                description="Spend arriving from several attributed workloads inside one short window is how fan-out looks once it reaches the billing data.",
                test=lambda m: m.get("distinct_sources", 0) >= 4 and m.get("burst_cost_usd", 0.0) >= 5.0,
                value_of=lambda m: m.get("distinct_sources", 0.0),
            ),
        ],
        recorded_incidents=11,
        source_keys=("arxiv",),
        remedy=(
            "Enforce the budget on the chain, not on each call: one ceiling for the whole "
            "delegation tree, checked before a sub-agent is created rather than after it "
            "reports. Token counts alone cannot distinguish this from several busy teams, "
            "so confirm it against your orchestration code before spending engineering time."
        ),
        effort="an architecture change",
        absent_when="spend arrives from one workload at a time",
    ),
    IncidentPattern(
        key="retry_storm",
        name="Retry storm",
        cluster="cascading provider failure and retry amplification",
        summary=(
            "A failing call is retried without backoff or a ceiling, and each retry is billed. "
            "A recorded incident reached 2.3 million erroneous calls. Most of the cost is "
            "latency, but the volume hides genuine spend and makes every other metric look "
            "worse than it is."
        ),
        signals=[
            Signal(
                name="failure rate",
                description="Share of calls returning an error. Failed calls still consume latency and often tokens.",
                test=lambda m: m.get("failure_rate", 0.0) >= 0.05,
                value_of=lambda m: m.get("failure_rate", 0.0),
            ),
        ],
        recorded_incidents=9,
        typical_loss_usd=47000.0,
        source_keys=("owasp", "arxiv"),
        remedy=(
            "Break failures down by status code before anything else. A majority of 429s means "
            "the client is ignoring rate limits; a majority of 5xx means the retry policy needs "
            "a ceiling and jitter. Both are cheap to fix and both distort every other number."
        ),
        effort="an afternoon",
        absent_when="the failure rate is under a few percent",
    ),
    IncidentPattern(
        key="premium_model_default",
        name="Premium model as the default",
        cluster="cost-blind model selection",
        summary=(
            "Every call routes to the most expensive available model because routing was never "
            "a decision anyone made. The pattern does not appear in the incident catalogue as a "
            "crash, because it never crashes; it just makes every other number larger."
        ),
        signals=[
            Signal(
                name="cost concentration in one expensive model",
                description="One model accounting for most spend while its per-call cost is many times the account average.",
                test=lambda m: m.get("peak_model_cost_multiple", 0.0) >= 8.0
                and m.get("peak_model_requests", 0.0) >= 3.0,
                value_of=lambda m: m.get("peak_model_cost_multiple", 0.0),
            ),
        ],
        recorded_incidents=0,
        source_keys=("owasp",),
        remedy=(
            "Classify calls by difficulty and send the easy ones elsewhere. Short prompts with "
            "no tools and low stakes rarely need a frontier model. Keep the expensive model for "
            "the calls that fail a cheap signal, and route on that signal rather than on intent."
        ),
        effort="a sprint",
        absent_when="no model costs several times the account average per call",
    ),
    IncidentPattern(
        key="uncached_prefix",
        name="Uncached repeated prefix",
        cluster="cache misconfiguration",
        summary=(
            "A stable system prompt, tool schema or retrieved document is re-sent on every call "
            "without a cache breakpoint, so it is billed at the full input rate every time. OWASP "
            "attributes roughly 62% of agent bills to re-sent context."
        ),
        signals=[
            Signal(
                name="cache hit rate",
                description="Share of input-side tokens served from cache. Zero across sustained traffic with a stable prompt is the tell.",
                test=lambda m: m.get("cache_hit_rate", 1.0) < 0.35
                and m.get("input_tokens", 0.0) >= 500_000,
                value_of=lambda m: m.get("cache_hit_rate", 0.0),
            ),
        ],
        recorded_incidents=0,
        source_keys=("owasp",),
        remedy=(
            "Put a cache breakpoint after the stable part of the prompt. Both providers bill "
            "cache reads at a fraction of input, but the fraction is per model and ranges from "
            "2.5% to 50%, so check the rate for the model in use before assuming a saving."
        ),
        effort="an afternoon",
        absent_when="cache reads are a meaningful share of input tokens",
    ),
    IncidentPattern(
        key="no_attribution",
        name="No cost attribution",
        cluster="governance gap (enabling condition)",
        summary=(
            "All spend lands on one key, so no spike can be traced to a team, customer or "
            "feature. Not an overrun by itself; it is the condition that lets every other "
            "pattern run undetected, and OWASP records runtime visibility in only about a "
            "fifth of organisations."
        ),
        signals=[
            Signal(
                name="distinct attribution sources",
                description="Number of distinct keys, projects or workspaces appearing in the data.",
                test=lambda m: m.get("distinct_sources", 0.0) <= 1.0 and m.get("row_count", 0.0) >= 20,
                value_of=lambda m: m.get("distinct_sources", 0.0),
            ),
        ],
        recorded_incidents=0,
        source_keys=("owasp",),
        remedy=(
            "Issue one key per consumer and record it. This does not reduce the bill directly, "
            "but it is the prerequisite for every other reduction, because a cost that cannot "
            "be attributed cannot be diagnosed during an incident."
        ),
        effort="an afternoon",
        absent_when="spend is already split across keys or projects",
    ),
    IncidentPattern(
        key="silent_unpriced",
        name="Unpriced traffic",
        cluster="measurement gap",
        summary=(
            "Traffic runs on models with no price on file, so its cost is recorded as unknown. "
            "Every total built on top is then a floor presented as a total, which is the failure "
            "mode most likely to make an engineer distrust the whole report."
        ),
        signals=[
            Signal(
                name="share of unpriced successful calls",
                description="Successful calls whose model has no rate on file.",
                test=lambda m: m.get("unpriced_share", 0.0) > 0.0,
                value_of=lambda m: m.get("unpriced_share", 0.0),
            ),
        ],
        recorded_incidents=0,
        source_keys=("owasp",),
        remedy=(
            "Add the rates, or route those models through something that can price them. Until "
            "then treat every figure in the report as a lower bound."
        ),
        effort="an afternoon",
        absent_when="every model in use has a rate on file",
    ),
    IncidentPattern(
        key="headless_spike",
        name="Velocity spike without a ceiling",
        cluster="denial-of-wallet",
        summary=(
            "Spend accelerates far above the account's own baseline with no ceiling in place. "
            "OWASP records Fortune 500 leakage of roughly $400M in unbudgeted spend from this "
            "family of failures, and the recorded median time from onset to detection is about "
            "4.5 hours."
        ),
        signals=[
            Signal(
                name="peak day against median day",
                description="A day costing several times the account's typical day indicates a burst rather than growth.",
                test=lambda m: m.get("peak_to_median_day", 0.0) >= 8.0
                and m.get("peak_day_cost_usd", 0.0) >= 5.0,
                value_of=lambda m: m.get("peak_to_median_day", 0.0),
            ),
        ],
        recorded_incidents=0,
        typical_loss_usd=400_000_000.0,
        source_keys=("owasp", "arxiv"),
        remedy=(
            "Put a hard ceiling on the key before the incident, not after. A limit that returns "
            "429 is the only control that works while nobody is watching, and detection lag is "
            "measured in hours."
        ),
        effort="an afternoon",
        absent_when="daily spend is steady",
    ),
]

PATTERNS_BY_KEY = {p.key: p for p in PATTERNS}


# ---------------------------------------------------------------------------
# Measuring, and matching
# ---------------------------------------------------------------------------
def measure(store: Store, *, window_sql: str, params: Sequence = (), granularity: str = "request") -> Dict[str, float]:
    """Compute every measure the patterns test against, from the customer's data."""
    head = store.one(
        f"""SELECT COUNT(*) AS row_count,
                   COALESCE(SUM(input_tokens),0)          AS input_tokens,
                   COALESCE(SUM(cached_input_tokens),0)   AS cached_input_tokens,
                   COALESCE(SUM(cache_write_tokens),0)    AS cache_write_tokens,
                   COALESCE(SUM(output_tokens),0)         AS output_tokens,
                   COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END),0) AS failures,
                   COALESCE(SUM(cost_usd),0)              AS cost_usd
            FROM requests WHERE {window_sql}""",
        params,
    )
    m: Dict[str, float] = {
        "row_count": float(head["row_count"]) if head else 0.0,
        "input_tokens": float(head["input_tokens"]) if head else 0.0,
        "cached_input_tokens": float(head["cached_input_tokens"]) if head else 0.0,
        "cache_write_tokens": float(head["cache_write_tokens"]) if head else 0.0,
        "output_tokens": float(head["output_tokens"]) if head else 0.0,
        "cost_usd": float(head["cost_usd"]) if head else 0.0,
    }
    rows = m["row_count"]
    m["failure_rate"] = (float(head["failures"]) / rows) if head and rows else 0.0

    input_side = m["input_tokens"] + m["cached_input_tokens"]
    m["cache_hit_rate"] = (m["cached_input_tokens"] / input_side) if input_side else 1.0

    sources = store.one(
        f"""SELECT COUNT(DISTINCT api_key_id) AS keys, COUNT(DISTINCT project) AS projects
            FROM requests WHERE {window_sql}""",
        params,
    )
    if sources:
        keys = int(sources["keys"] or 0)
        projects = int(sources["projects"] or 0)
        # An empty project string is not an attribution source.
        m["distinct_sources"] = float(max(keys, projects))
    else:
        m["distinct_sources"] = 0.0

    top = store.one(
        f"""SELECT model, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost
            FROM requests WHERE {window_sql} AND cost_usd IS NOT NULL
            GROUP BY model ORDER BY cost DESC LIMIT 1""",
        params,
    )
    total_cost = m["cost_usd"]
    if top and total_cost > 0:
        m["top_model_share"] = float(top["cost"]) / total_cost
        priced = store.one(
            f"""SELECT COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost
                FROM requests WHERE {window_sql} AND cost_usd IS NOT NULL""",
            params,
        )
        n = int(priced["n"]) if priced else 0
        if n:
            avg = total_cost / n
            m["top_model_cost_multiple"] = (
                (float(top["cost"]) / int(top["n"]) / avg) if avg and int(top["n"]) else 0.0
            )
        else:
            m["top_model_cost_multiple"] = 0.0
    else:
        m["top_model_share"] = 0.0
        m["top_model_cost_multiple"] = 0.0

    # Most expensive model by per-request cost, which is what routing is about.
    priced_rows = store.one(
        f"""SELECT COUNT(*) AS n FROM requests
            WHERE {window_sql} AND cost_usd IS NOT NULL""",
        params,
    )
    priced_n = float(priced_rows["n"]) if priced_rows else 0.0
    avg_cost = (total_cost / priced_n) if priced_n else 0.0
    worst = store.one(
        f"""SELECT model, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost
            FROM requests WHERE {window_sql} AND cost_usd IS NOT NULL
            GROUP BY model ORDER BY (SUM(cost_usd) / COUNT(*)) DESC LIMIT 1""",
        params,
    )
    if worst and avg_cost > 0 and int(worst["n"]):
        per_call = float(worst["cost"]) / int(worst["n"])
        m["peak_model_cost_multiple"] = per_call / avg_cost
        m["peak_model_requests"] = float(worst["n"])
    else:
        m["peak_model_cost_multiple"] = 0.0
        m["peak_model_requests"] = 0.0

    unpriced = store.one(
        f"""SELECT COUNT(*) AS n FROM requests
            WHERE {window_sql} AND cost_usd IS NULL AND status < 400 AND error = ''""",
        params,
    )
    successful = store.one(
        f"SELECT COUNT(*) AS n FROM requests WHERE {window_sql} AND status < 400", params
    )
    denom = float(successful["n"]) if successful else 0.0
    m["unpriced_count"] = float(unpriced["n"]) if unpriced else 0.0
    m["unpriced_share"] = (m["unpriced_count"] / denom) if denom else 0.0

    if granularity == "bucket":
        daily = store.query(
            f"""SELECT substr(ts,1,10) AS day, COALESCE(SUM(cost_usd),0) AS cost
                FROM requests WHERE {window_sql} GROUP BY day ORDER BY cost DESC""",
            params,
        )
        costs = sorted((float(r["cost"]) for r in daily), reverse=True)
        if costs:
            m["peak_day_cost_usd"] = costs[0]
            median = costs[len(costs) // 2]
            m["peak_to_median_day"] = (costs[0] / median) if median > 0 else 0.0
        else:
            m["peak_day_cost_usd"] = 0.0
            m["peak_to_median_day"] = 0.0
    else:
        # Per-request data: a burst shows as a spike in one hour's spend.
        hourly = store.query(
            f"""SELECT substr(ts,1,13) AS hour, COALESCE(SUM(cost_usd),0) AS cost
                FROM requests WHERE {window_sql} GROUP BY hour ORDER BY cost DESC""",
            params,
        )
        costs = sorted((float(r["cost"]) for r in hourly), reverse=True)
        if costs:
            m["peak_day_cost_usd"] = costs[0]
            median = costs[len(costs) // 2]
            m["peak_to_median_day"] = (costs[0] / median) if median > 0 else 0.0
        else:
            m["peak_day_cost_usd"] = 0.0
            m["peak_to_median_day"] = 0.0

    return m


@dataclass
class Match:
    pattern: IncidentPattern
    measures: Dict[str, float]
    fired: List[Signal] = field(default_factory=list)

    def evidence(self) -> Dict[str, object]:
        out: Dict[str, object] = {}
        for signal in self.fired:
            value = signal.value_of(self.measures)
            out[signal.name] = round(value, 4)
        if self.pattern.recorded_incidents:
            out["recorded_incidents"] = self.pattern.recorded_incidents
        if self.pattern.typical_loss_usd:
            out["largest_reported_loss_usd"] = self.pattern.typical_loss_usd
        out["source"] = " / ".join(self.pattern.cites())
        return out


def match_patterns(measures: Dict[str, float]) -> List[Match]:
    """Return every catalogued pattern whose signals fire on these measures."""
    out: List[Match] = []
    for pattern in PATTERNS:
        fired = [s for s in pattern.signals if s.test(measures)]
        if fired:
            out.append(Match(pattern=pattern, measures=measures, fired=fired))
    return out


def describe_catalogue() -> str:
    """Human-readable catalogue, for the CLI and for publishing."""
    lines = ["Catalogued agent cost-overrun patterns", ""]
    for p in PATTERNS:
        lines.append(f"  {p.name}")
        lines.append(f"    cluster   {p.cluster}")
        if p.recorded_incidents:
            lines.append(f"    recorded  {p.recorded_incidents} incident(s) in the published catalogue")
        for s in p.signals:
            lines.append(f"    signal    {s.name}")
        lines.append(f"    absent    {p.absent_when or 'n/a'}")
        for cite in p.cites():
            lines.append(f"    source    {cite}")
        lines.append("")
    return "\n".join(lines)
