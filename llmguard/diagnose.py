"""LLM cost diagnostics: find what is wrong with a bill, and put a number on it.

This is the module behind the paid deliverable. Everything here answers the same
question a client actually asks: *where is the money going, and what do I change
first?* Each finding carries:

  * the evidence it was derived from, so the claim can be checked
  * an estimated monthly saving, so findings can be ranked
  * a confidence level, because some of these are arithmetic and some are a
    judgement about your architecture

Deliberately conservative
-------------------------
Every saving estimate is a bound, not a promise, and the report says so. A
diagnostic that overstates savings gets found out on the first invoice, and then
nothing else in the document is trusted either.

Where a number cannot be known from the data (how much of a prompt is genuinely
repeated, how hard the requests are), the estimate is stated as a range and the
assumption is written next to it.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from .analytics import Window, _window_clause
from .pricing import get_price, resolve_model
from .storage import Store

# ---- thresholds, all tunable ------------------------------------------------
# Input/output ratio above which context is worth attacking. Normal is 5-15:1.
CONTEXT_RATIO = 20.0
# Share of input that is already cached, above which caching is "done".
GOOD_CACHE_HIT_RATE = 0.35
# A model has to be this much more expensive than the account average, with
# enough volume, before routing is worth proposing.
ROUTING_COST_MULTIPLE = 8.0
ROUTING_MIN_REQUESTS = 3
# Error rate worth reporting at all.
ERROR_RATE_FLOOR = 0.02
# Share of spend in one model above which concentration is a risk.
CONCENTRATION_SHARE = 0.6

CONFIDENCE_ARITHMETIC = "certain"      # pure arithmetic on your own data
CONFIDENCE_LIKELY = "likely"           # depends on a stated assumption
CONFIDENCE_JUDGEMENT = "worth testing"  # a hypothesis about your architecture


def resolve_window(days: int, window: Optional[Window] = None) -> Window:
    """Days, or an explicit window. Everything downstream takes the window."""
    if window is not None:
        return window
    return Window(sql=_window_clause(days)[0], params=list(_window_clause(days)[1]))


@dataclass
class Finding:
    """One diagnosed problem, with the money attached."""

    key: str
    title: str
    detail: str
    evidence: Dict[str, object] = field(default_factory=dict)
    monthly_saving_low: float = 0.0
    monthly_saving_high: float = 0.0
    confidence: str = CONFIDENCE_LIKELY
    remedy: str = ""
    effort: str = ""          # "an afternoon" | "a sprint" | "an architecture change"
    #: Set when this finding corresponds to a catalogued production failure
    #: pattern. Carries the citation, the incident count and the reported loss,
    #: so the report asserts a recognised pattern rather than our opinion.
    catalogue: Optional[dict] = None

    @property
    def mid_saving(self) -> float:
        return (self.monthly_saving_low + self.monthly_saving_high) / 2

    @property
    def has_money(self) -> bool:
        return self.monthly_saving_high > 0


@dataclass
class Diagnosis:
    """The whole picture, ready to render."""

    days: int
    total_spend: float
    total_requests: int
    total_tokens: int
    first_ts: str
    last_ts: str
    distinct_models: int
    distinct_keys: int
    error_requests: int
    unpriced_requests: int
    unpriced_models: List[str]
    findings: List[Finding] = field(default_factory=list)
    # "request" when the data came through the proxy, "bucket" when it came from a
    # provider export. Changes both the wording and the volume thresholds: 20
    # daily buckets is most of a month, 20 requests is nothing.
    granularity: str = "request"
    # What we could not determine, stated rather than hidden.
    gaps: List[str] = field(default_factory=list)

    @property
    def addressable_spend(self) -> float:
        """Sum of the high end of every saving. Never presented as achievable."""
        return sum(f.monthly_saving_high for f in self.findings)

    @property
    def ranked(self) -> List[Finding]:
        return sorted(
            self.findings,
            key=lambda f: (f.has_money, f.mid_saving),
            reverse=True,
        )

    @property
    def days_covered(self) -> int:
        return max(self.days, 1)

    @property
    def unit(self) -> str:
        return "daily buckets" if self.granularity == "bucket" else "requests"

    def volume_floor(self, for_requests: int) -> int:
        """Minimum volume before a finding is statistically worth reporting.

        Bucketed data has far fewer rows for the same history, so a floor tuned
        for per-request data silently suppresses every finding. Eight daily
        buckets is a week; eight requests is noise.
        """
        if self.granularity == "bucket":
            # Bucketed data has far fewer rows for the same history, so the floor
            # has to scale down or every finding is suppressed. At least 2, at
            # most 6, so a long enough window still clears it.
            return min(max(for_requests // 10, 2), 6)
        return for_requests

    def monthly(self, amount: float) -> float:
        """Scale a window figure to a 30-day month."""
        return amount / self.days_covered * 30.0


# ---------------------------------------------------------------------------
# individual checks
# ---------------------------------------------------------------------------
def check_context_bloat(
    store: Store, days: int, *, window: Optional[Window] = None, min_rows: int = 20, unit: str = "requests"
) -> Optional[Finding]:
    """Input-heavy traffic on one key: the prompt is being resent, not the answer made.

    Scoped per key rather than aggregated, because a global average dilutes the
    one workload that is actually looping. First run of this check aggregated
    everything and missed a 74:1 loop sitting alongside a lot of healthy chat
    traffic, which is exactly the failure a client is paying to have found.
    """
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    rows = store.query(
        f"""
        SELECT api_key_id,
               COALESCE(SUM(input_tokens), 0)        AS fresh_input,
               COALESCE(SUM(cached_input_tokens), 0) AS cached_input,
               COALESCE(SUM(cache_write_tokens), 0)  AS cache_write,
               COALESCE(SUM(output_tokens), 0)       AS output_tokens,
               COALESCE(SUM(cost_usd), 0)            AS cost_usd,
               COUNT(*)                              AS requests
        FROM requests
        WHERE {where} AND cost_usd IS NOT NULL AND model <> ''
        GROUP BY api_key_id
        HAVING output_tokens > 0 AND requests >= ?
        """,
        tuple(params) + (min_rows,),
    )
    if not rows:
        return None

    scored = []
    for r in rows:
        total_input = int(r["fresh_input"]) + int(r["cached_input"]) + int(r["cache_write"])
        output = int(r["output_tokens"])
        if total_input <= 0 or output <= 0:
            continue
        scored.append((total_input / output, r, total_input, output))
    if not scored:
        return None

    ratio, worst, total_input, output = max(scored, key=lambda x: x[0])
    if ratio < CONTEXT_RATIO:
        return None

    key = str(worst["api_key_id"])
    requests = int(worst["requests"])
    per_input = _blended_input_price(store, days, window=window)
    addressable = total_input * per_input
    low, high = addressable * 0.25, addressable * 0.55

    # Is this one key, or the whole account? The remedy differs.
    # "Systemic" should mean every key is bad, not merely that a second key
    # cleared the threshold. Claiming systemic when one outlier drove it would
    # point the client at the wrong remedy: a shared prompt template versus one
    # runaway chain.
    bad = sum(1 for r, *_ in scored if r >= CONTEXT_RATIO)
    if bad == len(scored) and len(scored) > 1:
        scope = f"All {len(scored)} attributed workloads show this shape, so it is systemic."
    elif bad > 1:
        scope = (
            f"{bad} of {len(scored)} attributed workloads show this shape; the worst is "
            f"{key}, so start there."
        )
    else:
        scope = f"The other workloads look normal, so this is specific to {key}."

    return Finding(
        key="context_bloat",
        title=f"{key} sends {ratio:.0f} input tokens per output token",
        detail=(
            f"Across {requests:,} priced {unit}, {key} sent {total_input:,} input tokens and "
            f"generated {output:,} output tokens. Normal chat traffic sits between 5:1 and 15:1. "
            f"A ratio this high normally means the same context is being resent on every step of "
            f"a chain rather than being summarised or trimmed. {scope}"
        ),
        evidence={
            "api_key_id": key,
            "input_tokens": total_input,
            "output_tokens": output,
            "ratio": round(ratio, 1),
            "requests": requests,
            "blended_input_price_per_mtok": round(per_input * 1_000_000, 4),
            "input_spend_usd": round(addressable, 2),
        },
        monthly_saving_low=low,
        monthly_saving_high=high,
        confidence=CONFIDENCE_LIKELY,
        remedy=(
            "Cap the context that gets resent: summarise completed turns, drop tool results that "
            "are no longer needed, and stop appending to a history that only grows. The "
            "input/output ratio is the cheapest proxy for whether the change worked, so record "
            "it before and after."
        ),
        effort="a sprint",
    )


def _blended_input_price(store: Store, days: int, *, window: Optional[Window] = None) -> float:
    """USD per input token, blended across models by actual token volume."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    rows = store.query(
        f"""SELECT model,
                   SUM(input_tokens + cached_input_tokens + cache_write_tokens) AS tok
            FROM requests WHERE {where} AND cost_usd IS NOT NULL
            GROUP BY model""",
        params,
    )
    total_tokens = 0
    weighted = 0.0
    for r in rows:
        tok = int(r["tok"] or 0)
        if tok <= 0:
            continue
        price = get_price(str(r["model"]))
        if price is None:
            continue
        total_tokens += tok
        weighted += tok * float(price.input)
    if total_tokens == 0:
        return 0.0
    return weighted / total_tokens / 1_000_000


def check_cache_opportunity(
    store: Store, days: int, *, window: Optional[Window] = None, min_tokens: int = 50_000
) -> Optional[Finding]:
    """Repeated prefixes that are not being cached, scoped per key."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    rows = store.query(
        f"""
        SELECT api_key_id,
               COALESCE(SUM(input_tokens), 0)        AS fresh,
               COALESCE(SUM(cached_input_tokens), 0) AS cached,
               COALESCE(SUM(cost_usd), 0)            AS cost
        FROM requests WHERE {where} AND cost_usd IS NOT NULL
        GROUP BY api_key_id
        HAVING (fresh + cached) >= ?
        """,
        tuple(params) + (min_tokens,),
    )
    if not rows:
        return None

    scored = []
    for r in rows:
        fresh, cached = int(r["fresh"]), int(r["cached"])
        denom = fresh + cached
        if denom <= 0:
            continue
        scored.append((cached / denom, r, fresh, cached))
    if not scored:
        return None

    hit, worst, fresh, cached = min(scored, key=lambda x: x[0])
    if hit >= GOOD_CACHE_HIT_RATE:
        return None

    key = str(worst["api_key_id"])
    per_input = _blended_input_price(store, days, window=window)
    converted = fresh * 0.35
    low = converted * per_input * 0.6
    high = converted * per_input * 0.9

    return Finding(
        key="cache_opportunity",
        title=f"{key} has a {hit * 100:.0f}% prompt cache hit rate",
        detail=(
            f"{fresh:,} input tokens were billed at the full input rate while only {cached:,} "
            f"were cache reads. Repeated prefixes are the usual cause: a system prompt, tool "
            f"schemas, or retrieved documents that are identical on every call but not marked "
            f"as cacheable."
        ),
        evidence={
            "api_key_id": key,
            "cache_hit_rate": round(hit, 4),
            "uncached_input_tokens": fresh,
            "cached_input_tokens": cached,
        },
        monthly_saving_low=low,
        monthly_saving_high=high,
        confidence=CONFIDENCE_LIKELY,
        remedy=(
            "Put a cache breakpoint after the stable part of the prompt. Both providers bill "
            "cache reads at a fraction of input, but the fraction is per model and runs from "
            "2.5% to 50%, so check the rate for your model before assuming a saving."
        ),
        effort="an afternoon",
    )


def check_routing(store: Store, days: int, *, window: Optional[Window] = None, min_rows: int = 50) -> Optional[Finding]:
    """One expensive model doing work a cheaper one could do."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    row = store.one(
        f"""SELECT COALESCE(SUM(cost_usd),0) AS cost, COUNT(*) AS n
            FROM requests WHERE {where} AND cost_usd IS NOT NULL""",
        params,
    )
    if row is None or int(row["n"]) < min_rows:
        return None
    total_cost, total_requests = float(row["cost"]), int(row["n"])
    avg = total_cost / total_requests

    rows = store.query(
        f"""SELECT model, COUNT(*) AS n, COALESCE(SUM(cost_usd),0) AS cost,
                   COALESCE(SUM(input_tokens+cached_input_tokens+output_tokens),0) AS tok
            FROM requests WHERE {where} AND cost_usd IS NOT NULL
            GROUP BY model HAVING n >= ? ORDER BY cost DESC""",
        params + [ROUTING_MIN_REQUESTS],
    )
    if not rows:
        return None

    worst = max(rows, key=lambda r: float(r["cost"]) / max(int(r["n"]), 1))
    worst_cost, worst_n = float(worst["cost"]), int(worst["n"])
    worst_avg = worst_cost / worst_n
    if worst_avg < avg * ROUTING_COST_MULTIPLE:
        return None

    # Assume a quarter to a half of that model's calls are easy enough to move.
    movable = worst_cost * 0.25
    low, high = movable * 0.4, movable * 0.75
    return Finding(
        key="routing",
        title=f"{worst['model']} costs {worst_avg / avg:.0f}x your average call",
        detail=(
            f"{worst['model']} handled {worst_n:,} requests at "
            f"${worst_avg:.4f} each, against a blended ${avg:.4f} across all "
            f"{total_requests:,} requests. A model that expensive is usually doing a mix of "
            f"work, only part of which needs it."
        ),
        evidence={
            "model": str(worst["model"]),
            "requests": worst_n,
            "cost_usd": round(worst_cost, 2),
            "cost_per_request": round(worst_avg, 6),
            "account_average": round(avg, 6),
            "multiple": round(worst_avg / avg, 1),
        },
        monthly_saving_low=low,
        monthly_saving_high=high,
        confidence=CONFIDENCE_JUDGEMENT,
        remedy=(
            "Classify these calls by difficulty and send the easy ones elsewhere. Short "
            "prompts with no tools and low stakes rarely need a frontier model. Route on a "
            "cheap signal (prompt length, whether tools are attached, a fast classifier) and "
            "keep the expensive model for the calls that fail the cheap one."
        ),
        effort="a sprint",
    )


def check_errors(store: Store, days: int, *, window: Optional[Window] = None, min_rows: int = 50) -> Optional[Finding]:
    """Failed calls: cost with no output."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    row = store.one(
        f"""SELECT COUNT(*) AS n,
                   COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END),0) AS bad,
                   COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN cost_usd ELSE 0 END),0) AS bad_cost,
                   COALESCE(AVG(latency_ms),0) AS avg_latency
            FROM requests WHERE {where}""",
        params,
    )
    if row is None or int(row["n"]) < min_rows:
        return None
    n, bad = int(row["n"]), int(row["bad"])
    if n == 0:
        return None
    rate = bad / n
    if rate < ERROR_RATE_FLOOR:
        return None

    bad_cost = float(row["bad_cost"])
    avg_latency = float(row["avg_latency"])
    # Failed calls mostly cost time, not tokens. Value the time at a nominal
    # engineer rate only if the volume is material; otherwise report it as a
    # reliability problem with no dollar figure.
    wasted_hours = bad * (avg_latency / 1000.0) / 3600.0
    return Finding(
        key="errors",
        title=f"{rate * 100:.0f}% of requests failed ({bad:,} calls)",
        detail=(
            f"{bad:,} of {n:,} requests returned an error. Failed calls usually cost little in "
            f"tokens but they cost latency and they hide real spend when a client retries "
            f"without backoff. Average request latency in this window was "
            f"{avg_latency:,.0f} ms, so roughly {wasted_hours:.1f} hours were spent on calls "
            f"that produced nothing."
        ),
        evidence={
            "requests": n,
            "failed": bad,
            "failure_rate": round(rate, 4),
            "tokens_spent_on_failures_usd": round(bad_cost, 2),
            "wasted_hours": round(wasted_hours, 2),
        },
        monthly_saving_low=0.0,
        monthly_saving_high=bad_cost,   # only claim the money we can see
        confidence=CONFIDENCE_ARITHMETIC,
        remedy=(
            "Break the failures down by status code before anything else. A majority of 429s "
            "means the client is ignoring rate limits; a majority of 5xx means retry policy. "
            "Both are cheap to fix and both distort every other number in this report."
        ),
        effort="an afternoon",
    )


def check_attribution(store: Store, days: int, *, window: Optional[Window] = None, min_rows: int = 100) -> Optional[Finding]:
    """No per-key attribution: the bill cannot be acted on."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    row = store.one(
        f"""SELECT COUNT(DISTINCT api_key_id) AS keys, COUNT(*) AS n
            FROM requests WHERE {where}""",
        params,
    )
    if row is None or int(row["n"]) < min_rows:
        return None
    keys = int(row["keys"])
    if keys > 1:
        return None
    return Finding(
        key="attribution",
        title="All spend is attributed to a single key",
        detail=(
            f"{int(row['n']):,} requests in this window but only {keys} distinct key. Without "
            f"per-key or per-project attribution there is no way to tell which team, customer "
            f"or feature caused a spike, which is the question that gets asked during an "
            f"incident."
        ),
        evidence={"distinct_keys": keys, "requests": int(row["n"])},
        monthly_saving_low=0.0,
        monthly_saving_high=0.0,   # enables action, does not itself save money
        confidence=CONFIDENCE_ARITHMETIC,
        remedy=(
            "Issue one key per consumer and record it. This does not reduce the bill directly, "
            "but it is the prerequisite for every other reduction, because you cannot fix a "
            "cost you cannot attribute."
        ),
        effort="an afternoon",
    )


def check_unpriced(store: Store, days: int, *, window: Optional[Window] = None) -> Tuple[Optional[Finding], List[str]]:
    """Traffic on models with no price on file: invisible spend."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    rows = store.query(
        f"""SELECT model, COUNT(*) AS n
            FROM requests
            WHERE {where} AND cost_usd IS NULL AND status < 400 AND error = ''
            GROUP BY model ORDER BY n DESC""",
        params,
    )
    models = [str(r["model"]) for r in rows]
    if not models:
        return None, []
    counted = sum(int(r["n"]) for r in rows)
    return (
        Finding(
            key="unpriced",
            title=f"{len(models)} model(s) have no price on file",
            detail=(
                f"{counted:,} successful requests used models that are not in the cost table, "
                f"so their spend is recorded as unknown. Every total in this report therefore "
                f"understates the real bill."
            ),
            evidence={"models": models[:10], "requests": counted},
            monthly_saving_low=0.0,
            monthly_saving_high=0.0,
            confidence=CONFIDENCE_ARITHMETIC,
            remedy=(
                "Add the rates, or route these models through a proxy that can price them. "
                "Until then treat the headline figure as a floor rather than a total."
            ),
            effort="an afternoon",
        ),
        models,
    )


def check_concentration(store: Store, days: int, *, window: Optional[Window] = None) -> Optional[Finding]:
    """Too much of the bill in one model: no leverage if its price changes."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    rows = store.query(
        f"""SELECT model, COALESCE(SUM(cost_usd),0) AS cost
            FROM requests WHERE {where} AND cost_usd IS NOT NULL
            GROUP BY model ORDER BY cost DESC""",
        params,
    )
    total = sum(float(r["cost"]) for r in rows)
    if total <= 0 or not rows:
        return None
    top = rows[0]
    share = float(top["cost"]) / total
    if share < CONCENTRATION_SHARE:
        return None
    return Finding(
        key="concentration",
        title=f"{share * 100:.0f}% of spend is in one model ({top['model']})",
        detail=(
            f"${float(top['cost']):,.2f} of ${total:,.2f} went to {top['model']}. That is not "
            f"wrong on its own, but it means a price change or a deprecation on that one model "
            f"moves your whole bill, and you have no cheaper path already wired up."
        ),
        evidence={
            "model": str(top["model"]),
            "share": round(share, 4),
            "cost_usd": round(float(top["cost"]), 2),
            "total_usd": round(total, 2),
        },
        monthly_saving_low=0.0,
        monthly_saving_high=0.0,
        confidence=CONFIDENCE_JUDGEMENT,
        remedy=(
            "Wire up a fallback to a second provider even if you never use it in anger. The "
            "point is not the saving today, it is that a pricing change or an outage does not "
            "become an emergency."
        ),
        effort="a sprint",
    )


# ---------------------------------------------------------------------------
# mapping our checks onto the published taxonomy
# ---------------------------------------------------------------------------
# Not every check corresponds to a catalogued incident pattern, and inventing a
# mapping for the ones that do not would weaken the ones that do.
PATTERN_FOR_FINDING = {
    "context_bloat": "context_loop",
    "cache_opportunity": "uncached_prefix",
    "routing": "premium_model_default",
    "errors": "retry_storm",
    "attribution": "no_attribution",
    "unpriced": "silent_unpriced",
    "concentration": "premium_model_default",
}


def attach_catalogue(findings: List[Finding], store: Store, days: int, granularity: str,
                     window: Optional[Window] = None) -> None:
    """Annotate findings with the catalogued pattern they correspond to.

    Scoped to the same subject the finding is about. A loop is usually confined to
    one key, so measuring the whole account would dilute it below the pattern's
    threshold and the citation would silently not appear, even though the loop is
    right there in the finding above it.

    Matching is by finding key rather than by re-deriving the signal, so a finding
    can only cite a pattern the checker actually tested for. A citation that was
    not earned by the data would be the worst kind of marketing.
    """
    from .analytics import _window_clause
    from .incidents import PATTERNS_BY_KEY, match_patterns, measure

    win = resolve_window(days, window)
    where, params = win.sql, win.params
    # Subjects named by the findings, so each can be measured on its own.
    subjects = []
    for finding in findings:
        subject = finding.evidence.get("api_key_id")
        if subject:
            subjects.append(str(subject))

    scopes: Dict[str, Tuple[str, Sequence]] = {"__all__": (where, params)}
    for subject in set(subjects):
        scopes[subject] = (f"{where} AND api_key_id = ?", tuple(params) + (subject,))

    fired: Dict[Tuple[str, str], object] = {}
    for scope_name, (sql, sql_params) in scopes.items():
        measures = measure(store, window_sql=sql, params=sql_params, granularity=granularity)
        for match in match_patterns(measures):
            fired[(scope_name, match.pattern.key)] = match

    for finding in findings:
        key = PATTERN_FOR_FINDING.get(finding.key)
        if not key:
            continue
        subject = str(finding.evidence.get("api_key_id") or "")
        match = fired.get((subject, key)) or fired.get(("__all__", key))
        if match is None:
            continue
        pattern = PATTERNS_BY_KEY[key]
        signal = match.fired[0]
        scope_note = (
            f"measured on {subject}" if (subject, key) in fired else "measured across the account"
        )
        finding.catalogue = {
            "name": pattern.name,
            "cluster": pattern.cluster,
            "recorded_incidents": pattern.recorded_incidents,
            "largest_reported_loss_usd": pattern.typical_loss_usd,
            "signal_measured": f"{signal.name} = {signal.value_of(match.measures):.4g} ({scope_note})",
            "absent_when": pattern.absent_when,
            "sources": pattern.cites(),
            "summary": pattern.summary,
        }


# ---------------------------------------------------------------------------
# assembly
# ---------------------------------------------------------------------------
def diagnose(store: Store, days: int = 30, *, window: Optional[Window] = None) -> Diagnosis:
    """Run every check and assemble the report."""
    win = resolve_window(days, window)
    where, params = win.sql, win.params
    head = store.one(
        f"""SELECT COUNT(*) AS n,
                   COALESCE(SUM(cost_usd),0) AS cost,
                   COALESCE(SUM(input_tokens+cached_input_tokens+cache_write_tokens+output_tokens),0) AS tok,
                   MIN(ts) AS lo, MAX(ts) AS hi,
                   COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END),0) AS bad
            FROM requests WHERE {where}""",
        params,
    )
    models = store.one(
        f"SELECT COUNT(DISTINCT model) AS m, COUNT(DISTINCT api_key_id) AS k "
        f"FROM requests WHERE {where}",
        params,
    )
    pricing = store.one(
        f"SELECT COALESCE(SUM(CASE WHEN cost_usd IS NULL AND status < 400 THEN 1 ELSE 0 END),0) AS unpriced "
        f"FROM requests WHERE {where}",
        params,
    )

    granularity = (store.get_meta("granularity") or "request").lower()

    d = Diagnosis(
        days=days,
        granularity=granularity,
        total_spend=float(head["cost"]) if head else 0.0,
        total_requests=int(head["n"]) if head else 0,
        total_tokens=int(head["tok"]) if head else 0,
        first_ts=str(head["lo"]) if head and head["lo"] else "",
        last_ts=str(head["hi"]) if head and head["hi"] else "",
        distinct_models=int(models["m"]) if models else 0,
        distinct_keys=int(models["k"]) if models else 0,
        error_requests=int(head["bad"]) if head else 0,
        unpriced_requests=int(pricing["unpriced"]) if pricing else 0,
        unpriced_models=[],
    )

    # Errors and attribution need per-request rows. On bucketed data they would
    # either find nothing or report something misleading, so they are skipped and
    # the absence is stated in the gaps rather than papered over.
    checks = [check_context_bloat, check_cache_opportunity, check_routing, check_concentration]
    if d.granularity != "bucket":
        checks += [check_errors, check_attribution]

    floor_ctx = d.volume_floor(20)
    floor_routing = d.volume_floor(50)
    for check in checks:
        kwargs = {}
        if check is check_context_bloat:
            kwargs = {"min_rows": floor_ctx, "unit": d.unit}
        elif check is check_routing:
            kwargs = {"min_rows": floor_routing}
        finding = check(store, days, window=win, **kwargs)
        if finding is not None:
            d.findings.append(finding)

    unpriced_finding, unpriced_models = check_unpriced(store, days)
    d.unpriced_models = unpriced_models
    if unpriced_finding is not None:
        d.findings.append(unpriced_finding)

    # State the limits of the analysis rather than letting the reader assume
    # the numbers are complete.
    if d.unpriced_requests:
        d.gaps.append(
            f"{d.unpriced_requests:,} successful requests used unpriced models. All totals "
            f"here are floors, not full spend."
        )
    if d.total_requests and d.distinct_keys <= 1:
        d.gaps.append(
            "No per-key or per-project attribution in this data, so spend cannot be split by "
            "team, customer or feature."
        )
    if d.days < 14:
        d.gaps.append(
            f"Only {d.days} days of data. Weekly patterns and month-end spikes will not show "
            f"up, and every projection here has wide error bars."
        )
    d.gaps.append(
        "Prompt and completion content is not part of this analysis, only token counts, "
        "latency and cost. Findings about *why* volume is high are inferred from shape, "
        "not from reading your calls."
    )
    d.gaps.append(
        "Where a finding cites a catalogued failure pattern, that is a pattern match and not "
        "a confirmed diagnosis. Several patterns are indistinguishable from token counts "
        "alone; the citation tells you which documented failure this most resembles."
    )
    if granularity == "bucket":
        d.gaps.append(
            "This data is aggregated by day rather than per request, so there is no "
            "latency, status code or end-user dimension. Error rates and per-customer "
            "unit economics cannot be derived from it, and the volume counts below are "
            "buckets rather than calls."
        )

    attach_catalogue(d.findings, store, days, granularity, window=win)
    return d
