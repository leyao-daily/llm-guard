"""Aggregations that turn stored requests into decisions.

The report deliberately answers "where is the money going and what should I
change", not "here is a dashboard". Every function returns plain dataclasses so
the CLI, the HTML dashboard and the tests all share one implementation.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import NamedTuple, Dict, List, Optional, Sequence

from .storage import Store, iso

# Cache reads are ~10% of input price on both OpenAI and Anthropic, so every
# input token that *could* have been a cache hit is worth ~90% of its input
# price. This constant is only used for opportunity sizing.
CACHE_READ_DISCOUNT = 0.10

# Alert when projected month-end spend exceeds budget by this factor.
BUDGET_WARN_RATIO = 0.80
BUDGET_BREACH_RATIO = 1.00


class Window(NamedTuple):
    """A time window, either trailing from now or two absolute dates.

    Needed because a before/after comparison reads two adjacent windows in the
    past, and a trailing window can only ever express one of them. Without this,
    a follow-up measurement looks at the same recent traffic the baseline just
    looked at and reports no change, or worse, reports a change that is really
    just the passage of time.
    """

    sql: str
    params: List[object]
    start: Optional[str] = None    # 'YYYY-MM-DD'
    end: Optional[str] = None

    @property
    def days(self) -> int:
        if self.start and self.end:
            a = datetime.strptime(self.start, "%Y-%m-%d")
            b = datetime.strptime(self.end, "%Y-%m-%d")
            return max((b - a).days, 1)
        return 0


def window_between(start: str, end: str) -> Window:
    """Half-open absolute window: start inclusive, end exclusive."""
    return Window(
        sql="ts >= ? AND ts < ?",
        params=[f"{start} 00:00:00", f"{end} 00:00:00"],
        start=start,
        end=end,
    )


def trailing_window(days: int, offset_days: int = 0) -> Window:
    where, params = _window_clause(days, offset_days=offset_days)
    return Window(sql=where, params=list(params))


def _window_clause(days: int, *, offset_days: int = 0) -> tuple[str, list]:
    """Return a WHERE fragment for a day-aligned trailing window.

    Both bounds are ``start of day`` in UTC. Using ``ts < datetime('now')``
    instead would exclude a row written in the current second, which makes
    freshly proxied traffic invisible to the report -- exactly the moment a
    user is most likely to look.

    ``offset_days`` shifts the window back, which is how the period-over-period
    comparison is built.
    """
    lower = f"-{offset_days + days} days"
    upper = f"-{offset_days} days"
    return (
        "ts >= datetime('now', ?, 'start of day') "
        "AND ts < datetime('now', ?, '+1 day', 'start of day')",
        [lower, upper],
    )


@dataclass
class Totals:
    requests: int = 0
    errors: int = 0
    cost_usd: float = 0.0
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0

    @property
    def total_tokens(self) -> int:
        return (
            self.input_tokens
            + self.cached_input_tokens
            + self.cache_write_tokens
            + self.output_tokens
        )

    @property
    def error_rate(self) -> float:
        return (self.errors / self.requests) if self.requests else 0.0

    @property
    def cost_per_request(self) -> float:
        return (self.cost_usd / self.requests) if self.requests else 0.0

    @property
    def cache_hit_rate(self) -> float:
        """Share of input-side tokens served from cache."""
        denom = self.input_tokens + self.cached_input_tokens
        return (self.cached_input_tokens / denom) if denom else 0.0


@dataclass
class BreakdownRow:
    key: str
    label: str
    requests: int
    cost_usd: float
    total_tokens: int
    errors: int
    share: float = 0.0

    @property
    def cost_per_request(self) -> float:
        return (self.cost_usd / self.requests) if self.requests else 0.0


@dataclass
class DayPoint:
    day: str
    requests: int
    cost_usd: float
    errors: int


@dataclass
class BudgetStatus:
    api_key_id: str
    daily_usd: Optional[float]
    monthly_usd: Optional[float]
    action: str
    spent_today: float
    spent_month: float
    projected_month: float

    @property
    def daily_ratio(self) -> Optional[float]:
        if not self.daily_usd:
            return None
        return self.spent_today / self.daily_usd

    @property
    def monthly_ratio(self) -> Optional[float]:
        if not self.monthly_usd:
            return None
        return self.projected_month / self.monthly_usd

    @property
    def state(self) -> str:
        """ok | warn | breach"""
        ratios = [r for r in (self.daily_ratio, self.monthly_ratio) if r is not None]
        if not ratios:
            return "ok"
        worst = max(ratios)
        if worst >= BUDGET_BREACH_RATIO:
            return "breach"
        if worst >= BUDGET_WARN_RATIO:
            return "warn"
        return "ok"


@dataclass
class ValueReport:
    """Everything the report renders, computed once."""

    days: int
    current: Totals
    previous: Totals
    by_model: List[BreakdownRow] = field(default_factory=list)
    by_key: List[BreakdownRow] = field(default_factory=list)
    by_project: List[BreakdownRow] = field(default_factory=list)
    by_end_user: List[BreakdownRow] = field(default_factory=list)
    daily: List[DayPoint] = field(default_factory=list)
    top_requests: List[dict] = field(default_factory=list)
    budgets: List[BudgetStatus] = field(default_factory=list)
    unpriced_models: List[str] = field(default_factory=list)
    total_requests_ever: int = 0

    @property
    def cost_delta_pct(self) -> Optional[float]:
        if not self.previous.cost_usd:
            return None
        return (self.current.cost_usd - self.previous.cost_usd) / self.previous.cost_usd

    @property
    def request_delta_pct(self) -> Optional[float]:
        if not self.previous.requests:
            return None
        return (self.current.requests - self.previous.requests) / self.previous.requests

    @property
    def projected_month_usd(self) -> float:
        """Naive linear projection from the current window."""
        if self.days <= 0:
            return 0.0
        return (self.current.cost_usd / self.days) * 30.0

    def cache_opportunity_usd(self) -> float:
        """Upper bound on monthly saving from caching all input tokens.

        Intentionally optimistic and labelled as such in the report: it prices
        every current input token at the cache-read rate. Real savings depend on
        how much of the prompt is genuinely repeated.
        """
        per_day = (
            self.current.input_tokens * (1.0 - CACHE_READ_DISCOUNT) * self._input_price_avg()
        )
        if self.days:
            per_day /= self.days
        return per_day * 30.0

    def _input_price_avg(self) -> float:
        """Cost per input token implied by priced rows (USD/token)."""
        priced = [r for r in self.by_model if r.total_tokens]
        if not priced:
            return 0.0
        # Blend by cost share so a cheap model cannot dominate the estimate.
        total_cost = sum(r.cost_usd for r in priced) or 1.0
        blended = 0.0
        for r in priced:
            blended += (r.cost_usd / total_cost) * (r.cost_usd / max(r.total_tokens, 1))
        return blended


def _totals(store: Store, days: int, offset_days: int = 0) -> Totals:
    where, params = _window_clause(days, offset_days=offset_days)
    row = store.one(
        f"""
        SELECT COUNT(*)                                   AS requests,
               COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END), 0) AS errors,
               COALESCE(SUM(cost_usd), 0)                  AS cost_usd,
               COALESCE(SUM(input_tokens), 0)              AS input_tokens,
               COALESCE(SUM(cached_input_tokens), 0)       AS cached_input_tokens,
               COALESCE(SUM(cache_write_tokens), 0)        AS cache_write_tokens,
               COALESCE(SUM(output_tokens), 0)             AS output_tokens
        FROM requests WHERE {where}
        """,
        params,
    )
    if row is None:
        return Totals()
    return Totals(
        requests=int(row["requests"]),
        errors=int(row["errors"]),
        cost_usd=float(row["cost_usd"]),
        input_tokens=int(row["input_tokens"]),
        cached_input_tokens=int(row["cached_input_tokens"]),
        cache_write_tokens=int(row["cache_write_tokens"]),
        output_tokens=int(row["output_tokens"]),
    )


def _breakdown(
    store: Store, column: str, days: int, limit: int = 8
) -> List[BreakdownRow]:
    where, params = _window_clause(days)
    rows = store.query(
        f"""
        SELECT {column} AS k,
               COUNT(*) AS requests,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COALESCE(SUM(input_tokens + cached_input_tokens
                            + cache_write_tokens + output_tokens), 0) AS total_tokens,
               COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END), 0) AS errors
        FROM requests
        WHERE {where}
        GROUP BY k
        ORDER BY cost_usd DESC
        LIMIT ?
        """,
        params + [limit],
    )
    total_cost = sum(float(r["cost_usd"]) for r in rows) or 0.0
    out: List[BreakdownRow] = []
    for r in rows:
        key = str(r["k"] or "(none)")
        cost = float(r["cost_usd"])
        out.append(
            BreakdownRow(
                key=key,
                label=key,
                requests=int(r["requests"]),
                cost_usd=cost,
                total_tokens=int(r["total_tokens"]),
                errors=int(r["errors"]),
                share=(cost / total_cost) if total_cost else 0.0,
            )
        )
    return out


def _daily(store: Store, days: int) -> List[DayPoint]:
    where, params = _window_clause(days)
    rows = store.query(
        f"""
        SELECT substr(ts, 1, 10) AS day,
               COUNT(*) AS requests,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END), 0) AS errors
        FROM requests WHERE {where}
        GROUP BY day ORDER BY day
        """,
        params,
    )
    return [
        DayPoint(
            day=str(r["day"]),
            requests=int(r["requests"]),
            cost_usd=float(r["cost_usd"]),
            errors=int(r["errors"]),
        )
        for r in rows
    ]


def _unpriced(store: Store, days: int) -> List[str]:
    """Models that produced a *successful* call we could not price.

    Failed requests legitimately carry no cost, so they are excluded here --
    otherwise every 429 would look like a missing price entry.
    """
    where, params = _window_clause(days)
    rows = store.query(
        f"""SELECT model, COUNT(*) AS n FROM requests
            WHERE cost_usd IS NULL AND status < 400 AND error = '' AND {where}
            GROUP BY model ORDER BY n DESC""",
        params,
    )
    return [str(r["model"]) for r in rows]


def _budget_status(store: Store, days: int) -> List[BudgetStatus]:
    """Compare each budget to actual spend today and projected month-end."""
    budgets = store.all_budgets()
    if not budgets:
        return []

    # Month-to-date spend per key.
    month_rows = store.query(
        """SELECT api_key_id,
                  COALESCE(SUM(cost_usd), 0) AS spent
           FROM requests
           WHERE ts >= datetime('now', 'start of month')
           GROUP BY api_key_id"""
    )
    month_map = {str(r["api_key_id"]): float(r["spent"]) for r in month_rows}

    today_rows = store.query(
        """SELECT api_key_id,
                  COALESCE(SUM(cost_usd), 0) AS spent
           FROM requests
           WHERE ts >= datetime('now', 'start of day')
           GROUP BY api_key_id"""
    )
    today_map = {str(r["api_key_id"]): float(r["spent"]) for r in today_rows}

    now = datetime.now(timezone.utc)
    days_elapsed = max(now.day, 1)
    days_in_month = 30.0
    out: List[BudgetStatus] = []
    for b in budgets:
        key = str(b["api_key_id"])
        spent_month = month_map.get(key, 0.0)
        projected = spent_month / days_elapsed * days_in_month
        out.append(
            BudgetStatus(
                api_key_id=key,
                daily_usd=b["daily_usd"],
                monthly_usd=b["monthly_usd"],
                action=str(b["action"]),
                spent_today=today_map.get(key, 0.0),
                spent_month=spent_month,
                projected_month=projected,
            )
        )
    return out


def build_report(store: Store, days: int = 30, top_n: int = 10) -> ValueReport:
    """Compute the full report. One entry point for CLI, dashboard and tests."""
    top_rows = store.query(
        f"""
        SELECT ts, provider, model, cost_usd, input_tokens, output_tokens,
               latency_ms, status, api_key_id, end_user, project, error
        FROM requests
        WHERE cost_usd IS NOT NULL AND {_window_clause(days)[0]}
        ORDER BY cost_usd DESC
        LIMIT ?
        """,
        _window_clause(days)[1] + [top_n],
    )

    return ValueReport(
        days=days,
        current=_totals(store, days),
        previous=_totals(store, days, offset_days=days),
        by_model=_breakdown(store, "model", days),
        by_key=_breakdown(store, "api_key_id", days),
        by_project=_breakdown(store, "project", days),
        by_end_user=_breakdown(store, "end_user", days),
        daily=_daily(store, days),
        top_requests=[dict(r) for r in top_rows],
        budgets=_budget_status(store, days),
        unpriced_models=_unpriced(store, days),
        total_requests_ever=store.count(),
    )


# --------------------------------------------------------------------------
# Budget enforcement (used by the proxy on the request path)
# --------------------------------------------------------------------------
@dataclass
class GuardDecision:
    allowed: bool
    reason: str = ""
    status: int = 200
    budget: Optional[BudgetStatus] = None


def evaluate_guard(store: Store, api_key_id: str) -> GuardDecision:
    """Decide whether to let a new request through for this API key.

    Called on the hot path for every proxied request, so it does two indexed
    aggregate reads and nothing else. Only keys with an explicit budget are
    ever blocked.
    """
    budget = store.get_budget(api_key_id)
    if budget is None:
        return GuardDecision(allowed=True)

    action = str(budget["action"] or "alert")

    if budget["daily_usd"]:
        row = store.one(
            """SELECT COALESCE(SUM(cost_usd),0) AS spent FROM requests
               WHERE api_key_id=? AND ts >= datetime('now','start of day')""",
            (api_key_id,),
        )
        spent_today = float(row["spent"]) if row else 0.0
        if spent_today >= float(budget["daily_usd"]):
            over = spent_today - float(budget["daily_usd"])
            return GuardDecision(
                allowed=(action != "block"),
                reason=(
                    f"daily budget ${float(budget['daily_usd']):.2f} exceeded "
                    f"by ${over:.2f} (spent ${spent_today:.2f})"
                ),
                status=429 if action == "block" else 200,
            )

    if budget["monthly_usd"]:
        row = store.one(
            """SELECT COALESCE(SUM(cost_usd),0) AS spent FROM requests
               WHERE api_key_id=? AND ts >= datetime('now','start of month')""",
            (api_key_id,),
        )
        spent_month = float(row["spent"]) if row else 0.0
        if spent_month >= float(budget["monthly_usd"]):
            over = spent_month - float(budget["monthly_usd"])
            return GuardDecision(
                allowed=(action != "block"),
                reason=(
                    f"monthly budget ${float(budget['monthly_usd']):.2f} exceeded "
                    f"by ${over:.2f} (spent ${spent_month:.2f})"
                ),
                status=429 if action == "block" else 200,
            )

    return GuardDecision(allowed=True)
