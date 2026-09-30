"""Detection of runaway agent spend patterns.

Why this module exists
----------------------
Budget caps catch *totals*. They are slow to fire on the failure mode that
actually produces five-figure bills: a single agent chain looping, retrying, or
fanning out. By the time a daily cap trips, the money is already gone.

The public research (OWASP AISVS C9.1, 2026-07) is specific about the shape of
these incidents, and each shape has a cheap, local signature:

  * **Unbounded context loops** re-send the whole conversation every step, so the
    input/output ratio explodes. A normal call sits at 5:1-15:1; the documented
    Claude Code incident hit 74:1 and 175:1. Hence: flag a sustained
    input/output ratio above ~30:1. OWASP calls this "a cheap way to surface
    this exact failure".
  * **Retry storms** produce bursts of failing calls in a short window.
  * **Runaway velocity** shows up as a chain whose spend rate is orders of
    magnitude above the account's own baseline -- not above some absolute
    number, which is why the baseline is derived from the account's history.

Everything here is advisory by default. Detectors *flag*; only an explicit
budget with ``action=block`` hard-stops traffic. A detector that silently killed
production traffic would be a worse failure than the overspend it prevents.
"""

from __future__ import annotations

import statistics
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Sequence

from .storage import Store

# Ratios above this are pathological. Normal chat traffic is 5:1-15:1; the
# documented unbounded-context incident reached 74:1 and 175:1.
DEFAULT_IO_RATIO = 30.0
# How many calls in the window must exceed the ratio before we call it a loop.
# One large call is normal; a sustained pattern is not.
DEFAULT_MIN_CALLS = 5
# A chain's spend rate versus the account baseline that counts as runaway.
DEFAULT_VELOCITY_MULTIPLIER = 25.0
# Minimum spend in a burst before velocity is worth reporting, to avoid noise
# from tiny accounts where every cent is hundreds of percent.
DEFAULT_VELOCITY_FLOOR_USD = 1.0
# Retry-storm thresholds.
DEFAULT_STORM_MIN_FAILURES = 20
DEFAULT_STORM_FAILURE_RATE = 0.5


@dataclass
class DetectionConfig:
    """Thresholds for the detectors. Override per deployment."""

    io_ratio: float = DEFAULT_IO_RATIO
    io_ratio_min_calls: int = DEFAULT_MIN_CALLS
    velocity_multiplier: float = DEFAULT_VELOCITY_MULTIPLIER
    velocity_floor_usd: float = DEFAULT_VELOCITY_FLOOR_USD
    storm_min_failures: int = DEFAULT_STORM_MIN_FAILURES
    storm_failure_rate: float = DEFAULT_STORM_FAILURE_RATE
    window_minutes: int = 15
    include_anonymous: bool = False


@dataclass
class Anomaly:
    """One detected pattern, with the evidence that produced it."""

    kind: str                  # io_ratio | velocity | retry_storm
    subject: str               # api key, project or end user it was seen on
    subject_type: str          # key | project | end_user
    severity: str              # warn | critical
    detail: str
    evidence: Dict[str, object] = field(default_factory=dict)

    @property
    def headline(self) -> str:
        return self.detail

    def remedy(self) -> str:
        if self.kind == "io_ratio":
            return (
                "The prompt is being re-sent in full on every step. Cap the "
                "context (summarise or truncate history), and enable prompt "
                "caching so the repeated prefix is billed at the cache rate. "
                "OWASP attributes ~62% of agent bills to re-sent context."
            )
        if self.kind == "velocity":
            return (
                "Spend rate is far above this account's own baseline. Check for "
                "a recursive tool call or an unbounded fan-out across sub-agents. "
                "Set a hard budget on this key to bound the blast radius."
            )
        if self.kind == "retry_storm":
            return (
                "Most calls in this window failed. Failed calls still cost "
                "latency and often tokens. Check the error mix before "
                "optimising spend further."
            )
        return ""


def _row_get(row, key, default=None):
    try:
        value = row[key]
    except (IndexError, KeyError):
        return default
    return default if value is None else value


def _baseline_cost_per_minute(
    store: Store, days: int = 7, exclude_recent_minutes: int = 0
) -> float:
    """This account's own typical spend rate, USD per minute.

    A fixed dollar threshold would be wrong for everyone: $50/hour is an
    emergency for one team and a quiet afternoon for another. The baseline is
    therefore derived from the account's own history.

    ``exclude_recent_minutes`` must cover the detection window. Including the
    burst in its own baseline is self-defeating -- a large enough spike raises
    the average enough that the ratio test never fires, which is exactly the
    incident we are trying to catch.
    """
    upper = "datetime('now')"
    params: list = [f"-{days} days"]
    if exclude_recent_minutes > 0:
        upper = "datetime('now', ?)"
        params.append(f"-{exclude_recent_minutes} minutes")

    row = store.one(
        f"""SELECT COALESCE(SUM(cost_usd), 0) AS total,
                   COUNT(*) AS n
            FROM requests
            WHERE ts >= datetime('now', ?)
              AND ts < {upper}""",
        tuple(params),
    )
    if not row:
        return 0.0
    total = float(_row_get(row, "total", 0.0))
    n = int(_row_get(row, "n", 0))
    if n == 0 or total <= 0:
        return 0.0

    span = store.one(
        f"""SELECT MIN(ts) AS lo, MAX(ts) AS hi FROM requests
            WHERE ts >= datetime('now', ?)
              AND ts < {upper}""",
        tuple(params),
    )
    minutes = float(days) * 24 * 60
    if span is not None:
        lo, hi = _row_get(span, "lo"), _row_get(span, "hi")
        if lo and hi:
            try:
                start = datetime.strptime(str(lo), "%Y-%m-%d %H:%M:%S")
                end = datetime.strptime(str(hi), "%Y-%m-%d %H:%M:%S")
                observed = max((end - start).total_seconds() / 60.0, 1.0)
                minutes = max(min(observed, minutes), 1.0)
            except ValueError:
                pass
    return total / minutes


def detect_io_ratio(
    store: Store, config: DetectionConfig, *, now: Optional[datetime] = None
) -> List[Anomaly]:
    """Sustained, pathological input-to-output token ratios.

    This is the single highest-signal detector for agent loops, because a loop
    re-sends its whole history every step. Grouped by key and by project because
    a loop is usually scoped to one workload.
    """
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(minutes=config.window_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    anomalies: List[Anomaly] = []

    for subject_type, column in (("key", "api_key_id"), ("project", "project")):
        rows = store.query(
            f"""
            SELECT {column} AS subject,
                   COUNT(*)                                  AS calls,
                   COALESCE(SUM(input_tokens), 0)             AS input_tokens,
                   COALESCE(SUM(cached_input_tokens), 0)      AS cached_tokens,
                   COALESCE(SUM(output_tokens), 0)            AS output_tokens,
                   COALESCE(SUM(cost_usd), 0)                 AS cost_usd
            FROM requests
            WHERE ts >= ?
              AND cost_usd IS NOT NULL
              AND model <> ''
              AND {column} <> ''
            GROUP BY subject
            HAVING calls >= ?
            """,
            (since, config.io_ratio_min_calls),
        )
        for row in rows:
            subject = str(_row_get(row, "subject", ""))
            if not config.include_anonymous and subject in ("anonymous", "(none)"):
                continue
            inputs = int(_row_get(row, "input_tokens", 0)) + int(
                _row_get(row, "cached_tokens", 0)
            )
            outputs = int(_row_get(row, "output_tokens", 0))
            if outputs <= 0 or inputs <= 0:
                continue
            ratio = inputs / outputs
            if ratio < config.io_ratio:
                continue
            calls = int(_row_get(row, "calls", 0))
            cost = float(_row_get(row, "cost_usd", 0.0))
            anomalies.append(
                Anomaly(
                    kind="io_ratio",
                    subject=subject,
                    subject_type=subject_type,
                    severity="critical" if ratio >= config.io_ratio * 2 else "warn",
                    detail=(
                        f"{subject} is running a {ratio:.0f}:1 input-to-output ratio "
                        f"across {calls} calls in the last {config.window_minutes} min "
                        f"({inputs:,} in / {outputs:,} out, ${cost:.2f}). "
                        f"Normal traffic sits at 5:1-15:1."
                    ),
                    evidence={
                        "io_ratio": round(ratio, 2),
                        "calls": calls,
                        "input_tokens": inputs,
                        "output_tokens": outputs,
                        "cost_usd": round(cost, 6),
                        "threshold": config.io_ratio,
                    },
                )
            )
    return anomalies


def detect_velocity(
    store: Store, config: DetectionConfig, *, now: Optional[datetime] = None
) -> List[Anomaly]:
    """Spend bursts far above this account's own baseline rate."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(minutes=config.window_minutes)).strftime("%Y-%m-%d %H:%M:%S")
    # Exclude the detection window itself: a spike must not raise the baseline it
    # is measured against.
    baseline = _baseline_cost_per_minute(
        store, exclude_recent_minutes=config.window_minutes
    )
    if baseline <= 0:
        return []

    expected = baseline * config.window_minutes
    threshold = max(expected * config.velocity_multiplier, config.velocity_floor_usd)

    anomalies: List[Anomaly] = []
    rows = store.query(
        """
        SELECT api_key_id AS subject,
               COALESCE(SUM(cost_usd), 0) AS cost_usd,
               COUNT(*) AS calls
        FROM requests
        WHERE ts >= ?
        GROUP BY subject
        HAVING cost_usd > ?
        ORDER BY cost_usd DESC
        """,
        (since, threshold),
    )
    for row in rows:
        subject = str(_row_get(row, "subject", ""))
        if not config.include_anonymous and subject == "anonymous":
            continue
        cost = float(_row_get(row, "cost_usd", 0.0))
        per_minute = cost / max(config.window_minutes, 1)
        multiple = per_minute / baseline if baseline else 0.0
        anomalies.append(
            Anomaly(
                kind="velocity",
                subject=subject,
                subject_type="key",
                severity="critical" if multiple >= config.velocity_multiplier * 4 else "warn",
                detail=(
                    f"{subject} spent ${cost:.2f} in {config.window_minutes} min "
                    f"({multiple:.0f}x its own baseline rate of "
                    f"${baseline * 60:.2f}/hour)."
                ),
                evidence={
                    "cost_usd": round(cost, 6),
                    "window_minutes": config.window_minutes,
                    "cost_per_minute": round(per_minute, 6),
                    "baseline_per_minute": round(baseline, 6),
                    "multiple": round(multiple, 1),
                    "calls": int(_row_get(row, "calls", 0)),
                },
            )
        )
    return anomalies


def detect_retry_storm(
    store: Store, config: DetectionConfig, *, now: Optional[datetime] = None
) -> List[Anomaly]:
    """Bursts of failing calls. Retry storms are the second-most-costly pattern."""
    now = now or datetime.now(timezone.utc)
    since = (now - timedelta(minutes=config.window_minutes)).strftime("%Y-%m-%d %H:%M:%S")

    rows = store.query(
        """
        SELECT api_key_id AS subject,
               COUNT(*) AS calls,
               COALESCE(SUM(CASE WHEN status >= 400 OR error <> '' THEN 1 ELSE 0 END), 0)
                   AS failures
        FROM requests
        WHERE ts >= ?
        GROUP BY subject
        HAVING failures >= ?
        """,
        (since, config.storm_min_failures),
    )
    anomalies: List[Anomaly] = []
    for row in rows:
        calls = int(_row_get(row, "calls", 0))
        failures = int(_row_get(row, "failures", 0))
        if calls <= 0:
            continue
        rate = failures / calls
        if rate < config.storm_failure_rate:
            continue
        subject = str(_row_get(row, "subject", ""))
        if not config.include_anonymous and subject == "anonymous":
            continue
        anomalies.append(
            Anomaly(
                kind="retry_storm",
                subject=subject,
                subject_type="key",
                severity="critical" if failures >= config.storm_min_failures * 5 else "warn",
                detail=(
                    f"{subject} had {failures:,} failed calls out of {calls:,} "
                    f"({rate * 100:.0f}%) in the last {config.window_minutes} min."
                ),
                evidence={
                    "failures": failures,
                    "calls": calls,
                    "failure_rate": round(rate, 4),
                },
            )
        )
    return anomalies


def detect_all(
    store: Store, config: Optional[DetectionConfig] = None
) -> List[Anomaly]:
    """Run every detector, most severe first."""
    config = config or DetectionConfig()
    found: List[Anomaly] = []
    found += detect_io_ratio(store, config)
    found += detect_velocity(store, config)
    found += detect_retry_storm(store, config)
    order = {"critical": 0, "warn": 1}
    found.sort(key=lambda a: (order.get(a.severity, 9), a.subject))
    return found


# ---------------------------------------------------------------------------
# Real-time guard used on the request path
# ---------------------------------------------------------------------------
@dataclass
class LiveGuard:
    """In-memory loop detector for the hot path.

    Keeps a small rolling window per key so the gateway can flag a loop *while it
    is happening* rather than after the next report. Deliberately in-memory and
    bounded: this is a tripwire, not a database.
    """

    io_ratio: float = DEFAULT_IO_RATIO
    min_samples: int = 8
    window: int = 64

    _recent: Dict[str, List[float]] = field(default_factory=dict)

    def observe(self, api_key_id: str, input_tokens: int, output_tokens: int) -> Optional[str]:
        """Record one call. Returns a warning string when a loop is suspected."""
        if output_tokens <= 0:
            return None
        history = self._recent.setdefault(api_key_id, [])
        history.append(input_tokens / output_tokens)
        if len(history) > self.window:
            del history[: len(history) - self.window]
        if len(history) < self.min_samples:
            return None
        # Median, not mean: one legitimate document-analysis call should not
        # trip the detector on its own.
        median_ratio = statistics.median(history)
        if median_ratio < self.io_ratio:
            return None
        return (
            f"sustained input/output ratio {median_ratio:.0f}:1 over the last "
            f"{len(history)} calls (threshold {self.io_ratio:.0f}:1) — suspected "
            f"agent loop"
        )
