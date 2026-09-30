"""Calibration: what we predicted, and what actually happened.

This is the only asset in the project that a competitor cannot copy by reading our
code or the published research.

The published incident catalogue records what failures *cost*. It says nothing
about whether the recommended fix worked. Neither does any vendor's
documentation, because nobody publishes a distribution of realised savings. So
when a report says "worth $13,000 to $28,000 a month", that range currently rests
on an assumption we chose. After a dozen engagements it rests on what actually
happened in the previous eleven.

That difference is the whole product. A free tool can compute a token ratio; it
cannot tell a prospect what fraction of the predicted saving is usually realised,
because it has never followed an engagement to completion.

How it works, in three commands:

    llm-guard engage new --client "Acme Corp" --baseline-days 30
        Snapshot the current diagnosis as a baseline. Records every finding, the
        estimate, and the window it was measured over.

    llm-guard engage measure --client "Acme Corp" --days 30
        Pull fresh usage and compare like for like against the baseline window.

    llm-guard outcomes
        Calibration across engagements: predicted versus realised.

One design decision worth stating: **we record the miss as well as the hit.** A
table that only ever shows successes is a marketing asset, not a measurement one,
and the first prospective client who asks to see the failures would find that out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Optional, Tuple

from .storage import Store, iso

SCHEMA = """
CREATE TABLE IF NOT EXISTS engagements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    client          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    baseline_from   TEXT,
    baseline_to     TEXT,
    baseline_days   INTEGER NOT NULL,
    baseline_spend  REAL NOT NULL DEFAULT 0,
    baseline_requests INTEGER NOT NULL DEFAULT 0,
    predicted_low   REAL NOT NULL DEFAULT 0,
    predicted_high  REAL NOT NULL DEFAULT 0,
    status          TEXT NOT NULL DEFAULT 'open',   -- open | measured | closed
    notes           TEXT NOT NULL DEFAULT '',
    UNIQUE(client, created_at)
);
CREATE TABLE IF NOT EXISTS engagement_findings (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    engagement_id   INTEGER NOT NULL,
    key             TEXT NOT NULL,
    title           TEXT NOT NULL,
    confidence      TEXT NOT NULL DEFAULT '',
    predicted_low   REAL NOT NULL DEFAULT 0,
    predicted_high  REAL NOT NULL DEFAULT 0,
    evidence        TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (engagement_id) REFERENCES engagements(id) ON DELETE CASCADE
);
CREATE TABLE IF NOT EXISTS measurements (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    engagement_id   INTEGER NOT NULL,
    measured_at     TEXT NOT NULL,
    window_from     TEXT,
    window_to       TEXT,
    days            INTEGER NOT NULL,
    spend           REAL NOT NULL DEFAULT 0,
    requests        INTEGER NOT NULL DEFAULT 0,
    tokens          INTEGER NOT NULL DEFAULT 0,
    -- Per-finding realised numbers, keyed by finding key:
    -- {"context_bloat": {"before": 74.0, "after": 12.0}, ...}
    realised        TEXT NOT NULL DEFAULT '{}',
    note            TEXT NOT NULL DEFAULT '',
    FOREIGN KEY (engagement_id) REFERENCES engagements(id) ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS idx_eng_client ON engagements(client);
CREATE INDEX IF NOT EXISTS idx_meas_eng ON measurements(engagement_id);
"""


@dataclass
class Engagement:
    id: int
    client: str
    created_at: str
    baseline_from: str
    baseline_to: str
    baseline_days: int
    baseline_spend: float
    baseline_requests: int
    predicted_low: float
    predicted_high: float
    status: str
    notes: str
    findings: List[dict] = field(default_factory=list)

    @property
    def predicted_mid(self) -> float:
        return (self.predicted_low + self.predicted_high) / 2


@dataclass
class Realised:
    """One client's realised result, and how it compares with the prediction."""

    engagement: Engagement
    days: int
    spend: float
    requests: int
    tokens: int
    window_from: str
    window_to: str
    #: Scaled to a 30-day month for comparability with the prediction.
    monthly_spend: float
    monthly_baseline: float
    monthly_change: float          # negative means spend fell
    note: str = ""

    @property
    def realised_saving(self) -> float:
        """Monthly saving actually observed. Negative means spend rose."""
        return -self.monthly_change

    @property
    def ratio_to_prediction(self) -> Optional[float]:
        """Realised saving as a fraction of the midpoint predicted."""
        if self.engagement.predicted_mid <= 0:
            return None
        return self.realised_saving / self.engagement.predicted_mid


def _day(stamp: Optional[str]) -> str:
    """'YYYY-MM-DD' from any of the timestamp shapes this module passes around."""
    if not stamp:
        return ""
    return str(stamp)[:10] if " " not in str(stamp) else str(stamp)[:10]


def ensure_schema(store: Store) -> None:
    for statement in SCHEMA.strip().split(";"):
        if statement.strip():
            store.execute(statement)


# ---------------------------------------------------------------------------
# recording
# ---------------------------------------------------------------------------
def open_engagement(
    store: Store,
    client: str,
    *,
    baseline_days: int = 30,
    notes: str = "",
    now: Optional[datetime] = None,
) -> Engagement:
    """Snapshot the current diagnosis as this engagement's baseline.

    The baseline is the diagnosis itself, not a fresh computation: whatever the
    report claimed is what we later hold ourselves to. Recomputing at measurement
    time would let the goalposts move.
    """
    from .analytics import window_between
    from .diagnose import diagnose
    from .storage import utcnow

    ensure_schema(store)
    moment = now or utcnow()

    # Pin the baseline to explicit dates. A trailing window would silently follow
    # the clock forward, so a measurement taken three weeks later would compare
    # fresh traffic against fresh traffic and report no change.
    end = moment.astimezone(timezone.utc).date()
    start = end - timedelta(days=baseline_days)
    win = window_between(start.isoformat(), end.isoformat())
    d = diagnose(store, days=win.days, window=win)

    low = sum(f.monthly_saving_low for f in d.findings)
    high = sum(f.monthly_saving_high for f in d.findings)

    cur = store.execute(
        """INSERT INTO engagements
           (client, created_at, baseline_from, baseline_to, baseline_days,
            baseline_spend, baseline_requests, predicted_low, predicted_high,
            status, notes)
           VALUES (?,?,?,?,?,?,?,?,?,?,?)""",
        (client, iso(moment), d.first_ts, d.last_ts, baseline_days,
         d.total_spend, d.total_requests, low, high, "open", notes),
    )
    row = store.one("SELECT id FROM engagements WHERE client = ? ORDER BY id DESC LIMIT 1", (client,))
    engagement_id = int(row["id"])

    for f in d.ranked:
        store.execute(
            """INSERT INTO engagement_findings
               (engagement_id, key, title, confidence, predicted_low, predicted_high, evidence)
               VALUES (?,?,?,?,?,?,?)""",
            (engagement_id, f.key, f.title, f.confidence,
             f.monthly_saving_low, f.monthly_saving_high,
             json.dumps(f.evidence, default=str)),
        )

    return get_engagement(store, engagement_id)  # type: ignore[return-value]


def get_engagement(store: Store, engagement_id: int) -> Optional[Engagement]:
    ensure_schema(store)
    row = store.one("SELECT * FROM engagements WHERE id = ?", (engagement_id,))
    if row is None:
        return None
    findings = [
        dict(r) for r in store.query(
            "SELECT * FROM engagement_findings WHERE engagement_id = ? ORDER BY predicted_high DESC",
            (engagement_id,),
        )
    ]
    return Engagement(
        id=int(row["id"]), client=str(row["client"]), created_at=str(row["created_at"]),
        baseline_from=str(row["baseline_from"] or ""), baseline_to=str(row["baseline_to"] or ""),
        baseline_days=int(row["baseline_days"]), baseline_spend=float(row["baseline_spend"]),
        baseline_requests=int(row["baseline_requests"]),
        predicted_low=float(row["predicted_low"]), predicted_high=float(row["predicted_high"]),
        status=str(row["status"]), notes=str(row["notes"]), findings=findings,
    )


def latest_engagement(store: Store, client: str, *, status: Optional[str] = None) -> Optional[Engagement]:
    ensure_schema(store)
    sql = "SELECT id FROM engagements WHERE client = ?"
    params: List[object] = [client]
    if status:
        sql += " AND status = ?"
        params.append(status)
    sql += " ORDER BY id DESC LIMIT 1"
    row = store.one(sql, tuple(params))
    return get_engagement(store, int(row["id"])) if row else None


def list_engagements(store: Store) -> List[Engagement]:
    ensure_schema(store)
    rows = store.query("SELECT id FROM engagements ORDER BY id DESC")
    out = []
    for r in rows:
        e = get_engagement(store, int(r["id"]))
        if e:
            out.append(e)
    return out


def measure_engagement(
    store: Store,
    client: str,
    *,
    days: int = 30,
    since_baseline: Optional["object"] = None,
    note: str = "",
    now: Optional[datetime] = None,
) -> Tuple[Optional[Realised], List[str]]:
    """Compare the current window against the baseline, like for like.

    Returns the measurement and a list of warnings about comparability. The
    warnings matter more than the number: a spend drop caused by a traffic drop
    says nothing about whether the fix worked, and reporting it as a saving would
    be the kind of thing that ends a client relationship.
    """
    from .analytics import window_between
    from .diagnose import diagnose
    from .storage import utcnow

    engagement = latest_engagement(store, client, status="open") or latest_engagement(store, client)
    if engagement is None:
        return None, [f"no engagement recorded for {client!r}"]

    moment = now or utcnow()

    # Measure the window immediately after the baseline by default. This is the
    # whole point: a follow-up has to look at different traffic from the baseline,
    # otherwise the comparison measures the passage of time rather than the fix.
    warnings: List[str] = []
    if since_baseline is None:
        if not engagement.baseline_to:
            return None, ["baseline has no recorded end date; pass --start/--end explicitly"]
        start = engagement.baseline_to
        end = (datetime.strptime(start, "%Y-%m-%d") + timedelta(days=days)).date().isoformat()
        since_baseline = window_between(start, end)
        today = moment.astimezone(timezone.utc).date().isoformat()
        if end > today:
            inside = window_between(start, today)
            if inside.days <= 1:
                return None, [
                    f"the follow-up window starts {start} and today is {today}; there is "
                    f"not enough traffic after the baseline yet to measure anything"
                ]
            warnings.append(
                f"only {inside.days} day(s) of the {days}-day follow-up window have elapsed. "
                f"The result is partial and scaled up, so treat it as a direction, not a number"
            )
            since_baseline = inside
    win = since_baseline

    d = diagnose(store, days=win.days, window=win)
    if d.total_requests == 0:
        return None, [f"no traffic between {win.start} and {win.end}"]

    # A follow-up window with a fraction of the baseline's traffic cannot support
    # a before/after comparison. Dividing a handful of requests by the window
    # length and scaling to 30 days produces a large, confident and meaningless
    # number, which is worse than declining to answer.
    if engagement.baseline_requests:
        coverage = d.total_requests / engagement.baseline_requests
        if coverage < 0.25:
            return None, [
                f"only {d.total_requests:,} request(s) between {_day(win.start)} and "
                f"{_day(win.end)}, against {engagement.baseline_requests:,} in the baseline "
                f"({coverage * 100:.0f}%). Not enough comparable traffic to measure a change"
            ]

    monthly_spend = d.monthly(d.total_spend)
    monthly_baseline = engagement.baseline_spend / max(engagement.baseline_days, 1) * 30.0

    if win.days != engagement.baseline_days:
        warnings.append(
            f"measurement window is {win.days} days against a {engagement.baseline_days}-day "
            f"baseline; both are scaled to 30 days but seasonality will not cancel"
        )
    # Volume is the control. A spend drop caused by a traffic drop says nothing
    # about whether the fix worked, and reporting it as a saving is the kind of
    # thing that ends a client relationship.
    if engagement.baseline_requests:
        per_month_base = engagement.baseline_requests / max(engagement.baseline_days, 1) * 30.0
        per_month_now = d.total_requests / max(win.days, 1) * 30.0
        if per_month_base > 0:
            volume_change = (per_month_now - per_month_base) / per_month_base
            if abs(volume_change) >= 0.15:
                direction = "fell" if volume_change < 0 else "rose"
                warnings.append(
                    f"request volume {direction} {abs(volume_change) * 100:.0f}% against the "
                    f"baseline. Spend moved for reasons other than the fix, so attribute with care"
                )
    else:
        warnings.append("no baseline request count; volume change cannot be assessed")

    realised_evidence = _per_finding_change(store, engagement, win)

    store.execute(
        """INSERT INTO measurements
           (engagement_id, measured_at, window_from, window_to, days, spend, requests,
            tokens, realised, note)
           VALUES (?,?,?,?,?,?,?,?,?,?)""",
        (engagement.id, iso(moment), _day(win.start), _day(win.end), win.days,
         d.total_spend, d.total_requests, d.total_tokens,
         json.dumps(realised_evidence, default=str), note),
    )
    store.execute("UPDATE engagements SET status = 'measured' WHERE id = ?", (engagement.id,))

    return Realised(
        engagement=engagement, days=win.days, spend=d.total_spend, requests=d.total_requests,
        tokens=d.total_tokens, window_from=_day(win.start), window_to=_day(win.end),
        monthly_spend=monthly_spend, monthly_baseline=monthly_baseline,
        monthly_change=monthly_spend - monthly_baseline, note=note,
    ), warnings


def _per_finding_change(store: Store, engagement: Engagement, win) -> Dict[str, dict]:
    """Re-measure the specific signal behind each finding, before and after.

    The headline spend number is confounded by traffic changes. The signal behind
    a finding usually is not: an input/output ratio is a property of the workload,
    not of its volume, so it moves only if the workload changed.
    """
    from .incidents import measure

    after = measure(store, window_sql=win.sql, params=win.params)

    out: Dict[str, dict] = {}
    for finding in engagement.findings:
        key = str(finding["key"])
        evidence = {}
        try:
            evidence = json.loads(finding.get("evidence") or "{}")
        except (TypeError, ValueError):
            evidence = {}

        pair: Dict[str, object] = {}
        if key == "context_bloat":
            if evidence.get("ratio"):
                pair["input_output_ratio"] = {"before": evidence["ratio"]}
            out.setdefault(key, {})["signals"] = pair
        elif key == "cache_opportunity":
            if "cache_hit_rate" in evidence:
                pair["cache_hit_rate"] = {"before": evidence["cache_hit_rate"]}
            out.setdefault(key, {})["signals"] = pair

    # Attach the current value of the same global measures.
    for key, payload in out.items():
        signals = payload.get("signals", {})
        if "input_output_ratio" in signals and after.get("output_tokens"):
            signals["input_output_ratio"]["after"] = round(
                after["input_tokens"] / after["output_tokens"], 2)
        if "cache_hit_rate" in signals:
            signals["cache_hit_rate"]["after"] = round(after.get("cache_hit_rate", 0.0), 4)
    return out


# ---------------------------------------------------------------------------
# calibration
# ---------------------------------------------------------------------------
@dataclass
class Calibration:
    measured: int
    total: int
    ratios: List[float] = field(default_factory=list)

    @property
    def median_ratio(self) -> Optional[float]:
        if not self.ratios:
            return None
        s = sorted(self.ratios)
        n = len(s)
        return s[n // 2] if n % 2 else (s[n // 2 - 1] + s[n // 2]) / 2

    @property
    def mean_ratio(self) -> Optional[float]:
        return sum(self.ratios) / len(self.ratios) if self.ratios else None

    @property
    def overpredicted(self) -> int:
        return sum(1 for r in self.ratios if r < 1.0)

    @property
    def has_sample(self) -> bool:
        return len(self.ratios) >= 3


def calibrate(store: Store) -> Calibration:
    """How well do our predictions hold up, across every measured engagement?"""
    ensure_schema(store)
    total_row = store.one("SELECT COUNT(*) AS n FROM engagements")
    total = int(total_row["n"]) if total_row else 0

    ratios: List[float] = []
    for row in store.query("SELECT id FROM engagements WHERE status IN ('measured','closed')"):
        e = get_engagement(store, int(row["id"]))
        if e is None or e.predicted_mid <= 0:
            continue
        m = store.one(
            "SELECT * FROM measurements WHERE engagement_id = ? ORDER BY id DESC LIMIT 1",
            (e.id,),
        )
        if m is None:
            continue
        monthly = float(m["spend"]) / max(int(m["days"]), 1) * 30.0
        baseline = e.baseline_spend / max(e.baseline_days, 1) * 30.0
        saving = baseline - monthly
        ratios.append(saving / e.predicted_mid)
    return Calibration(measured=len(ratios), total=total, ratios=ratios)


def describe_calibration(store: Store) -> str:
    c = calibrate(store)
    lines = ["Prediction calibration", ""]
    lines.append(f"  engagements recorded     {c.total}")
    lines.append(f"  with a measurement       {c.measured}")
    if not c.ratios:
        lines.append("")
        lines.append("  Nothing measured yet. Until there is, every saving estimate in a")
        lines.append("  report is an assumption we chose, not a rate we have observed.")
        return "\n".join(lines)
    assert c.median_ratio is not None
    lines.append(f"  median realised/predicted {c.median_ratio:.2f}")
    if c.mean_ratio is not None:
        lines.append(f"  mean realised/predicted   {c.mean_ratio:.2f}")
    lines.append(f"  over-predicted            {c.overpredicted} of {c.measured}")
    lines.append("")
    if not c.has_sample:
        lines.append(
            f"  {c.measured} measurement(s). Too few to quote. Three is the minimum worth"
        )
        lines.append("  reporting and ten is where it starts to mean something.")
    else:
        lines.append(
            f"  Based on {c.measured} measurement(s), a finding predicted at $X has"
        )
        lines.append(f"  historically realised about ${c.median_ratio:.2f}X.")
    return "\n".join(lines)
