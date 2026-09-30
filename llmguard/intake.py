"""Get a customer's usage data in, without asking them to deploy anything first.

This is the piece that decides whether the service is sellable. If the intake step
is "install our gateway in your VPC", the conversation ends there, because a
prospect will not deploy infrastructure before they know whether the diagnosis is
worth paying for. So the first thing built here is the path of least resistance:

  1. **Their admin API.** Both providers expose organisation-level usage and cost
     endpoints. One key, one read-only request, no deployment.
  2. **A plain file.** CSV or JSON in a documented shape, for anyone who would
     rather not hand over a key, or who already exports this somewhere.
  3. **An existing llm-guard database.** For anyone already running the gateway.

Three honest limitations of provider-level data, which the diagnosis has to
respect and the report has to state:

  * **It is bucketed, not per request.** You get daily or hourly totals, so
    latency, status codes and exact request counts are not available. The checks
    that depend on those are skipped rather than guessed at.
  * **Attribution is coarse.** Grouping is by model, project or workspace, not by
    end user. Per-customer unit economics cannot be derived from it.
  * **There is no way to tell why.** Token totals show the shape of a problem,
    never its cause.

That is still enough to find a loop, because the shape is the point: a context
loop shows up as a lopsided input/output ratio whether you measure it per request
or per day.
"""

from __future__ import annotations

import csv
import io
import json
import urllib.error
import urllib.parse
import urllib.request
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from decimal import Decimal
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from .pricing import TokenUsage, compute_cost, get_price
from .storage import RequestRecord, Store, iso

# ---------------------------------------------------------------------------
# Canonical shape
# ---------------------------------------------------------------------------
# Everything is normalised into this before it reaches the database. One row is
# one bucket: a day, an hour, or a workspace/model combination.
CANONICAL_FIELDS = (
    "bucket_start",      # ISO-8601, UTC
    "provider",          # openai | anthropic | google | other
    "model",
    "input_tokens",          # uncached input
    "cached_input_tokens",   # cache reads
    "cache_write_tokens",    # cache creation
    "output_tokens",
    "cost_usd",          # optional; blank means "compute from the price table"
    "workspace",         # optional attribution
    "api_key_id",        # optional attribution
)


@dataclass
class ImportResult:
    rows_in: int
    rows_kept: int
    rows_skipped: int
    total_tokens: int
    total_cost_usd: float
    unpriced_models: List[str]
    notes: List[str]
    source: str

    def describe(self) -> str:
        lines = [
            f"source            {self.source}",
            f"rows read         {self.rows_in:,}",
            f"rows imported     {self.rows_kept:,}",
            f"tokens            {self.total_tokens:,}",
            f"cost              ${self.total_cost_usd:,.2f}",
        ]
        if self.rows_skipped:
            lines.append(f"rows skipped      {self.rows_skipped:,} (see notes)")
        if self.unpriced_models:
            lines.append(
                "unpriced          " + ", ".join(sorted(set(self.unpriced_models))[:8])
            )
            lines.append(
                "                  their cost is recorded as unknown, not zero"
            )
        for note in self.notes:
            lines.append(f"note              {note}")
        return "\n".join(lines)


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
def _int(value) -> int:
    if value is None or value == "":
        return 0
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return 0


def _num(value) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _parse_time(value: str) -> Optional[datetime]:
    """Accept the handful of timestamp shapes these APIs actually emit."""
    if not value:
        return None
    text = str(value).strip().replace("Z", "+00:00")
    for fmt in (
        "%Y-%m-%dT%H:%M:%S%z",
        "%Y-%m-%dT%H:%M:%S.%f%z",
        "%Y-%m-%dT%H:%M:%S",
        "%Y-%m-%d %H:%M:%S",
        "%Y-%m-%d",
        "%Y-%m",
    ):
        try:
            parsed = datetime.strptime(text, fmt)
            if parsed.tzinfo is None:
                parsed = parsed.replace(tzinfo=timezone.utc)
            return parsed.astimezone(timezone.utc)
        except ValueError:
            continue
    # Unix seconds, which some exports use
    try:
        return datetime.fromtimestamp(int(float(text)), tz=timezone.utc)
    except (TypeError, ValueError):
        return None


def _bucket_to_record(
    row: Dict[str, object], *, default_provider: str, default_model: str = ""
) -> Tuple[Optional[RequestRecord], Optional[str]]:
    """Turn one canonical row into a RequestRecord.

    Returns (record, skip_reason). A bucket becomes a single synthetic request
    carrying the bucket's totals: the schema is per request, but the arithmetic
    only cares about token counts, and pretending a day is one call would be a
    lie the report would then repeat.

    To stay honest, a bucket record is clearly marked and its latency and status
    are left neutral.
    """
    when = _parse_time(str(row.get("bucket_start", "")))
    if when is None:
        return None, "unparseable bucket_start"

    provider = str(row.get("provider") or default_provider or "other").lower()
    model = str(row.get("model") or default_model or "").strip()
    if not model:
        return None, "no model"

    usage = TokenUsage(
        input=_int(row.get("input_tokens")),
        cached_input=_int(row.get("cached_input_tokens")),
        cache_write=_int(row.get("cache_write_tokens")),
        output=_int(row.get("output_tokens")),
    )
    if usage.total == 0:
        return None, "no tokens"

    cost = _num(row.get("cost_usd"))
    if cost is None:
        computed = compute_cost(model, usage)
        cost = None if computed is None else float(computed)

    return (
        RequestRecord(
            provider=provider,
            model=model,
            input_tokens=usage.input,
            cached_input_tokens=usage.cached_input,
            cache_write_tokens=usage.cache_write,
            output_tokens=usage.output,
            cost_usd=cost,
            latency_ms=0,
            status=200,
            streamed=False,
            api_key_id=str(row.get("api_key_id") or row.get("workspace") or "imported"),
            end_user="",
            project=str(row.get("workspace") or ""),
            request_id="",
            error="",
            ts=iso(when),
        ),
        None,
    )


def _ingest(
    store: Store,
    rows: Iterable[Dict[str, object]],
    *,
    source: str,
    default_provider: str,
) -> ImportResult:
    kept: List[RequestRecord] = []
    skipped = 0
    notes: List[str] = []
    unpriced: List[str] = []
    tokens = 0
    cost = 0.0
    total = 0

    for row in rows:
        total += 1
        record, reason = _bucket_to_record(row, default_provider=default_provider)
        if record is None:
            skipped += 1
            if reason and reason not in notes:
                notes.append(f"{skipped} row(s) skipped: {reason}")
            continue
        if record.cost_usd is None and record.model not in unpriced:
            unpriced.append(record.model)
        tokens += record.input_tokens + record.cached_input_tokens + \
            record.cache_write_tokens + record.output_tokens
        cost += record.cost_usd or 0.0
        kept.append(record)

    if kept:
        store.insert_many(kept)
        # Record that this data is bucketed, not per request. The diagnosis reads
        # this: thresholds and wording both have to change, because 8 daily
        # buckets is a month of history while 8 requests is nothing.
        store.set_meta("granularity", "bucket")

    return ImportResult(
        rows_in=total,
        rows_kept=len(kept),
        rows_skipped=skipped,
        total_tokens=tokens,
        total_cost_usd=cost,
        unpriced_models=unpriced,
        notes=notes,
        source=source,
    )


# ---------------------------------------------------------------------------
# 1. file intake
# ---------------------------------------------------------------------------
def import_csv(store: Store, text: str, *, default_provider: str = "openai") -> ImportResult:
    reader = csv.DictReader(io.StringIO(text))
    if not reader.fieldnames:
        return ImportResult(0, 0, 0, 0, 0.0, [], ["file has no header row"], "csv")
    # Tolerate a couple of common header spellings so people do not have to
    # rename columns by hand.
    alias = {
        "date": "bucket_start",
        "timestamp": "bucket_start",
        "time": "bucket_start",
        "start_time": "bucket_start",
        "start": "bucket_start",
        "prompt_tokens": "input_tokens",
        "input": "input_tokens",
        "uncached_input_tokens": "input_tokens",
        "cached_tokens": "cached_input_tokens",
        "cache_read_input_tokens": "cached_input_tokens",
        "cache_read": "cached_input_tokens",
        "cache_creation_input_tokens": "cache_write_tokens",
        "cache_write": "cache_write_tokens",
        "completion_tokens": "output_tokens",
        "output": "output_tokens",
        "cost": "cost_usd",
        "usd": "cost_usd",
        "amount_usd": "cost_usd",
    }
    normalised = []
    for raw in reader:
        row = {}
        for key, value in raw.items():
            if key is None:
                continue
            clean = key.strip().lower().replace(" ", "_")
            row[alias.get(clean, clean)] = value
        normalised.append(row)
    return _ingest(store, normalised, source="csv", default_provider=default_provider)


def import_json(store: Store, text: str, *, default_provider: str = "openai") -> ImportResult:
    try:
        payload = json.loads(text)
    except ValueError as exc:
        return ImportResult(0, 0, 0, 0, 0.0, [], [f"invalid json: {exc}"], "json")

    if isinstance(payload, dict):
        rows = payload.get("data") or payload.get("rows") or payload.get("results") or []
    elif isinstance(payload, list):
        rows = payload
    else:
        rows = []

    normalised = []
    for raw in rows:
        if not isinstance(raw, dict):
            continue
        row = {str(k).strip().lower(): v for k, v in raw.items()}
        normalised.append(row)
    return _ingest(store, normalised, source="json", default_provider=default_provider)


# ---------------------------------------------------------------------------
# 2. Anthropic Admin API
# ---------------------------------------------------------------------------
ANTHROPIC_BASE = "https://api.anthropic.com/v1/organizations"


def fetch_anthropic_usage(
    admin_key: str,
    *,
    days: int = 30,
    bucket_width: str = "1d",
    timeout: float = 60.0,
) -> List[Dict[str, object]]:
    """Fetch token usage from the Anthropic Admin API, grouped by model.

    Endpoint and response shape verified against Anthropic's own cookbook
    (platform.claude.com/cookbook/observability-usage-cost-api). Token fields:
    ``uncached_input_tokens``, ``output_tokens``, ``cache_read_input_tokens``,
    and ``cache_creation.{ephemeral_5m_input_tokens,ephemeral_1h_input_tokens}``.
    """
    if not admin_key.startswith("sk-ant-admin"):
        raise ValueError(
            "this needs an Anthropic Admin API key (sk-ant-admin...), not a normal API key"
        )
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)

    rows: List[Dict[str, object]] = []
    page: Optional[str] = None
    for _ in range(40):   # hard stop; a bad cursor must not loop forever
        params = {
            "starting_at": start.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "ending_at": end.strftime("%Y-%m-%dT%H:%M:%SZ"),
            "bucket_width": bucket_width,
            "group_by[]": "model",
            "limit": "31",
        }
        if page:
            params["page"] = page
        query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        request = urllib.request.Request(
            f"{ANTHROPIC_BASE}/usage_report/messages?{query}",
            headers={
                "x-api-key": admin_key,
                "anthropic-version": "2023-06-01",
                "accept": "application/json",
            },
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        for bucket in payload.get("data", []):
            when = bucket.get("starting_at", "")
            for result in bucket.get("results", []):
                creation = result.get("cache_creation") or {}
                rows.append(
                    {
                        "bucket_start": when,
                        "provider": "anthropic",
                        "model": result.get("model", ""),
                        "input_tokens": result.get("uncached_input_tokens", 0),
                        "cached_input_tokens": result.get("cache_read_input_tokens", 0),
                        "cache_write_tokens": _int(creation.get("ephemeral_5m_input_tokens"))
                        + _int(creation.get("ephemeral_1h_input_tokens")),
                        "output_tokens": result.get("output_tokens", 0),
                        "workspace": result.get("workspace_id", "") or "",
                        "api_key_id": result.get("api_key_id", "") or "",
                    }
                )
        if not payload.get("has_more"):
            break
        page = payload.get("next_page")
        if not page:
            break
    return rows


def fetch_anthropic_cost(
    admin_key: str, *, days: int = 30, timeout: float = 60.0
) -> Dict[str, float]:
    """Daily cost from the Anthropic cost endpoint, keyed by YYYY-MM-DD.

    Amounts come back in minor units (cents) as strings, hence the /100.
    """
    end = datetime.now(timezone.utc).replace(hour=0, minute=0, second=0, microsecond=0)
    start = end - timedelta(days=days)
    out: Dict[str, float] = {}
    query = (
        f"starting_at={urllib.parse.quote(start.strftime('%Y-%m-%dT%H:%M:%SZ'))}"
        f"&ending_at={urllib.parse.quote(end.strftime('%Y-%m-%dT%H:%M:%SZ'))}"
        f"&bucket_width=1d&limit=31"
    )
    request = urllib.request.Request(
        f"{ANTHROPIC_BASE}/cost_report?{query}",
        headers={
            "x-api-key": admin_key,
            "anthropic-version": "2023-06-01",
            "accept": "application/json",
        },
    )
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    for bucket in payload.get("data", []):
        day = str(bucket.get("starting_at", ""))[:10]
        total = 0.0
        for result in bucket.get("results", []):
            amount = result.get("amount")
            if isinstance(amount, str):
                try:
                    amount = float(amount)
                except ValueError:
                    amount = 0.0
            total += float(amount or 0.0)
        if day:
            out[day] = total / 100.0
    return out


def import_anthropic_admin(
    store: Store, admin_key: str, *, days: int = 30
) -> ImportResult:
    """Full Anthropic intake: usage for the shape, cost for the truth."""
    rows = fetch_anthropic_usage(admin_key, days=days)
    result = _ingest(store, rows, source="anthropic admin api", default_provider="anthropic")
    try:
        costs = fetch_anthropic_cost(admin_key, days=days)
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"cost endpoint unavailable ({exc}); costs computed from the price table")
        return result
    if costs:
        _reconcile_costs(store, costs)
        billed = sum(costs.values())
        result.notes.append(
            f"cost endpoint reported ${billed:,.2f}; per-row costs recomputed against it"
        )
        result.total_cost_usd = billed
    return result


def _reconcile_costs(store: Store, daily: Dict[str, float]) -> None:
    """Replace computed daily cost with what the provider actually billed.

    The provider's own cost figure outranks our price table, always. If the two
    disagree, the table is wrong, and a diagnosis built on a wrong table would
    recommend changes that do not save what it claims.
    """
    for day, billed in daily.items():
        rows = store.query(
            "SELECT id, cost_usd FROM requests WHERE substr(ts,1,10) = ?", (day,)
        )
        if not rows:
            continue
        computed = sum(float(r["cost_usd"] or 0.0) for r in rows)
        if computed <= 0:
            # Nothing priced: spread the billed amount by token share.
            total_tokens = sum(
                _int(r["input_tokens"]) + _int(r["cached_input_tokens"])
                + _int(r["cache_write_tokens"]) + _int(r["output_tokens"])
                for r in store.query(
                    "SELECT input_tokens,cached_input_tokens,cache_write_tokens,"
                    "output_tokens FROM requests WHERE substr(ts,1,10) = ?",
                    (day,),
                )
            )
            if total_tokens <= 0:
                continue
            for row in store.query(
                "SELECT id, input_tokens, cached_input_tokens, cache_write_tokens, "
                "output_tokens FROM requests WHERE substr(ts,1,10) = ?",
                (day,),
            ):
                share = (
                    _int(row["input_tokens"]) + _int(row["cached_input_tokens"])
                    + _int(row["cache_write_tokens"]) + _int(row["output_tokens"])
                ) / total_tokens
                store.execute(
                    "UPDATE requests SET cost_usd = ? WHERE id = ?",
                    (billed * share, row["id"]),
                )
            continue
        factor = billed / computed
        store.execute(
            "UPDATE requests SET cost_usd = cost_usd * ? WHERE substr(ts,1,10) = ?",
            (factor, day),
        )


# ---------------------------------------------------------------------------
# 3. OpenAI Admin API
# ---------------------------------------------------------------------------
OPENAI_BASE = "https://api.openai.com/v1/organization"


def fetch_openai_usage(
    admin_key: str,
    *,
    days: int = 30,
    bucket_width: str = "1d",
    timeout: float = 60.0,
) -> List[Dict[str, object]]:
    """Fetch token usage from the OpenAI organisation usage endpoint.

    Grouped by model so the output matches the Anthropic path. The response shape
    used here (``data[].results[]`` with ``input_tokens``, ``output_tokens`` and
    ``input_cached_tokens``) follows the documented organisation usage API. If a
    given organisation returns a different shape, the caller gets zero rows and a
    note rather than a silently wrong import, so verify on the first run.
    """
    if not admin_key.startswith("sk-admin"):
        # Not fatal: some orgs still use a project key with org read scope.
        pass
    end = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp())
    start = end - days * 86400

    rows: List[Dict[str, object]] = []
    page: Optional[str] = None
    for _ in range(40):
        params = {
            "start_time": start,
            "end_time": end,
            "bucket_width": bucket_width,
            "group_by": "model",
            "limit": 31,
        }
        if page:
            params["page"] = page
        query = "&".join(f"{k}={urllib.parse.quote(str(v))}" for k, v in params.items())
        request = urllib.request.Request(
            f"{OPENAI_BASE}/usage/completions?{query}",
            headers={"Authorization": f"Bearer {admin_key}", "accept": "application/json"},
        )
        with urllib.request.urlopen(request, timeout=timeout) as response:
            payload = json.loads(response.read())
        for bucket in payload.get("data", []):
            when = bucket.get("start_time")
            stamp = (
                datetime.fromtimestamp(int(when), tz=timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")
                if when else ""
            )
            for result in bucket.get("results", []):
                rows.append(
                    {
                        "bucket_start": stamp or str(bucket.get("start_time", "")),
                        "provider": "openai",
                        "model": result.get("model", ""),
                        "input_tokens": _int(result.get("input_tokens"))
                        - _int(result.get("input_cached_tokens")),
                        "cached_input_tokens": _int(result.get("input_cached_tokens")),
                        "cache_write_tokens": 0,
                        "output_tokens": result.get("output_tokens", 0),
                        "project": result.get("project_id", "") or "",
                        "api_key_id": result.get("api_key_id", "") or "",
                    }
                )
        if not payload.get("has_more"):
            break
        page = payload.get("next_page")
        if not page:
            break
    return rows


def fetch_openai_cost(admin_key: str, *, days: int = 30, timeout: float = 60.0) -> Dict[str, float]:
    """Daily cost from the OpenAI costs endpoint. Amounts are in dollars."""
    end = int(datetime.now(timezone.utc).replace(
        hour=0, minute=0, second=0, microsecond=0).timestamp())
    start = end - days * 86400
    query = f"start_time={start}&end_time={end}&bucket_width=1d&limit=31"
    request = urllib.request.Request(
        f"{OPENAI_BASE}/costs?{query}",
        headers={"Authorization": f"Bearer {admin_key}", "accept": "application/json"},
    )
    out: Dict[str, float] = {}
    with urllib.request.urlopen(request, timeout=timeout) as response:
        payload = json.loads(response.read())
    for bucket in payload.get("data", []):
        when = bucket.get("start_time")
        day = (
            datetime.fromtimestamp(int(when), tz=timezone.utc).strftime("%Y-%m-%d")
            if when else ""
        )
        total = 0.0
        for result in bucket.get("results", []):
            amount = result.get("amount") or {}
            value = amount.get("value") if isinstance(amount, dict) else amount
            total += float(value or 0.0)
        if day:
            out[day] = total
    return out


def import_openai_admin(store: Store, admin_key: str, *, days: int = 30) -> ImportResult:
    rows = fetch_openai_usage(admin_key, days=days)
    result = _ingest(store, rows, source="openai admin api", default_provider="openai")
    if not rows:
        result.notes.append(
            "the usage endpoint returned no buckets. Check that the key has "
            "organisation read scope and that there is traffic in the window."
        )
        return result
    try:
        costs = fetch_openai_cost(admin_key, days=days)
    except Exception as exc:  # noqa: BLE001
        result.notes.append(f"cost endpoint unavailable ({exc}); costs computed from the price table")
        return result
    if costs:
        _reconcile_costs(store, costs)
        billed = sum(costs.values())
        result.notes.append(
            f"cost endpoint reported ${billed:,.2f}; per-row costs recomputed against it"
        )
        result.total_cost_usd = billed
    return result

