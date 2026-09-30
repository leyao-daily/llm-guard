"""Render the value report as terminal text, JSON or CSV.

The terminal output is the primary surface: it must read like advice, not like
a dashboard. Every section ends with what to do about it.
"""

from __future__ import annotations

import csv
import io
import json
import shutil
from dataclasses import asdict, is_dataclass
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional

from .analytics import BudgetStatus, BreakdownRow, Totals, ValueReport

# ---------------------------------------------------------------------------
# terminal styling (no dependencies; degrade to plain text when not a TTY)
# ---------------------------------------------------------------------------
_ANSI = {
    "reset": "\033[0m",
    "bold": "\033[1m",
    "dim": "\033[2m",
    "red": "\033[31m",
    "green": "\033[32m",
    "yellow": "\033[33m",
    "blue": "\033[34m",
    "magenta": "\033[35m",
    "cyan": "\033[36m",
}


class Style:
    def __init__(self, enabled: bool):
        self.enabled = enabled

    def __call__(self, text: str, *codes: str) -> str:
        if not self.enabled or not codes:
            return text
        prefix = "".join(_ANSI.get(c, "") for c in codes)
        return f"{prefix}{text}{_ANSI['reset']}"


def _width(default: int = 88) -> int:
    try:
        return max(60, min(shutil.get_terminal_size((default, 24)).columns, 120))
    except Exception:  # noqa: BLE001
        return default


def money(value: float, places: int = 2) -> str:
    if value == 0:
        return "$0.00"
    if abs(value) < 0.01:
        return f"${value:.5f}"
    return f"${value:,.{places}f}"


def _pct(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    return f"{value * 100:+.1f}%"


def _bar(fraction: float, width: int = 18, char: str = "█") -> str:
    fraction = max(0.0, min(fraction, 1.0))
    filled = int(round(fraction * width))
    return char * filled + "·" * (width - filled)


def sparkline(values: List[float]) -> str:
    """Unicode sparkline; good enough to see a trend in one terminal line."""
    if not values:
        return ""
    blocks = "▁▂▃▄▅▆▇█"
    lo, hi = min(values), max(values)
    if hi - lo < 1e-12:
        return blocks[3] * len(values)
    out = []
    for v in values:
        idx = int(round((v - lo) / (hi - lo) * (len(blocks) - 1)))
        out.append(blocks[idx])
    return "".join(out)


# ---------------------------------------------------------------------------
# serialisation
# ---------------------------------------------------------------------------
def report_to_dict(report: Any) -> Dict[str, Any]:
    """Convert a report (or any dataclass tree) to plain JSON-safe types."""
    if is_dataclass(report) and not isinstance(report, type):
        return {k: report_to_dict(v) for k, v in asdict(report).items()}
    if isinstance(report, dict):
        return {k: report_to_dict(v) for k, v in report.items()}
    if isinstance(report, (list, tuple)):
        return [report_to_dict(v) for v in report]
    return report


def render_json(report: ValueReport) -> str:
    payload = report_to_dict(report)
    payload["generated_at"] = datetime.now(timezone.utc).isoformat(timespec="seconds")
    payload["derived"] = {
        "cost_delta_pct": report.cost_delta_pct,
        "request_delta_pct": report.request_delta_pct,
        "projected_month_usd": report.projected_month_usd,
        "cache_opportunity_usd": report.cache_opportunity_usd(),
        "cache_hit_rate": report.current.cache_hit_rate,
        "cost_per_request": report.current.cost_per_request,
    }
    return json.dumps(payload, ensure_ascii=False, indent=2, default=str)


def render_csv(report: ValueReport) -> str:
    """Flat CSV of the per-model breakdown plus a TOTAL row (spreadsheet-ready)."""
    buf = io.StringIO()
    writer = csv.writer(buf)
    writer.writerow(
        ["model", "requests", "cost_usd", "cost_per_request", "total_tokens", "share"]
    )
    for row in report.by_model:
        writer.writerow(
            [
                row.label,
                row.requests,
                f"{row.cost_usd:.6f}",
                f"{row.cost_per_request:.6f}",
                row.total_tokens,
                f"{row.share:.4f}",
            ]
        )
    t = report.current
    writer.writerow(
        ["TOTAL", t.requests, f"{t.cost_usd:.6f}", f"{t.cost_per_request:.6f}", t.total_tokens, "1.0000"]
    )
    return buf.getvalue()


# ---------------------------------------------------------------------------
# terminal report
# ---------------------------------------------------------------------------
def render_terminal(
    report: ValueReport,
    *,
    color: bool = True,
    verbose: bool = False,
    anomalies: Optional[List[Any]] = None,
) -> str:
    s = Style(color)
    w = _width()
    out: List[str] = []
    add = out.append

    cur, prev = report.current, report.previous

    # ---- header ----
    add("")
    add(s("  LLM Spend Report", "bold"))
    add(s(f"  window: last {report.days} days   ·   {report.total_requests_ever:,} requests recorded in total", "dim"))
    add(s("  " + "─" * (w - 4), "dim"))
    add("")

    # ---- headline ----
    add(f"  {'Total spend':<22}{s(money(cur.cost_usd), 'bold', 'cyan')}")
    delta = report.cost_delta_pct
    if delta is None:
        add(s(f"  {'vs previous period':<22}n/a (no prior data)", "dim"))
    else:
        tone = "red" if delta > 0 else "green"
        arrow = "▲" if delta > 0 else "▼"
        add(
            f"  {'vs previous period':<22}{s(f'{arrow} {_pct(delta)}', tone)}"
            f"{s(f'  ({money(prev.cost_usd)} before)', 'dim')}"
        )
    add(f"  {'Requests':<22}{cur.requests:,}" + s(f"   ({_pct(report.request_delta_pct)} vs prior)", "dim"))
    add(f"  {'Cost per request':<22}{money(cur.cost_per_request, 4)}")
    add(f"  {'Month-end projection':<22}{s(money(report.projected_month_usd), 'yellow')}"
        + s("   (linear, from this window)", "dim"))
    if cur.errors:
        add(f"  {'Errors':<22}{s(f'{cur.errors:,} ({cur.error_rate * 100:.1f}%)', 'red')}")
    add("")

    # ---- action items first: this is the point of the tool ----
    actions = _action_items(report)
    if actions:
        add(s("  What to do about it", "bold"))
        add("")
        for i, (title, detail) in enumerate(actions, 1):
            add(f"  {s(str(i) + '.', 'bold')} {s(title, 'bold')}")
            for line in _wrap(detail, w - 7):
                add(f"     {s(line, 'dim')}")
            add("")

    # ---- runaway-spend anomalies -------------------------------------------------
    # Placed near the top because a live loop outranks any optimisation advice:
    # it is spending money right now.
    if anomalies:
        add(s("  Runaway spend detected", "bold"))
        add("")
        for a in anomalies[:4]:
            tone = "red" if a.severity == "critical" else "yellow"
            tag = s(f"{a.severity.upper():<9}", tone)
            add(f"  {tag}{s(a.kind, 'bold')}"
                + s(f"  [{a.subject_type}: {a.subject}]", "dim"))
            for line in _wrap(a.detail, w - 7):
                add(f"     {line}")
            remedy = a.remedy().splitlines()[0] if a.remedy() else ""
            for i, line in enumerate(_wrap(remedy, w - 10)):
                prefix = "     → " if i == 0 else "       "
                add(s(f"{prefix}{line}", "dim"))
            add("")

    # ---- spend by model ----
    if report.by_model:
        add(s("  Where the money went", "bold"))
        add("")
        add(s(f"  {'model':<26}{'cost':>11}{'share':>8}{'reqs':>8}  {'cost/req':>10}", "dim"))
        for row in report.by_model[:8]:
            add(
                f"  {_trunc(row.label, 26):<26}"
                f"{money(row.cost_usd):>11}"
                f"{row.share * 100:>7.1f}%"
                f"{row.requests:>8,}"
                f"  {money(row.cost_per_request, 4):>10}"
            )
        add("")

    # ---- spend by key (only if meaningful) ----
    if len(report.by_key) > 1:
        add(s("  Who spent it", "bold"))
        add("")
        add(s(f"  {'api key':<26}{'cost':>11}{'share':>8}{'reqs':>8}  {'errors':>7}", "dim"))
        for row in report.by_key[:8]:
            err = s(f"{row.errors:>7,}", "red" if row.errors else "dim")
            add(
                f"  {_trunc(row.label, 26):<26}"
                f"{money(row.cost_usd):>11}"
                f"{row.share * 100:>7.1f}%"
                f"{row.requests:>8,}  {err}"
            )
        add("")

    if verbose and len(report.by_project) > 1:
        add(s("  By project", "bold"))
        add("")
        for row in report.by_project[:8]:
            add(f"  {_trunc(row.label, 30):<30}{money(row.cost_usd):>11}{row.share * 100:>7.1f}%")
        add("")

    # ---- daily trend ----
    if report.daily:
        values = [d.cost_usd for d in report.daily]
        add(s("  Daily trend", "bold"))
        add("")
        add(f"  {sparkline(values)}   {money(min(values))} – {money(max(values))}")
        busiest = max(report.daily, key=lambda d: d.cost_usd)
        add(s(f"  Peak day: {busiest.day} at {money(busiest.cost_usd)}", "dim"))
        add("")

    # ---- cache opportunity ----
    opp = report.cache_opportunity_usd()
    if opp > 0.01 and cur.input_tokens:
        add(s("  Cache opportunity", "bold"))
        add("")
        add(f"  Current cache hit rate:   {s(f'{cur.cache_hit_rate * 100:.1f}%', 'yellow')}")
        add(f"  Uncached input tokens:    {cur.input_tokens:,}")
        add(
            f"  Upper bound on saving:    {s(money(opp) + '/month', 'green')}"
            + s("  (if every input token were a cache hit)", "dim")
        )
        add("")

    # ---- biggest single requests ----
    if report.top_requests:
        add(s("  Most expensive single requests", "bold"))
        add("")
        add(s(f"  {'when':<20}{'model':<24}{'cost':>10}{'in':>9}{'out':>8}", "dim"))
        for r in report.top_requests[:5]:
            when = str(r.get("ts", ""))[:19].replace("T", " ")
            model = _trunc(str(r.get("model") or "(unknown)"), 24)
            cost = r.get("cost_usd") or 0.0
            add(
                f"  {when:<20}{model:<24}{money(float(cost)):>10}"
                f"{int(r.get('input_tokens') or 0):>9,}{int(r.get('output_tokens') or 0):>8,}"
            )
        add("")
        if report.top_requests:
            top = report.top_requests[0]
            share = (float(top.get("cost_usd") or 0) / cur.cost_usd * 100) if cur.cost_usd else 0
            if share >= 5:
                add(
                    s(
                        f"  Note: a single request accounts for {share:.1f}% of the window's spend.",
                        "yellow",
                    )
                )
                add("")

    # ---- budgets ----
    if report.budgets:
        add(s("  Budgets", "bold"))
        add("")
        add(s(f"  {'api key':<24}{'today':>16}{'month (projected)':>24}  {'action':<8}state", "dim"))
        for b in report.budgets:
            add(f"  {_budget_line(b, s)}")
        add("")

    # ---- data quality ----
    if report.unpriced_models:
        add(s("  Unpriced models (cost recorded as unknown)", "bold"))
        add("")
        for m in report.unpriced_models[:6]:
            add(f"  {s('•', 'yellow')} {m}")
        add(s("    Add them to llmguard/pricing.py or spend is under-reported.", "dim"))
        add("")

    if cur.requests == 0:
        add(s("  No traffic in this window.", "yellow"))
        add(s("  Try:  llm-guard seed --days 30    to load a realistic demo dataset.", "dim"))
        add("")

    add(s("  " + "─" * (w - 4), "dim"))
    add(s("  Every figure above is computed from the usage blocks your providers returned.", "dim"))
    add("")
    return "\n".join(out)


def _budget_line(b: BudgetStatus, s: Style) -> str:
    key = _trunc(b.api_key_id, 24)
    if b.daily_usd:
        today = f"{money(b.spent_today)} / {money(b.daily_usd)}"
    else:
        today = "—"
    if b.monthly_usd:
        month = f"{money(b.projected_month)} / {money(b.monthly_usd)}"
    else:
        month = "—"
    tone = {"ok": "green", "warn": "yellow", "breach": "red"}[b.state]
    state = {"ok": "ok", "warn": "warning", "breach": "OVER BUDGET"}[b.state]
    return (
        f"{key:<24}{today:>16}{month:>24}  {b.action:<8}{s(state, tone)}"
    )


def _action_items(report: ValueReport) -> List[tuple[str, str]]:
    """Turn the numbers into concrete, ordered recommendations."""
    items: List[tuple[str, str]] = []
    cur = report.current

    if cur.requests == 0:
        return items

    # 1. model concentration -> negotiation / routing
    if report.by_model:
        top = report.by_model[0]
        if top.share >= 0.5 and top.requests > 5:
            items.append(
                (
                    f"One model is {top.share * 100:.0f}% of your bill ({top.label}, {money(top.cost_usd)}).",
                    f"Routing the easy traffic to a cheaper tier is the single highest-leverage change. "
                    f"It costs {money(top.cost_per_request, 4)} per request today. "
                    f"Classify requests by difficulty (short prompts, no tools, low stakes) and send those "
                    f"elsewhere; keep the expensive model for the hard 20%.",
                )
            )

    # 2. cost per request outlier
    if report.by_model:
        blended = cur.cost_per_request
        worst = max(report.by_model, key=lambda r: r.cost_per_request)
        if blended and worst.cost_per_request > blended * 3 and worst.requests >= 3:
            items.append(
                (
                    f"{worst.label} costs {worst.cost_per_request / blended:.1f}x your average per request.",
                    f"{money(worst.cost_per_request, 4)} vs a blended {money(blended, 4)}. "
                    f"Check whether its prompts carry large fixed context (system prompts, few-shot "
                    f"examples, retrieved documents). Trimming fixed context pays off on every call.",
                )
            )

    # 3. cache opportunity
    opp = report.cache_opportunity_usd()
    if opp > max(1.0, cur.cost_usd * 0.05) and cur.input_tokens > 1000:
        items.append(
            (
                f"Prompt caching is worth up to {money(opp)}/month on this traffic.",
                f"Your cache hit rate is {cur.cache_hit_rate * 100:.1f}% across {cur.input_tokens:,} uncached "
                f"input tokens. Repeated prefixes (system prompts, tool schemas, retrieved docs) are the usual "
                f"culprit. Cache reads are billed at 2.5%-10% of the input rate depending on the model "
                f"(see `llm-guard models`); treat this as an upper bound and measure after the change.",
            )
        )

    # 4. budget state
    for b in report.budgets:
        if b.state == "breach":
            items.append(
                (
                    f"{b.api_key_id} is projected to finish the month over budget.",
                    f"Projected {money(b.projected_month)} against a {money(b.monthly_usd or 0)} limit "
                    f"(action: {b.action}). "
                    + (
                        "Set action=block if you want the gateway to hard-stop this key."
                        if b.action != "block"
                        else "This key is set to block; the gateway will return HTTP 429 once the limit is hit."
                    ),
                )
            )
            break
        if b.state == "warn":
            items.append(
                (
                    f"{b.api_key_id} is tracking close to its monthly budget.",
                    f"Projected {money(b.projected_month)} against {money(b.monthly_usd or 0)}. "
                    f"Worth a look before it becomes an incident.",
                )
            )
            break

    # 5. errors are spend with no value
    if cur.error_rate >= 0.05 and cur.errors >= 5:
        items.append(
            (
                f"{cur.error_rate * 100:.0f}% of requests failed ({cur.errors:,}).",
                "Failed calls still cost you latency and sometimes tokens. Check the error column for the "
                "dominant status code before optimising spend further.",
            )
        )

    # 6. attribution gap
    if len(report.by_key) <= 1 and cur.requests > 20:
        items.append(
            (
                "All traffic is attributed to a single key.",
                "Per-key attribution is what makes a bill actionable: without it you cannot tell which "
                "team, customer or feature drove the spend. Issue one gateway key per consumer and pass "
                "--key-map, or send an x-project header.",
            )
        )

    # 7. unpriced traffic is invisible spend
    if report.unpriced_models:
        items.append(
            (
                f"{len(report.unpriced_models)} model(s) have no price entry.",
                "Their cost is recorded as unknown, so the headline number under-reports. Add the rates to "
                "llmguard/pricing.py — this is a one-line change per model.",
            )
        )

    return items


def _trunc(text: str, width: int) -> str:
    text = str(text)
    return text if len(text) <= width else text[: width - 1] + "…"


def _wrap(text: str, width: int) -> List[str]:
    words = str(text).split()
    lines: List[str] = []
    cur = ""
    for word in words:
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines
