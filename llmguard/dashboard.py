"""Self-contained HTML dashboard.

No CDN, no build step, no JavaScript dependency for the charts: the SVG is
generated server-side. That means it renders identically inside an air-gapped
VPC, which is the whole point of shipping a self-hosted gateway.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import List, Sequence

from .analytics import BudgetStatus, BreakdownRow, ValueReport
from .report import money

CSS = """
:root{
  --bg:#0b1020; --panel:#141a2e; --panel2:#1b2340; --line:#26304f;
  --ink:#e8ecf6; --muted:#8b97b8; --accent:#4f8cff; --good:#2fd08a;
  --warn:#ffc14d; --bad:#ff6b6b;
}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--ink);
  font:14px/1.55 -apple-system,BlinkMacSystemFont,"Segoe UI",Inter,Roboto,Arial,sans-serif}
.wrap{max-width:1180px;margin:0 auto;padding:28px 22px 64px}
header{display:flex;justify-content:space-between;align-items:baseline;
  border-bottom:1px solid var(--line);padding-bottom:16px;margin-bottom:24px;flex-wrap:wrap;gap:8px}
h1{font-size:20px;margin:0;letter-spacing:-.01em}
h2{font-size:15px;margin:0 0 14px;color:var(--muted);font-weight:600;
  text-transform:uppercase;letter-spacing:.06em}
.sub{color:var(--muted);font-size:13px}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(180px,1fr));gap:14px;margin-bottom:26px}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:16px}
.tile .k{color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.05em}
.tile .v{font-size:25px;font-weight:700;margin-top:7px;letter-spacing:-.02em}
.tile .d{font-size:12px;color:var(--muted);margin-top:5px}
.up{color:var(--bad)} .down{color:var(--good)}
.grid{display:grid;grid-template-columns:1fr 1fr;gap:16px}
@media(max-width:860px){.grid{grid-template-columns:1fr}}
.card{background:var(--panel);border:1px solid var(--line);border-radius:10px;padding:18px;margin-bottom:16px}
table{width:100%;border-collapse:collapse;font-size:13px}
th{text-align:left;color:var(--muted);font-weight:600;font-size:11.5px;
  text-transform:uppercase;letter-spacing:.05em;padding:6px 8px;border-bottom:1px solid var(--line)}
td{padding:8px;border-bottom:1px solid rgba(38,48,79,.55)}
td.num{text-align:right;font-variant-numeric:tabular-nums}
tr:last-child td{border-bottom:none}
.bar{height:7px;border-radius:4px;background:var(--panel2);overflow:hidden;min-width:60px}
.bar > i{display:block;height:100%;background:var(--accent)}
.pill{display:inline-block;padding:2px 9px;border-radius:999px;font-size:11.5px;font-weight:600}
.pill.ok{background:rgba(47,208,138,.15);color:var(--good)}
.pill.warn{background:rgba(255,193,77,.15);color:var(--warn)}
.pill.bad{background:rgba(255,107,107,.15);color:var(--bad)}
.actions{counter-reset:a;list-style:none;padding:0;margin:0}
.actions li{counter-increment:a;position:relative;padding:12px 0 12px 40px;
  border-bottom:1px solid rgba(38,48,79,.55)}
.actions li:last-child{border-bottom:none}
.actions li::before{content:counter(a);position:absolute;left:0;top:12px;width:26px;height:26px;
  border-radius:50%;background:var(--accent);color:#fff;display:flex;align-items:center;
  justify-content:center;font-size:12.5px;font-weight:700}
.actions .t{font-weight:650;margin-bottom:4px}
.actions .d{color:var(--muted);font-size:13px}
.mono{font-family:ui-monospace,SFMono-Regular,Menlo,monospace;font-size:12.4px}
footer{color:var(--muted);font-size:12px;border-top:1px solid var(--line);padding-top:16px;margin-top:26px}
.empty{color:var(--muted);padding:24px;text-align:center}
"""


def _e(text: object) -> str:
    return html.escape(str(text), quote=True)


def _spark_svg(values: Sequence[float], width: int = 1060, height: int = 120) -> str:
    """Inline SVG area chart for the daily spend series."""
    if not values:
        return '<div class="empty">no data</div>'
    n = len(values)
    lo = min(min(values), 0.0)
    hi = max(values) or 1.0
    span = (hi - lo) or 1.0
    pad_l, pad_r, pad_t, pad_b = 46, 12, 14, 26
    plot_w = max(width - pad_l - pad_r, 10)
    plot_h = max(height - pad_t - pad_b, 10)

    def x(i: int) -> float:
        return pad_l + (plot_w * i / max(n - 1, 1))

    def y(v: float) -> float:
        return pad_t + plot_h * (1 - (v - lo) / span)

    pts = [(x(i), y(v)) for i, v in enumerate(values)]
    line = " ".join(f"{px:.1f},{py:.1f}" for px, py in pts)
    area = (
        f"{pad_l},{pad_t + plot_h} " + line + f" {pad_l + plot_w},{pad_t + plot_h}"
    )

    # horizontal guide lines at 0/50/100% of the range
    guides = []
    for frac in (0.0, 0.5, 1.0):
        gy = pad_t + plot_h * (1 - frac)
        val = lo + span * frac
        guides.append(
            f'<line x1="{pad_l}" y1="{gy:.1f}" x2="{pad_l + plot_w}" y2="{gy:.1f}" '
            f'stroke="#26304f" stroke-width="1" stroke-dasharray="3 4"/>'
            f'<text x="{pad_l - 8}" y="{gy + 4:.1f}" fill="#8b97b8" font-size="10.5" '
            f'text-anchor="end">{money(val)}</text>'
        )

    dots = "".join(
        f'<circle cx="{px:.1f}" cy="{py:.1f}" r="2.6" fill="#4f8cff"/>' for px, py in pts
    )
    last_x, last_y = pts[-1]
    return f"""<svg viewBox="0 0 {width} {height}" width="100%" height="{height}"
 role="img" aria-label="daily spend">
 <defs><linearGradient id="g" x1="0" y1="0" x2="0" y2="1">
   <stop offset="0%" stop-color="#4f8cff" stop-opacity="0.42"/>
   <stop offset="100%" stop-color="#4f8cff" stop-opacity="0.02"/>
 </linearGradient></defs>
 {''.join(guides)}
 <polygon points="{area}" fill="url(#g)"/>
 <polyline points="{line}" fill="none" stroke="#4f8cff" stroke-width="2"/>
 {dots}
 <circle cx="{last_x:.1f}" cy="{last_y:.1f}" r="4.5" fill="#4f8cff" stroke="#0b1020" stroke-width="2"/>
</svg>"""


def _bar_table(rows: List[BreakdownRow], label_head: str, limit: int = 8) -> str:
    if not rows:
        return '<div class="empty">no data</div>'
    out = [
        f"<table><thead><tr><th>{_e(label_head)}</th><th class='num'>cost</th>"
        "<th class='num'>share</th><th class='num'>reqs</th>"
        "<th class='num'>cost/req</th><th></th></tr></thead><tbody>"
    ]
    for row in rows[:limit]:
        out.append(
            "<tr>"
            f"<td class='mono'>{_e(row.label)}</td>"
            f"<td class='num'>{money(row.cost_usd)}</td>"
            f"<td class='num'>{row.share * 100:.1f}%</td>"
            f"<td class='num'>{row.requests:,}</td>"
            f"<td class='num'>{money(row.cost_per_request, 4)}</td>"
            f"<td style='width:90px'><div class='bar'><i style='width:{max(row.share * 100, 1):.1f}%'></i></div></td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _budget_table(budgets: List[BudgetStatus]) -> str:
    if not budgets:
        return '<div class="empty">No budgets configured. Set one with '
        '<span class="mono">llm-guard budget set &lt;key&gt; --monthly 500</span></div>'
    out = [
        "<table><thead><tr><th>api key</th><th class='num'>today</th>"
        "<th class='num'>month to date</th><th class='num'>projected</th>"
        "<th class='num'>limit</th><th>action</th><th>state</th></tr></thead><tbody>"
    ]
    for b in budgets:
        cls = {"ok": "ok", "warn": "warn", "breach": "bad"}[b.state]
        text = {"ok": "ok", "warn": "close", "breach": "OVER"}[b.state]
        daily = f"{money(b.spent_today)} / {money(b.daily_usd)}" if b.daily_usd else "—"
        out.append(
            "<tr>"
            f"<td class='mono'>{_e(b.api_key_id)}</td>"
            f"<td class='num'>{daily}</td>"
            f"<td class='num'>{money(b.spent_month)}</td>"
            f"<td class='num'>{money(b.projected_month)}</td>"
            f"<td class='num'>{money(b.monthly_usd) if b.monthly_usd else '—'}</td>"
            f"<td>{_e(b.action)}</td>"
            f"<td><span class='pill {cls}'>{text}</span></td>"
            "</tr>"
        )
    out.append("</tbody></table>")
    return "".join(out)


def _actions_html(actions: List[tuple[str, str]]) -> str:
    if not actions:
        return '<div class="empty">Nothing to flag yet.</div>'
    items = "".join(
        f"<li><div class='t'>{_e(t)}</div><div class='d'>{_e(d)}</div></li>"
        for t, d in actions
    )
    return f"<ol class='actions'>{items}</ol>"


def render_dashboard(report: ValueReport) -> str:
    from .report import _action_items  # local import avoids a cycle

    cur, prev = report.current, report.previous
    delta = report.cost_delta_pct
    if delta is None:
        delta_html = '<div class="d">no prior period</div>'
    else:
        cls = "up" if delta > 0 else "down"
        arrow = "▲" if delta > 0 else "▼"
        delta_html = (
            f'<div class="d"><span class="{cls}">{arrow} {delta * 100:+.1f}%</span> '
            f"vs {money(prev.cost_usd)} prior</div>"
        )

    generated = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")
    daily_values = [d.cost_usd for d in report.daily]
    opp = report.cache_opportunity_usd()

    tiles = f"""
    <div class="tiles">
      <div class="tile"><div class="k">Total spend</div>
        <div class="v">{money(cur.cost_usd)}</div>{delta_html}</div>
      <div class="tile"><div class="k">Requests</div>
        <div class="v">{cur.requests:,}</div>
        <div class="d">{money(cur.cost_per_request, 4)} per request</div></div>
      <div class="tile"><div class="k">Month-end projection</div>
        <div class="v">{money(report.projected_month_usd)}</div>
        <div class="d">linear from last {report.days} days</div></div>
      <div class="tile"><div class="k">Cache hit rate</div>
        <div class="v">{cur.cache_hit_rate * 100:.1f}%</div>
        <div class="d">up to {money(opp)}/mo recoverable</div></div>
      <div class="tile"><div class="k">Errors</div>
        <div class="v">{cur.errors:,}</div>
        <div class="d">{cur.error_rate * 100:.1f}% of requests</div></div>
    </div>
    """

    top_rows = ""
    for r in report.top_requests[:8]:
        cost = float(r.get("cost_usd") or 0.0)
        top_rows += (
            "<tr>"
            f"<td class='mono'>{_e(str(r.get('ts',''))[:19].replace('T',' '))}</td>"
            f"<td class='mono'>{_e(r.get('model') or '(unknown)')}</td>"
            f"<td class='mono'>{_e(r.get('api_key_id') or '')}</td>"
            f"<td class='num'>{int(r.get('input_tokens') or 0):,}</td>"
            f"<td class='num'>{int(r.get('output_tokens') or 0):,}</td>"
            f"<td class='num'>{money(cost)}</td>"
            f"<td class='num'>{int(r.get('latency_ms') or 0):,} ms</td>"
            "</tr>"
        )
    top_table = (
        "<table><thead><tr><th>when</th><th>model</th><th>key</th>"
        "<th class='num'>in tok</th><th class='num'>out tok</th>"
        "<th class='num'>cost</th><th class='num'>latency</th></tr></thead>"
        f"<tbody>{top_rows}</tbody></table>"
        if top_rows
        else '<div class="empty">no priced requests</div>'
    )

    unpriced = ""
    if report.unpriced_models:
        items = "".join(f"<li class='mono'>{_e(m)}</li>" for m in report.unpriced_models[:10])
        unpriced = (
            "<div class='card'><h2>Unpriced models</h2>"
            f"<ul style='color:var(--warn);margin:0;padding-left:20px'>{items}</ul>"
            "<p class='sub' style='margin-top:10px'>Cost is recorded as unknown for these, so the "
            "headline under-reports. Add rates to <span class='mono'>llmguard/pricing.py</span>.</p></div>"
        )

    return f"""<!DOCTYPE html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Spend Dashboard</title>
<style>{CSS}</style></head>
<body><div class="wrap">
  <header>
    <div>
      <h1>LLM Spend Dashboard</h1>
      <div class="sub">last {report.days} days &middot; {report.total_requests_ever:,} requests recorded</div>
    </div>
    <div class="sub">generated {generated}</div>
  </header>

  {tiles}

  <div class="card">
    <h2>What to do about it</h2>
    {_actions_html(_action_items(report))}
  </div>

  <div class="card">
    <h2>Daily spend</h2>
    {_spark_svg(daily_values)}
  </div>

  <div class="grid">
    <div class="card"><h2>By model</h2>{_bar_table(report.by_model, "model")}</div>
    <div class="card"><h2>By API key</h2>{_bar_table(report.by_key, "api key")}</div>
  </div>

  <div class="grid">
    <div class="card"><h2>By project</h2>{_bar_table(report.by_project, "project", limit=6)}</div>
    <div class="card"><h2>By end user</h2>{_bar_table(report.by_end_user, "end user", limit=6)}</div>
  </div>

  <div class="card">
    <h2>Budgets</h2>
    {_budget_table(report.budgets)}
  </div>

  <div class="card">
    <h2>Most expensive requests</h2>
    {top_table}
  </div>

  {unpriced}

  <footer>
    Every figure is derived from the <span class="mono">usage</span> blocks your providers returned.
    Token counts are never estimated locally &mdash; unpriced models are reported as unknown rather than guessed.
    <br>Rendered offline by llm-guard. No external requests, no CDN, no telemetry.
  </footer>
</div></body></html>"""
