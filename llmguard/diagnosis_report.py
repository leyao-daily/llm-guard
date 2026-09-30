"""Render a diagnosis as the document that gets delivered to a client.

Design constraints, in order of importance:

1. **It has to survive scrutiny.** Every number is shown with the evidence it came
   from, and every estimate is labelled with its confidence. A report that
   overstates savings is found out on the next invoice, and after that nothing in
   it is believed.
2. **It has to be printable.** Clients forward these. A4 margins, page-break
   control, black-on-white, and no interactive-only elements.
3. **It has to lead with the decision.** The first thing on page one is what to do
   first and what it is worth, not a chart of what already happened.
"""

from __future__ import annotations

import html
from datetime import datetime, timezone
from typing import List, Optional

from .diagnose import (
    CONFIDENCE_ARITHMETIC,
    CONFIDENCE_JUDGEMENT,
    CONFIDENCE_LIKELY,
    Diagnosis,
    Finding,
)

CONFIDENCE_LABEL = {
    CONFIDENCE_ARITHMETIC: ("Certain", "This is arithmetic on your own data."),
    CONFIDENCE_LIKELY: ("Likely", "Depends on one stated assumption, shown below."),
    CONFIDENCE_JUDGEMENT: ("Worth testing", "A hypothesis about your architecture, not a certainty."),
}

CSS = """
@page { size: A4; margin: 18mm 16mm; }
* { box-sizing: border-box; }
body {
  margin: 0; background: #fff; color: #14181f;
  font: 15px/1.62 -apple-system, BlinkMacSystemFont, "Segoe UI", Inter, Roboto,
        "Helvetica Neue", Arial, "PingFang SC", sans-serif;
  -webkit-font-smoothing: antialiased;
}
.page { max-width: 800px; margin: 0 auto; padding: 40px 32px 72px; }

/* ---- masthead ---- */
.masthead { border-bottom: 2px solid #14181f; padding-bottom: 16px; margin-bottom: 30px;
  display: flex; justify-content: space-between; align-items: baseline; gap: 20px; flex-wrap: wrap; }
.masthead .brand { font-weight: 800; letter-spacing: -0.02em; font-size: 16px; }
.masthead .brand span { color: #1b4dff; }
.masthead .meta { font-size: 12.5px; color: #667085; text-align: right; }

h1 { font-size: 30px; line-height: 1.2; letter-spacing: -0.02em; margin: 0 0 10px; }
h2 { font-size: 19px; letter-spacing: -0.01em; margin: 40px 0 14px; padding-bottom: 7px;
     border-bottom: 1px solid #e4e8ef; }
h3 { font-size: 16px; margin: 0 0 6px; }
p { margin: 0 0 14px; }
.lede { font-size: 17px; color: #3d4657; margin-bottom: 22px; }
small, .fine { font-size: 12.5px; color: #667085; line-height: 1.55; }
code { font-family: ui-monospace, SFMono-Regular, Menlo, monospace; font-size: 0.88em;
       background: #f4f6fb; padding: 1px 4px; border-radius: 3px; }

/* ---- summary strip ---- */
.strip { display: grid; grid-template-columns: repeat(4, 1fr); gap: 1px;
         background: #e4e8ef; border: 1px solid #e4e8ef; border-radius: 8px; overflow: hidden;
         margin: 0 0 26px; }
.strip div { background: #fff; padding: 14px 14px 15px; }
.strip .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.06em;
            color: #667085; margin-bottom: 5px; }
.strip .v { font-size: 21px; font-weight: 700; letter-spacing: -0.02em; }
.strip .v.money { color: #14181f; }
.strip .v.save { color: #0a7c53; }
@media print { .strip { break-inside: avoid; } }

/* ---- verdict ---- */
.verdict { border: 1px solid #e4e8ef; border-left: 4px solid #1b4dff; border-radius: 6px;
           padding: 16px 18px; background: #f7f9fc; margin-bottom: 26px; }
.verdict p:last-child { margin-bottom: 0; }

/* ---- findings ---- */
.finding { border: 1px solid #e4e8ef; border-radius: 8px; padding: 18px 20px; margin-bottom: 16px;
           break-inside: avoid; }
.finding .top { display: flex; justify-content: space-between; gap: 16px; align-items: flex-start;
                margin-bottom: 10px; flex-wrap: wrap; }
.finding .num { display: inline-flex; width: 22px; height: 22px; border-radius: 50%;
                background: #14181f; color: #fff; font-size: 12px; font-weight: 700;
                align-items: center; justify-content: center; margin-right: 9px;
                flex: 0 0 auto; }
.finding h3 { display: inline; }
.finding .money { font-size: 15px; font-weight: 700; color: #0a7c53; white-space: nowrap; }
.finding .money.zero { color: #667085; font-weight: 500; }
.badges { margin: 0 0 10px; }
.badge { display: inline-block; font-size: 11px; font-weight: 600; padding: 2px 8px;
         border-radius: 999px; margin-right: 6px; letter-spacing: 0.02em; }
.badge.certain { background: #e6f5ee; color: #0a7c53; }
.badge.likely { background: #fff4e0; color: #8a5a00; }
.badge.test { background: #eef2ff; color: #1b4dff; }
.badge.effort { background: #f2f4f8; color: #46506a; }
.evidence { background: #f7f9fc; border: 0; border-radius: 6px; padding: 11px 13px; margin: 12px 0 0; }
.evidence .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
               color: #667085; margin-bottom: 4px; }
.evidence-lines { font-family: ui-monospace, SFMono-Regular, Menlo, monospace;
                  font-size: 12.5px; color: #3d4657; white-space: pre-wrap; }
.remedy { margin: 12px 0 0; padding: 12px 14px; background: #fff; border: 1px solid #e4e8ef;
          border-radius: 6px; }
.remedy .k { font-size: 11px; text-transform: uppercase; letter-spacing: 0.05em;
             color: #667085; margin-bottom: 5px; }
.remedy p { margin: 0; font-size: 14.5px; }

/* ---- lists ---- */
ul { margin: 0 0 14px; padding-left: 20px; }
li { margin-bottom: 6px; }
.good li { color: #2b3a2f; }
.warn { border: 1px solid #f0d9a8; background: #fffaf0; border-radius: 6px; padding: 14px 16px; }
.warn ul { margin-bottom: 0; }

footer { margin-top: 44px; padding-top: 16px; border-top: 1px solid #e4e8ef; }
"""


def _e(x: object) -> str:
    return html.escape(str(x), quote=True)


def _money(v: float, places: int = 0) -> str:
    return f"${v:,.{places}f}"


def _range(f: Finding) -> str:
    if not f.has_money:
        return "No direct saving"
    if abs(f.monthly_saving_high - f.monthly_saving_low) < 1:
        return f"{_money(f.mid_saving)}/mo"
    return f"{_money(f.monthly_saving_low)}–{_money(f.monthly_saving_high)}/mo"


def _badge_class(confidence: str) -> str:
    return {
        CONFIDENCE_ARITHMETIC: "certain",
        CONFIDENCE_LIKELY: "likely",
        CONFIDENCE_JUDGEMENT: "test",
    }.get(confidence, "likely")


def _evidence_block(f: Finding) -> str:
    if not f.evidence:
        return ""
    lines = "\n".join(f"{k}: {v}" for k, v in f.evidence.items())
    return (
        '<div class="evidence">'
        '<div class="k">Evidence</div>'
        f'<div class="evidence-lines">{_e(lines)}</div>'
        "</div>"
    )


def _finding_html(f: Finding, index: int) -> str:
    label, explanation = CONFIDENCE_LABEL.get(f.confidence, ("", ""))
    money_cls = "money" if f.has_money else "money zero"
    effort = f'<span class="badge effort">{_e(f.effort)}</span>' if f.effort else ""
    return f"""      <div class="finding">
        <div class="top">
          <h3><span class="num">{index}</span>{_e(f.title)}</h3>
          <span class="{money_cls}">{_e(_range(f))}</span>
        </div>
        <div class="badges">
          <span class="badge {_badge_class(f.confidence)}" title="{_e(explanation)}">{_e(label)}</span>
          {effort}
        </div>
        <p>{_e(f.detail)}</p>
        {_evidence_block(f)}
        <div class="remedy">
          <div class="k">What to do</div>
          <p>{_e(f.remedy)}</p>
        </div>
      </div>"""


def render_diagnosis_html(
    d: Diagnosis,
    *,
    client: str = "",
    prepared_by: str = "LYE LABS LIMITED",
    contact: str = "hello@lye-labs.com",
) -> str:
    """Render the client-facing diagnosis document."""
    generated = datetime.now(timezone.utc).strftime("%d %B %Y")
    findings = d.ranked
    with_money = [f for f in findings if f.has_money]
    spend_month = d.monthly(d.total_spend)
    save_low = sum(f.monthly_saving_low for f in findings)
    save_high = sum(f.monthly_saving_high for f in findings)
    pct = (save_high / spend_month * 100) if spend_month else 0.0

    # --- verdict paragraph: say the conclusion in words, not just numbers ---
    if with_money:
        top = with_money[0]
        verdict = (
            f"Two things are driving cost here, and both are in the same place. "
            f"The largest single item is <strong>{_e(top.title)}</strong>, worth "
            f"{_e(_range(top))} if the underlying pattern is fixed. "
            f"Across everything found, the addressable spend is between "
            f"{_money(save_low)} and {_money(save_high)} a month, against a measured "
            f"{_money(spend_month)} a month, so roughly {pct:.0f}% of the bill is in scope."
            if len(with_money) > 1
            else f"The main finding is <strong>{_e(top.title)}</strong>, worth "
                 f"{_e(_range(top))}. Against a measured {_money(spend_month)} a month that is "
                 f"{pct:.0f}% of the bill."
        )
    else:
        verdict = (
            "No single dominant cost problem stands out in this window. The findings below are "
            "still worth acting on, but this bill is not being driven by one runaway pattern."
        )

    findings_html = "\n".join(_finding_html(f, i) for i, f in enumerate(findings, 1))
    if not findings_html:
        findings_html = '<p class="fine">No findings. Either the data is too small to analyse, or this is a clean bill.</p>'

    # --- what is already fine ---
    good: List[str] = []
    if d.distinct_keys > 1:
        good.append(
            f"Spend is split across {d.distinct_keys} keys, so cost can be attributed to a "
            f"team, customer or feature rather than being one undifferentiated total."
        )
    if not any(f.key == "cache_opportunity" for f in findings):
        good.append(
            "Prompt cache hit rates are reasonable, so repeated prefixes are already being "
            "billed at the cheaper rate."
        )
    if not any(f.key == "errors" for f in findings):
        good.append("Error rates are low, so retries are not inflating the bill.")
    if not any(f.key == "concentration" for f in findings):
        good.append("Spend is spread across models rather than concentrated in one, so a price "
                    "change or deprecation on a single model is not an emergency.")
    if d.unpriced_requests == 0:
        good.append("Every model in use has a price on file, so these totals are complete "
                    "rather than a floor.")
    good_html = "\n".join(f"<li>{_e(g)}</li>" for g in good) if good else "<li>No clear strengths identified in this window.</li>"

    gaps_html = "\n".join(f"<li>{_e(g)}</li>" for g in d.gaps)

    return f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>LLM Cost Diagnosis{(' — ' + _e(client)) if client else ''}</title>
<style>{CSS}</style>
</head>
<body>
<div class="page">

  <div class="masthead">
    <div class="brand">LYE <span>LABS</span></div>
    <div class="meta">
      LLM Cost Diagnosis<br>
      {_e(generated)}
    </div>
  </div>

  <h1>Where your LLM spend is going</h1>
  <p class="lede">
    {('Prepared for <strong>' + _e(client) + '</strong>. ') if client else ''}
    An analysis of {d.total_requests:,} API calls over {d.days} days
    {('from ' + _e(d.first_ts[:10]) + ' to ' + _e(d.last_ts[:10])) if d.first_ts else ''}.
  </p>

  <div class="strip">
    <div><div class="k">Measured spend</div><div class="v money">{_money(spend_month)}</div>
      <small>per 30 days</small></div>
    <div><div class="k">Addressable</div><div class="v save">{_money(save_low)}–{_money(save_high)}</div>
      <small>per 30 days, upper bound</small></div>
    <div><div class="k">Requests</div><div class="v">{d.total_requests:,}</div>
      <small>{d.days} days</small></div>
    <div><div class="k">Models / keys</div><div class="v">{d.distinct_models} / {d.distinct_keys}</div><small>attribution sources</small></div>
  </div>

  <div class="verdict">
    <p>{verdict}</p>
    <p class="fine" style="margin-bottom:0">
      Addressable spend adds every estimate together, which overstates what any single change
      delivers: trimming context and improving cache hit rates act on <em>the same tokens</em>,
      so they cannot both be collected in full. Treat it as the size of the opportunity, not a
      forecast. Each finding carries its own confidence level and its own assumption.
    </p>
  </div>

  <h2>What to do, in order</h2>
{findings_html}

  <h2>What is already in good shape</h2>
  <ul class="good">
{good_html}
  </ul>

  <h2>What this analysis cannot tell you</h2>
  <div class="warn">
    <ul>
{gaps_html}
    </ul>
  </div>

  <h2>Method</h2>
  <p class="fine">
    Costs were computed from the <code>usage</code> block each provider returned, never from a
    local token estimate, using per-model rates including separate cached-input and cache-write
    categories. Findings are derived from token counts, latency, status codes and attribution
    labels. Prompt and completion content was not read at any point.
  </p>
  <p class="fine">
    Saving estimates are bounds. Where a figure depends on an assumption about your
    architecture, that assumption is stated in the finding. Confidence labels mean:
    <strong>Certain</strong> — arithmetic on your data;
    <strong>Likely</strong> — depends on one stated assumption;
    <strong>Worth testing</strong> — a hypothesis, worth a bounded experiment but not a
    committed budget.
  </p>

  <footer>
    <p class="fine">
      Prepared by <strong>{_e(prepared_by)}</strong>{(' · ' + _e(contact)) if contact else ''}<br>
      This report analyses metadata only. No prompt or completion content was accessed.
    </p>
  </footer>

</div>
</body>
</html>
"""
