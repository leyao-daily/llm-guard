# Pricing sources and maintenance

`llmguard/pricing.py` is the one part of this project that **goes wrong silently**. If a rate
is stale or missing, every cost figure downstream is wrong while still looking plausible. This
document records where each number came from and how to bring it up to date.

**Table verified:** 2026-09-30 (`VERIFIED_AT` in `pricing.py`)

---

## Sources read (full text, not search snippets)

| Provider | URL | Notes |
|---|---|---|
| OpenAI | <https://developers.openai.com/api/docs/pricing> | Markdown sibling at `.../pricing.md` is machine-readable. **Standard** tier is used; Batch / Flex / Fast / Ultrafast tiers are not modelled. |
| Anthropic | <https://platform.claude.com/docs/en/about-claude/pricing> | Markdown sibling at `.../pricing.md`. 5-minute cache TTL is modelled (1-hour TTL is 2x and is not modelled separately). |
| Google | *not read* | ⚠️ Gemini rows are **legacy placeholders**, retained only so old traffic does not report as unpriced. Verify before trusting any Gemini number. |

---

## Three things this table gets right that are easy to get wrong

### 1. Cache multipliers are per-model, not a constant

| Model | Input /MTok | Cache read /MTok | Multiplier |
|---|---|---|---|
| Claude Fable 5.1 | $10.00 | $0.25 | **0.025x** |
| Claude Opus 5.5 | $4.00 | $0.20 | **0.05x** |
| Claude Sonnet 4.5 | $3.00 | $0.30 | 0.1x |
| Claude Haiku 4.5 | $1.00 | $0.10 | 0.1x |
| gpt-4o | $2.50 | $1.25 | **0.5x** |
| gpt-6.1-sol | $2.00 | $0.10 | 0.05x |
| gpt-5.5 | $5.00 | $0.50 | 0.1x |

A single global "cache is 10% of input" constant would be wrong for most of the current
lineup. Hence `_anthropic()` derives the rate from an explicit per-model multiplier, and
`_openai()` takes the published figure directly.

### 2. Cache **writes** are a separately billed category

OpenAI bills cache writes on the gpt-6 family (e.g. `gpt-6-astra`: $12.50/MTok write vs
$10.00/MTok input). Anthropic bills 5-minute writes at 1.25x input and 1-hour writes at 2x.
The schema has a `cache_write` token class for this reason; collapsing it into `input` would
misprice every cache-creation request.

### 3. Context length changes the rate

Current OpenAI flagships bill >272K-token prompts at roughly **2x** the short-context rate
(e.g. `gpt-6-astra`: $10 → $20 input, $50 → $75 output). This module models the short-context
tier only. Requests above 272K input tokens will be **understated by up to 2x**. If you route
long-context traffic, gate on request size or add a second tier.

---

## Maintenance procedure

1. Open the two source URLs above (append `.md` for the machine-readable version).
2. Update the figures in `PRICES`.
3. Update `VERIFIED_AT`.
4. Run `python3 -m unittest discover -s tests` — `test_current_flagships_are_covered` fails if
   a headline model drops out of the table.
5. Run `python3 -m llmguard models` and sanity-check a few rows by eye.

**Cadence: quarterly.** The observed churn is aggressive — between mid-2025 and
September 2026 the OpenAI flagship line moved from the gpt-4.1/gpt-4o generation through
gpt-5, gpt-5.4/5.5/5.6 and on to gpt-6, and Anthropic's went from Claude 3.5/4 to
Sonnet 5.5 / Opus 5.5 / Fable 5.1. A table that is a year old does not merely cost a little
precision; it reports every current model as unpriced.

## Why this is a product feature, not just a chore

The staleness above is the honest reason a self-hosted cost tool needs a *maintained* price
table, and it is also the strongest argument for a subscription rather than a one-off script:
rates move, models retire, and cache economics change. If you fork this, the maintenance
procedure above is the part that has to survive.
