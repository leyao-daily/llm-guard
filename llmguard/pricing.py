"""Model pricing table and cost computation.

Prices are USD per 1,000,000 tokens, read from the providers' own pricing pages
(see ``PRICE_SOURCES`` and ``docs/PRICING_SOURCES.md``).

Read this before editing
------------------------
Two things are easy to get wrong and both produce silently wrong bills:

1. **Cache multipliers are not a constant.** They differ per model *and* change
   over time. OpenAI prices some models' cache reads at 0.05x input and others
   at 0.10x, and introduced *cache writes* as a distinct billed category.
   Anthropic uses 0.1x for most models but 0.025x for Fable 5.1 / Mythos 5.1 and
   0.05x for Opus 5.5. The table therefore stores explicit cached-input and
   cache-write rates per model rather than deriving them from a global constant.

2. **Context length changes the rate.** For current OpenAI flagships, prompts
   over 272K tokens are billed at roughly 2x the short-context rate. This module
   models the *short-context* tier, which covers the overwhelming majority of
   traffic. If you route long-context work through the gateway, its cost will be
   understated by up to 2x -- gate on request size, or add a long-context tier.

Unknown models return ``None`` rather than a guess, so reports surface
"unpriced" traffic instead of quietly under-reporting spend.
"""

from __future__ import annotations

import re

from dataclasses import dataclass
from decimal import Decimal, ROUND_HALF_UP
from typing import Dict, Iterable, Optional

# When this table was last checked against the providers' pricing pages.
# Treat prices older than ~90 days as suspect: the model lineup turns over fast.
VERIFIED_AT = "2026-09-30"

PRICE_SOURCES = (
    "https://developers.openai.com/api/docs/pricing",
    "https://platform.claude.com/docs/en/about-claude/pricing",
)

MILLION = Decimal(1_000_000)


@dataclass(frozen=True)
class ModelPrice:
    """Per-million-token USD prices for one model (short-context tier)."""

    model: str
    provider: str
    input: Decimal
    output: Decimal
    cached_input: Optional[Decimal] = None
    cache_write: Optional[Decimal] = None
    context_note: str = ""


def _d(value: str) -> Decimal:
    return Decimal(value)


PRICES: Dict[str, ModelPrice] = {}


def _register(price: ModelPrice) -> None:
    PRICES[price.model] = price


def _openai(model: str, inp: str, cached: str, write: str, out: str, *, long_context: bool = False) -> None:
    """Register an OpenAI model. ``cached``/``write`` of '' mean not offered."""
    _register(
        ModelPrice(
            model,
            "openai",
            _d(inp),
            _d(out),
            _d(cached) if cached else None,
            _d(write) if write else None,
            context_note="short context (<=272K); long context is ~2x" if long_context else "",
        )
    )


def _anthropic(model: str, inp: str, out: str, *, cache_read_mult: str = "0.1",
               cache_write_mult: str = "1.25") -> None:
    """Register an Anthropic model, deriving cache rates from the multipliers."""
    base = _d(inp)
    _register(
        ModelPrice(
            model,
            "anthropic",
            base,
            _d(out),
            (base * _d(cache_read_mult)).quantize(Decimal("0.000001")),
            (base * _d(cache_write_mult)).quantize(Decimal("0.000001")),
        )
    )


# ---------------------------------------------------------------------------
# OpenAI  (developers.openai.com/api/docs/pricing, Standard tier)
# ---------------------------------------------------------------------------
_openai("gpt-6-astra", "10.00", "1.00", "12.50", "50.00", long_context=True)
_openai("gpt-6.1-sol", "2.00", "0.10", "2.50", "10.00", long_context=True)
_openai("gpt-6-luna", "0.10", "0.01", "0.125", "0.50", long_context=True)
_openai("gpt-6-sol", "2.00", "0.20", "2.50", "10.00", long_context=True)
_openai("gpt-5.6-sol", "4.00", "0.40", "5.00", "20.00", long_context=True)
_openai("gpt-5.6-terra", "2.00", "0.20", "2.50", "12.00", long_context=True)
_openai("gpt-5.6-luna", "0.20", "0.02", "0.25", "1.20", long_context=True)
_openai("gpt-5.5", "5.00", "0.50", "", "30.00", long_context=True)
_openai("gpt-5.5-pro", "30.00", "", "", "180.00", long_context=True)
_openai("gpt-5.4", "2.50", "0.25", "", "15.00", long_context=True)
_openai("gpt-5.4-mini", "0.75", "0.075", "", "4.50")
_openai("gpt-5.4-nano", "0.20", "0.02", "", "1.25")
_openai("gpt-5.4-pro", "30.00", "", "", "180.00", long_context=True)
_openai("gpt-5.2", "1.75", "0.175", "", "14.00")
_openai("gpt-5.2-pro", "21.00", "", "", "168.00")
_openai("gpt-5.1", "1.25", "0.125", "", "10.00")
_openai("gpt-5", "1.25", "0.125", "", "10.00")
_openai("gpt-5-mini", "0.25", "0.025", "", "2.00")
_openai("gpt-5-nano", "0.05", "0.005", "", "0.40")
_openai("gpt-5-pro", "15.00", "", "", "120.00")
_openai("gpt-5.3-codex", "1.75", "0.175", "", "14.00")
_openai("gpt-5-search-api", "1.25", "0.125", "", "10.00")
# Still widely deployed; kept so existing projects price correctly.
_openai("gpt-4.1", "2.00", "0.50", "", "8.00")
_openai("gpt-4.1-mini", "0.40", "0.10", "", "1.60")
_openai("gpt-4.1-nano", "0.10", "0.025", "", "0.40")
_openai("gpt-4o", "2.50", "1.25", "", "10.00")
_openai("gpt-4o-mini", "0.15", "0.075", "", "0.60")
_openai("o3", "2.00", "0.50", "", "8.00")
_openai("o3-pro", "20.00", "", "", "80.00")
_openai("o4-mini", "1.10", "0.275", "", "4.40")
_openai("o3-mini", "1.10", "0.55", "", "4.40")
_openai("o1", "15.00", "7.50", "", "60.00")
_openai("gpt-3.5-turbo", "0.50", "", "", "1.50")
# Embeddings bill input only.
_openai("text-embedding-3-small", "0.02", "", "", "0.00")
_openai("text-embedding-3-large", "0.13", "", "", "0.00")

# ---------------------------------------------------------------------------
# Anthropic  (platform.claude.com/docs/en/about-claude/pricing)
# ---------------------------------------------------------------------------
# Cache-write figures are the 5-minute TTL (1.25x). The 1-hour TTL is 2x and is
# not modelled separately.
_anthropic("claude-fable-5.1", "10.00", "50.00", cache_read_mult="0.025", cache_write_mult="1.25")
_anthropic("claude-fable-5", "10.00", "50.00", cache_read_mult="0.1", cache_write_mult="1.25")
_anthropic("claude-opus-5.5", "4.00", "20.00", cache_read_mult="0.05", cache_write_mult="1.25")
_anthropic("claude-opus-5", "5.00", "25.00")
_anthropic("claude-opus-4.8", "5.00", "25.00")
_anthropic("claude-opus-4.7", "5.00", "25.00")
_anthropic("claude-opus-4.6", "5.00", "25.00")
_anthropic("claude-opus-4.5", "5.00", "25.00")
_anthropic("claude-sonnet-5.5", "2.00", "10.00")
_anthropic("claude-sonnet-5", "2.00", "10.00")
_anthropic("claude-sonnet-4.6", "3.00", "15.00")
_anthropic("claude-sonnet-4.5", "3.00", "15.00")
_anthropic("claude-haiku-4.5", "1.00", "5.00")
# Retired on the first-party API but still served via Bedrock/Vertex.
_anthropic("claude-opus-4.1", "15.00", "75.00")
_anthropic("claude-opus-4", "15.00", "75.00")
_anthropic("claude-sonnet-4", "3.00", "15.00")
_anthropic("claude-haiku-3.5", "0.80", "4.00")

# ---------------------------------------------------------------------------
# Google
# ---------------------------------------------------------------------------
# NOT refreshed in the 2026-09-30 pass: the Gemini pricing page was not read.
# These entries are retained so older traffic does not report as unpriced, but
# verify them before relying on Gemini numbers. See docs/PRICING_SOURCES.md.
_register(ModelPrice("gemini-2.0-flash", "google", _d("0.10"), _d("0.40")))
_register(ModelPrice("gemini-1.5-pro", "google", _d("1.25"), _d("5.00")))
_register(ModelPrice("gemini-1.5-flash", "google", _d("0.075"), _d("0.30")))


# ---------------------------------------------------------------------------
# Resolution of concrete model ids
# ---------------------------------------------------------------------------
_ALIASES = {
    # Older Anthropic ids -> current pricing rows with the same economics.
    "claude-3-5-sonnet": "claude-sonnet-4",
    "claude-3-5-sonnet-20240620": "claude-sonnet-4",
    "claude-3-5-sonnet-20241022": "claude-sonnet-4",
    "claude-3-5-haiku": "claude-haiku-3.5",
    "claude-3-5-haiku-20241022": "claude-haiku-3.5",
    "claude-3-opus": "claude-opus-4",
    "claude-3-opus-20240229": "claude-opus-4",
    "claude-3-haiku": "claude-haiku-3.5",
    "claude-3-haiku-20240307": "claude-haiku-3.5",
    "claude-sonnet-4-20250514": "claude-sonnet-4",
    "claude-opus-4-20250514": "claude-opus-4",
    # OpenAI dated snapshots whose base is no longer the headline row.
    "gpt-4o-2024-05-13": "gpt-4o",
    "gpt-4-turbo": "gpt-4o",
    "gpt-4-turbo-2024-04-09": "gpt-4o",
}

# Some provider ids are written with a hyphen between version components
# (claude-sonnet-4-5-20250929) while the docs and API return a dot
# (claude-sonnet-4.5). Normalising lets either spelling resolve without
# maintaining an alias per model. The head is non-greedy so multi-word families
# such as ``claude-sonnet`` are captured whole.
_HYPHENATED_VERSION = re.compile(
    r"^(?P<head>.+?)-(?P<major>\d+)-(?P<minor>\d+)(?P<tail>.*)$"
)
# ``claude-haiku-3-20240307`` style: bare major version plus a date suffix.
_MAJOR_ONLY_VERSION = re.compile(r"^(?P<head>.+?)-(?P<major>\d+)(?P<tail>-\d{6,}.*)$")


def _normalise(name: str) -> str:
    """Join a hyphenated version into the dotted form used in this table."""
    match = _HYPHENATED_VERSION.match(name)
    if match:
        return (
            f"{match.group('head')}-{match.group('major')}."
            f"{match.group('minor')}{match.group('tail')}"
        )
    match = _MAJOR_ONLY_VERSION.match(name)
    if match:
        return f"{match.group('head')}-{match.group('major')}{match.group('tail')}"
    return name


# Trailing release dates, e.g. ``-20250929`` or ``-20240620``.
_DATE_SUFFIX = re.compile(r"-\d{6,}$")


def _normalise_variants(name: str) -> list:
    """Candidate table keys for a raw provider model id, best first.

    Handles the three id shapes actually seen in the wild:
      claude-sonnet-4.5                -> itself
      claude-sonnet-4-5-20250929       -> claude-sonnet-4.5  (dotted, date stripped)
      claude-3-5-sonnet-20241022       -> claude-sonnet-4     (via alias)
    """
    variants = []
    dotted = _normalise(name)
    for candidate in (dotted, _DATE_SUFFIX.sub("", dotted)):
        if candidate and candidate not in variants:
            variants.append(candidate)
    return variants

_CANDIDATES: list = []


def _candidates() -> list:
    """(match_key, resolved_key) pairs, longest match key first.

    Candidates are built only from *resolvable* keys so a prefix match can never
    point at something missing from the price table. A regression here once made
    correctly-priced models report as unpriced.
    """
    pairs = [(key, key) for key in PRICES]
    pairs += [(alias, target) for alias, target in _ALIASES.items() if target in PRICES]
    pairs.sort(key=lambda p: len(p[0]), reverse=True)
    return pairs


def resolve_model(model: str) -> str:
    """Map a concrete/datestamped model id to a pricing-table key."""
    global _CANDIDATES
    if not model:
        return ""
    name = model.strip()
    if name in PRICES:
        return name
    if name in _ALIASES and _ALIASES[name] in PRICES:
        return _ALIASES[name]
    for variant in _normalise_variants(name):
        if variant in PRICES:
            return variant
        if variant in _ALIASES and _ALIASES[variant] in PRICES:
            return _ALIASES[variant]
    if not _CANDIDATES:
        _CANDIDATES = _candidates()
    for match_key, resolved in _CANDIDATES:
        if name.startswith(match_key):
            return resolved
    return name


def get_price(model: str) -> Optional[ModelPrice]:
    return PRICES.get(resolve_model(model))


@dataclass
class TokenUsage:
    """Token counts for a single request."""

    input: int = 0
    output: int = 0
    cached_input: int = 0
    cache_write: int = 0

    @property
    def total(self) -> int:
        return self.input + self.output + self.cached_input + self.cache_write


def compute_cost(model: str, usage: TokenUsage, *, quantize: bool = True) -> Optional[Decimal]:
    """Cost in USD for one request, or None when the model is unpriced.

    ``usage.input`` must *exclude* cached tokens, which is how OpenAI reports it
    under ``prompt_tokens_details.cached_tokens`` and how Anthropic reports
    ``input_tokens`` alongside ``cache_read_input_tokens``.
    """
    price = get_price(model)
    if price is None:
        return None

    cost = (
        Decimal(max(int(usage.input), 0)) * price.input
        + Decimal(int(usage.output)) * price.output
    ) / MILLION

    if usage.cached_input:
        rate = price.cached_input if price.cached_input is not None else price.input
        cost += (Decimal(int(usage.cached_input)) * rate) / MILLION

    if usage.cache_write:
        rate = price.cache_write if price.cache_write is not None else price.input
        cost += (Decimal(int(usage.cache_write)) * rate) / MILLION

    if quantize:
        cost = cost.quantize(Decimal("0.000001"), rounding=ROUND_HALF_UP)
    return cost


def estimate_cache_saving(model: str, usage: TokenUsage) -> Decimal:
    """USD that would have been saved if input tokens had been cache hits."""
    price = get_price(model)
    if price is None or not usage.input or price.cached_input is None:
        return Decimal(0)
    return (Decimal(int(usage.input)) * (price.input - price.cached_input)) / MILLION


def known_models() -> Iterable[str]:
    return sorted(PRICES)
