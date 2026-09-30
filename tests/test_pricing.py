"""Pricing and cost-computation tests.

These encode the billing rules that are easy to get wrong and expensive to get
wrong:

* cached tokens are billed at a *per-model* discount (0.025x-0.10x), not a global
  constant,
* cache writes are a separate billed category at a premium,
* datestamped model ids must resolve to the right base rate,
* an unknown model must return ``None``, never a plausible number.

Prices verified 2026-09-30 against the providers' own pricing pages.
"""

from __future__ import annotations

import unittest
from decimal import Decimal

from llmguard.pricing import (
    PRICES,
    VERIFIED_AT,
    TokenUsage,
    compute_cost,
    estimate_cache_saving,
    get_price,
    known_models,
    resolve_model,
)


class TestModelResolution(unittest.TestCase):
    def test_exact_match(self):
        self.assertEqual(resolve_model("gpt-4o"), "gpt-4o")
        self.assertEqual(resolve_model("claude-sonnet-4.5"), "claude-sonnet-4.5")

    def test_dated_snapshot_resolves_to_base(self):
        self.assertEqual(resolve_model("gpt-4o-2024-08-06"), "gpt-4o")
        self.assertEqual(resolve_model("gpt-5.5-2026-01-01"), "gpt-5.5")
        self.assertEqual(resolve_model("claude-sonnet-4-5-20250929"), "claude-sonnet-4.5")

    def test_prefix_match_does_not_steal_a_sibling(self):
        """The longer key must win: gpt-4o-mini is not gpt-4o."""
        self.assertEqual(resolve_model("gpt-4o-mini"), "gpt-4o-mini")
        self.assertEqual(resolve_model("gpt-4o-mini-2024-07-18"), "gpt-4o-mini")
        self.assertEqual(resolve_model("gpt-5.4-mini"), "gpt-5.4-mini")
        self.assertEqual(resolve_model("gpt-5.4-nano"), "gpt-5.4-nano")
        self.assertEqual(resolve_model("gpt-5.5-pro"), "gpt-5.5-pro")

    def test_legacy_anthropic_ids_map_to_current_rows(self):
        self.assertEqual(resolve_model("claude-3-5-sonnet-20241022"), "claude-sonnet-4")
        self.assertEqual(resolve_model("claude-3-opus-20240229"), "claude-opus-4")
        self.assertEqual(resolve_model("claude-3-haiku-20240307"), "claude-haiku-3.5")

    def test_unknown_model_unchanged_and_unpriced(self):
        self.assertEqual(resolve_model("ft:acme-support-v2"), "ft:acme-support-v2")
        self.assertIsNone(get_price("ft:acme-support-v2"))
        self.assertIsNone(get_price("some-internal-model-v3"))

    def test_every_resolved_key_exists_in_the_table(self):
        """Regression: prefix matching once returned alias keys absent from PRICES."""
        for name in [
            "gpt-4o",
            "gpt-4o-2024-08-06",
            "gpt-4o-mini-2024-07-18",
            "gpt-5.5",
            "gpt-6.1-sol",
            "claude-3-5-sonnet-20240620",
            "claude-3-5-sonnet-20241022",
            "claude-3-opus-20240229",
            "claude-haiku-4.5",
            "claude-fable-5.1",
            "gemini-1.5-flash",
        ]:
            resolved = resolve_model(name)
            self.assertIn(resolved, PRICES, f"{name} resolved to missing key {resolved}")

    def test_empty_model_is_safe(self):
        self.assertEqual(resolve_model(""), "")
        self.assertIsNone(get_price(""))


class TestCostComputation(unittest.TestCase):
    def test_simple_input_output(self):
        # gpt-4o: $2.50/1M in, $10.00/1M out
        cost = compute_cost("gpt-4o", TokenUsage(input=1_000_000, output=1_000_000))
        self.assertEqual(cost, Decimal("12.500000"))

    def test_cached_input_uses_the_models_own_rate(self):
        # gpt-4o cache read is 0.5x input = $1.25/1M
        cost = compute_cost(
            "gpt-4o", TokenUsage(input=0, output=1_000_000, cached_input=1_000_000)
        )
        self.assertEqual(cost, Decimal("11.250000"))

    def test_caching_is_cheaper_than_not_caching(self):
        uncached = compute_cost("gpt-4o", TokenUsage(input=100_000, output=1000))
        cached = compute_cost("gpt-4o", TokenUsage(output=1000, cached_input=100_000))
        self.assertLess(cached, uncached)

    def test_cache_multipliers_differ_per_model(self):
        """Fable 5.1 reads cache at 0.025x; Sonnet 4.5 at 0.1x. Not a constant."""
        fable = get_price("claude-fable-5.1")
        sonnet = get_price("claude-sonnet-4.5")
        self.assertEqual(fable.cached_input, Decimal("0.250000"))   # 10.00 * 0.025
        self.assertEqual(sonnet.cached_input, Decimal("0.300000"))  # 3.00 * 0.1

        # Opus 5.5 is the third multiplier: 0.05x
        opus = get_price("claude-opus-5.5")
        self.assertEqual(opus.cached_input, Decimal("0.200000"))    # 4.00 * 0.05

    def test_anthropic_cache_write_premium_is_1_25x(self):
        price = get_price("claude-sonnet-4.5")
        self.assertEqual(price.cache_write, Decimal("3.750000"))    # 3.00 * 1.25
        cost = compute_cost(
            "claude-sonnet-4.5", TokenUsage(cache_write=1_000_000)
        )
        self.assertEqual(cost, Decimal("3.750000"))

    def test_openai_cache_write_is_a_distinct_category(self):
        # gpt-6.1-sol: write $2.50/1M, read $0.10/1M, input $2.00/1M
        cost = compute_cost("gpt-6.1-sol", TokenUsage(cache_write=1_000_000))
        self.assertEqual(cost, Decimal("2.500000"))
        cost = compute_cost("gpt-6.1-sol", TokenUsage(cached_input=1_000_000))
        self.assertEqual(cost, Decimal("0.100000"))

    def test_cheap_models_are_actually_cheap(self):
        # gpt-6-luna: $0.10 in / $0.50 out
        cost = compute_cost("gpt-6-luna", TokenUsage(input=1_000_000, output=1_000_000))
        self.assertEqual(cost, Decimal("0.600000"))

    def test_embedding_model_bills_input_only(self):
        cost = compute_cost("text-embedding-3-small", TokenUsage(input=1_000_000))
        self.assertEqual(cost, Decimal("0.020000"))

    def test_unpriced_model_returns_none_not_zero(self):
        """None means 'unknown'; 0 would silently under-report spend."""
        self.assertIsNone(compute_cost("unknown-model-xyz", TokenUsage(input=1000)))
        self.assertIsNone(compute_cost("ft:acme-support-v2", TokenUsage(input=1000)))

    def test_zero_usage_costs_zero(self):
        self.assertEqual(compute_cost("gpt-4o", TokenUsage()), Decimal("0.000000"))

    def test_negative_input_is_clamped(self):
        self.assertEqual(
            compute_cost("gpt-4o", TokenUsage(input=-500)), Decimal("0.000000")
        )

    def test_dated_model_prices_like_its_base(self):
        a = compute_cost("gpt-4o", TokenUsage(input=5000, output=500))
        b = compute_cost("gpt-4o-2024-08-06", TokenUsage(input=5000, output=500))
        self.assertEqual(a, b)

    def test_precision_is_six_decimals(self):
        cost = compute_cost("gpt-4o-mini", TokenUsage(input=1))
        self.assertEqual(cost.as_tuple().exponent, -6)

    def test_all_four_token_classes_are_summed(self):
        usage = TokenUsage(input=1_000_000, output=1_000_000,
                           cached_input=1_000_000, cache_write=1_000_000)
        cost = compute_cost("gpt-6.1-sol", usage)
        # 2.00 + 10.00 + 0.10 + 2.50
        self.assertEqual(cost, Decimal("14.600000"))


class TestCacheSavingEstimate(unittest.TestCase):
    def test_saving_uses_the_models_cache_rate(self):
        saving = estimate_cache_saving("gpt-4o", TokenUsage(input=1_000_000))
        self.assertEqual(saving, Decimal("1.250000"))   # 2.50 - 1.25

        saving = estimate_cache_saving("claude-fable-5.1", TokenUsage(input=1_000_000))
        self.assertEqual(saving, Decimal("9.750000"))   # 10.00 - 0.25

    def test_no_saving_for_model_without_cache_pricing(self):
        self.assertEqual(
            estimate_cache_saving("gpt-3.5-turbo", TokenUsage(input=1000)), Decimal(0)
        )

    def test_no_saving_without_input_tokens(self):
        self.assertEqual(
            estimate_cache_saving("gpt-4o", TokenUsage(output=500)), Decimal(0)
        )


class TestPriceTableIntegrity(unittest.TestCase):
    def test_all_input_and_output_prices_are_non_negative(self):
        for name in known_models():
            p = PRICES[name]
            self.assertGreaterEqual(p.input, 0, name)
            self.assertGreaterEqual(p.output, 0, name)

    def test_embedding_models_are_input_only(self):
        for name in ("text-embedding-3-small", "text-embedding-3-large"):
            self.assertEqual(PRICES[name].output, Decimal("0"), name)

    def test_cache_read_is_cheaper_than_input(self):
        for name in known_models():
            p = PRICES[name]
            if p.cached_input is not None:
                self.assertLess(p.cached_input, p.input, name)

    def test_cache_write_is_at_least_input_price(self):
        for name in known_models():
            p = PRICES[name]
            if p.cache_write is not None:
                self.assertGreaterEqual(p.cache_write, p.input, name)

    def test_models_sorted_and_unique(self):
        models = list(known_models())
        self.assertEqual(models, sorted(models))
        self.assertEqual(len(models), len(set(models)))

    def test_table_is_not_stale_by_construction(self):
        """A tripwire: the table must carry a verification date we can audit."""
        self.assertRegex(VERIFIED_AT, r"^\d{4}-\d{2}-\d{2}$")

    def test_current_flagships_are_covered(self):
        """If these drop out, the table has drifted from the live lineup."""
        for name in ("gpt-6.1-sol", "gpt-5.5", "claude-sonnet-4.5",
                     "claude-opus-5.5", "claude-haiku-4.5"):
            self.assertIn(name, PRICES, f"{name} missing from the price table")


if __name__ == "__main__":
    unittest.main()
