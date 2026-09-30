"""Tests for the catalogued failure taxonomy and its wiring into diagnoses.

The risk this file guards against is a citation that was not earned. Claiming a
finding "matches a documented failure pattern" when the pattern's own signal was
never tested would be the worst possible failure in this codebase: it turns
evidence into marketing, and the first technical reader would notice.
"""

from __future__ import annotations

import unittest

from llmguard.diagnose import PATTERN_FOR_FINDING, diagnose
from llmguard.incidents import (
    PATTERNS,
    PATTERNS_BY_KEY,
    SOURCES,
    describe_catalogue,
    match_patterns,
    measure,
)
from llmguard.storage import RequestRecord, Store, iso
from datetime import datetime, timedelta, timezone


def _ago(minutes: int = 30) -> str:
    return iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))


class TestCatalogueIntegrity(unittest.TestCase):
    def test_every_pattern_has_a_signal_and_a_remedy(self):
        for p in PATTERNS:
            self.assertTrue(p.signals, p.key)
            self.assertTrue(p.remedy, p.key)
            self.assertTrue(p.summary, p.key)
            self.assertTrue(p.absent_when, p.key)

    def test_every_pattern_cites_a_real_source(self):
        for p in PATTERNS:
            self.assertTrue(p.cites(), p.key)
            for citation in p.cites():
                self.assertIn(citation, SOURCES.values())

    def test_keys_are_unique(self):
        keys = [p.key for p in PATTERNS]
        self.assertEqual(len(keys), len(set(keys)))

    def test_taxonomy_mapping_points_at_real_patterns(self):
        for finding_key, pattern_key in PATTERN_FOR_FINDING.items():
            self.assertIn(pattern_key, PATTERNS_BY_KEY, finding_key)

    def test_incident_counts_match_the_published_catalogue(self):
        """The paper reports 11 delegation-fanout and 11 context-loop incidents."""
        self.assertEqual(PATTERNS_BY_KEY["context_loop"].recorded_incidents, 11)
        self.assertEqual(PATTERNS_BY_KEY["delegation_fanout"].recorded_incidents, 11)

    def test_description_lists_every_pattern(self):
        text = describe_catalogue()
        for p in PATTERNS:
            self.assertIn(p.name, text)


class TestSignalsFire(unittest.TestCase):
    def _m(self, **kw):
        base = {
            "input_tokens": 1000.0, "cached_input_tokens": 500.0,
            "cache_write_tokens": 0.0, "output_tokens": 200.0,
            "cost_usd": 1.0, "row_count": 100.0, "failure_rate": 0.0,
            "cache_hit_rate": 0.33, "distinct_sources": 5.0,
            "top_model_share": 0.3, "top_model_cost_multiple": 1.0,
            "peak_model_cost_multiple": 1.0, "peak_model_requests": 10.0,
            "unpriced_share": 0.0, "peak_to_median_day": 1.0,
            "peak_day_cost_usd": 1.0,
        }
        base.update(kw)
        return base

    def test_healthy_measures_match_nothing(self):
        self.assertEqual(match_patterns(self._m()), [])

    def test_context_loop_fires_on_high_ratio(self):
        m = self._m(input_tokens=100_000.0, output_tokens=1_000.0)
        keys = {x.pattern.key for x in match_patterns(m)}
        self.assertIn("context_loop", keys)

    def test_retry_storm_fires_on_failure_rate(self):
        keys = {x.pattern.key for x in match_patterns(self._m(failure_rate=0.3))}
        self.assertIn("retry_storm", keys)

    def test_uncached_prefix_fires_on_zero_hits(self):
        keys = {x.pattern.key for x in match_patterns(
            self._m(cache_hit_rate=0.0, input_tokens=2_000_000.0))}
        self.assertIn("uncached_prefix", keys)

    def test_uncached_prefix_needs_volume(self):
        """A tiny amount of traffic with no cache hits is not worth reporting."""
        keys = {x.pattern.key for x in match_patterns(
            self._m(cache_hit_rate=0.0, input_tokens=1_000.0))}
        self.assertNotIn("uncached_prefix", keys)

    def test_no_attribution_needs_enough_rows(self):
        keys = {x.pattern.key for x in match_patterns(
            self._m(distinct_sources=1.0, row_count=5.0))}
        self.assertNotIn("no_attribution", keys)
        keys = {x.pattern.key for x in match_patterns(
            self._m(distinct_sources=1.0, row_count=500.0))}
        self.assertIn("no_attribution", keys)

    def test_premium_model_fires_on_per_call_multiple(self):
        keys = {x.pattern.key for x in match_patterns(
            self._m(peak_model_cost_multiple=20.0, peak_model_requests=10.0))}
        self.assertIn("premium_model_default", keys)

    def test_headless_spike_fires_on_peak_to_median(self):
        keys = {x.pattern.key for x in match_patterns(
            self._m(peak_to_median_day=40.0, peak_day_cost_usd=500.0))}
        self.assertIn("headless_spike", keys)

    def test_match_carries_evidence_with_the_source(self):
        m = self._m(input_tokens=200_000.0, output_tokens=1_000.0)
        match = [x for x in match_patterns(m) if x.pattern.key == "context_loop"][0]
        ev = match.evidence()
        self.assertIn("source", ev)
        self.assertIn("arXiv", str(ev["source"]))
        self.assertEqual(ev["recorded_incidents"], 11)


class TestMeasure(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def _add(self, *, key="prod", model="gpt-6.1-sol", inp=1000, out=200,
             cached=0, cost=0.01, status=200, projects=("web",), minutes=30):
        for project in projects:
            self.store.insert(RequestRecord(
                provider="openai", model=model, input_tokens=inp, output_tokens=out,
                cached_input_tokens=cached, cost_usd=cost, status=status,
                api_key_id=key, project=project, ts=_ago(minutes),
            ))

    def test_measures_reflect_the_rows(self):
        self._add(inp=2000, out=500, cached=1000, cost=0.5)
        self._add(inp=2000, out=500, cached=1000, cost=0.5)
        m = measure(self.store, window_sql="1=1")
        self.assertEqual(m["input_tokens"], 4000.0)
        self.assertEqual(m["cached_input_tokens"], 2000.0)
        self.assertEqual(m["output_tokens"], 1000.0)
        self.assertAlmostEqual(m["cost_usd"], 1.0)
        self.assertAlmostEqual(m["cache_hit_rate"], 2000 / 6000, places=4)

    def test_distinct_sources_counts_keys_and_projects(self):
        self._add(key="a", projects=("web",))
        self._add(key="b", projects=("batch",))
        m = measure(self.store, window_sql="1=1")
        self.assertEqual(m["distinct_sources"], 2.0)

    def test_unpriced_share_is_measured(self):
        self._add(cost=None)
        self._add(cost=0.5)
        m = measure(self.store, window_sql="1=1")
        self.assertAlmostEqual(m["unpriced_share"], 0.5, places=4)

    def test_peak_to_median_uses_hours_for_request_data(self):
        self._add(cost=0.01, minutes=30)
        self._add(cost=0.01, minutes=90)
        self._add(cost=5.0, minutes=200)
        m = measure(self.store, window_sql="1=1", granularity="request")
        self.assertGreaterEqual(m["peak_to_median_day"], 1.0)


class TestCitationsAreEarned(unittest.TestCase):
    """The core guarantee: no citation without a measured signal."""

    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_citation_appears_only_when_the_signal_fires(self):
        # A clean account: two keys, cache reads present, steady spend.
        for i in range(40):
            self.store.insert(RequestRecord(
                provider="openai", model="gpt-6.1-sol", input_tokens=2000,
                cached_input_tokens=4000, output_tokens=500, cost_usd=0.01,
                api_key_id=f"k{i % 3}", project=f"p{i % 3}", ts=_ago(30 + i),
            ))
        d = diagnose(self.store, days=30)
        for finding in d.findings:
            self.assertIsNone(
                finding.catalogue,
                f"{finding.key} cited a pattern its data does not support",
            )

    def test_loop_is_cited_with_the_specific_signal(self):
        for i in range(40):
            self.store.insert(RequestRecord(
                provider="openai", model="gpt-6.1-sol", input_tokens=100_000,
                output_tokens=1_000, cost_usd=0.4, api_key_id="looping",
                project="agent", ts=_ago(20 + i),
            ))
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "context_bloat"]
        self.assertTrue(found)
        c = found[0].catalogue
        self.assertIsNotNone(c, "loop finding should cite the catalogued pattern")
        self.assertEqual(c["name"], "Unbounded context loop")
        self.assertIn("ratio", c["signal_measured"])
        self.assertIn("looping", c["signal_measured"])
        self.assertTrue(c["sources"])

    def test_citation_records_which_scope_was_measured(self):
        for i in range(40):
            self.store.insert(RequestRecord(
                provider="openai", model="gpt-6.1-sol", input_tokens=100_000,
                output_tokens=1_000, cost_usd=0.4, api_key_id="looping",
                ts=_ago(20 + i),
            ))
        d = diagnose(self.store, days=30)
        c = [f for f in d.findings if f.key == "context_bloat"][0].catalogue
        self.assertIn("measured on", c["signal_measured"])

    def test_report_states_that_a_match_is_not_a_diagnosis(self):
        from llmguard.diagnosis_report import render_diagnosis_html
        for i in range(40):
            self.store.insert(RequestRecord(
                provider="openai", model="gpt-6.1-sol", input_tokens=100_000,
                output_tokens=1_000, cost_usd=0.4, api_key_id="looping",
                ts=_ago(20 + i),
            ))
        doc = render_diagnosis_html(diagnose(self.store, days=30))
        self.assertIn("not a confirmed diagnosis", doc.replace("\n", " "))


if __name__ == "__main__":
    unittest.main()
