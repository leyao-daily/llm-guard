"""Tests for the cost diagnostics and the client-facing report.

The diagnostics are the paid deliverable, so the bar is higher than for the rest
of the codebase: every finding has to fire on the shape it claims to detect, stay
quiet on a healthy bill, and carry evidence that matches what it says.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from llmguard.diagnose import (
    CONFIDENCE_ARITHMETIC,
    diagnose,
)
from llmguard.diagnosis_report import render_diagnosis_html
from llmguard.storage import RequestRecord, Store, iso


def _ago(minutes: int = 30) -> str:
    return iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))


class DiagnosisBase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def add(self, *, key="prod-web", model="gpt-6.1-sol", inp=1000, out=200,
            cached=0, cache_write=0, cost=None, status=200, error="", minutes=30):
        if cost is None:
            from llmguard.pricing import TokenUsage, compute_cost
            c = compute_cost(model, TokenUsage(input=inp, output=out,
                                               cached_input=cached, cache_write=cache_write))
            cost = None if c is None else float(c)
        self.store.insert(RequestRecord(
            provider="openai", model=model, input_tokens=inp, output_tokens=out,
            cached_input_tokens=cached, cache_write_tokens=cache_write,
            cost_usd=cost, status=status, error=error, api_key_id=key,
            project="web", ts=_ago(minutes),
        ))

    def healthy(self, n=60):
        for i in range(n):
            self.add(inp=2000, out=400, cached=1200, minutes=30 + i)


class TestContextBloat(DiagnosisBase):
    def test_detects_a_loop_on_one_key(self):
        for i in range(30):
            self.add(key="looping", inp=100_000, out=1_300, cost=0.30, minutes=20 + i)
        self.healthy()
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "context_bloat"]
        self.assertTrue(found, "loop not detected")
        self.assertIn("looping", found[0].title)

    def test_healthy_ratio_is_not_flagged(self):
        self.healthy(80)
        d = diagnose(self.store, days=30)
        self.assertFalse([f for f in d.findings if f.key == "context_bloat"])

    def test_one_looping_key_is_named_not_averaged_away(self):
        """The bug this check was rewritten for: a global average hid the loop."""
        for i in range(60):
            self.add(key="looping", inp=100_000, out=1_000, cost=0.4, minutes=20 + i)
        for i in range(60):
            self.add(key="normal", inp=2_000, out=500, cached=1_200, cost=0.002, minutes=90 + i)
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "context_bloat"]
        self.assertTrue(found)
        self.assertIn("looping", found[0].title)
        self.assertNotIn("normal", found[0].title)

    def test_finding_carries_evidence_matching_its_claim(self):
        for i in range(30):
            self.add(key="looping", inp=100_000, out=1_000, cost=0.4, minutes=20 + i)
        d = diagnose(self.store, days=30)
        f = [x for x in d.findings if x.key == "context_bloat"][0]
        self.assertEqual(f.evidence["ratio"], round(100_000 / 1_000, 1))
        self.assertIn("input_tokens", f.evidence)
        self.assertTrue(f.remedy)


class TestCacheOpportunity(DiagnosisBase):
    def test_detects_zero_cache_hits(self):
        for i in range(40):
            self.add(inp=50_000, out=500, cached=0, cost=0.2, minutes=20 + i)
        d = diagnose(self.store, days=30)
        self.assertTrue([f for f in d.findings if f.key == "cache_opportunity"])

    def test_good_cache_rate_is_not_flagged(self):
        for i in range(60):
            self.add(inp=2_000, out=400, cached=5_000, cost=0.01, minutes=20 + i)
        d = diagnose(self.store, days=30)
        self.assertFalse([f for f in d.findings if f.key == "cache_opportunity"])


class TestRouting(DiagnosisBase):
    def test_expensive_model_is_flagged(self):
        for i in range(80):
            self.add(model="gpt-6.1-sol", inp=1_000, out=200, cost=0.004, minutes=40 + i)
        for i in range(6):
            self.add(model="gpt-6-astra", inp=9_000, out=2_000, cost=0.30, minutes=10 + i)
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "routing"]
        self.assertTrue(found)
        self.assertIn("astra", found[0].title)
        self.assertEqual(found[0].confidence, "worth testing")

    def test_uniform_costs_are_not_flagged(self):
        for i in range(80):
            self.add(inp=1_000, out=200, cost=0.004, minutes=30 + i)
        d = diagnose(self.store, days=30)
        self.assertFalse([f for f in d.findings if f.key == "routing"])


class TestErrorsAndUnpriced(DiagnosisBase):
    def test_error_rate_is_reported(self):
        for i in range(200):
            bad = i % 10 == 0
            self.add(status=503 if bad else 200, error="upstream status 503" if bad else "",
                     cost=0.0 if bad else 0.004, minutes=30 + i)
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "errors"]
        self.assertTrue(found)
        self.assertEqual(found[0].confidence, CONFIDENCE_ARITHMETIC)
        self.assertGreaterEqual(found[0].evidence["failure_rate"], 0.02)

    def test_unpriced_models_are_surfaced_and_listed(self):
        for i in range(40):
            self.add(model="ft:internal-v3", inp=1000, out=200, cost=None, minutes=10 + i)
        for i in range(40):
            self.add(inp=1000, out=200, cost=0.004, minutes=60 + i)
        d = diagnose(self.store, days=30)
        self.assertIn("ft:internal-v3", d.unpriced_models)
        self.assertTrue([f for f in d.findings if f.key == "unpriced"])
        self.assertTrue(any("floor" in g for g in d.gaps))


class TestAttributionAndConcentration(DiagnosisBase):
    def test_single_key_is_flagged(self):
        for i in range(150):
            self.add(key="only", inp=1000, out=200, cost=0.004, minutes=30)
        d = diagnose(self.store, days=30)
        self.assertTrue([f for f in d.findings if f.key == "attribution"])

    def test_multi_key_is_not_flagged(self):
        for i in range(80):
            self.add(key=f"k{i % 5}", inp=1000, out=200, cost=0.004, minutes=30 + i)
        d = diagnose(self.store, days=30)
        self.assertFalse([f for f in d.findings if f.key == "attribution"])

    def test_concentration_is_flagged(self):
        for i in range(50):
            self.add(model="gpt-6-astra", inp=1000, out=200, cost=1.0, minutes=30 + i)
        for i in range(50):
            self.add(model="gpt-6-luna", inp=1000, out=200, cost=0.001, minutes=60 + i)
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "concentration"]
        self.assertTrue(found)


class TestDiagnosisShape(DiagnosisBase):
    def test_empty_database_does_not_crash(self):
        d = diagnose(self.store, days=30)
        self.assertEqual(d.total_requests, 0)
        self.assertEqual(d.findings, [])

    def test_findings_are_ranked_by_money(self):
        for i in range(30):
            self.add(key="looping", inp=100_000, out=1_000, cost=0.4, minutes=20 + i)
        self.healthy()
        d = diagnose(self.store, days=30)
        money = [f.has_money for f in d.ranked]
        # every moneyed finding sorts before every unmoneyed one
        self.assertEqual(money, sorted(money, reverse=True))

    def test_every_finding_has_remedy_and_confidence(self):
        for i in range(30):
            self.add(key="looping", inp=100_000, out=1_000, cost=0.4, minutes=20 + i)
        d = diagnose(self.store, days=30)
        self.assertTrue(d.findings)
        for f in d.findings:
            self.assertTrue(f.remedy, f.key)
            self.assertTrue(f.detail, f.key)
            self.assertIn(f.confidence, ("certain", "likely", "worth testing"))

    def test_short_window_is_declared_as_a_gap(self):
        self.healthy(40)
        d = diagnose(self.store, days=7)
        self.assertTrue(any("7 days" in g for g in d.gaps))

    def test_monthly_scaling_is_proportional(self):
        for i in range(60):
            self.add(inp=1000, out=200, cost=0.01, minutes=30 + i)
        d = diagnose(self.store, days=30)
        self.assertAlmostEqual(d.monthly(30.0), 30.0, places=6)
        d7 = diagnose(self.store, days=7)
        self.assertGreater(d7.monthly(7.0), 7.0)   # 7 days scaled to 30


class TestReportRendering(DiagnosisBase):
    def _report(self, **kw):
        for i in range(30):
            self.add(key="looping", inp=100_000, out=1_000, cost=0.4, minutes=20 + i)
        self.healthy()
        d = diagnose(self.store, days=30)
        return render_diagnosis_html(d, **kw)

    def test_html_is_well_formed_and_self_contained(self):
        doc = self._report(client="Acme Corp")
        self.assertIn("<!DOCTYPE html>", doc)
        self.assertIn("</html>", doc)
        self.assertIn("Acme Corp", doc)
        # no external assets
        self.assertNotIn("http://", doc.replace("http://www.w3.org", ""))
        self.assertNotIn("<script", doc)
        self.assertNotIn("cdn.", doc)

    def test_renders_evidence_and_confidence(self):
        doc = self._report()
        self.assertIn("Evidence", doc)
        self.assertIn("What to do", doc)
        self.assertIn("Likely", doc)

    def test_states_that_estimates_overlap(self):
        doc = self._report()
        self.assertIn("the same tokens", doc)

    def test_empty_diagnosis_still_renders(self):
        d = diagnose(self.store, days=30)
        doc = render_diagnosis_html(d)
        self.assertIn("<!DOCTYPE html>", doc)

    def test_client_name_is_escaped(self):
        doc = self._report(client='<script>alert(1)</script>')
        self.assertNotIn("<script>alert(1)</script>", doc)
        self.assertIn("&lt;script&gt;", doc)


if __name__ == "__main__":
    unittest.main()
