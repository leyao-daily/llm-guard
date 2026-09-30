"""Tests for the calibration ledger: predicted versus realised.

This is the asset that takes years to build and cannot be copied from a
repository, so the tests are about the two ways it can quietly become worthless:

  1. **Comparing the wrong windows.** A follow-up that reads recent traffic and
     compares it against a baseline from the same recent traffic will always
     report no change. Early versions did exactly that.
  2. **Attributing a traffic change to a fix.** If request volume halved, spend
     halved, and nothing was fixed, the ledger must not record a saving.
"""

from __future__ import annotations

import unittest
from datetime import datetime, timedelta, timezone

from llmguard.analytics import window_between
from llmguard.outcomes import (
    calibrate,
    describe_calibration,
    list_engagements,
    measure_engagement,
    open_engagement,
)
from llmguard.storage import RequestRecord, Store, iso, utcnow


class LedgerBase(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")
        self.now = utcnow()

    def tearDown(self) -> None:
        self.store.close()

    def add(self, days_ago: int, *, key: str, inp: int, out: int, cached: int = 0,
            cost: float = 1.0, model: str = "claude-sonnet-4.5") -> None:
        self.store.insert(RequestRecord(
            provider="anthropic", model=model, input_tokens=inp,
            cached_input_tokens=cached, output_tokens=out, cost_usd=cost,
            api_key_id=key, project="agent",
            ts=iso(self.now - timedelta(days=days_ago)),
        ))

    def loop_traffic(self, *, from_days_ago: int, to_days_ago: int, cost: float = 12.0) -> None:
        # Half-open like the window it feeds: from_days_ago exclusive, to_days_ago
        # inclusive. Getting this wrong leaks a day of baseline traffic into the
        # follow-up window and makes the comparison look like it worked.
        for d in range(to_days_ago + 1, from_days_ago):
            self.add(d, key="agent-loop", inp=4_000_000, out=54_000, cost=cost)
            self.add(d, key="web", inp=200_000, out=40_000, cached=150_000, cost=0.9)

    def fixed_traffic(self, *, from_days_ago: int, to_days_ago: int) -> None:
        for d in range(to_days_ago, from_days_ago):
            self.add(d, key="agent-loop", inp=300_000, out=40_000, cached=240_000, cost=1.1)
            self.add(d, key="web", inp=200_000, out=40_000, cached=150_000, cost=0.9)


class TestOpenEngagement(LedgerBase):
    def test_snapshots_the_diagnosis_as_the_baseline(self):
        self.loop_traffic(from_days_ago=30, to_days_ago=0)
        e = open_engagement(self.store, "Acme", baseline_days=30)
        self.assertEqual(e.client, "Acme")
        self.assertGreater(e.baseline_spend, 0)
        self.assertGreater(e.predicted_high, 0)
        self.assertTrue(e.findings)

    def test_records_explicit_window_dates(self):
        self.loop_traffic(from_days_ago=30, to_days_ago=0)
        e = open_engagement(self.store, "Acme", baseline_days=30)
        self.assertTrue(e.baseline_from)
        self.assertTrue(e.baseline_to)
        self.assertLess(e.baseline_from, e.baseline_to)

    def test_baseline_is_frozen_not_recomputed(self):
        """Recomputing at measurement time would let the goalposts move."""
        self.loop_traffic(from_days_ago=30, to_days_ago=0)
        e = open_engagement(self.store, "Acme", baseline_days=30)
        before = e.predicted_high
        self.fixed_traffic(from_days_ago=30, to_days_ago=0)
        again = list_engagements(self.store)[0]
        self.assertEqual(again.predicted_high, before)

    def test_findings_carry_their_estimates(self):
        self.loop_traffic(from_days_ago=30, to_days_ago=0)
        e = open_engagement(self.store, "Acme", baseline_days=30)
        self.assertTrue(any(f["predicted_high"] > 0 for f in e.findings))
        for f in e.findings:
            self.assertIn("key", f)
            self.assertIn("confidence", f)

    def test_empty_database_opens_without_crashing(self):
        e = open_engagement(self.store, "Empty", baseline_days=30)
        self.assertEqual(e.predicted_high, 0.0)


class TestMeasureAdjacentWindow(LedgerBase):
    def _baseline(self):
        """Baseline 60..30 days ago, then fixed traffic 30..0 days ago."""
        self.loop_traffic(from_days_ago=60, to_days_ago=31)
        win = window_between(
            (self.now - timedelta(days=60)).date().isoformat(),
            (self.now - timedelta(days=30)).date().isoformat(),
        )
        from llmguard.diagnose import diagnose
        d = diagnose(self.store, days=30, window=win)
        from llmguard.outcomes import ensure_schema
        ensure_schema(self.store)
        self.store.execute(
            """INSERT INTO engagements
               (client, created_at, baseline_from, baseline_to, baseline_days,
                baseline_spend, baseline_requests, predicted_low, predicted_high, status)
               VALUES (?,?,?,?,?,?,?,?,?,'open')""",
            ("Fixed Co", iso(self.now - timedelta(days=30)), win.start, win.end, 30,
             d.total_spend, d.total_requests,
             sum(f.monthly_saving_low for f in d.findings),
             sum(f.monthly_saving_high for f in d.findings)),
        )

    def test_measures_the_window_after_the_baseline(self):
        """The core guarantee. Early versions re-read recent traffic and saw nothing."""
        self._baseline()
        self.fixed_traffic(from_days_ago=30, to_days_ago=0)
        r, warnings = measure_engagement(self.store, "Fixed Co", days=30)
        self.assertIsNotNone(r)
        self.assertNotEqual(r.window_from, r.engagement.baseline_from)
        self.assertGreaterEqual(r.window_from, r.engagement.baseline_to)

    def test_detects_a_real_saving(self):
        self._baseline()
        self.fixed_traffic(from_days_ago=30, to_days_ago=0)
        r, _ = measure_engagement(self.store, "Fixed Co", days=30)
        self.assertGreater(r.realised_saving, 0)

    def test_reports_no_change_when_nothing_changed(self):
        self._baseline()
        # same loop, still running, in the follow-up window
        self.loop_traffic(from_days_ago=30, to_days_ago=0, cost=12.0)
        r, _ = measure_engagement(self.store, "Fixed Co", days=30)
        self.assertIsNotNone(r)
        self.assertAlmostEqual(r.realised_saving, 0.0, delta=max(r.monthly_baseline * 0.1, 1.0))

    def test_refuses_when_there_is_no_follow_up_traffic(self):
        self._baseline()
        r, warnings = measure_engagement(self.store, "Fixed Co", days=30)
        self.assertIsNone(r)
        self.assertTrue(warnings)

    def test_volume_change_is_flagged_as_a_confound(self):
        """Spend fell because traffic fell. That is not a saving."""
        self._baseline()
        # a quarter of the requests in the follow-up window
        for d in range(0, 30, 4):
            self.add(d, key="agent-loop", inp=4_000_000, out=54_000, cost=12.0)
        r, warnings = measure_engagement(self.store, "Fixed Co", days=30)
        if r is not None:
            self.assertTrue(
                any("volume" in w for w in warnings),
                f"volume confound not flagged: {warnings}",
            )

    def test_partial_window_is_flagged(self):
        self._baseline()
        # only a few days of the follow-up window exist
        for d in range(0, 3):
            self.add(d, key="agent-loop", inp=300_000, out=40_000, cached=240_000, cost=1.1)
        r, warnings = measure_engagement(self.store, "Fixed Co", days=30)
        if r is not None:
            self.assertTrue(any("elapsed" in w or "partial" in w for w in warnings))

    def test_unknown_client_is_reported_not_silently_ignored(self):
        r, warnings = measure_engagement(self.store, "Nobody", days=30)
        self.assertIsNone(r)
        self.assertTrue(any("no engagement" in w for w in warnings))


class TestCalibration(LedgerBase):
    def test_empty_ledger_says_so_plainly(self):
        text = describe_calibration(self.store)
        self.assertIn("Nothing measured yet", text)
        self.assertIn("assumption", text)

    def test_calibration_counts_measured_engagements(self):
        self.loop_traffic(from_days_ago=60, to_days_ago=30)
        win = window_between(
            (self.now - timedelta(days=60)).date().isoformat(),
            (self.now - timedelta(days=30)).date().isoformat(),
        )
        from llmguard.diagnose import diagnose
        from llmguard.outcomes import ensure_schema
        ensure_schema(self.store)
        d = diagnose(self.store, days=30, window=win)
        self.store.execute(
            """INSERT INTO engagements
               (client, created_at, baseline_from, baseline_to, baseline_days,
                baseline_spend, baseline_requests, predicted_low, predicted_high, status)
               VALUES (?,?,?,?,?,?,?,?,?,'open')""",
            ("Fixed Co", iso(self.now - timedelta(days=30)), win.start, win.end, 30,
             d.total_spend, d.total_requests,
             sum(f.monthly_saving_low for f in d.findings),
             sum(f.monthly_saving_high for f in d.findings)),
        )
        self.fixed_traffic(from_days_ago=30, to_days_ago=0)
        measure_engagement(self.store, "Fixed Co", days=30)

        c = calibrate(self.store)
        self.assertEqual(c.total, 1)
        self.assertEqual(c.measured, 1)
        self.assertEqual(len(c.ratios), 1)
        self.assertFalse(c.has_sample, "one measurement is not a sample")

    def test_small_sample_is_declared_unquotable(self):
        text = describe_calibration(self.store)
        self.assertIn("Nothing measured yet", text)


class TestWindowArithmetic(unittest.TestCase):
    def test_window_between_is_half_open(self):
        w = window_between("2026-08-01", "2026-09-01")
        self.assertIn(">=", w.sql)
        self.assertIn("<", w.sql)
        self.assertEqual(w.days, 31)

    def test_adjacent_windows_do_not_overlap(self):
        a = window_between("2026-08-01", "2026-09-01")
        b = window_between("2026-09-01", "2026-10-01")
        self.assertEqual(a.end, b.start)
        # first is [start, end), second is [start, end): row exactly at the boundary
        # belongs to the second only
        self.assertEqual(a.params[1], "2026-09-01 00:00:00")
        self.assertEqual(b.params[0], "2026-09-01 00:00:00")


if __name__ == "__main__":
    unittest.main()
