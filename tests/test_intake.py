"""Tests for getting a customer's data in.

The intake path decides whether this service is sellable at all: if a prospect has
to deploy infrastructure before they learn whether the diagnosis is worth paying
for, the conversation ends. So the file path has to work with whatever a provider
dashboard actually exports, and it has to be honest about what that data cannot
support.
"""

from __future__ import annotations

import json
import unittest

from llmguard.diagnose import diagnose
from llmguard.intake import (
    CANONICAL_FIELDS,
    import_csv,
    import_json,
)
from llmguard.storage import Store

CLIENT_CSV = """bucket_start,provider,model,input_tokens,cached_input_tokens,cache_write_tokens,output_tokens,cost_usd,workspace
2026-09-01T00:00:00Z,openai,gpt-6.1-sol,1000000,500000,0,80000,12.50,prod-web
2026-09-02T00:00:00Z,openai,gpt-6.1-sol,1100000,550000,0,88000,13.75,prod-web
2026-09-03T00:00:00Z,openai,gpt-6.1-sol,900000,450000,0,72000,11.25,prod-web
2026-09-01T00:00:00Z,anthropic,claude-sonnet-4.5,9000000,0,0,120000,28.00,prod-agent
2026-09-02T00:00:00Z,anthropic,claude-sonnet-4.5,9500000,0,0,128000,29.50,prod-agent
2026-09-03T00:00:00Z,anthropic,claude-sonnet-4.5,10200000,0,0,138000,31.00,prod-agent
"""


class TestCsvImport(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_imports_a_provider_style_export(self):
        result = import_csv(self.store, CLIENT_CSV)
        self.assertEqual(result.rows_in, 6)
        self.assertEqual(result.rows_kept, 6)
        self.assertEqual(result.rows_skipped, 0)
        self.assertGreater(result.total_cost_usd, 0)
        self.assertEqual(self.store.count(), 6)

    def test_falls_back_to_computed_cost_when_blank(self):
        text = CLIENT_CSV.replace("12.50", "").replace("13.75", "").replace("11.25", "")
        result = import_csv(self.store, text)
        self.assertEqual(result.rows_kept, 6)
        # computed from the price table, so still non-zero
        self.assertGreater(result.total_cost_usd, 0)

    def test_column_aliases_are_accepted(self):
        text = (
            "date,model,prompt_tokens,completion_tokens,cached_tokens\n"
            "2026-09-01,gpt-6.1-sol,1000,200,400\n"
        )
        result = import_csv(self.store, text)
        self.assertEqual(result.rows_kept, 1)
        row = self.store.one("SELECT * FROM requests")
        self.assertEqual(row["input_tokens"], 1000)
        self.assertEqual(row["cached_input_tokens"], 400)
        self.assertEqual(row["output_tokens"], 200)

    def test_rows_without_tokens_are_skipped_with_a_reason(self):
        text = (
            "bucket_start,model,input_tokens,output_tokens\n"
            "2026-09-01,gpt-6.1-sol,0,0\n"
            "2026-09-02,gpt-6.1-sol,1000,200\n"
        )
        result = import_csv(self.store, text)
        self.assertEqual(result.rows_kept, 1)
        self.assertEqual(result.rows_skipped, 1)
        self.assertTrue(result.notes)

    def test_unparseable_date_is_skipped(self):
        text = (
            "bucket_start,model,input_tokens,output_tokens\n"
            "not-a-date,gpt-6.1-sol,1000,200\n"
        )
        result = import_csv(self.store, text)
        self.assertEqual(result.rows_kept, 0)
        self.assertEqual(result.rows_skipped, 1)

    def test_missing_model_is_skipped(self):
        text = "bucket_start,model,input_tokens,output_tokens\n2026-09-01,,1000,200\n"
        result = import_csv(self.store, text)
        self.assertEqual(result.rows_kept, 0)

    def test_unpriced_models_are_reported_not_silently_zeroed(self):
        text = (
            "bucket_start,model,input_tokens,output_tokens\n"
            "2026-09-01,ft:internal-v2,1000,200\n"
        )
        result = import_csv(self.store, text)
        self.assertIn("ft:internal-v2", result.unpriced_models)
        row = self.store.one("SELECT cost_usd FROM requests")
        self.assertIsNone(row["cost_usd"])

    def test_empty_file_does_not_crash(self):
        result = import_csv(self.store, "")
        self.assertEqual(result.rows_kept, 0)
        self.assertTrue(result.notes)

    def test_marks_the_data_as_bucketed(self):
        import_csv(self.store, CLIENT_CSV)
        self.assertEqual(self.store.get_meta("granularity"), "bucket")

    def test_timestamps_land_in_sqlite_comparable_format(self):
        import_csv(self.store, CLIENT_CSV)
        row = self.store.one("SELECT ts FROM requests LIMIT 1")
        self.assertEqual(row["ts"][10], " ")          # space, not a T
        self.assertEqual(len(row["ts"]), 19)

    def test_various_timestamp_shapes(self):
        shapes = [
            "2026-09-01T00:00:00Z",
            "2026-09-01T00:00:00+00:00",
            "2026-09-01T00:00:00",
            "2026-09-01 00:00:00",
            "2026-09-01",
        ]
        for shape in shapes:
            store = Store(":memory:")
            text = (
                "bucket_start,model,input_tokens,output_tokens\n"
                f"{shape},gpt-6.1-sol,1000,200\n"
            )
            result = import_csv(store, text)
            self.assertEqual(result.rows_kept, 1, f"failed on {shape}")
            store.close()


class TestJsonImport(unittest.TestCase):
    def setUp(self):
        self.store = Store(":memory:")

    def tearDown(self):
        self.store.close()

    def test_list_payload(self):
        payload = [
            {"bucket_start": "2026-09-01", "model": "gpt-6.1-sol",
             "input_tokens": 1000, "output_tokens": 200}
        ]
        result = import_json(self.store, json.dumps(payload))
        self.assertEqual(result.rows_kept, 1)

    def test_wrapped_data_payload(self):
        payload = {"data": [
            {"bucket_start": "2026-09-01", "model": "gpt-6.1-sol",
             "input_tokens": 1000, "output_tokens": 200}
        ]}
        result = import_json(self.store, json.dumps(payload))
        self.assertEqual(result.rows_kept, 1)

    def test_invalid_json_reports_rather_than_raising(self):
        result = import_json(self.store, "{not json")
        self.assertEqual(result.rows_kept, 0)
        self.assertTrue(any("invalid json" in n for n in result.notes))


class TestBucketedDiagnosis(unittest.TestCase):
    """Aggregate data must change the diagnosis, not silently break it."""

    def setUp(self):
        self.store = Store(":memory:")
        import_csv(self.store, CLIENT_CSV)

    def tearDown(self):
        self.store.close()

    def test_granularity_is_read_from_the_data(self):
        d = diagnose(self.store, days=30)
        self.assertEqual(d.granularity, "bucket")
        self.assertIn("bucket", d.unit)

    def test_volume_floor_is_lower_for_bucketed_data(self):
        d = diagnose(self.store, days=30)
        self.assertLess(d.volume_floor(20), 20)
        self.assertGreaterEqual(d.volume_floor(20), 2)

    def test_a_loop_is_still_found_in_daily_buckets(self):
        """Six daily buckets is a week of history. The threshold must allow it."""
        d = diagnose(self.store, days=30)
        found = [f for f in d.findings if f.key == "context_bloat"]
        self.assertTrue(found, "loop missed on bucketed data")
        self.assertIn("prod-agent", found[0].title)

    def test_wording_says_buckets_not_requests(self):
        d = diagnose(self.store, days=30)
        f = [x for x in d.findings if x.key == "context_bloat"][0]
        self.assertIn("bucket", f.detail)

    def test_checks_that_need_per_request_rows_are_skipped(self):
        d = diagnose(self.store, days=30)
        keys = {f.key for f in d.findings}
        self.assertNotIn("errors", keys)
        self.assertNotIn("attribution", keys)

    def test_the_gap_is_stated_explicitly(self):
        d = diagnose(self.store, days=30)
        self.assertTrue(any("aggregated by day" in g for g in d.gaps))
        self.assertTrue(any("latency" in g for g in d.gaps))

    def test_report_renders_the_bucket_wording(self):
        from llmguard.diagnosis_report import render_diagnosis_html
        d = diagnose(self.store, days=30)
        doc = render_diagnosis_html(d, client="Prospect Ltd")
        self.assertIn("daily buckets", doc)
        self.assertIn("aggregated by day", doc)


class TestPerRequestDataUnaffected(unittest.TestCase):
    """The proxy path still reports in requests."""

    def setUp(self):
        self.store = Store(":memory:")
        self.store.set_meta("granularity", "request")

    def tearDown(self):
        self.store.close()

    def test_granularity_defaults_to_request(self):
        d = diagnose(self.store, days=30)
        self.assertEqual(d.granularity, "request")
        self.assertEqual(d.unit, "requests")
        self.assertEqual(d.volume_floor(20), 20)

    def test_missing_marker_falls_back_to_request(self):
        store = Store(":memory:")
        d = diagnose(store, days=30)
        self.assertEqual(d.granularity, "request")
        store.close()


class TestSampleDocumentation(unittest.TestCase):
    def test_canonical_fields_are_documented(self):
        self.assertIn("bucket_start", CANONICAL_FIELDS)
        self.assertIn("model", CANONICAL_FIELDS)
        self.assertIn("output_tokens", CANONICAL_FIELDS)


if __name__ == "__main__":
    unittest.main()
