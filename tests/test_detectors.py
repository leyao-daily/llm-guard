"""Tests for runaway-agent detection and mid-stream cancellation.

These cover the two failure modes the public research identifies as the ones
that actually produce five-figure bills, and which a per-key budget cannot stop:

1. **Runaway loops** -- detected from a sustained pathological input/output
   token ratio, spend velocity against the account's own baseline, or a burst of
   failing calls.
2. **The request that crosses the budget** -- a budget checked between requests
   cannot stop the streaming request that is crossing the line right now.
"""

from __future__ import annotations

import asyncio
import json
import socket
import time
import unittest
from datetime import datetime, timedelta, timezone

from llmguard.detectors import (
    DetectionConfig,
    LiveGuard,
    detect_all,
    detect_io_ratio,
    detect_retry_storm,
    detect_velocity,
)
from llmguard.gateway import Gateway, GatewayConfig, Route, StreamBudget, _SseUsageDecoder
from llmguard.pricing import TokenUsage
from llmguard.storage import RequestRecord, Store, iso

from test_gateway_e2e import MockUpstream, http_request


def _recent(minutes: int = 1) -> str:
    return iso(datetime.now(timezone.utc) - timedelta(minutes=minutes))


class TestLiveLoopGuard(unittest.TestCase):
    def test_no_warning_before_enough_samples(self):
        guard = LiveGuard(io_ratio=30.0, min_samples=8)
        for _ in range(7):
            self.assertIsNone(guard.observe("k", input_tokens=10_000, output_tokens=100))

    def test_warns_on_sustained_high_ratio(self):
        guard = LiveGuard(io_ratio=30.0, min_samples=8)
        warning = None
        for _ in range(10):
            warning = guard.observe("k", input_tokens=10_000, output_tokens=100)
        self.assertIsNotNone(warning)
        self.assertIn("agent loop", warning)

    def test_does_not_warn_on_normal_traffic(self):
        guard = LiveGuard(io_ratio=30.0, min_samples=8)
        for _ in range(20):
            self.assertIsNone(guard.observe("k", input_tokens=1000, output_tokens=200))

    def test_single_outlier_does_not_trip_the_detector(self):
        """Median, not mean: one document-analysis call is normal."""
        guard = LiveGuard(io_ratio=30.0, min_samples=8)
        for _ in range(10):
            guard.observe("k", input_tokens=1000, output_tokens=200)
        # One pathological call among healthy ones must not raise the median.
        self.assertIsNone(guard.observe("k", input_tokens=500_000, output_tokens=10))

    def test_keys_are_tracked_independently(self):
        guard = LiveGuard(io_ratio=30.0, min_samples=8)
        for _ in range(10):
            guard.observe("noisy", input_tokens=10_000, output_tokens=100)
            guard.observe("healthy", input_tokens=1000, output_tokens=200)
        # The healthy key must still be clean.
        self.assertIsNone(guard.observe("healthy", input_tokens=1000, output_tokens=200))

    def test_zero_output_is_ignored(self):
        guard = LiveGuard(io_ratio=30.0, min_samples=1)
        self.assertIsNone(guard.observe("k", input_tokens=10_000, output_tokens=0))


class TestDetectionQueries(unittest.TestCase):
    def setUp(self) -> None:
        self.store = Store(":memory:")

    def tearDown(self) -> None:
        self.store.close()

    def _add(self, *, key="prod", model="gpt-4o", inp=1000, out=200, cost=0.01,
             status=200, error="", project="web", minutes_ago=1):
        self.store.insert(
            RequestRecord(
                provider="openai", model=model, input_tokens=inp, output_tokens=out,
                cost_usd=cost, status=status, error=error, api_key_id=key,
                project=project, ts=_recent(minutes_ago),
            )
        )

    # -- io ratio ---------------------------------------------------------
    def test_io_ratio_detected_on_sustained_loop(self):
        for _ in range(10):
            self._add(inp=50_000, out=500, cost=0.05)
        found = detect_io_ratio(self.store, DetectionConfig())
        # Reported twice on purpose: once scoped to the key, once to the
        # project, because a loop is usually confined to one workload.
        by_key = [a for a in found if a.subject_type == "key"]
        self.assertEqual(len(by_key), 1)
        self.assertEqual(by_key[0].kind, "io_ratio")
        self.assertEqual(by_key[0].subject, "prod")
        self.assertGreater(by_key[0].evidence["io_ratio"], 30)

    def test_io_ratio_critical_at_double_threshold(self):
        for _ in range(10):
            self._add(inp=200_000, out=500, cost=0.05)
        found = detect_io_ratio(self.store, DetectionConfig())
        self.assertTrue(found)
        self.assertEqual(found[0].severity, "critical")

    def test_normal_traffic_is_not_flagged(self):
        for _ in range(20):
            self._add(inp=2000, out=400)
        self.assertEqual(detect_io_ratio(self.store, DetectionConfig()), [])

    def test_io_ratio_needs_enough_calls(self):
        self._add(inp=100_000, out=100, cost=0.5)
        self.assertEqual(detect_io_ratio(self.store, DetectionConfig()), [])

    def test_io_ratio_ignores_old_traffic(self):
        for _ in range(10):
            self._add(inp=100_000, out=100, cost=0.5, minutes_ago=90)
        self.assertEqual(detect_io_ratio(self.store, DetectionConfig()), [])

    def test_io_ratio_detected_by_project_too(self):
        for _ in range(10):
            self._add(inp=100_000, out=100, cost=0.5, project="nightly")
        kinds = {(a.subject_type, a.subject) for a in detect_io_ratio(self.store, DetectionConfig())}
        self.assertIn(("project", "nightly"), kinds)

    # -- velocity ---------------------------------------------------------
    def test_velocity_burst_detected(self):
        # Establish a quiet baseline over the past week...
        for i in range(1, 60):
            self._add(cost=0.001, minutes_ago=60 * 20 + i)
        # ...then a burst inside the detection window.
        for _ in range(30):
            self._add(cost=5.0, minutes_ago=2)
        found = detect_velocity(self.store, DetectionConfig())
        self.assertTrue(found)
        self.assertEqual(found[0].kind, "velocity")
        self.assertGreater(found[0].evidence["multiple"], 10)

    def test_no_velocity_warning_without_baseline(self):
        for _ in range(30):
            self._add(cost=5.0, minutes_ago=2)
        self.assertEqual(detect_velocity(self.store, DetectionConfig()), [])

    def test_steady_spend_does_not_trip_velocity(self):
        # Uniform spend across the whole week: no burst, no warning.
        for i in range(1, 400):
            self._add(cost=0.01, minutes_ago=i * 25)
        self.assertEqual(detect_velocity(self.store, DetectionConfig()), [])

    # -- retry storm ------------------------------------------------------
    def test_retry_storm_detected(self):
        for _ in range(40):
            self._add(status=503, error="upstream status 503", cost=None, inp=0, out=0)
        found = detect_retry_storm(self.store, DetectionConfig())
        self.assertTrue(found)
        self.assertEqual(found[0].kind, "retry_storm")
        self.assertGreaterEqual(found[0].evidence["failure_rate"], 0.5)

    def test_low_failure_rate_not_flagged(self):
        for _ in range(50):
            self._add(status=200)
        for _ in range(3):
            self._add(status=500, error="upstream status 500")
        self.assertEqual(detect_retry_storm(self.store, DetectionConfig()), [])

    def test_storm_below_minimum_count_not_flagged(self):
        for _ in range(10):
            self._add(status=500, error="upstream status 500")
        self.assertEqual(detect_retry_storm(self.store, DetectionConfig()), [])

    # -- aggregate --------------------------------------------------------
    def test_detect_all_sorts_critical_first(self):
        for _ in range(10):
            self._add(inp=200_000, out=100, cost=0.5)      # critical io ratio
        for _ in range(25):
            self._add(status=503, error="x", cost=None, inp=0, out=0)
        found = detect_all(self.store, DetectionConfig())
        self.assertGreaterEqual(len(found), 2)
        self.assertEqual(found[0].severity, "critical")

    def test_every_anomaly_carries_a_remedy(self):
        for _ in range(10):
            self._add(inp=100_000, out=100, cost=0.5)
        for anomaly in detect_all(self.store, DetectionConfig()):
            self.assertTrue(anomaly.remedy(), anomaly.kind)
            self.assertTrue(anomaly.detail)


class TestStreamBudget(unittest.TestCase):
    def _budget(self, **kw):
        base = dict(max_cost_usd=None, max_tokens=None, initial_tokens=0,
                    model="gpt-4o", chars_per_token=4.0)
        base.update(kw)
        return StreamBudget(**base)

    def test_no_cap_never_trips(self):
        budget = self._budget()
        budget.observe_content("x" * 100_000)
        self.assertIsNone(budget.check())

    def test_token_cap_trips(self):
        budget = self._budget(max_tokens=1000, initial_tokens=900)
        budget.observe_content("y" * 2000)   # ~500 more tokens
        self.assertIsNotNone(budget.check())

    def test_cost_cap_trips(self):
        # gpt-4o output is $10/1M, so 1M estimated output tokens costs ~$10.
        budget = self._budget(max_cost_usd=0.01)
        budget.observe_content("z" * 400_000)   # ~100k tokens -> ~$1
        reason = budget.check()
        self.assertIsNotNone(reason)
        self.assertIn("stream cost cap", reason)

    def test_cap_is_sticky(self):
        budget = self._budget(max_tokens=10, initial_tokens=10)
        first = budget.check()
        second = budget.check()
        self.assertEqual(first, second)

    def test_provider_usage_replaces_the_estimate(self):
        budget = self._budget(initial_tokens=999_999)
        budget.observe_usage(TokenUsage(input=100, output=50, cached_input=900))
        self.assertEqual(budget.input_tokens, 1000)   # 100 + 900
        self.assertEqual(budget.output_tokens, 50)

    def test_abort_frame_is_valid_json_and_signals_error(self):
        budget = self._budget(max_tokens=1, initial_tokens=1)
        budget.check()
        frame = budget.abort_frame()
        self.assertIn(b"[DONE]", frame)
        payload = json.loads(frame.split(b"data: ", 1)[1].split(b"\n", 1)[0])
        self.assertEqual(payload["error"]["type"], "budget_exceeded")
        self.assertIn("llm_guard", payload["error"])

    def test_unpriced_model_estimates_zero_cost(self):
        budget = self._budget(model="ft:unknown", max_cost_usd=0.01)
        budget.observe_content("z" * 400_000)
        self.assertIsNone(budget.check())


class TestSseUsageDecoder(unittest.TestCase):
    def test_extracts_openai_content_and_usage(self):
        decoder = _SseUsageDecoder()
        decoder.feed(
            b'data: {"choices":[{"delta":{"content":"hello "}}]}\n\n'
            b'data: {"choices":[{"delta":{"content":"world"}}]}\n\n'
        )
        self.assertEqual(decoder.take_content(), "hello world")

        decoder.feed(
            b'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":20,'
            b'"prompt_tokens_details":{"cached_tokens":40}}}\n\n'
        )
        usage = decoder.take_usage()
        self.assertEqual(usage.input, 60)      # 100 - 40
        self.assertEqual(usage.output, 20)
        self.assertEqual(usage.cached_input, 40)

    def test_extracts_anthropic_deltas(self):
        decoder = _SseUsageDecoder()
        decoder.feed(b'data: {"type":"content_block_delta","delta":{"text":"abc"}}\n\n')
        self.assertEqual(decoder.take_content(), "abc")

    def test_frame_split_across_reads_is_not_lost(self):
        """A frame straddling two TCP reads must still be counted."""
        decoder = _SseUsageDecoder()
        frame = b'data: {"choices":[{"delta":{"content":"split-frame"}}]}\n\n'
        decoder.feed(frame[:20])
        decoder.feed(frame[20:])
        self.assertEqual(decoder.take_content(), "split-frame")

    def test_done_sentinel_ignored(self):
        decoder = _SseUsageDecoder()
        decoder.feed(b"data: [DONE]\n\n")
        self.assertEqual(decoder.take_content(), "")
        self.assertIsNone(decoder.take_usage())

    def test_malformed_frame_does_not_raise(self):
        decoder = _SseUsageDecoder()
        decoder.feed(b"data: {not json\n\n")
        self.assertEqual(decoder.take_content(), "")


# ---------------------------------------------------------------------------
# end-to-end: the stream is actually cut off on the wire
# ---------------------------------------------------------------------------
class LongStreamUpstream(MockUpstream):
    """Mock upstream that streams many frames so a cap can trip mid-flight."""

    async def _handle(self, reader, writer):  # type: ignore[override]
        try:
            await reader.readline()
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
            body = await reader.read(65536)
            self.received.append(("/", {}, body))

            writer.write(
                b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
                b"Connection: close\r\n\r\n"
            )
            await writer.drain()
            # 300 frames of 4000 chars each -- far more than any cap under test.
            for i in range(300):
                frame = {
                    "model": "gpt-4o",
                    "choices": [{"delta": {"content": "x" * 4000}}],
                }
                writer.write(f"data: {json.dumps(frame)}\n\n".encode())
                await writer.drain()
            writer.write(b"data: [DONE]\n\n")
            await writer.drain()
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
            except Exception:  # noqa: BLE001
                pass


class TestMidStreamCancellation(unittest.TestCase):
    """The headline capability: cut a runaway response off while it is running."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.upstream = LongStreamUpstream()
        self.store = Store(":memory:")
        self.loop.run_until_complete(self.upstream.start())

    def tearDown(self) -> None:
        self.loop.run_until_complete(self.upstream.stop())
        self.store.close()
        self.loop.close()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def _run(self, **config_kw):
        gateway = Gateway(
            self.store,
            GatewayConfig(
                host="127.0.0.1",
                port=0,
                write_behind=False,
                loop_detection=False,
                routes=[
                    Route(
                        prefix="/v1",
                        upstream=self.upstream.base_url(),
                        provider="openai",
                        strip_prefix="/v1",
                    )
                ],
                **config_kw,
            ),
        )
        port = self._free_port()

        async def scenario():
            server = await asyncio.start_server(gateway.handle, "127.0.0.1", port)
            started = time.perf_counter()
            status, _headers, body = await http_request(
                port, "POST", "/v1/chat/completions",
                json.dumps({"model": "gpt-4o", "stream": True,
                            "messages": [{"role": "user", "content": "go"}]}).encode(),
            )
            elapsed = time.perf_counter() - started
            server.close()
            await server.wait_closed()
            return status, body, elapsed, gateway

        return self.loop.run_until_complete(scenario())

    def test_stream_is_aborted_when_the_token_cap_is_crossed(self):
        status, body, elapsed, gateway = self._run(stream_max_tokens=5000)

        self.assertEqual(status, 200)
        # The client is told why, in a shape it already parses.
        self.assertIn(b"budget_exceeded", body)
        self.assertIn(b"stream token cap", body)
        self.assertIn(b"[DONE]", body)

        # It stopped early rather than forwarding all 300 frames (1.2M chars).
        self.assertLess(len(body), 400_000, "stream was not truncated")
        self.assertEqual(gateway.counters["aborted"], 1)

        row = self.store.query("SELECT error, streamed FROM requests")[0]
        self.assertEqual(row["streamed"], 1)
        self.assertIn("stream token cap", row["error"])

    def test_stream_is_aborted_when_the_cost_cap_is_crossed(self):
        _status, body, _elapsed, gateway = self._run(stream_max_cost_usd=0.05)
        self.assertIn(b"budget_exceeded", body)
        self.assertEqual(gateway.counters["aborted"], 1)

    def test_no_cap_configured_forwards_the_whole_stream(self):
        _status, body, _elapsed, gateway = self._run()
        self.assertNotIn(b"budget_exceeded", body)
        self.assertIn(b"[DONE]", body)
        self.assertEqual(gateway.counters["aborted"], 0)
        # All 300 frames forwarded.
        self.assertGreater(body.count(b"data: "), 300)

    def test_abort_is_recorded_so_the_report_can_see_it(self):
        self._run(stream_max_tokens=2000)
        row = self.store.query("SELECT error FROM requests")[0]
        self.assertIn("stream token cap", row["error"])


class TestAnomaliesEndpoint(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.upstream = MockUpstream()
        self.store = Store(":memory:")
        self.loop.run_until_complete(self.upstream.start())

    def tearDown(self) -> None:
        self.loop.run_until_complete(self.upstream.stop())
        self.store.close()
        self.loop.close()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def test_anomalies_endpoint_returns_thresholds_and_findings(self):
        # Seed a loop directly into the store.
        for _ in range(10):
            self.store.insert(
                RequestRecord(
                    provider="openai", model="gpt-4o", input_tokens=100_000,
                    output_tokens=100, cost_usd=0.5, api_key_id="prod",
                    project="web", ts=_recent(),
                )
            )

        gateway = Gateway(
            self.store,
            GatewayConfig(
                host="127.0.0.1", port=0, write_behind=False,
                routes=[Route(prefix="/v1", upstream=self.upstream.base_url(),
                              provider="openai", strip_prefix="/v1")],
            ),
        )
        port = self._free_port()

        async def scenario():
            server = await asyncio.start_server(gateway.handle, "127.0.0.1", port)
            status, _headers, body = await http_request(port, "GET", "/-/anomalies")
            server.close()
            await server.wait_closed()
            return status, json.loads(body)

        status, payload = self.loop.run_until_complete(scenario())
        self.assertEqual(status, 200)
        self.assertIn("detected", payload)
        self.assertIn("thresholds", payload)
        kinds = {d["kind"] for d in payload["detected"]}
        self.assertIn("io_ratio", kinds)
        self.assertTrue(all(d["remedy"] for d in payload["detected"]))


if __name__ == "__main__":
    unittest.main()
