"""Response parsing tests.

The gateway must read usage from real provider payload shapes, including
streamed ones, and must never invent a number when the payload is unusable.
"""

from __future__ import annotations

import json
import unittest

from llmguard.parsers import (
    extract_end_user,
    extract_project,
    is_streaming_response,
    iter_sse_events,
    parse_buffered,
    parse_streamed,
)


class TestBufferedParsing(unittest.TestCase):
    def test_openai_chat_completion(self):
        body = json.dumps(
            {
                "model": "gpt-4o-2024-08-06",
                "usage": {
                    "prompt_tokens": 1200,
                    "completion_tokens": 300,
                    "total_tokens": 1500,
                },
            }
        ).encode()
        model, usage = parse_buffered("openai", body)
        self.assertEqual(model, "gpt-4o-2024-08-06")
        self.assertEqual(usage.input, 1200)
        self.assertEqual(usage.output, 300)
        self.assertEqual(usage.cached_input, 0)

    def test_openai_cached_tokens_are_split_out_of_prompt_tokens(self):
        """prompt_tokens is inclusive; the cached part must not be double billed."""
        body = json.dumps(
            {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 1000,
                    "completion_tokens": 100,
                    "prompt_tokens_details": {"cached_tokens": 800},
                },
            }
        ).encode()
        _, usage = parse_buffered("openai", body)
        self.assertEqual(usage.input, 200)       # 1000 - 800
        self.assertEqual(usage.cached_input, 800)
        self.assertEqual(usage.output, 100)

    def test_cached_never_exceeds_prompt(self):
        body = json.dumps(
            {
                "model": "gpt-4o",
                "usage": {
                    "prompt_tokens": 10,
                    "completion_tokens": 1,
                    "prompt_tokens_details": {"cached_tokens": 999},
                },
            }
        ).encode()
        _, usage = parse_buffered("openai", body)
        self.assertEqual(usage.cached_input, 10)
        self.assertEqual(usage.input, 0)

    def test_anthropic_messages(self):
        body = json.dumps(
            {
                "model": "claude-3-5-sonnet-20241022",
                "usage": {
                    "input_tokens": 500,
                    "output_tokens": 250,
                    "cache_read_input_tokens": 4000,
                    "cache_creation_input_tokens": 900,
                },
            }
        ).encode()
        model, usage = parse_buffered("anthropic", body)
        self.assertEqual(model, "claude-3-5-sonnet-20241022")
        self.assertEqual(usage.input, 500)
        self.assertEqual(usage.cached_input, 4000)
        self.assertEqual(usage.cache_write, 900)
        self.assertEqual(usage.output, 250)

    def test_responses_api_shape(self):
        body = json.dumps(
            {"model": "gpt-4.1", "usage": {"input_tokens": 700, "output_tokens": 90}}
        ).encode()
        _, usage = parse_buffered("openai", body)
        self.assertEqual(usage.input, 700)
        self.assertEqual(usage.output, 90)

    def test_malformed_json_yields_empty_usage(self):
        _, usage = parse_buffered("openai", b"{not json")
        self.assertEqual(usage.total, 0)

    def test_empty_body_yields_empty_usage(self):
        _, usage = parse_buffered("openai", b"")
        self.assertEqual(usage.total, 0)

    def test_error_payload_has_no_usage(self):
        body = json.dumps(
            {"error": {"message": "rate limited", "type": "rate_limit_error"}}
        ).encode()
        _, usage = parse_buffered("openai", body)
        self.assertEqual(usage.total, 0)


class TestStreamParsing(unittest.TestCase):
    def _sse(self, *objs) -> str:
        return "".join(f"data: {json.dumps(o)}\n\n" for o in objs)

    def test_openai_stream_with_include_usage(self):
        text = self._sse(
            {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "hi"}}]},
            {
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {
                    "prompt_tokens": 900,
                    "completion_tokens": 120,
                    "prompt_tokens_details": {"cached_tokens": 400},
                },
            },
        )
        model, usage = parse_streamed("openai", text)
        self.assertEqual(model, "gpt-4o-mini")
        self.assertEqual(usage.input, 500)      # 900 - 400
        self.assertEqual(usage.cached_input, 400)
        self.assertEqual(usage.output, 120)

    def test_anthropic_stream_merges_message_start_and_delta(self):
        text = self._sse(
            {
                "type": "message_start",
                "message": {
                    "model": "claude-3-5-sonnet-20241022",
                    "usage": {
                        "input_tokens": 300,
                        "output_tokens": 1,
                        "cache_read_input_tokens": 1500,
                        "cache_creation_input_tokens": 200,
                    },
                },
            },
            {"type": "content_block_delta", "delta": {"text": "hello"}},
            {"type": "message_delta", "usage": {"output_tokens": 777}},
        )
        model, usage = parse_streamed("anthropic", text)
        self.assertEqual(model, "claude-3-5-sonnet-20241022")
        self.assertEqual(usage.input, 300)
        self.assertEqual(usage.output, 777)
        self.assertEqual(usage.cached_input, 1500)
        self.assertEqual(usage.cache_write, 200)

    def test_stream_without_usage_reports_zero(self):
        text = self._sse({"model": "gpt-4o", "choices": [{"delta": {"content": "x"}}]})
        _, usage = parse_streamed("openai", text)
        self.assertEqual(usage.total, 0)

    def test_done_sentinel_is_ignored(self):
        text = "data: {\"model\":\"gpt-4o\",\"choices\":[]}\n\ndata: [DONE]\n\n"
        _, usage = parse_streamed("openai", text)
        self.assertEqual(usage.total, 0)

    def test_broken_frame_does_not_abort_the_stream(self):
        text = (
            "data: {broken\n\n"
            'data: {"model":"gpt-4o","usage":{"prompt_tokens":10,"completion_tokens":5}}\n\n'
        )
        model, usage = parse_streamed("openai", text)
        self.assertEqual(model, "gpt-4o")
        self.assertEqual(usage.input, 10)
        self.assertEqual(usage.output, 5)

    def test_multiline_data_block(self):
        text = 'data: {"model":"gpt-4o",\ndata: "usage":{"prompt_tokens":7,"completion_tokens":3}}\n\n'
        _, usage = parse_streamed("openai", text)
        self.assertEqual(usage.input, 7)


class TestSseIterator(unittest.TestCase):
    def test_yields_only_dicts(self):
        text = 'data: {"a":1}\n\ndata: [1,2]\n\ndata: {"b":2}\n\n'
        got = list(iter_sse_events(text))
        self.assertEqual(got, [{"a": 1}, {"b": 2}])


class TestHeaderHelpers(unittest.TestCase):
    def test_streaming_detection(self):
        self.assertTrue(is_streaming_response({"content-type": "text/event-stream"}))
        self.assertTrue(
            is_streaming_response({"Content-Type": "text/event-stream; charset=utf-8"})
        )
        self.assertFalse(is_streaming_response({"content-type": "application/json"}))

    def test_end_user_from_header_wins(self):
        self.assertEqual(
            extract_end_user({"x-end-user": "u_1"}, b'{"user":"u_2"}'), "u_1"
        )

    def test_end_user_from_body_field(self):
        self.assertEqual(extract_end_user({}, b'{"user":"u_42"}'), "u_42")

    def test_end_user_from_metadata_dict(self):
        self.assertEqual(
            extract_end_user({}, b'{"metadata":{"customer_id":"cust_9"}}'), "cust_9"
        )

    def test_no_end_user(self):
        self.assertEqual(extract_end_user({}, b'{"model":"gpt-4o"}'), "")

    def test_project_header(self):
        self.assertEqual(extract_project({"x-project": "checkout"}), "checkout")
        self.assertEqual(extract_project({}), "")


if __name__ == "__main__":
    unittest.main()
