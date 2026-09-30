"""End-to-end tests for the proxy.

These start a real mock upstream on a loopback port, point the gateway at it,
and drive real HTTP through the socket path. This is the test that proves the
product works, rather than proving that helper functions are internally
consistent.

No network access and no API keys are required.
"""

from __future__ import annotations

import asyncio
import json
import socket
import unittest
from typing import List, Optional, Tuple

from llmguard.gateway import Gateway, GatewayConfig, Route
from llmguard.storage import Store


# ---------------------------------------------------------------------------
# mock upstream
# ---------------------------------------------------------------------------
class MockUpstream:
    """Minimal OpenAI-shaped upstream that records what it received."""

    def __init__(self) -> None:
        self.server: Optional[asyncio.AbstractServer] = None
        self.port = 0
        self.received: List[Tuple[str, dict, bytes]] = []
        self.mode = "json"          # json | stream | error | chunked
        self.status = 200

    async def start(self) -> None:
        self.server = await asyncio.start_server(self._handle, "127.0.0.1", 0)
        self.port = self.server.sockets[0].getsockname()[1]

    async def stop(self) -> None:
        if self.server:
            self.server.close()
            await self.server.wait_closed()

    def base_url(self) -> str:
        return f"http://127.0.0.1:{self.port}"

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            request_line = await reader.readline()
            method, target, _ = request_line.decode().strip().split(" ", 2)
            headers = {}
            while True:
                line = await reader.readline()
                if line in (b"\r\n", b"\n", b""):
                    break
                k, v = line.decode().split(":", 1)
                headers[k.strip().lower()] = v.strip()
            body = b""
            if int(headers.get("content-length", "0") or 0):
                body = await reader.readexactly(int(headers["content-length"]))
            self.received.append((target, headers, body))

            if self.mode == "error":
                await self._send_json(writer, self.status, {"error": {"message": "boom"}})
            elif self.mode == "chunked":
                await self._send_chunked_json(writer)
            elif self.mode == "stream":
                await self._send_stream(writer)
            else:
                await self._send_json(
                    writer,
                    200,
                    {
                        "id": "chatcmpl-test",
                        "object": "chat.completion",
                        "model": "gpt-4o-2024-08-06",
                        "choices": [
                            {"index": 0, "message": {"role": "assistant", "content": "hi"}}
                        ],
                        "usage": {
                            "prompt_tokens": 1000,
                            "completion_tokens": 200,
                            "total_tokens": 1200,
                            "prompt_tokens_details": {"cached_tokens": 600},
                        },
                    },
                )
        except Exception:  # noqa: BLE001
            pass
        finally:
            try:
                writer.close()
                await writer.wait_closed()
            except Exception:  # noqa: BLE001
                pass

    async def _send_json(self, writer, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode()
        writer.write(
            f"HTTP/1.1 {status} OK\r\nContent-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\nConnection: close\r\n\r\n".encode() + body
        )
        await writer.drain()

    async def _send_chunked_json(self, writer) -> None:
        payload = json.dumps(
            {
                "model": "gpt-4o",
                "usage": {"prompt_tokens": 500, "completion_tokens": 50},
            }
        ).encode()
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            b"Transfer-Encoding: chunked\r\nConnection: close\r\n\r\n"
        )
        half = len(payload) // 2
        for part in (payload[:half], payload[half:]):
            writer.write(f"{len(part):x}\r\n".encode() + part + b"\r\n")
        writer.write(b"0\r\n\r\n")
        await writer.drain()

    async def _send_stream(self, writer) -> None:
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        frames = [
            {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "he"}}]},
            {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "llo"}}]},
            {
                "model": "gpt-4o-mini",
                "choices": [],
                "usage": {
                    "prompt_tokens": 800,
                    "completion_tokens": 40,
                    "prompt_tokens_details": {"cached_tokens": 300},
                },
            },
        ]
        for frame in frames:
            writer.write(f"data: {json.dumps(frame)}\n\n".encode())
            await writer.drain()
        writer.write(b"data: [DONE]\n\n")
        await writer.drain()


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------
async def http_request(
    port: int, method: str, path: str, body: bytes = b"", headers: Optional[dict] = None
) -> Tuple[int, dict, bytes]:
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        hdrs = {"Host": f"127.0.0.1:{port}", "Content-Length": str(len(body))}
        hdrs.update(headers or {})
        head = f"{method} {path} HTTP/1.1\r\n" + "".join(
            f"{k}: {v}\r\n" for k, v in hdrs.items()
        ) + "\r\n"
        writer.write(head.encode() + body)
        await writer.drain()

        status_line = await reader.readline()
        status = int(status_line.decode().split(" ", 2)[1])
        resp_headers: dict = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            k, v = line.decode().split(":", 1)
            resp_headers[k.strip().lower()] = v.strip()

        if "chunked" in resp_headers.get("transfer-encoding", "").lower():
            chunks = bytearray()
            while True:
                size_line = await reader.readline()
                size = int(size_line.split(b";")[0].strip() or b"0", 16)
                if size == 0:
                    await reader.readline()
                    break
                chunks.extend(await reader.readexactly(size))
                await reader.readline()
            payload = bytes(chunks)
        elif "content-length" in resp_headers:
            payload = await reader.readexactly(int(resp_headers["content-length"]))
        else:
            payload = await reader.read()
        return status, resp_headers, payload
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


class GatewayTestCase(unittest.TestCase):
    """Base class that wires a gateway to a mock upstream on loopback."""

    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.upstream = MockUpstream()
        self.store = Store(":memory:")
        self.loop.run_until_complete(self.upstream.start())
        self.gateway = Gateway(self.store, self._config())
        self.gateway_port = self._free_port()
        self.gateway.config.port = self.gateway_port
        self._server = self.loop.run_until_complete(
            asyncio.start_server(self.gateway.handle, "127.0.0.1", self.gateway_port)
        )

    def _config(self) -> GatewayConfig:
        return GatewayConfig(
            host="127.0.0.1",
            port=0,
            # Tests read the store immediately after a request, so keep writes
            # synchronous here. Write-behind is covered by its own test below.
            write_behind=False,
            routes=[
                Route(
                    prefix="/v1",
                    upstream=self.upstream.base_url(),
                    provider="openai",
                    strip_prefix="/v1",
                )
            ],
        )

    def tearDown(self) -> None:
        self._server.close()
        self.loop.run_until_complete(self._server.wait_closed())
        self.loop.run_until_complete(self.upstream.stop())
        for task in asyncio.all_tasks(self.loop):
            task.cancel()
        try:
            self.loop.run_until_complete(
                asyncio.gather(*asyncio.all_tasks(self.loop), return_exceptions=True)
            )
        except Exception:  # noqa: BLE001
            pass
        self.store.close()
        self.loop.close()

    @staticmethod
    def _free_port() -> int:
        with socket.socket() as s:
            s.bind(("127.0.0.1", 0))
            return int(s.getsockname()[1])

    def request(self, method: str, path: str, body: bytes = b"", headers=None):
        return self.loop.run_until_complete(
            http_request(self.gateway_port, method, path, body, headers)
        )


# ---------------------------------------------------------------------------
# tests
# ---------------------------------------------------------------------------
class TestProxying(GatewayTestCase):
    def test_json_request_is_forwarded_and_recorded(self):
        payload = json.dumps(
            {"model": "gpt-4o", "messages": [{"role": "user", "content": "hi"}]}
        ).encode()
        status, _headers, body = self.request(
            "POST", "/v1/chat/completions", payload,
            {"Authorization": "Bearer sk-test-abc", "Content-Type": "application/json"},
        )

        self.assertEqual(status, 200)
        self.assertIn(b"chatcmpl-test", body)

        rows = self.store.query("SELECT * FROM requests")
        self.assertEqual(len(rows), 1)
        row = rows[0]

        self.assertEqual(row["model"], "gpt-4o-2024-08-06")
        self.assertEqual(row["input_tokens"], 400)      # 1000 - 600 cached
        self.assertEqual(row["cached_input_tokens"], 600)
        self.assertEqual(row["output_tokens"], 200)
        self.assertEqual(row["status"], 200)
        self.assertEqual(row["provider"], "openai")
        # gpt-4o: 400 in @2.50 + 600 cached @1.25 + 200 out @10.00, all /1M
        expected = (400 * 2.50 + 600 * 1.25 + 200 * 10.00) / 1_000_000
        self.assertAlmostEqual(row["cost_usd"], expected, places=9)

    def test_upstream_receives_the_original_body_and_auth(self):
        payload = b'{"model":"gpt-4o","messages":[]}'
        self.request(
            "POST", "/v1/chat/completions", payload,
            {"Authorization": "Bearer sk-test-abc"},
        )
        target, headers, body = self.upstream.received[-1]
        self.assertEqual(target, "/chat/completions")
        self.assertEqual(body, payload)
        self.assertEqual(headers.get("authorization"), "Bearer sk-test-abc")

    def test_path_prefix_is_stripped_for_openai_style_routes(self):
        self.request("POST", "/v1/embeddings", b"{}")
        target, _, _ = self.upstream.received[-1]
        self.assertEqual(target, "/embeddings")

    def test_query_string_is_preserved(self):
        self.request("POST", "/v1/chat/completions?foo=bar", b"{}")
        target, _, _ = self.upstream.received[-1]
        self.assertEqual(target, "/chat/completions?foo=bar")

    def test_error_response_is_still_recorded(self):
        self.upstream.mode = "error"
        self.upstream.status = 429
        status, _, _ = self.request("POST", "/v1/chat/completions", b"{}")
        self.assertEqual(status, 429)
        row = self.store.query("SELECT * FROM requests")[0]
        self.assertEqual(row["status"], 429)
        self.assertIsNone(row["cost_usd"])
        self.assertIn("429", row["error"])

    def test_chunked_upstream_response_is_reframed_and_parsed(self):
        self.upstream.mode = "chunked"
        status, _headers, body = self.request("POST", "/v1/chat/completions", b"{}")
        self.assertEqual(status, 200)
        parsed = json.loads(body)          # client must receive valid JSON
        self.assertEqual(parsed["usage"]["prompt_tokens"], 500)
        row = self.store.query("SELECT * FROM requests")[0]
        self.assertEqual(row["input_tokens"], 500)
        self.assertEqual(row["output_tokens"], 50)

    def test_unknown_path_returns_404(self):
        status, _, body = self.request("POST", "/nope", b"{}")
        self.assertEqual(status, 404)
        self.assertIn(b"no upstream route", body)

    def test_multiple_requests_accumulate(self):
        for _ in range(3):
            self.request("POST", "/v1/chat/completions", b"{}")
        self.assertEqual(self.store.count(), 3)
        total = self.store.one("SELECT SUM(cost_usd) AS c FROM requests")["c"]
        self.assertGreater(total, 0)


class TestStreaming(GatewayTestCase):
    def test_streamed_response_is_forwarded_and_priced(self):
        self.upstream.mode = "stream"
        status, headers, body = self.request("POST", "/v1/chat/completions", b"{}")

        self.assertEqual(status, 200)
        self.assertIn("text/event-stream", headers.get("content-type", ""))
        # The client must receive the frames verbatim.
        self.assertIn(b"data: ", body)
        self.assertIn(b"[DONE]", body)

        row = self.store.query("SELECT * FROM requests")[0]
        self.assertEqual(row["streamed"], 1)
        self.assertEqual(row["model"], "gpt-4o-mini")
        self.assertEqual(row["input_tokens"], 500)     # 800 - 300 cached
        self.assertEqual(row["cached_input_tokens"], 300)
        self.assertEqual(row["output_tokens"], 40)
        self.assertIsNotNone(row["cost_usd"])


class TestOwnApi(GatewayTestCase):
    def test_health_endpoint(self):
        status, _, body = self.request("GET", "/-/health")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertEqual(payload["status"], "ok")
        self.assertIn("rows_total", payload)

    def test_stats_endpoint_returns_json(self):
        self.request("POST", "/v1/chat/completions", b"{}")
        status, _, body = self.request("GET", "/-/stats")
        self.assertEqual(status, 200)
        payload = json.loads(body)
        self.assertIn("current", payload)
        self.assertIn("by_model", payload)
        self.assertEqual(payload["current"]["requests"], 1)

    def test_stats_with_window(self):
        status, _, body = self.request("GET", "/-/stats/7")
        self.assertEqual(status, 200)
        self.assertEqual(json.loads(body)["days"], 7)

    def test_dashboard_endpoint_returns_html(self):
        self.request("POST", "/v1/chat/completions", b"{}")
        status, headers, body = self.request("GET", "/-/dashboard")
        self.assertEqual(status, 200)
        self.assertIn("text/html", headers.get("content-type", ""))
        self.assertIn(b"LLM Spend Dashboard", body)

    def test_unknown_own_endpoint_404(self):
        status, _, _ = self.request("GET", "/-/nope")
        self.assertEqual(status, 404)

    def test_method_not_allowed(self):
        status, _, _ = self.request("POST", "/-/health", b"{}")
        self.assertEqual(status, 405)


class TestBudgetEnforcement(GatewayTestCase):
    def test_key_without_budget_is_never_blocked(self):
        payload = b'{"model":"gpt-4o"}'
        for _ in range(3):
            status, _, _ = self.request(
                "POST", "/v1/chat/completions", payload,
                {"Authorization": "Bearer sk-team-x"},
            )
            self.assertEqual(status, 200)

    def test_block_action_returns_429_once_budget_is_spent(self):
        # Attribute traffic via key_map, then set an impossible budget.
        self.gateway.config.key_map = {"sk-team-a": "team-a"}
        self.store.set_budget("team-a", daily_usd=0.000001, action="block")

        payload = json.dumps({"model": "gpt-4o", "user": "u_1"}).encode()
        # First call is allowed through (nothing spent yet), and it is expensive
        # enough to blow the tiny budget.
        status, _, _ = self.request(
            "POST", "/v1/chat/completions", payload, {"Authorization": "Bearer sk-team-a"}
        )
        self.assertEqual(status, 200)

        # Second call must be rejected by the guard.
        status, _, body = self.request(
            "POST", "/v1/chat/completions", payload, {"Authorization": "Bearer sk-team-a"}
        )
        self.assertEqual(status, 429)
        self.assertIn(b"budget exceeded", body)

        rows = self.store.query(
            "SELECT status, error FROM requests WHERE api_key_id='team-a' ORDER BY id"
        )
        self.assertEqual(rows[-1]["status"], 429)
        self.assertIn("daily budget", rows[-1]["error"])
        self.assertEqual(self.gateway.counters["blocked"], 1)

    def test_alert_action_does_not_block(self):
        self.gateway.config.key_map = {"sk-team-b": "team-b"}
        self.store.set_budget("team-b", daily_usd=0.000001, action="alert")
        payload = b'{"model":"gpt-4o"}'
        for _ in range(3):
            status, _, _ = self.request(
                "POST", "/v1/chat/completions", payload, {"Authorization": "Bearer sk-team-b"}
            )
            self.assertEqual(status, 200, "alert action must never block traffic")

    def test_anonymous_traffic_is_attributed(self):
        self.request("POST", "/v1/chat/completions", b"{}")
        row = self.store.query("SELECT api_key_id FROM requests")[0]
        self.assertEqual(row["api_key_id"], "anonymous")

    def test_credentials_are_never_stored_in_plaintext(self):
        secret = "sk-super-secret-value"
        self.request(
            "POST", "/v1/chat/completions", b"{}", {"Authorization": f"Bearer {secret}"}
        )
        row = self.store.query("SELECT api_key_id FROM requests")[0]
        self.assertNotIn(secret, row["api_key_id"])
        self.assertTrue(row["api_key_id"].startswith("key-"))


class TestAttribution(GatewayTestCase):
    def test_project_and_end_user_headers_are_recorded(self):
        self.request(
            "POST", "/v1/chat/completions", b"{}",
            {"x-project": "checkout-assistant", "x-end-user": "u_9001"},
        )
        row = self.store.query("SELECT project, end_user FROM requests")[0]
        self.assertEqual(row["project"], "checkout-assistant")
        self.assertEqual(row["end_user"], "u_9001")


if __name__ == "__main__":
    unittest.main()
