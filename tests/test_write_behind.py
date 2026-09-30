"""Tests for buffered accounting writes.

Any design that buffers spend data must prove it does not lose it. These tests
cover the batch flusher directly, plus a gateway configured with write-behind
enabled going through its shutdown drain.
"""

from __future__ import annotations

import asyncio
import json
import socket
import unittest

from llmguard.gateway import Gateway, GatewayConfig, Route
from llmguard.storage import RequestRecord, Store

from test_gateway_e2e import MockUpstream, http_request


class TestWriteBehind(unittest.TestCase):
    def setUp(self) -> None:
        self.loop = asyncio.new_event_loop()
        self.store = Store(":memory:")

    def tearDown(self) -> None:
        # Cancel any flusher a test left running before the loop dies, otherwise
        # the orphaned task is reported at GC time on newer interpreters.
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

    def test_records_are_flushed_by_the_timer(self):
        from llmguard.gateway import _WriteBehind

        async def scenario():
            wb = _WriteBehind(self.store, interval=0.05, max_batch=100)
            wb.start()
            for i in range(5):
                wb.submit(RequestRecord(provider="openai", model="gpt-4o", input_tokens=i))
            self.assertEqual(self.store.count(), 0, "should still be buffered")
            await asyncio.sleep(0.25)          # let the timer fire
            flushed = self.store.count()
            await wb.drain()
            return flushed

        flushed = self.loop.run_until_complete(scenario())
        self.assertEqual(flushed, 5)
        self.assertEqual(self.store.count(), 5)

    def test_drain_persists_after_a_single_submit(self):
        from llmguard.gateway import _WriteBehind

        async def scenario():
            wb = _WriteBehind(self.store, interval=30.0, max_batch=100)
            wb.start()
            wb.submit(RequestRecord(provider="openai", model="gpt-4o", input_tokens=42))
            await asyncio.sleep(0.05)
            await wb.drain()                   # must not wait for the timer

        self.loop.run_until_complete(scenario())
        self.assertEqual(self.store.count(), 1)

    def test_batch_size_triggers_an_immediate_flush(self):
        from llmguard.gateway import _WriteBehind

        async def scenario():
            wb = _WriteBehind(self.store, interval=30.0, max_batch=10)
            wb.start()
            for i in range(10):
                wb.submit(RequestRecord(provider="openai", model="gpt-4o", input_tokens=i))
            await asyncio.sleep(0.15)
            return self.store.count()

        # max_batch is reached, so a flush happens without waiting 30s.
        self.assertEqual(self.loop.run_until_complete(scenario()), 10)

    def test_drain_without_start_still_flushes(self):
        from llmguard.gateway import _WriteBehind

        async def scenario():
            wb = _WriteBehind(self.store, interval=30.0, max_batch=100)
            for i in range(3):
                wb.submit(RequestRecord(provider="openai", model="gpt-4o", input_tokens=i))
            await wb.drain()

        self.loop.run_until_complete(scenario())
        self.assertEqual(self.store.count(), 3)


class TestGatewayDrain(unittest.TestCase):
    """End-to-end: write-behind enabled, then drain, then assert."""

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

    def test_traffic_survives_a_drain(self):
        gateway = Gateway(
            self.store,
            GatewayConfig(
                host="127.0.0.1",
                port=0,
                write_behind=True,
                write_behind_interval=30.0,     # never fires on its own
                write_behind_max_batch=10_000,
                routes=[
                    Route(
                        prefix="/v1",
                        upstream=self.upstream.base_url(),
                        provider="openai",
                        strip_prefix="/v1",
                    )
                ],
            ),
        )
        port = self._free_port()

        async def scenario():
            server = await asyncio.start_server(gateway.handle, "127.0.0.1", port)
            gateway.writer.start()
            for _ in range(7):
                await http_request(port, "POST", "/v1/chat/completions", b"{}")
            buffered = self.store.count()
            await gateway.drain()
            server.close()
            await server.wait_closed()
            return buffered

        buffered = self.loop.run_until_complete(scenario())
        self.assertEqual(buffered, 0, "rows should still be in the buffer pre-drain")
        self.assertEqual(self.store.count(), 7, "drain must persist every row")

        # And the persisted rows must be priced, not just present.
        priced = self.store.one(
            "SELECT COUNT(*) n FROM requests WHERE cost_usd IS NOT NULL"
        )["n"]
        self.assertEqual(priced, 7)


if __name__ == "__main__":
    unittest.main()
