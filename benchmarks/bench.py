"""Throughput and overhead benchmark.

Measures what the gateway actually costs you: added latency per proxied
request, and how many requests per second it can sustain. Run it on the machine
you intend to deploy on; the numbers move with disk and CPU.

    python benchmarks/bench.py            # default: 2000 requests, concurrency 32
    python benchmarks/bench.py --requests 5000 --concurrency 64

Method: a local mock upstream (so upstream latency is ~0), then the same
workload with and without the gateway in the path. The difference is the
gateway's cost. This is the only honest way to state overhead.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import platform
import statistics
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from llmguard.gateway import Gateway, GatewayConfig, Route  # noqa: E402
from llmguard.storage import Store  # noqa: E402

RESPONSE = {
    "id": "chatcmpl-bench",
    "object": "chat.completion",
    "model": "gpt-4o-mini",
    "choices": [{"index": 0, "message": {"role": "assistant", "content": "ok"}}],
    "usage": {
        "prompt_tokens": 800,
        "completion_tokens": 120,
        "total_tokens": 920,
        "prompt_tokens_details": {"cached_tokens": 300},
    },
}
RESPONSE_BYTES = json.dumps(RESPONSE).encode()
REQUEST_BODY = json.dumps(
    {"model": "gpt-4o-mini", "messages": [{"role": "user", "content": "hello"}]}
).encode()


async def mock_upstream(reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
    try:
        await reader.readline()
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        head = (
            b"HTTP/1.1 200 OK\r\nContent-Type: application/json\r\n"
            + f"Content-Length: {len(RESPONSE_BYTES)}\r\n".encode()
            + b"Connection: close\r\n\r\n"
        )
        writer.write(head + RESPONSE_BYTES)
        await writer.drain()
    except Exception:  # noqa: BLE001
        pass
    finally:
        try:
            writer.close()
        except Exception:  # noqa: BLE001
            pass


STREAM_FRAMES = [
    {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "hel"}}]},
    {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "lo "}}]},
    {"model": "gpt-4o-mini", "choices": [{"delta": {"content": "world"}}]},
    {
        "model": "gpt-4o-mini",
        "choices": [],
        "usage": {
            "prompt_tokens": 800,
            "completion_tokens": 120,
            "prompt_tokens_details": {"cached_tokens": 300},
        },
    },
]


async def mock_upstream_stream(
    reader: asyncio.StreamReader, writer: asyncio.StreamWriter
) -> None:
    """Mock OpenAI streaming endpoint: text/event-stream, no Content-Length."""
    try:
        await reader.readline()
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
        writer.write(
            b"HTTP/1.1 200 OK\r\nContent-Type: text/event-stream\r\n"
            b"Cache-Control: no-cache\r\nConnection: close\r\n\r\n"
        )
        await writer.drain()
        for frame in STREAM_FRAMES:
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


async def one_request(port: int, path: str = "/v1/chat/completions") -> float:
    """Send one request, return elapsed milliseconds."""
    started = time.perf_counter()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        writer.write(
            f"POST {path} HTTP/1.1\r\nHost: 127.0.0.1\r\n"
            f"Content-Type: application/json\r\n"
            f"Authorization: Bearer sk-bench\n"
            f"Content-Length: {len(REQUEST_BODY)}\r\n\r\n".encode()
            + REQUEST_BODY
        )
        await writer.drain()
        await reader.read()
    finally:
        writer.close()
    return (time.perf_counter() - started) * 1000.0


async def run_workload(port: int, total: int, concurrency: int) -> list[float]:
    latencies: list[float] = []
    semaphore = asyncio.Semaphore(concurrency)

    async def worker() -> None:
        async with semaphore:
            latencies.append(await one_request(port))

    started = time.perf_counter()
    await asyncio.gather(*(worker() for _ in range(total)))
    elapsed = time.perf_counter() - started
    return latencies


def summarise(name: str, latencies: list[float], wall_seconds: float) -> dict:
    ordered = sorted(latencies)
    n = len(ordered)

    def pct(p: float) -> float:
        idx = min(int(p / 100 * n), n - 1)
        return ordered[idx]

    return {
        "name": name,
        "requests": n,
        "wall_seconds": round(wall_seconds, 3),
        "rps": round(n / wall_seconds, 1) if wall_seconds else 0.0,
        "mean_ms": round(statistics.fmean(ordered), 3),
        "p50_ms": round(pct(50), 3),
        "p95_ms": round(pct(95), 3),
        "p99_ms": round(pct(99), 3),
    }


async def main_async(args: argparse.Namespace) -> int:
    upstream = await asyncio.start_server(mock_upstream, "127.0.0.1", 0)
    up_port = upstream.sockets[0].getsockname()[1]

    # --- baseline: talk to the mock upstream directly ---
    started = time.perf_counter()
    lat_direct = await run_workload(up_port, args.requests, args.concurrency)
    direct = summarise("direct to upstream (no gateway)", lat_direct, time.perf_counter() - started)

    # --- through the gateway ---
    db_path = Path(args.db)
    if db_path.exists():
        db_path.unlink()
    store = Store(str(db_path))
    gateway = Gateway(
        store,
        GatewayConfig(
            host="127.0.0.1",
            port=0,
            routes=[
                Route(
                    prefix="/v1",
                    upstream=f"http://127.0.0.1:{up_port}",
                    provider="openai",
                    strip_prefix="/v1",
                )
            ],
        ),
    )
    gw_server = await asyncio.start_server(gateway.handle, "127.0.0.1", 0)
    gw_port = gw_server.sockets[0].getsockname()[1]

    started = time.perf_counter()
    lat_gw = await run_workload(gw_port, args.requests, args.concurrency)
    gated = summarise("through llm-guard", lat_gw, time.perf_counter() - started)

    # Drain before counting: accounting is batched, so the rows land on a timer.
    await gateway.drain()
    rows = store.count()

    # --- streaming scenario: usage arrives incrementally ---
    upstream_streaming = await asyncio.start_server(
        mock_upstream_stream, "127.0.0.1", 0
    )
    up_stream_port = upstream_streaming.sockets[0].getsockname()[1]
    gateway.config.routes = [
        Route(
            prefix="/v1",
            upstream=f"http://127.0.0.1:{up_stream_port}",
            provider="openai",
            strip_prefix="/v1",
        )
    ]
    stream_count = max(args.requests // 4, 100)
    started = time.perf_counter()
    lat_stream = await run_workload(gw_port, stream_count, args.concurrency)
    streaming = summarise("through llm-guard (SSE stream)", lat_stream, time.perf_counter() - started)

    await gateway.drain()
    rows = store.count()
    streamed_rows = store.one("SELECT COUNT(*) n FROM requests WHERE streamed=1")["n"]
    priced_rows = store.one("SELECT COUNT(*) n FROM requests WHERE cost_usd IS NOT NULL")["n"]

    store.close()
    upstream_streaming.close()
    await upstream_streaming.wait_closed()

    # --- report ---
    overhead = gated["mean_ms"] - direct["mean_ms"]
    overhead_pct = (overhead / direct["mean_ms"] * 100) if direct["mean_ms"] else 0.0
    rps_ratio = (gated["rps"] / direct["rps"]) if direct["rps"] else 0.0

    print()
    print("=" * 74)
    print("  llm-guard benchmark")
    print("=" * 74)
    print(f"  python           {platform.python_version()} on {platform.system()} {platform.machine()}")
    print(f"  requests         {args.requests:,} per scenario, concurrency {args.concurrency}")
    print(f"  upstream         local mock (JSON, ~{len(RESPONSE_BYTES)} byte response)")
    print(f"  rows recorded    {rows:,} ({streamed_rows:,} streamed, {priced_rows:,} priced)")
    print()
    header = f"  {'scenario':<34}{'rps':>9}{'mean':>9}{'p50':>9}{'p95':>9}{'p99':>9}"
    print(header)
    print("  " + "-" * 70)
    for s in (direct, gated, streaming):
        print(
            f"  {s['name']:<34}{s['rps']:>9,.1f}{s['mean_ms']:>8.3f}m"
            f"{s['p50_ms']:>8.3f}m{s['p95_ms']:>8.3f}m{s['p99_ms']:>8.3f}m"
        )
    print("  " + "-" * 70)
    print(f"  added latency    {overhead:+.3f} ms per request ({overhead_pct:+.1f}%)")
    print(f"  throughput       {rps_ratio * 100:.1f}% of direct")
    print()
    print("  Context: a single LLM API call takes 400-4000 ms. This overhead is")
    print("  measured against a zero-latency local upstream, so it is the worst")
    print("  case ratio. In production it is in the noise.")
    print()

    gw_server.close()
    await gw_server.wait_closed()
    upstream.close()
    await upstream.wait_closed()

    if args.json:
        Path(args.json).write_text(
            json.dumps(
                {
                    "direct": direct,
                    "through_gateway": gated,
                    "overhead_ms": round(overhead, 4),
                    "overhead_pct": round(overhead_pct, 3),
                    "requests": args.requests,
                    "concurrency": args.concurrency,
                    "rows_recorded": rows,
                    "streamed_rows": streamed_rows,
                    "priced_rows": priced_rows,
                    "streaming": streaming,
                    "python": platform.python_version(),
                },
                indent=2,
            )
        )
        print(f"  wrote {args.json}")
    return 0


def main() -> int:
    parser = argparse.ArgumentParser(description="Benchmark llm-guard proxy overhead")
    parser.add_argument("--requests", type=int, default=2000)
    parser.add_argument("--concurrency", type=int, default=32)
    parser.add_argument("--db", default="/tmp/llmguard-bench.db")
    parser.add_argument("--json", default="")
    args = parser.parse_args()
    return asyncio.run(main_async(args))


if __name__ == "__main__":
    raise SystemExit(main())
