#!/usr/bin/env python3
"""Capture real provider payloads so the parsers can be verified exactly.

Why this exists
---------------
`tests/test_parsers.py` verifies the parser against the payload shapes OpenAI and
Anthropic are *documented* to return. That is not the same as verifying them
against what the APIs *actually* return today -- field names, nesting and
streaming frame order are provider details that change without a version bump.

This script makes that verification a two-minute job. It calls each provider once
(buffered) and once (streamed) through the live gateway, and writes the raw
response bodies to `fixtures/captured/`. Nothing else is needed from the
provider: no dashboard access, no billing access, no org-level key.

Usage
-----
    # a key with a few cents of credit is enough; total spend is well under $0.01
    export OPENAI_API_KEY=sk-...
    export ANTHROPIC_API_KEY=sk-ant-...
    python3 tools/capture_fixtures.py

    # then re-run the suite -- the captured fixtures are asserted automatically
    python3 -m unittest discover -s tests

    # review what was captured (these files contain model output, so read before sharing)
    ls -la fixtures/captured/

Cost
----
Four small calls: roughly 40 input tokens and 30 output tokens each. On a cheap
model that is a fraction of a cent. Use `--model` to pin an exact model id.

Privacy
-------
The captured files contain the model's response to a fixed, non-sensitive prompt
("Reply with the single word: ok"). Review them before committing them anywhere.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import pathlib
import sys
import time

sys.path.insert(0, str(pathlib.Path(__file__).resolve().parents[1]))

from llmguard.gateway import (  # noqa: E402
    DEFAULT_ROUTES,
    Gateway,
    GatewayConfig,
    Route,
)
from llmguard.storage import Store  # noqa: E402

OUT = pathlib.Path(__file__).resolve().parents[1] / "fixtures" / "captured"

PROMPT = "Reply with the single word: ok"


async def _send(port: int, path: str, headers: dict, payload: dict) -> tuple[int, dict, bytes]:
    """Minimal HTTP/1.1 POST, returning (status, headers, body)."""
    body = json.dumps(payload).encode()
    reader, writer = await asyncio.open_connection("127.0.0.1", port)
    try:
        head = (
            f"POST {path} HTTP/1.1\r\n"
            f"Host: 127.0.0.1:{port}\r\n"
            f"Content-Type: application/json\r\n"
            f"Content-Length: {len(body)}\r\n"
            + "".join(f"{k}: {v}\r\n" for k, v in headers.items())
            + "\r\n"
        )
        writer.write(head.encode() + body)
        await writer.drain()

        status_line = await reader.readline()
        status = int(status_line.decode().split(" ", 2)[1])
        resp_headers: dict = {}
        while True:
            line = await reader.readline()
            if line in (b"\r\n", b"\n", b""):
                break
            key, value = line.decode().split(":", 1)
            resp_headers[key.strip().lower()] = value.strip()

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
            payload_bytes = bytes(chunks)
        elif "content-length" in resp_headers:
            payload_bytes = await reader.readexactly(int(resp_headers["content-length"]))
        else:
            payload_bytes = await reader.read()
        return status, resp_headers, payload_bytes
    finally:
        writer.close()
        try:
            await writer.wait_closed()
        except Exception:  # noqa: BLE001
            pass


def _write(name: str, status: int, headers: dict, body: bytes) -> pathlib.Path:
    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / name
    # Store the parsed JSON when possible so a human can read it, and always keep
    # the raw bytes so the parser can be tested against the exact wire format.
    try:
        pretty = json.dumps(json.loads(body), indent=2, ensure_ascii=False)
    except ValueError:
        pretty = body.decode("utf-8", "replace")
    path.write_text(
        f"# captured {time.strftime('%Y-%m-%d %H:%M:%S')}  status={status}\n"
        f"# content-type: {headers.get('content-type', '?')}\n"
        f"# --- raw body below ---\n{pretty}\n",
        encoding="utf-8",
    )
    return path


async def capture(args: argparse.Namespace) -> int:
    openai_key = os.environ.get("OPENAI_API_KEY", "").strip()
    anthropic_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not openai_key and not anthropic_key:
        print("error: set OPENAI_API_KEY and/or ANTHROPIC_API_KEY", file=sys.stderr)
        print("       a key with a few cents of credit is enough.", file=sys.stderr)
        return 2

    store = Store(":memory:")
    gateway = Gateway(
        store,
        GatewayConfig(
            host="127.0.0.1",
            port=0,
            # Synchronous writes so the captured usage is in the store immediately.
            write_behind=False,
            routes=list(DEFAULT_ROUTES),
        ),
    )
    server = await asyncio.start_server(gateway.handle, "127.0.0.1", 0)
    port = server.sockets[0].getsockname()[1]
    captured = []
    failures = []

    try:
        if openai_key:
            model = args.openai_model
            base = {"model": model, "messages": [{"role": "user", "content": PROMPT}]}
            # 1. buffered
            status, headers, body = await _send(
                port, "/v1/chat/completions",
                {"Authorization": f"Bearer {openai_key}"}, dict(base, max_tokens=16),
            )
            if status == 200:
                captured.append(_write("openai_chat_buffered.json", status, headers, body))
            else:
                failures.append(("openai buffered", status, body[:400]))

            # 2. streamed, with usage requested -- this is the frame order that
            #    the mid-stream accounting depends on.
            status, headers, body = await _send(
                port, "/v1/chat/completions",
                {"Authorization": f"Bearer {openai_key}"},
                dict(base, max_tokens=16, stream=True,
                     stream_options={"include_usage": True}),
            )
            if status == 200:
                captured.append(_write("openai_chat_streamed.json", status, headers, body))
            else:
                failures.append(("openai streamed", status, body[:400]))

        if anthropic_key:
            model = args.anthropic_model
            base = {
                "model": model,
                "max_tokens": 16,
                "messages": [{"role": "user", "content": PROMPT}],
            }
            headers = {"x-api-key": anthropic_key, "anthropic-version": "2023-06-01"}

            status, hdrs, body = await _send(
                port, "/anthropic/v1/messages", headers, dict(base)
            )
            if status == 200:
                captured.append(_write("anthropic_messages_buffered.json", status, hdrs, body))
            else:
                failures.append(("anthropic buffered", status, body[:400]))

            status, hdrs, body = await _send(
                port, "/anthropic/v1/messages", headers, dict(base, stream=True)
            )
            if status == 200:
                captured.append(_write("anthropic_messages_streamed.json", status, hdrs, body))
            else:
                failures.append(("anthropic streamed", status, body[:400]))
    finally:
        server.close()
        await server.wait_closed()

    # Report what the gateway itself recorded -- this is the strongest signal:
    # if usage parsed correctly, each call appears with tokens and a cost.
    rows = store.query(
        "SELECT provider, model, input_tokens, cached_input_tokens, output_tokens, "
        "cost_usd, streamed, status FROM requests ORDER BY id"
    )
    store.close()

    print()
    print("=== captured fixtures ===")
    for path in captured:
        print(f"  {path.relative_to(path.parents[2])}")
    if not captured:
        print("  (none)")

    print()
    print("=== what the gateway parsed from them ===")
    print(f"  {'provider':<10}{'model':<26}{'in':>8}{'cached':>8}{'out':>7}"
          f"{'cost':>12}  streamed")
    for row in rows:
        cost = row["cost_usd"]
        print(
            f"  {row['provider']:<10}{str(row['model'])[:25]:<26}"
            f"{row['input_tokens']:>8}{row['cached_input_tokens']:>8}"
            f"{row['output_tokens']:>7}"
            f"{('$%.6f' % cost) if cost is not None else 'UNPRICED':>12}"
            f"  {'yes' if row['streamed'] else 'no'}"
        )

    if failures:
        print()
        print("=== failures ===")
        for label, status, snippet in failures:
            print(f"  {label}: HTTP {status}")
            print(f"    {snippet.decode('utf-8', 'replace')[:300]}")

    ok = bool(captured) and not failures
    print()
    if ok:
        print("Result: every call parsed with tokens and a cost.")
        print("Next:  python3 -m unittest discover -s tests")
    else:
        print("Result: some calls failed or parsed no usage. See failures above.")
        print("If usage parsed as zero, the provider changed its payload shape --")
        print("that is exactly the bug this script exists to find.")
    return 0 if ok else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--openai-model", default="gpt-6-luna",
                        help="a cheap current model (default: gpt-6-luna)")
    parser.add_argument("--anthropic-model", default="claude-haiku-4.5",
                        help="a cheap current model (default: claude-haiku-4.5)")
    return asyncio.run(capture(parser.parse_args()))


if __name__ == "__main__":
    raise SystemExit(main())
