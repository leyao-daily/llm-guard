"""Command line interface.

    llm-guard seed          load a realistic demo dataset (no API key needed)
    llm-guard report        show where the money went and what to change
    llm-guard serve         run the proxy in front of your LLM calls
    llm-guard dashboard     write a standalone HTML dashboard
    llm-guard budget        set a per-key daily/monthly limit
    llm-guard models        list the built-in pricing table
    llm-guard cost          price a single call from the shell
"""

from __future__ import annotations

import argparse
import asyncio
import json
import logging
import os
import sys
import urllib.error
from pathlib import Path
from typing import List, Optional

from .analytics import build_report
from .pricing import TokenUsage, compute_cost, get_price, known_models, resolve_model
from .report import (
    money,
    render_csv,
    render_json,
    render_terminal,
    report_to_dict,
)
from .storage import Store

DEFAULT_DB = os.environ.get("LLMGUARD_DB", str(Path.home() / ".llm-guard" / "usage.db"))


def _store(args: argparse.Namespace) -> Store:
    return Store(getattr(args, "db", None) or DEFAULT_DB)


def _is_tty() -> bool:
    return sys.stdout.isatty()


# ---------------------------------------------------------------------------
# commands
# ---------------------------------------------------------------------------
def cmd_seed(args: argparse.Namespace) -> int:
    from .demo import seed as do_seed

    store = _store(args)
    total_days = args.days + args.compare_days
    try:
        summary = do_seed(
            store,
            days=total_days,
            seed_value=args.seed_value,
            requests_per_day=args.requests_per_day,
            reset=args.reset,
            with_budgets=not args.no_budgets,
        )
    finally:
        store.close()

    print(
        f"Seeded {summary['inserted']:,} requests spanning {total_days} days "
        f"({money(float(summary['total_cost_usd']))} of simulated spend)."
    )
    if args.compare_days:
        print(
            f"  Last {args.days} days is the reporting window; the preceding "
            f"{args.compare_days} days gives the period-over-period delta."
        )
    if summary["budgets"]:
        print(f"Added {summary['budgets']} demo budgets.")
    print(f"\nDatabase: {getattr(args, 'db', None) or DEFAULT_DB}")
    print("\nNext:  llm-guard report")
    return 0


def cmd_report(args: argparse.Namespace) -> int:
    from .detectors import detect_all

    store = _store(args)
    try:
        report = build_report(store, days=args.days, top_n=args.top)
        # Runaway detection shares the connection: it reads the same tables.
        anomalies = [] if args.no_detection else detect_all(store)
    finally:
        store.close()

    if args.format == "json":
        payload = json.loads(render_json(report))
        payload["anomalies"] = [
            {
                "kind": a.kind,
                "subject": a.subject,
                "subject_type": a.subject_type,
                "severity": a.severity,
                "detail": a.detail,
                "remedy": a.remedy(),
                "evidence": a.evidence,
            }
            for a in anomalies
        ]
        print(json.dumps(payload, ensure_ascii=False, indent=2, default=str))
    elif args.format == "csv":
        sys.stdout.write(render_csv(report))
    else:
        print(
            render_terminal(
                report,
                color=_is_tty() and not args.no_color,
                verbose=args.verbose,
                anomalies=anomalies,
            )
        )
    return 0


def cmd_dashboard(args: argparse.Namespace) -> int:
    from .dashboard import render_dashboard

    store = _store(args)
    try:
        report = build_report(store, days=args.days)
    finally:
        store.close()

    html = render_dashboard(report)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html, encoding="utf-8")
    print(f"Wrote {out}  ({len(html):,} bytes, self-contained — no CDN, no JS deps)")
    return 0


def cmd_export(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        rows = store.query(
            """SELECT ts, provider, model, input_tokens, cached_input_tokens,
                      cache_write_tokens, output_tokens, cost_usd, latency_ms,
                      status, streamed, api_key_id, end_user, project, error
               FROM requests WHERE ts >= datetime('now', ?)
               ORDER BY ts DESC""",
            (f"-{args.days} days",),
        )
    finally:
        store.close()

    if args.format == "json":
        print(json.dumps([dict(r) for r in rows], indent=2, default=str))
    else:
        import csv as _csv
        import io

        buf = io.StringIO()
        if rows:
            writer = _csv.DictWriter(buf, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            for r in rows:
                writer.writerow(dict(r))
        sys.stdout.write(buf.getvalue())
    return 0


def cmd_budget(args: argparse.Namespace) -> int:
    store = _store(args)
    try:
        if args.action_cmd == "set":
            if args.daily is None and args.monthly is None:
                print("error: give --daily and/or --monthly", file=sys.stderr)
                return 2
            store.set_budget(
                args.key,
                daily_usd=args.daily,
                monthly_usd=args.monthly,
                action=args.on_exceed,
            )
            parts = []
            if args.daily is not None:
                parts.append(f"daily {money(args.daily)}")
            if args.monthly is not None:
                parts.append(f"monthly {money(args.monthly)}")
            print(f"Budget set for {args.key}: {', '.join(parts)} (action: {args.on_exceed})")
            if args.on_exceed == "block":
                print("The proxy will return HTTP 429 for this key once the limit is reached.")
            return 0

        if args.action_cmd == "rm":
            store.query("DELETE FROM budgets WHERE api_key_id=?", (args.key,))
            print(f"Removed budget for {args.key}")
            return 0

        # list
        rows = store.all_budgets()
        if not rows:
            print("No budgets configured.")
            print("Set one:  llm-guard budget set prod-web --monthly 500 --on-exceed block")
            return 0
        report = build_report(store, days=30)
        by_key = {b.api_key_id: b for b in report.budgets}
        print(f"{'api key':<26}{'daily':>14}{'monthly':>14}{'action':>10}  state")
        for row in rows:
            key = str(row["api_key_id"])
            status = by_key.get(key)
            state = status.state if status else "ok"
            daily = money(row["daily_usd"]) if row["daily_usd"] else "—"
            monthly = money(row["monthly_usd"]) if row["monthly_usd"] else "—"
            print(f"{key:<26}{daily:>14}{monthly:>14}{str(row['action']):>10}  {state}")
        return 0
    finally:
        store.close()


def cmd_import(args: argparse.Namespace) -> int:
    """Bring a prospect's usage data in, before they commit to anything."""
    from .intake import (
        CANONICAL_FIELDS,
        import_anthropic_admin,
        import_csv,
        import_json,
        import_openai_admin,
    )

    if args.sample:
        print("bucket_start,provider,model,input_tokens,cached_input_tokens,"
              "cache_write_tokens,output_tokens,cost_usd,workspace,api_key_id")
        print("2026-09-01T00:00:00Z,openai,gpt-6.1-sol,180000,90000,0,12000,,prod-web")
        print("2026-09-01T00:00:00Z,anthropic,claude-sonnet-4.5,220000,150000,4000,15000,,prod-agent")
        print()
        print("Required: " + ", ".join(CANONICAL_FIELDS))
        print()
        print("cost_usd is optional. Leave it blank and it is computed from the price")
        print("table, which is fine for a first pass. If you can supply what the")
        print("provider actually billed, do: it outranks our own arithmetic and the")
        print("diagnosis will say so.")
        return 0

    store = _store(args)
    try:
        if args.source == "file":
            text = Path(args.path).read_text(encoding="utf-8")
            if args.path.lower().endswith(".json") or text.lstrip()[:1] in "[{":
                result = import_json(store, text, default_provider=args.provider)
            else:
                result = import_csv(store, text, default_provider=args.provider)
        elif args.source == "anthropic":
            result = import_anthropic_admin(store, args.key, days=args.days)
        elif args.source == "openai":
            result = import_openai_admin(store, args.key, days=args.days)
        else:
            print(f"unknown source {args.source!r}", file=sys.stderr)
            return 2
    except urllib.error.HTTPError as exc:
        body = exc.read().decode("utf-8", "replace")[:400]
        print(f"HTTP {exc.code} from the provider:\n{body}", file=sys.stderr)
        print()
        if exc.code in (401, 403):
            print("This usually means the key is not an organisation/admin key, or it", file=sys.stderr)
            print("lacks read scope on usage. A normal API key will not work here.", file=sys.stderr)
        return 1
    except urllib.error.URLError as exc:
        print(f"could not reach the provider: {exc.reason}", file=sys.stderr)
        return 1
    except ValueError as exc:
        print(f"{exc}", file=sys.stderr)
        return 1
    finally:
        store.close()

    print()
    print(result.describe())
    print()
    if result.rows_kept:
        print("Next:  llm-guard diagnose --days %d" % args.days)
    else:
        print("Nothing imported. Run with --sample to see the expected columns.")
    return 0 if result.rows_kept else 1


def cmd_diagnose(args: argparse.Namespace) -> int:
    """Produce the client-facing cost diagnosis."""
    from .diagnose import diagnose
    from .diagnosis_report import render_diagnosis_html

    store = _store(args)
    try:
        d = diagnose(store, days=args.days)
    finally:
        store.close()

    if args.format == "json":
        print(json.dumps({
            "client": args.client,
            "days": d.days,
            "total_spend_usd": round(d.total_spend, 4),
            "monthly_spend_usd": round(d.monthly(d.total_spend), 2),
            "total_requests": d.total_requests,
            "total_tokens": d.total_tokens,
            "distinct_models": d.distinct_models,
            "distinct_keys": d.distinct_keys,
            "addressable_monthly_low": round(sum(f.monthly_saving_low for f in d.findings), 2),
            "addressable_monthly_high": round(sum(f.monthly_saving_high for f in d.findings), 2),
            "findings": [
                {
                    "key": f.key, "title": f.title, "detail": f.detail,
                    "confidence": f.confidence, "effort": f.effort,
                    "monthly_saving_low": round(f.monthly_saving_low, 2),
                    "monthly_saving_high": round(f.monthly_saving_high, 2),
                    "evidence": f.evidence, "remedy": f.remedy,
                }
                for f in d.ranked
            ],
            "gaps": d.gaps,
        }, indent=2, default=str))
        return 0

    if args.format == "text":
        color = _is_tty() and not args.no_color
        from .report import Style
        s = Style(color)
        print()
        print(s("  LLM Cost Diagnosis", "bold"))
        print(s(f"  {d.total_requests:,} {d.unit} over {d.days} days"
                + (f"   ·   client: {args.client}" if args.client else ""), "dim"))
        print(s("  " + "─" * 74, "dim"))
        print()
        spend_m = d.monthly(d.total_spend)
        lo = sum(f.monthly_saving_low for f in d.findings)
        hi = sum(f.monthly_saving_high for f in d.findings)
        print(f"  {'Measured spend':<26}{money(spend_m)}/month")
        print(f"  {'Addressable (upper bound)':<26}{money(lo)} – {money(hi)}/month"
              + s(f"   ({hi / spend_m * 100:.0f}% of bill)" if spend_m else "", "dim"))
        print()
        if not d.ranked:
            print(s("  No findings. Data may be too small to analyse.", "yellow"))
        for i, f in enumerate(d.ranked, 1):
            tone = "green" if f.has_money else "dim"
            print(f"  {s(str(i) + '.', 'bold')} {s(f.title, 'bold')}")
            if f.has_money:
                print(f"     {s(_money_range(f), tone)}   {s('[' + f.confidence + ']', 'dim')}")
            else:
                print(s(f"     [{'no direct saving'} · {f.confidence}]", "dim"))
            for line in _wrap_text(f.detail, 70):
                print(f"     {line}")
            for line in _wrap_text("→ " + f.remedy, 70):
                print(s(f"     {line}", "dim"))
            print()
        if d.gaps:
            print(s("  What this cannot tell you", "bold"))
            for g in d.gaps:
                for line in _wrap_text("• " + g, 70):
                    print(s(f"  {line}", "dim"))
            print()
        return 0

    html_doc = render_diagnosis_html(d, client=args.client or "")
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(html_doc, encoding="utf-8")
    print(f"Wrote {out}  ({len(html_doc):,} bytes)")
    print()
    print("  Open it in a browser and print to PDF, or send the HTML as-is.")
    print("  The layout is set for A4 with page breaks; no external assets.")
    return 0


def _money_range(f) -> str:
    if abs(f.monthly_saving_high - f.monthly_saving_low) < 1:
        return f"${f.mid_saving:,.0f}/mo"
    return f"${f.monthly_saving_low:,.0f}–${f.monthly_saving_high:,.0f}/mo"


def cmd_anomalies(args: argparse.Namespace) -> int:
    """Report suspected runaway agent spend."""
    from .detectors import DetectionConfig, detect_all

    config = DetectionConfig(
        io_ratio=args.io_ratio,
        velocity_multiplier=args.velocity_multiplier,
        window_minutes=args.window,
    )
    store = _store(args)
    try:
        found = detect_all(store, config)
    finally:
        store.close()

    if args.format == "json":
        print(
            json.dumps(
                {
                    "window_minutes": config.window_minutes,
                    "thresholds": {
                        "io_ratio": config.io_ratio,
                        "velocity_multiplier": config.velocity_multiplier,
                        "storm_min_failures": config.storm_min_failures,
                    },
                    "detected": [
                        {
                            "kind": a.kind,
                            "subject": a.subject,
                            "subject_type": a.subject_type,
                            "severity": a.severity,
                            "detail": a.detail,
                            "remedy": a.remedy(),
                            "evidence": a.evidence,
                        }
                        for a in found
                    ],
                },
                indent=2,
                default=str,
            )
        )
        return 0

    color = _is_tty() and not args.no_color
    from .report import Style

    s = Style(color)
    print()
    print(s("  Runaway-spend detection", "bold"))
    print(
        s(
            f"  window: last {config.window_minutes} min   ·   "
            f"io ratio > {config.io_ratio:.0f}:1   ·   "
            f"velocity > {config.velocity_multiplier:.0f}x baseline",
            "dim",
        )
    )
    print(s("  " + "─" * 72, "dim"))
    print()
    if not found:
        print(s("  Nothing anomalous detected in this window.", "green"))
        print()
        return 0

    for anomaly in found:
        tone = "red" if anomaly.severity == "critical" else "yellow"
        label = s(f"{anomaly.severity.upper():<9}", tone)
        scope = s(f"  [{anomaly.subject_type}: {anomaly.subject}]", "dim")
        print(f"  {label}{s(anomaly.kind, 'bold')}{scope}")
        for line in _wrap_text(anomaly.detail, 68):
            print(f"    {line}")
        print()
        print(s("    → what to do", "dim"))
        for line in _wrap_text(anomaly.remedy(), 68):
            print(f"      {line}")
        print()
    print(s("  " + "─" * 72, "dim"))
    print(s("  Detection is advisory. Only a budget with action=block stops traffic.", "dim"))
    print()
    return 0


def _wrap_text(text: str, width: int) -> list:
    words, lines, cur = str(text).split(), [], ""
    for word in words:
        if not cur:
            cur = word
        elif len(cur) + 1 + len(word) <= width:
            cur += " " + word
        else:
            lines.append(cur)
            cur = word
    if cur:
        lines.append(cur)
    return lines


def cmd_models(args: argparse.Namespace) -> int:
    from .pricing import PRICES, VERIFIED_AT

    print(f"Built-in pricing table (USD per 1M tokens) — verified {VERIFIED_AT}\n")
    print(f"{'model':<26}{'provider':<11}{'input':>9}{'output':>9}{'cache rd':>10}{'cache wr':>10}")
    print("-" * 76)
    for name in known_models():
        p = PRICES[name]
        cr = f"{p.cached_input:.3f}" if p.cached_input is not None else "—"
        cw = f"{p.cache_write:.3f}" if p.cache_write is not None else "—"
        print(f"{name:<26}{p.provider:<11}{p.input:>9.3f}{p.output:>9.3f}{cr:>10}{cw:>10}")
    print(
        "\nEdit llmguard/pricing.py to add or correct a rate. "
        "Unpriced models are reported as unknown rather than estimated."
    )
    return 0


def cmd_cost(args: argparse.Namespace) -> int:
    resolved = resolve_model(args.model)
    price = get_price(args.model)
    usage = TokenUsage(
        input=args.input,
        output=args.output,
        cached_input=args.cached,
        cache_write=args.cache_write,
    )
    cost = compute_cost(args.model, usage)
    if cost is None:
        print(f"Model {args.model!r} is not in the pricing table (resolved to {resolved!r}).")
        print("Add it to llmguard/pricing.py, or run `llm-guard models` to see what is covered.")
        return 1
    print(f"model           {args.model}  (price entry: {resolved})")
    print(f"input tokens    {usage.input:,} @ ${price.input}/1M")
    if usage.cached_input:
        print(f"cached input    {usage.cached_input:,} @ ${price.cached_input}/1M")
    if usage.cache_write:
        print(f"cache write     {usage.cache_write:,} @ ${price.cache_write}/1M")
    print(f"output tokens   {usage.output:,} @ ${price.output}/1M")
    print(f"{'-' * 40}")
    print(f"cost            {money(float(cost), 6)}")
    if args.count > 1:
        print(f"× {args.count:,} calls      {money(float(cost) * args.count)}")
    return 0


def cmd_serve(args: argparse.Namespace) -> int:
    from .gateway import DEFAULT_ROUTES, Gateway, GatewayConfig, Route

    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%H:%M:%S",
    )

    routes: List[Route] = list(DEFAULT_ROUTES)
    if args.route:
        routes = []
        for spec in args.route:
            # form:  /v1=https://api.openai.com[=provider]
            try:
                prefix, upstream = spec.split("=", 1)
            except ValueError:
                print(f"error: --route expects PREFIX=URL, got {spec!r}", file=sys.stderr)
                return 2
            provider = "openai"
            if "=" in upstream:
                upstream, provider = upstream.split("=", 1)
            if prefix == "/anthropic":
                provider = "anthropic"
            routes.append(
                Route(
                    prefix=prefix.rstrip("/") or "/",
                    upstream=upstream.rstrip("/"),
                    provider=provider,
                    strip_prefix=prefix.rstrip("/") or "/",
                )
            )

    upstream_keys = {}
    if args.openai_key:
        upstream_keys["openai"] = args.openai_key
    if args.anthropic_key:
        upstream_keys["anthropic"] = args.anthropic_key
    if args.google_key:
        upstream_keys["google"] = args.google_key

    key_map = {}
    for pair in args.key_map or []:
        if "=" not in pair:
            print(f"error: --key-map expects TOKEN=LABEL, got {pair!r}", file=sys.stderr)
            return 2
        token, label = pair.split("=", 1)
        key_map[token] = label

    store = _store(args)
    config = GatewayConfig(
        host=args.host,
        port=args.port,
        routes=routes,
        upstream_keys=upstream_keys,
        key_map=key_map,
        verify_upstream_tls=not args.no_verify_upstream_tls,
        upstream_ca_file=args.upstream_ca,
        loop_detection=not args.no_loop_detection,
        stream_max_tokens=args.stream_max_tokens,
        stream_max_cost_usd=args.stream_max_cost,
    )
    gateway = Gateway(store, config)

    print(f"llm-guard gateway listening on http://{args.host}:{args.port}")
    print(f"  database        {getattr(args, 'db', None) or DEFAULT_DB}")
    for route in routes:
        print(f"  {route.prefix:<12} -> {route.upstream}  ({route.provider})")
    print(f"  health          http://{args.host}:{args.port}/-/health")
    print(f"  live dashboard  http://{args.host}:{args.port}/-/dashboard")
    print(f"  json stats      http://{args.host}:{args.port}/-/stats")
    print(f"  anomalies       http://{args.host}:{args.port}/-/anomalies")
    print()
    print("  Point your SDK at it:")
    print(f"    export OPENAI_BASE_URL=http://{args.host}:{args.port}/v1")
    print(f"    export ANTHROPIC_BASE_URL=http://{args.host}:{args.port}/anthropic")
    if not upstream_keys:
        print()
        print("  No --openai-key/--anthropic-key given: the gateway forwards your own")
        print("  Authorization header upstream. That is the normal local-dev setup.")
    print()

    try:
        asyncio.run(gateway.serve_forever())
    except KeyboardInterrupt:
        print("\nshutting down")
    finally:
        store.close()
    return 0


def cmd_doctor(args: argparse.Namespace) -> int:
    """Sanity-check the install and the database."""
    import platform
    import sqlite3

    ok = True
    print("llm-guard doctor")
    print(f"  python            {platform.python_version()} ({platform.system()})")
    print(f"  sqlite            {sqlite3.sqlite_version}")
    db = getattr(args, "db", None) or DEFAULT_DB
    print(f"  database          {db}")
    try:
        store = Store(db)
        n = store.count()
        report = build_report(store, days=30)
        store.close()
        print(f"  rows              {n:,}")
        print(f"  priced models     {len(report.by_model)}")
        print(f"  budgets           {len(report.budgets)}")
        if report.unpriced_models:
            print(f"  unpriced models   {', '.join(report.unpriced_models)}")
        if n == 0:
            print("\n  Database is empty. Run:  llm-guard seed")
    except Exception as exc:  # noqa: BLE001
        ok = False
        print(f"  database error    {exc}")
    print("\n  " + ("all checks passed" if ok else "problems found"))
    return 0 if ok else 1


# ---------------------------------------------------------------------------
# parser
# ---------------------------------------------------------------------------
def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llm-guard",
        description=(
            "Self-hosted LLM cost attribution and budget enforcement gateway. "
            "Zero runtime dependencies: stdlib Python only."
        ),
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=(
            "quickstart:\n"
            "  llm-guard seed                  # demo data, no API key needed\n"
            "  llm-guard report                # where the money went\n"
            "  llm-guard dashboard             # standalone HTML\n"
            "  llm-guard serve --openai-key $OPENAI_API_KEY\n"
        ),
    )
    parser.add_argument(
        "--db", default=None, help=f"SQLite path (default: {DEFAULT_DB}, env LLMGUARD_DB)"
    )
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("seed", help="load a realistic demo dataset")
    p.add_argument("--days", type=int, default=30, help="reporting window length")
    p.add_argument(
        "--compare-days",
        type=int,
        default=0,
        help="also seed this many earlier days so report shows a delta",
    )
    p.add_argument("--requests-per-day", type=int, default=260)
    p.add_argument("--seed-value", type=int, default=7, help="RNG seed (deterministic)")
    p.add_argument("--reset", action="store_true", help="delete existing rows first")
    p.add_argument("--no-budgets", action="store_true", help="skip demo budgets")
    p.set_defaults(func=cmd_seed)

    p = sub.add_parser("report", help="show spend and what to change")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--format", choices=["text", "json", "csv"], default="text")
    p.add_argument("--top", type=int, default=10, help="how many expensive requests to show")
    p.add_argument("--no-color", action="store_true")
    p.add_argument("--verbose", action="store_true", help="extra breakdowns")
    p.add_argument("--no-detection", action="store_true",
                   help="skip the runaway-spend detectors")
    p.set_defaults(func=cmd_report)

    p = sub.add_parser("dashboard", help="write a self-contained HTML dashboard")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--out", default="llm-guard-dashboard.html")
    p.set_defaults(func=cmd_dashboard)

    p = sub.add_parser("export", help="dump raw request rows as csv or json")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--format", choices=["csv", "json"], default="csv")
    p.set_defaults(func=cmd_export)

    p = sub.add_parser("budget", help="manage per-key budgets")
    bsub = p.add_subparsers(dest="action_cmd", required=True)
    b = bsub.add_parser("set", help="set a budget")
    b.add_argument("key")
    b.add_argument("--daily", type=float, default=None)
    b.add_argument("--monthly", type=float, default=None)
    b.add_argument("--on-exceed", choices=["alert", "block"], default="alert")
    b = bsub.add_parser("rm", help="remove a budget")
    b.add_argument("key")
    bsub.add_parser("list", help="list budgets")
    p.set_defaults(func=cmd_budget)




    p = sub.add_parser("import", help="load usage data from a file or a provider's admin API")
    p.add_argument("--source", choices=["file", "anthropic", "openai"], default="file")
    p.add_argument("--path", default="", help="csv or json file, when --source file")
    p.add_argument("--key", default="", help="admin api key, when --source anthropic|openai")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--provider", default="openai", help="default provider for file imports")
    p.add_argument("--sample", action="store_true", help="print the expected columns and exit")
    p.set_defaults(func=cmd_import)

    p = sub.add_parser("diagnose", help="produce the client-facing LLM cost diagnosis")
    p.add_argument("--days", type=int, default=30)
    p.add_argument("--client", default="", help="client name, appears on the report")
    p.add_argument("--format", choices=["html", "json", "text"], default="html")
    p.add_argument("--out", default="llm-cost-diagnosis.html")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_diagnose)

    p = sub.add_parser("anomalies", help="detect runaway agent spend (loops, storms, bursts)")
    p.add_argument("--window", type=int, default=15, help="lookback window in minutes")
    p.add_argument("--io-ratio", type=float, default=30.0,
                   help="input/output ratio that indicates a context loop (normal is 5-15)")
    p.add_argument("--velocity-multiplier", type=float, default=25.0,
                   help="multiple of the account's own baseline rate that counts as runaway")
    p.add_argument("--format", choices=["text", "json"], default="text")
    p.add_argument("--no-color", action="store_true")
    p.set_defaults(func=cmd_anomalies)

    p = sub.add_parser("models", help="list the built-in pricing table")
    p.set_defaults(func=cmd_models)

    p = sub.add_parser("cost", help="price a single call")
    p.add_argument("model")
    p.add_argument("--input", type=int, default=1000)
    p.add_argument("--output", type=int, default=500)
    p.add_argument("--cached", type=int, default=0)
    p.add_argument("--cache-write", type=int, default=0)
    p.add_argument("--count", type=int, default=1, help="multiply for N identical calls")
    p.set_defaults(func=cmd_cost)

    p = sub.add_parser("serve", help="run the proxy gateway")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8787)
    p.add_argument("--openai-key", default=os.environ.get("OPENAI_API_KEY"))
    p.add_argument("--anthropic-key", default=os.environ.get("ANTHROPIC_API_KEY"))
    p.add_argument("--google-key", default=os.environ.get("GOOGLE_API_KEY"))
    p.add_argument(
        "--route",
        action="append",
        help="override routes, e.g. --route /v1=http://127.0.0.1:9000=openai",
    )
    p.add_argument(
        "--key-map", action="append", help="TOKEN=LABEL, for stable per-team attribution"
    )
    p.add_argument("--stream-max-tokens", type=int, default=None,
                   help="abort a streaming response once it exceeds this many tokens (local estimate)")
    p.add_argument("--stream-max-cost", type=float, default=None,
                   help="abort a streaming response once it exceeds this many USD (local estimate)")
    p.add_argument("--upstream-ca", default="", metavar="PATH",
                   help="CA bundle for upstream TLS (use behind a corporate MITM proxy)")
    p.add_argument("--no-verify-upstream-tls", action="store_true",
                   help="DANGEROUS: skip upstream certificate verification")
    p.add_argument("--no-loop-detection", action="store_true",
                   help="disable the live input/output ratio tripwire")
    p.add_argument("--verbose", action="store_true")
    p.set_defaults(func=cmd_serve)

    p = sub.add_parser("doctor", help="check the install and database")
    p.set_defaults(func=cmd_doctor)

    return parser


def main(argv: Optional[List[str]] = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return int(args.func(args) or 0)
    except KeyboardInterrupt:
        return 130


if __name__ == "__main__":
    raise SystemExit(main())
