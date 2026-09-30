"""SQLite storage for request records.

One row per proxied API call. WAL mode keeps the proxy writing while the CLI
reads, which matters because `serve` and `report` normally run at the same time.

Everything is stdlib: sqlite3 ships with CPython, so the gateway has no
third-party runtime dependency at all.
"""

from __future__ import annotations

import os
import sqlite3
import threading
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

SCHEMA = """
CREATE TABLE IF NOT EXISTS requests (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    ts                TEXT    NOT NULL,          -- ISO-8601 UTC
    provider          TEXT    NOT NULL,
    model             TEXT    NOT NULL,
    input_tokens      INTEGER NOT NULL DEFAULT 0,
    cached_input_tokens INTEGER NOT NULL DEFAULT 0,
    cache_write_tokens  INTEGER NOT NULL DEFAULT 0,
    output_tokens     INTEGER NOT NULL DEFAULT 0,
    cost_usd          REAL,                      -- NULL when model is unpriced
    latency_ms        INTEGER NOT NULL DEFAULT 0,
    status            INTEGER NOT NULL DEFAULT 200,
    streamed          INTEGER NOT NULL DEFAULT 0,
    api_key_id        TEXT    NOT NULL DEFAULT 'anonymous',
    end_user          TEXT    NOT NULL DEFAULT '',
    project           TEXT    NOT NULL DEFAULT '',
    request_id        TEXT    NOT NULL DEFAULT '',
    error             TEXT    NOT NULL DEFAULT ''
);

CREATE INDEX IF NOT EXISTS idx_requests_ts        ON requests(ts);
CREATE INDEX IF NOT EXISTS idx_requests_key_ts    ON requests(api_key_id, ts);
CREATE INDEX IF NOT EXISTS idx_requests_model     ON requests(model);
CREATE INDEX IF NOT EXISTS idx_requests_project   ON requests(project, ts);

CREATE TABLE IF NOT EXISTS budgets (
    api_key_id  TEXT PRIMARY KEY,
    daily_usd   REAL,
    monthly_usd REAL,
    action      TEXT NOT NULL DEFAULT 'alert'    -- alert | block
);

CREATE TABLE IF NOT EXISTS meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""


def utcnow() -> datetime:
    return datetime.now(timezone.utc)


def iso(dt: Optional[datetime] = None) -> str:
    """Timestamp string, formatted for SQLite's own date functions.

    Deliberately ``YYYY-MM-DD HH:MM:SS`` rather than a full ISO-8601 string:
    SQLite's ``datetime('now')`` emits a space separator, and every window
    filter compares against it textually. Using ``T`` and a ``+00:00`` suffix
    here silently makes every ``ts >= datetime('now', ...)`` comparison fail.
    Always UTC.
    """
    return (dt or utcnow()).astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")


def iso_display(dt: Optional[datetime] = None) -> str:
    """Full ISO-8601 with offset, for JSON output and human-facing APIs."""
    return (dt or utcnow()).astimezone(timezone.utc).isoformat(timespec="seconds")


@dataclass
class RequestRecord:
    provider: str
    model: str
    input_tokens: int = 0
    cached_input_tokens: int = 0
    cache_write_tokens: int = 0
    output_tokens: int = 0
    cost_usd: Optional[float] = None
    latency_ms: int = 0
    status: int = 200
    streamed: bool = False
    api_key_id: str = "anonymous"
    end_user: str = ""
    project: str = ""
    request_id: str = ""
    error: str = ""
    ts: Optional[str] = None


class Store:
    """Thin wrapper around a SQLite file.

    A single connection guarded by a lock is enough: write volume here is one
    row per LLM call, and reads are short. If you ever outgrow it, switch to a
    connection pool or ship rows to ClickHouse -- the schema is portable.
    """

    def __init__(self, path: str | os.PathLike[str]):
        self.path = str(path)
        if self.path != ":memory:":
            Path(self.path).parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._conn = sqlite3.connect(self.path, check_same_thread=False)
        self._conn.row_factory = sqlite3.Row
        with self._lock:
            self._conn.executescript(SCHEMA)
            try:
                self._conn.execute("PRAGMA journal_mode=WAL")
                self._conn.execute("PRAGMA synchronous=NORMAL")
            except sqlite3.DatabaseError:
                pass  # e.g. :memory: in some builds
            self._conn.commit()

    # -- writes ------------------------------------------------------------
    def insert(self, rec: RequestRecord) -> int:
        with self._lock:
            cur = self._conn.execute(
                """
                INSERT INTO requests (
                    ts, provider, model, input_tokens, cached_input_tokens,
                    cache_write_tokens, output_tokens, cost_usd, latency_ms,
                    status, streamed, api_key_id, end_user, project,
                    request_id, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                (
                    rec.ts or iso(),
                    rec.provider,
                    rec.model,
                    int(rec.input_tokens),
                    int(rec.cached_input_tokens),
                    int(rec.cache_write_tokens),
                    int(rec.output_tokens),
                    None if rec.cost_usd is None else float(rec.cost_usd),
                    int(rec.latency_ms),
                    int(rec.status),
                    1 if rec.streamed else 0,
                    rec.api_key_id or "anonymous",
                    rec.end_user or "",
                    rec.project or "",
                    rec.request_id or "",
                    rec.error or "",
                ),
            )
            self._conn.commit()
            return int(cur.lastrowid or 0)

    def insert_many(self, recs: Iterable[RequestRecord]) -> int:
        rows = [
            (
                r.ts or iso(), r.provider, r.model, int(r.input_tokens),
                int(r.cached_input_tokens), int(r.cache_write_tokens),
                int(r.output_tokens),
                None if r.cost_usd is None else float(r.cost_usd),
                int(r.latency_ms), int(r.status), 1 if r.streamed else 0,
                r.api_key_id or "anonymous", r.end_user or "", r.project or "",
                r.request_id or "", r.error or "",
            )
            for r in recs
        ]
        if not rows:
            return 0
        self._insert_rows(rows)
        return len(rows)

    def _insert_rows(self, rows: list[tuple]) -> None:
        with self._lock:
            self._conn.executemany(
                """
                INSERT INTO requests (
                    ts, provider, model, input_tokens, cached_input_tokens,
                    cache_write_tokens, output_tokens, cost_usd, latency_ms,
                    status, streamed, api_key_id, end_user, project,
                    request_id, error
                ) VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?,?,?,?)
                """,
                rows,
            )
            self._conn.commit()

    @staticmethod
    def record_to_row(r: RequestRecord) -> tuple:
        return (
            r.ts or iso(), r.provider, r.model, int(r.input_tokens),
            int(r.cached_input_tokens), int(r.cache_write_tokens),
            int(r.output_tokens),
            None if r.cost_usd is None else float(r.cost_usd),
            int(r.latency_ms), int(r.status), 1 if r.streamed else 0,
            r.api_key_id or "anonymous", r.end_user or "", r.project or "",
            r.request_id or "", r.error or "",
        )

    # -- budgets -----------------------------------------------------------
    def set_budget(
        self,
        api_key_id: str,
        *,
        daily_usd: Optional[float] = None,
        monthly_usd: Optional[float] = None,
        action: str = "alert",
    ) -> None:
        with self._lock:
            self._conn.execute(
                """INSERT INTO budgets (api_key_id, daily_usd, monthly_usd, action)
                   VALUES (?,?,?,?)
                   ON CONFLICT(api_key_id) DO UPDATE SET
                     daily_usd=excluded.daily_usd,
                     monthly_usd=excluded.monthly_usd,
                     action=excluded.action""",
                (api_key_id, daily_usd, monthly_usd, action),
            )
            self._conn.commit()

    def get_budget(self, api_key_id: str) -> Optional[sqlite3.Row]:
        with self._lock:
            cur = self._conn.execute(
                "SELECT * FROM budgets WHERE api_key_id=?", (api_key_id,)
            )
            return cur.fetchone()

    def all_budgets(self) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute("SELECT * FROM budgets ORDER BY api_key_id"))

    # -- reads -------------------------------------------------------------
    def execute(self, sql: str, params: Sequence[Any] = ()) -> int:
        """Run a statement that changes rows. Returns the affected count.

        Separate from query() so a read cannot accidentally be a write, and so
        callers do not have to know that query() happens to commit too.
        """
        with self._lock:
            cur = self._conn.execute(sql, tuple(params))
            self._conn.commit()
            return int(cur.rowcount if cur.rowcount is not None else 0)

    def query(self, sql: str, params: Sequence[Any] = ()) -> List[sqlite3.Row]:
        with self._lock:
            return list(self._conn.execute(sql, tuple(params)))

    def one(self, sql: str, params: Sequence[Any] = ()) -> Optional[sqlite3.Row]:
        rows = self.query(sql, params)
        return rows[0] if rows else None

    def count(self) -> int:
        row = self.one("SELECT COUNT(*) AS n FROM requests")
        return int(row["n"]) if row else 0

    def set_meta(self, key: str, value: str) -> None:
        with self._lock:
            self._conn.execute(
                "INSERT INTO meta (key,value) VALUES (?,?) "
                "ON CONFLICT(key) DO UPDATE SET value=excluded.value",
                (key, value),
            )
            self._conn.commit()

    def get_meta(self, key: str) -> Optional[str]:
        row = self.one("SELECT value FROM meta WHERE key=?", (key,))
        return row["value"] if row else None

    def close(self) -> None:
        with self._lock:
            self._conn.close()
