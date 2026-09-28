"""SQLite-backed store for raw fetched articles.

This is the hand-off point between fetcher agents and the processing agent.
Using a real (if tiny) DB rather than in-memory lists means: fetchers can run
on a schedule and exit; the processing agent can run later and pick up
whatever is new; and URL-level dedupe is enforced by a UNIQUE constraint
instead of by convention.
"""

from __future__ import annotations

import sqlite3
from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator, Optional

from src.models import RawArticle

SCHEMA = """
CREATE TABLE IF NOT EXISTS raw_articles (
    id TEXT PRIMARY KEY,
    url TEXT UNIQUE NOT NULL,
    headline TEXT NOT NULL,
    body TEXT NOT NULL,
    source TEXT NOT NULL,
    category TEXT NOT NULL,
    published_at TEXT,
    fetched_at TEXT NOT NULL,
    processed INTEGER NOT NULL DEFAULT 0
);
CREATE INDEX IF NOT EXISTS idx_raw_processed ON raw_articles(processed);
CREATE INDEX IF NOT EXISTS idx_raw_category ON raw_articles(category);

CREATE TABLE IF NOT EXISTS run_log (
    run_id TEXT NOT NULL,
    agent TEXT NOT NULL,
    started_at TEXT NOT NULL,
    finished_at TEXT,
    status TEXT NOT NULL,
    fetched_count INTEGER NOT NULL DEFAULT 0,
    kept_count INTEGER NOT NULL DEFAULT 0,
    error TEXT
);
"""


class RawStore:
    def __init__(self, db_path: str):
        self.db_path = db_path
        with self._connect() as conn:
            conn.executescript(SCHEMA)

    @contextmanager
    def _connect(self) -> Iterator[sqlite3.Connection]:
        conn = sqlite3.connect(self.db_path)
        try:
            yield conn
            conn.commit()
        finally:
            conn.close()

    def insert_if_new(self, article: RawArticle) -> bool:
        """Returns True if the article was newly inserted, False if it was
        already present (i.e. a duplicate URL we've seen before)."""
        with self._connect() as conn:
            try:
                conn.execute(
                    """INSERT INTO raw_articles
                       (id, url, headline, body, source, category, published_at, fetched_at, processed)
                       VALUES (?, ?, ?, ?, ?, ?, ?, ?, 0)""",
                    (
                        article.id,
                        article.url,
                        article.headline,
                        article.body,
                        article.source,
                        article.category,
                        article.published_at.isoformat() if article.published_at else None,
                        article.fetched_at.isoformat(),
                    ),
                )
                return True
            except sqlite3.IntegrityError:
                return False

    def unprocessed(self, limit: Optional[int] = None) -> list[RawArticle]:
        query = "SELECT id, url, headline, body, source, category, published_at, fetched_at FROM raw_articles WHERE processed = 0"
        if limit:
            query += f" LIMIT {int(limit)}"
        with self._connect() as conn:
            rows = conn.execute(query).fetchall()
        out = []
        for r in rows:
            pub = datetime.fromisoformat(r[6]) if r[6] else None
            out.append(
                RawArticle(
                    url=r[1],
                    headline=r[2],
                    body=r[3],
                    source=r[4],
                    category=r[5],
                    published_at=pub,
                    fetched_at=datetime.fromisoformat(r[7]),
                )
            )
        return out

    def mark_processed(self, article_ids: list[str]) -> None:
        if not article_ids:
            return
        with self._connect() as conn:
            conn.executemany(
                "UPDATE raw_articles SET processed = 1 WHERE id = ?",
                [(i,) for i in article_ids],
            )

    def log_run_start(self, run_id: str, agent: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO run_log (run_id, agent, started_at, status) VALUES (?, ?, ?, 'running')",
                (run_id, agent, datetime.now(timezone.utc).isoformat()),
            )

    def log_run_end(
        self,
        run_id: str,
        agent: str,
        status: str,
        fetched_count: int = 0,
        kept_count: int = 0,
        error: Optional[str] = None,
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                """UPDATE run_log SET finished_at = ?, status = ?, fetched_count = ?,
                   kept_count = ?, error = ? WHERE run_id = ? AND agent = ?""",
                (
                    datetime.now(timezone.utc).isoformat(),
                    status,
                    fetched_count,
                    kept_count,
                    error,
                    run_id,
                    agent,
                ),
            )

    def recent_runs(self, limit: int = 20) -> list[dict]:
        with self._connect() as conn:
            conn.row_factory = sqlite3.Row
            rows = conn.execute(
                "SELECT * FROM run_log ORDER BY started_at DESC LIMIT ?", (limit,)
            ).fetchall()
            return [dict(r) for r in rows]
