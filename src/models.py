"""Shared data structures used across fetcher, processing, and RAG layers."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Optional


def url_hash(url: str) -> str:
    return hashlib.sha256(url.strip().lower().encode("utf-8")).hexdigest()[:24]


@dataclass
class RawArticle:
    """An article exactly as collected by a fetcher agent, before cleaning."""

    url: str
    headline: str
    body: str
    source: str
    category: str
    published_at: Optional[datetime]
    fetched_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

    @property
    def id(self) -> str:
        return url_hash(self.url)


@dataclass
class ProcessedArticle:
    """Output of the processing agent: one deduped, cleaned article."""

    id: str
    url: str
    headline: str
    clean_body: str
    summary: str
    entities: list[str]
    category: str
    source: str
    published_at: Optional[datetime]


@dataclass
class Chunk:
    """One retrieval unit stored in the vector database."""

    id: str
    article_id: str
    text: str
    headline: str
    url: str
    source: str
    category: str
    published_at: Optional[str]  # ISO date string, or None
    chunk_index: int
