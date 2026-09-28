"""Generates the optional "Today's Briefing" panel: one short, cited
summary per category, built from the most recent chunks in the knowledge
base. Meant to be run once each morning by the scheduler."""

from __future__ import annotations

import json
from datetime import datetime

import cohere

from src import config
from src.kb.vector_store import VectorStore
from src.rag.generate import generate_answer


def _most_recent_chunks(store: VectorStore, category: str, limit: int = 8) -> list[dict]:
    where = {"category": category}
    pool = store.all_chunks_in(where=where, limit=500)

    def pub_key(c: dict):
        raw = c["metadata"].get("published_at") or ""
        try:
            return datetime.fromisoformat(raw)
        except ValueError:
            return datetime.min

    pool.sort(key=pub_key, reverse=True)
    return pool[:limit]


def generate_briefing(client: cohere.ClientV2, store: VectorStore) -> dict:
    briefing = {"generated_at": datetime.utcnow().isoformat(), "sections": {}}
    for category in config.CATEGORIES:
        chunks = _most_recent_chunks(store, category)
        result = generate_answer(
            client,
            f"Summarize today's most important {category} news in 3-4 sentences.",
            chunks,
        )
        briefing["sections"][category] = {
            "summary": result.answer,
            "citations": [
                {
                    "headline": c.headline,
                    "url": c.url,
                    "source": c.source,
                    "published_at": c.published_at,
                }
                for c in result.citations
            ],
        }
    return briefing


def save_briefing(briefing: dict, path: str = config.BRIEFING_PATH) -> None:
    with open(path, "w") as f:
        json.dump(briefing, f, indent=2)


def load_briefing(path: str = config.BRIEFING_PATH) -> dict | None:
    try:
        with open(path) as f:
            return json.load(f)
    except FileNotFoundError:
        return None
