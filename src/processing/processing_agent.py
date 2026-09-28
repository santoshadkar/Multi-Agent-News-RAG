"""Processing agent: raw articles in, clean/deduped/tagged/chunked out.

Pipeline per the brief:
  1. dedupe (near-duplicate stories across outlets)
  2. strip boilerplate
  3. summarize
  4. extract entities
  5. tag by category (already known from the fetcher, kept for clarity)
  6. split into ~300-500 token chunks
"""

from __future__ import annotations

import logging

from src.models import ProcessedArticle, RawArticle
from src.processing.chunking import chunk_article
from src.processing.clean import clean_body, extractive_summary
from src.processing.dedupe import dedupe_near_duplicates
from src.processing.tagging import extract_entities
from src.storage.raw_store import RawStore

logger = logging.getLogger(__name__)


def process_raw_article(raw: RawArticle) -> ProcessedArticle:
    body = clean_body(raw.body)
    return ProcessedArticle(
        id=raw.id,
        url=raw.url,
        headline=raw.headline.strip(),
        clean_body=body,
        summary=extractive_summary(body),
        entities=extract_entities(f"{raw.headline} {body}"),
        category=raw.category,
        source=raw.source,
        published_at=raw.published_at,
    )


class ProcessingAgent:
    def __init__(self, store: RawStore):
        self.store = store

    def run(self, dedupe_threshold: float = 0.6) -> dict:
        raw_articles = self.store.unprocessed()
        if not raw_articles:
            return {"input": 0, "after_dedupe": 0, "chunks": 0, "processed": []}

        deduped = dedupe_near_duplicates(raw_articles, threshold=dedupe_threshold)
        dropped_ids = {a.id for a in raw_articles} - {a.id for a in deduped}

        processed: list[ProcessedArticle] = [process_raw_article(a) for a in deduped]
        all_chunks = []
        for article in processed:
            all_chunks.extend(chunk_article(article))

        # Mark ALL originally-fetched raw rows as processed, including the
        # near-duplicates we dropped -- otherwise the dropped copies would
        # be re-considered (and re-dropped) on every future run.
        self.store.mark_processed([a.id for a in raw_articles])

        logger.info(
            "Processed %d raw articles -> %d unique -> %d chunks (%d near-dup dropped)",
            len(raw_articles),
            len(processed),
            len(all_chunks),
            len(dropped_ids),
        )
        return {
            "input": len(raw_articles),
            "after_dedupe": len(processed),
            "dropped_near_duplicates": len(dropped_ids),
            "chunks": len(all_chunks),
            "processed": processed,
            "chunk_objects": all_chunks,
        }
