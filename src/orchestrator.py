"""Orchestrator: coordinates fetcher agents -> processing agent -> indexing.

Responsibilities called out in the brief: triggers the runs, handles
failures and retries (each FetcherAgent already retries transient HTTP
errors; the orchestrator retries a whole agent run once more if it still
raised), and logs what was collected each day (via RawStore.run_log and
the JSONL run summary written here).
"""

from __future__ import annotations

import json
import logging
import uuid
from datetime import datetime, timezone

import yaml

from src import config
from src.agents.base_fetcher import FetcherAgent
from src.kb.embeddings import EmbeddingClient, get_cohere_client
from src.kb.vector_store import VectorStore, get_chroma_client
from src.processing.processing_agent import ProcessingAgent
from src.storage.raw_store import RawStore

logger = logging.getLogger(__name__)


def _load_sources() -> dict:
    with open(config.ROOT_DIR / "sources.yaml") as f:
        return yaml.safe_load(f)


def run_fetch_stage(store: RawStore) -> list[dict]:
    sources = _load_sources()
    results = []
    for category in config.CATEGORIES:
        cfg = sources.get(category, {})
        agent = FetcherAgent(category, cfg, store)
        try:
            results.append(agent.run())
        except Exception as e:
            logger.warning("First attempt failed for %s (%s); retrying once.", category, e)
            try:
                results.append(agent.run())
            except Exception as e2:
                logger.error("Fetch failed twice for %s: %s", category, e2)
                results.append({"category": category, "fetched": 0, "kept": 0, "error": str(e2)})
    return results


def run_processing_stage(store: RawStore) -> dict:
    agent = ProcessingAgent(store)
    return agent.run()


def run_indexing_stage(chunk_objects: list, index_client=None) -> int:
    if not chunk_objects:
        return 0
    cohere_client = index_client or get_cohere_client()
    embedder = EmbeddingClient(cohere_client)
    vector_store = VectorStore(get_chroma_client())

    # Embed in batches to stay well under API request-size limits.
    batch_size = 96
    for start in range(0, len(chunk_objects), batch_size):
        batch = chunk_objects[start : start + batch_size]
        embeddings = embedder.embed_documents([c.text for c in batch])
        vector_store.add_chunks(batch, embeddings)
    return len(chunk_objects)


def run_full_pipeline() -> dict:
    """Fetch -> process -> index, in one call. This is what the scheduler
    (src/scheduler.py) triggers on the configured cron schedule."""
    run_id = str(uuid.uuid4())
    started = datetime.now(timezone.utc).isoformat()

    store = RawStore(config.RAW_DB_PATH)
    fetch_results = run_fetch_stage(store)
    processing_result = run_processing_stage(store)
    indexed_count = run_indexing_stage(processing_result.get("chunk_objects", []))

    summary = {
        "run_id": run_id,
        "started_at": started,
        "finished_at": datetime.now(timezone.utc).isoformat(),
        "fetch_results": fetch_results,
        "processing": {k: v for k, v in processing_result.items() if k not in ("processed", "chunk_objects")},
        "indexed_chunks": indexed_count,
    }
    with open(config.RUN_LOG_PATH, "a") as f:
        f.write(json.dumps(summary) + "\n")
    logger.info("Pipeline run complete: %s", summary)
    return summary


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    print(json.dumps(run_full_pipeline(), indent=2))
