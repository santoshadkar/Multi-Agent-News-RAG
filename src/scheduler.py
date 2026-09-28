"""APScheduler-based scheduling, per the "Suggested Tech Stack" (cron,
APScheduler, or n8n). Run with: python -m src.scheduler
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.blocking import BlockingScheduler
from apscheduler.triggers.cron import CronTrigger

from src import config
from src.kb.embeddings import get_cohere_client
from src.kb.vector_store import VectorStore, get_chroma_client
from src.orchestrator import run_full_pipeline
from src.rag.briefing import generate_briefing, save_briefing

logger = logging.getLogger(__name__)


def _cron_trigger(expr: str) -> CronTrigger:
    minute, hour, day, month, day_of_week = expr.split()
    return CronTrigger(minute=minute, hour=hour, day=day, month=month, day_of_week=day_of_week)


def fetch_job() -> None:
    logger.info("Scheduled fetch/process/index run starting.")
    run_full_pipeline()


def briefing_job() -> None:
    logger.info("Scheduled briefing generation starting.")
    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    briefing = generate_briefing(client, store)
    save_briefing(briefing)


def start() -> None:
    logging.basicConfig(level=logging.INFO)
    scheduler = BlockingScheduler()
    scheduler.add_job(fetch_job, _cron_trigger(config.FETCH_CRON), id="fetch_pipeline")
    scheduler.add_job(briefing_job, _cron_trigger(config.BRIEFING_CRON), id="daily_briefing")
    logger.info(
        "Scheduler started. Fetch cron: '%s', briefing cron: '%s'.",
        config.FETCH_CRON,
        config.BRIEFING_CRON,
    )
    scheduler.start()


if __name__ == "__main__":
    start()
