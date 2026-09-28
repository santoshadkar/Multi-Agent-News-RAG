"""Command-line entry points.

  python -m src.cli run            -- one fetch+process+index cycle
  python -m src.cli briefing        -- generate today's briefing panel
  python -m src.cli check-sources   -- verify every feed/page in sources.yaml resolves
  python -m src.cli ask "question"  -- ask a one-off question from the terminal
"""

from __future__ import annotations

import argparse
import json
import logging

import requests
import yaml

from src import config


def cmd_run(_args) -> None:
    from src.orchestrator import run_full_pipeline

    print(json.dumps(run_full_pipeline(), indent=2))


def cmd_briefing(_args) -> None:
    from src.kb.embeddings import get_cohere_client
    from src.kb.vector_store import VectorStore, get_chroma_client
    from src.rag.briefing import generate_briefing, save_briefing

    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    briefing = generate_briefing(client, store)
    save_briefing(briefing)
    print(json.dumps(briefing, indent=2))


def cmd_check_sources(_args) -> None:
    with open(config.ROOT_DIR / "sources.yaml") as f:
        sources = yaml.safe_load(f)

    ok, failed = [], []
    for category, cfg in sources.items():
        for feed in cfg.get("feeds", []):
            try:
                resp = requests.get(feed["url"], timeout=config.REQUEST_TIMEOUT_SECS, headers={"User-Agent": config.USER_AGENT})
                (ok if resp.ok else failed).append(f"{category}/{feed['name']}: HTTP {resp.status_code}")
            except requests.RequestException as e:
                failed.append(f"{category}/{feed['name']}: {e}")
        for page in cfg.get("official_pages", []):
            try:
                resp = requests.get(page["url"], timeout=config.REQUEST_TIMEOUT_SECS, headers={"User-Agent": config.USER_AGENT})
                (ok if resp.ok else failed).append(f"{category}/{page['name']}: HTTP {resp.status_code}")
            except requests.RequestException as e:
                failed.append(f"{category}/{page['name']}: {e}")

    print(f"OK: {len(ok)}")
    for line in ok:
        print("  " + line)
    print(f"FAILED: {len(failed)}")
    for line in failed:
        print("  " + line)
    if failed:
        print("\nReplace failing entries in sources.yaml with a working feed for that outlet.")


def cmd_ask(args) -> None:
    from src.kb.embeddings import get_cohere_client
    from src.kb.vector_store import VectorStore, get_chroma_client
    from src.rag.query_engine import QueryEngine

    client = get_cohere_client()
    store = VectorStore(get_chroma_client())
    engine = QueryEngine(client, store)
    result = engine.answer(args.question)

    print(result.answer.answer)
    if result.answer.citations:
        print("\nSources:")
        for c in result.answer.citations:
            date_str = f" ({c.published_at})" if c.published_at else ""
            print(f"  - {c.headline}{date_str} — {c.source} — {c.url}")


def main() -> None:
    logging.basicConfig(level=logging.INFO)
    parser = argparse.ArgumentParser(description="News RAG pipeline CLI")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("run", help="Run one fetch+process+index cycle").set_defaults(func=cmd_run)
    sub.add_parser("briefing", help="Generate today's briefing").set_defaults(func=cmd_briefing)
    sub.add_parser("check-sources", help="Verify sources.yaml feeds resolve").set_defaults(func=cmd_check_sources)
    ask_parser = sub.add_parser("ask", help="Ask a one-off question")
    ask_parser.add_argument("question")
    ask_parser.set_defaults(func=cmd_ask)

    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
