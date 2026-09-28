from datetime import datetime, timezone

from src.models import RawArticle
from src.processing.processing_agent import ProcessingAgent
from src.storage.raw_store import RawStore


def make_article(url, headline, body, published_at=None, category="finance"):
    return RawArticle(
        url=url,
        headline=headline,
        body=body,
        source="test-source",
        category=category,
        published_at=published_at or datetime(2026, 1, 1, tzinfo=timezone.utc),
    )


def test_insert_if_new_rejects_duplicate_url(tmp_path):
    store = RawStore(str(tmp_path / "raw.db"))
    article = make_article("https://a.com/1", "Headline", "Body text about markets and finance news.")
    assert store.insert_if_new(article) is True
    assert store.insert_if_new(article) is False  # same URL, second insert is a no-op


def test_unprocessed_returns_only_new_rows(tmp_path):
    store = RawStore(str(tmp_path / "raw.db"))
    store.insert_if_new(make_article("https://a.com/1", "H1", "Body one about finance."))
    store.insert_if_new(make_article("https://a.com/2", "H2", "Body two about finance."))
    assert len(store.unprocessed()) == 2

    ids = [a.id for a in store.unprocessed()]
    store.mark_processed(ids)
    assert store.unprocessed() == []


def test_processing_agent_dedupes_and_chunks(tmp_path):
    store = RawStore(str(tmp_path / "raw.db"))
    long_body = ("The central bank announced a new policy today affecting markets nationwide. " * 20).strip()
    store.insert_if_new(make_article("https://a.com/1", "Central bank policy news", long_body))
    store.insert_if_new(make_article("https://b.com/1", "Central bank policy news", long_body))  # near-dup
    store.insert_if_new(make_article("https://c.com/2", "Unrelated tech story", "A startup launched a new gadget today."))

    result = ProcessingAgent(store).run()

    assert result["input"] == 3
    assert result["after_dedupe"] == 2  # one near-duplicate dropped
    assert result["dropped_near_duplicates"] == 1
    assert result["chunks"] >= 2
    assert store.unprocessed() == []  # everything, including the dropped dup, marked processed
