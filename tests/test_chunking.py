from datetime import datetime, timezone

from src.models import ProcessedArticle
from src.processing.chunking import chunk_article, chunk_text


def test_short_text_yields_single_chunk():
    text = "word " * 50
    chunks = chunk_text(text.strip(), min_words=220, max_words=380, overlap_words=40)
    assert len(chunks) == 1


def test_long_text_splits_into_multiple_chunks_within_bounds():
    text = "word " * 1000
    chunks = chunk_text(text.strip(), min_words=220, max_words=380, overlap_words=40)
    assert len(chunks) > 1
    for c in chunks[:-1]:  # last chunk may be a short remainder/merged tail
        n_words = len(c.split())
        assert n_words <= 380


def test_consecutive_chunks_overlap():
    text = " ".join(f"w{i}" for i in range(1000))
    chunks = chunk_text(text, min_words=220, max_words=380, overlap_words=40)
    first_words = chunks[0].split()
    second_words = chunks[1].split()
    overlap = set(first_words[-40:]) & set(second_words[:40])
    assert len(overlap) > 0


def test_chunk_article_sets_metadata():
    article = ProcessedArticle(
        id="abc123",
        url="https://example.com/story",
        headline="Test headline",
        clean_body="word " * 30,
        summary="word word word",
        entities=["Test Corp"],
        category="technology",
        source="Example News",
        published_at=datetime(2026, 3, 1, tzinfo=timezone.utc),
    )
    chunks = chunk_article(article)
    assert len(chunks) == 1
    c = chunks[0]
    assert c.article_id == "abc123"
    assert c.id == "abc123_0"
    assert c.url == "https://example.com/story"
    assert c.published_at == "2026-03-01"
