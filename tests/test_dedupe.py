from datetime import datetime, timezone

from src.models import RawArticle
from src.processing.dedupe import dedupe_near_duplicates


def make_article(url, headline, body, published_at=None):
    return RawArticle(
        url=url,
        headline=headline,
        body=body,
        source="test",
        category="finance",
        published_at=published_at,
    )


def test_exact_duplicate_headline_across_outlets_is_dropped():
    articles = [
        make_article(
            "https://a.com/1",
            "RBI cuts repo rate by 25 basis points",
            "The Reserve Bank of India cut the repo rate by 25 basis points on Wednesday, citing easing inflation.",
            datetime(2026, 1, 1, tzinfo=timezone.utc),
        ),
        make_article(
            "https://b.com/1",
            "RBI cuts repo rate by 25 basis points",
            "The Reserve Bank of India cut the repo rate by 25 basis points on Wednesday, citing easing inflation.",
            datetime(2026, 1, 1, 10, tzinfo=timezone.utc),
        ),
    ]
    result = dedupe_near_duplicates(articles)
    assert len(result) == 1
    assert result[0].url == "https://a.com/1"  # earliest-published copy wins


def test_unrelated_articles_are_both_kept():
    articles = [
        make_article("https://a.com/1", "RBI cuts rates", "Central bank policy news about interest rates."),
        make_article("https://b.com/2", "Startup raises funding", "A tech startup announced a new funding round today."),
    ]
    result = dedupe_near_duplicates(articles)
    assert len(result) == 2


def test_empty_input_returns_empty():
    assert dedupe_near_duplicates([]) == []
