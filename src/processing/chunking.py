"""Split cleaned article text into ~300-500 token chunks.

Token counts aren't computed exactly (that would require the model's
tokenizer); word count is used as a proxy at roughly 0.75 words/token,
which maps the 300-500 token target in the brief to ~220-380 words. This
is called out in README so it isn't mistaken for an exact token count.
"""

from __future__ import annotations

from src import config
from src.models import Chunk, ProcessedArticle


def chunk_text(text: str, min_words: int, max_words: int, overlap_words: int) -> list[str]:
    words = text.split()
    if not words:
        return []
    if len(words) <= max_words:
        return [text]

    chunks = []
    start = 0
    step = max(max_words - overlap_words, 1)
    while start < len(words):
        end = min(start + max_words, len(words))
        chunk_words = words[start:end]
        if len(chunk_words) < min_words and chunks:
            # Too small a tail chunk -- merge into the previous one instead
            # of emitting a near-empty fragment.
            chunks[-1] = chunks[-1] + " " + " ".join(chunk_words)
            break
        chunks.append(" ".join(chunk_words))
        if end == len(words):
            break
        start += step
    return chunks


def chunk_article(article: ProcessedArticle) -> list[Chunk]:
    pieces = chunk_text(
        article.clean_body,
        config.CHUNK_MIN_WORDS,
        config.CHUNK_MAX_WORDS,
        config.CHUNK_OVERLAP_WORDS,
    )
    published = article.published_at.date().isoformat() if article.published_at else None
    chunks = []
    for i, piece in enumerate(pieces):
        chunks.append(
            Chunk(
                id=f"{article.id}_{i}",
                article_id=article.id,
                text=piece,
                headline=article.headline,
                url=article.url,
                source=article.source,
                category=article.category,
                published_at=published,
                chunk_index=i,
            )
        )
    return chunks
