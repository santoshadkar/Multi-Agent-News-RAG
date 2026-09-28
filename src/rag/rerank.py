"""Step 3b: cross-encoder rerank via Cohere's rerank endpoint.

Reranking after hybrid fusion catches cases where the embedding + BM25
blend still let a topically-similar-but-not-quite-relevant chunk through;
a cross-encoder scores query and passage jointly, which is more accurate
but too slow/expensive to run over the whole candidate pool up front.
"""

from __future__ import annotations

import cohere

from src import config


def rerank_chunks(client: cohere.ClientV2, query: str, chunks: list[dict], top_n: int) -> list[dict]:
    if not chunks:
        return []
    texts = [c["text"] for c in chunks]
    resp = client.rerank(model=config.RERANK_MODEL, query=query, documents=texts, top_n=min(top_n, len(texts)))
    out = []
    for item in resp.results:
        chunk = dict(chunks[item.index])
        chunk["rerank_score"] = item.relevance_score
        out.append(chunk)
    return out
