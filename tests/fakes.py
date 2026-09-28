"""Fake Cohere client so the RAG pipeline can be unit-tested without a
real API key or network access. Mimics the shapes of cohere.ClientV2's
embed/chat/rerank responses that the rest of the code relies on (verified
against the installed cohere==7.1.1 SDK's response types)."""

from __future__ import annotations

import re

_VOCAB_DIM = 64


def _hash_embed(text: str) -> list[float]:
    """Deterministic bag-of-words-via-hashing 'embedding' -- good enough
    for testing that similar text scores higher than dissimilar text,
    without needing a real model."""
    vec = [0.0] * _VOCAB_DIM
    for word in re.findall(r"[a-z0-9]+", text.lower()):
        vec[hash(word) % _VOCAB_DIM] += 1.0
    norm = sum(v * v for v in vec) ** 0.5
    return [v / norm for v in vec] if norm else vec


class FakeEmbeddings:
    def __init__(self, texts: list[str]):
        self.float_ = [_hash_embed(t) for t in texts]


class FakeEmbedResponse:
    def __init__(self, texts: list[str]):
        self.embeddings = FakeEmbeddings(texts)


class FakeRerankResultItem:
    def __init__(self, index: int, relevance_score: float):
        self.index = index
        self.relevance_score = relevance_score


class FakeRerankResponse:
    def __init__(self, results):
        self.results = results


class FakeCitationSource:
    def __init__(self, doc_id: str, document: dict):
        self.type = "document"
        self.id = doc_id
        self.document = document


class FakeCitation:
    def __init__(self, text: str, sources):
        self.text = text
        self.sources = sources


class FakeContentItem:
    def __init__(self, text: str):
        self.text = text


class FakeAssistantMessage:
    def __init__(self, content, citations):
        self.role = "assistant"
        self.content = content
        self.citations = citations


class FakeChatResponse:
    def __init__(self, message):
        self.message = message


class FakeCohereClient:
    """Drop-in fake for cohere.ClientV2 covering .embed, .chat, .rerank."""

    def __init__(self):
        self.chat_calls = []
        self.rerank_calls = []
        self.embed_calls = []

    def embed(self, *, model, input_type, texts, embedding_types=None, **kwargs):
        self.embed_calls.append({"model": model, "input_type": input_type, "texts": texts})
        return FakeEmbedResponse(texts)

    def rerank(self, *, model, query, documents, top_n=None, **kwargs):
        self.rerank_calls.append({"model": model, "query": query, "documents": documents})
        query_words = set(re.findall(r"[a-z0-9]+", query.lower()))
        scored = []
        for i, doc in enumerate(documents):
            doc_words = set(re.findall(r"[a-z0-9]+", doc.lower()))
            overlap = len(query_words & doc_words)
            score = min(0.99, 0.2 + 0.15 * overlap)
            scored.append((i, score))
        scored.sort(key=lambda t: t[1], reverse=True)
        top_n = top_n or len(scored)
        results = [FakeRerankResultItem(i, s) for i, s in scored[:top_n]]
        return FakeRerankResponse(results)

    def chat(self, *, model, messages, documents=None, citation_options=None, **kwargs):
        self.chat_calls.append({"model": model, "messages": messages, "documents": documents})
        if not documents:
            content = [FakeContentItem("I don't have news on that.")]
            return FakeChatResponse(FakeAssistantMessage(content, citations=[]))

        first = documents[0]
        text = f"Based on the retrieved articles: {first.data.get('headline', '')}."
        citation = FakeCitation(
            text=text,
            sources=[FakeCitationSource(first.id, dict(first.data))],
        )
        content = [FakeContentItem(text)]
        return FakeChatResponse(FakeAssistantMessage(content, citations=[citation]))
