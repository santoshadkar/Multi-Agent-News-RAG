"""Thin wrapper around Cohere embeddings (v2 client).

The client is injected rather than constructed internally so tests can pass
a fake client and never touch the network. See tests/conftest.py.
"""

from __future__ import annotations

import cohere

from src import config


def get_cohere_client() -> cohere.ClientV2:
    if not config.COHERE_API_KEY:
        raise RuntimeError(
            "COHERE_API_KEY is not set. Copy .env.example to .env, add your "
            "key from https://dashboard.cohere.com/api-keys, and export it "
            "into your shell before running the pipeline."
        )
    return cohere.ClientV2(api_key=config.COHERE_API_KEY)


class EmbeddingClient:
    def __init__(self, client: cohere.ClientV2, model: str = config.EMBED_MODEL):
        self.client = client
        self.model = model

    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        if not texts:
            return []
        resp = self.client.embed(
            model=self.model,
            input_type="search_document",
            texts=texts,
            embedding_types=["float"],
        )
        return resp.embeddings.float_

    def embed_query(self, text: str) -> list[float]:
        resp = self.client.embed(
            model=self.model,
            input_type="search_query",
            texts=[text],
            embedding_types=["float"],
        )
        return resp.embeddings.float_[0]
